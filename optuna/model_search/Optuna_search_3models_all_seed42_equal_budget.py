# ============================================================
# Equal-budget Optuna search for:
#   1. TransformerFixed
#   2. iTransformer
#   3. HybridHQ
#
# Fair tuning policy:
# - tuning dataset      : all-season only
# - tuning seed         : 42
# - target split        : strict warm 80/10/10
# - objective           : validation MSE in standardized scale
# - total trial budget  : same for every model
# - max epochs/patience : same for every model
# - sampler/pruner      : same for every model
# - input information   : past demand + weather + calendar
# - test set            : NOT used during tuning
#
# NOTE:
# Architecture-specific search spaces are allowed.
# Equal effort does not mean identical architecture parameters.
#
# Install:
#   pip install optuna
#
# Run all three:
#   python Optuna_search_3models_all_seed42_equal_budget.py
# ============================================================

import os
import gc
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "Optunaがありません。仮想環境で `pip install optuna` を実行してください。"
    ) from exc


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]  # project root (clean_code/)
ALL_DATA_FILE = BASE_DIR / "data" / "demand_holiday_with_solar_mean.csv"

OUTPUT_DIR = BASE_DIR / "result" / "result_csv" / "optuna_equal_budget_seed42"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# このリストを1モデルだけにすれば単独探索にも使えます。
MODELS_TO_TUNE = [
    "TransformerFixed",
    "iTransformer",
    "HybridHQ",
]

TUNING_SEED = 42
OPTUNA_SAMPLER_SEED = 42

# 各モデルの「総trial数」です。
# SQLiteが残っている場合、既存trialを数えて不足分だけ追加します。
TOTAL_TRIALS_PER_MODEL = 30

MAX_EPOCHS = 80
EARLY_PATIENCE = 10

SEQ_LEN = 72
HORIZON = 24
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

MAX_GRAD_NORM = 1.0
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

# MedianPrunerも全モデルで統一
PRUNER_STARTUP_TRIALS = 5
PRUNER_WARMUP_EPOCHS = 10

# Hybridの構造上の条件は、提案モデルのFull構成として固定
HYBRID_CNN_MODE = "linear"
HYBRID_HQ_COND_MODE = "shared"
HYBRID_USE_Q_SKIP = True
HYBRID_FILM_USE_TANH = True
HYBRID_HQ_L2_WEIGHT = 0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(TUNING_SEED)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        generator=generator if shuffle else None,
    )


# ============================================================
# Dataset
# ============================================================

class SequenceDataset(Dataset):
    def __init__(
        self,
        X_full: np.ndarray,
        X_exog: np.ndarray,
        y_past: np.ndarray,
        y_future: np.ndarray,
    ):
        self.X_full = torch.tensor(X_full, dtype=torch.float32)
        self.X_exog = torch.tensor(X_exog, dtype=torch.float32)
        self.y_past = torch.tensor(y_past, dtype=torch.float32)
        self.y_future = torch.tensor(y_future, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y_future)

    def __getitem__(self, idx: int):
        return (
            self.X_full[idx],
            self.X_exog[idx],
            self.y_past[idx],
            self.y_future[idx],
        )


# ============================================================
# Shared data preparation
# ============================================================

def infer_step_minutes(df: pd.DataFrame) -> Tuple[int, int]:
    per_day = int(
        df.groupby(df["datetime"].dt.date)["time"]
        .nunique()
        .mode()
        .iloc[0]
    )

    if 1440 % per_day == 0:
        step_minutes = 1440 // per_day
    else:
        step_minutes = 60 if int(df["time"].max()) <= 23 else 30

    return per_day, step_minutes


def build_sequences(
    full_points: np.ndarray,
    exog_points: np.ndarray,
    target_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_full, X_exog, y_past, y_future = [], [], [], []

    n_sequences = len(target_points) - SEQ_LEN - HORIZON + 1

    for start in range(n_sequences):
        target_start = start + SEQ_LEN
        target_end = target_start + HORIZON

        X_full.append(full_points[start:target_start])
        X_exog.append(exog_points[start:target_start])
        y_past.append(target_points[start:target_start])
        y_future.append(target_points[target_start:target_end].reshape(-1))

    return (
        np.asarray(X_full, dtype=np.float32),
        np.asarray(X_exog, dtype=np.float32),
        np.asarray(y_past, dtype=np.float32),
        np.asarray(y_future, dtype=np.float32),
    )


def prepare_all_season_data() -> Dict[str, object]:
    if not ALL_DATA_FILE.exists():
        raise FileNotFoundError(f"all-season CSVがありません: {ALL_DATA_FILE}")

    df = pd.read_csv(ALL_DATA_FILE)

    required = [
        "datetime",
        "time",
        "power_demand",
        "temperature",
        "precipitation",
        "solar_mean",
        "restday",
        "holiday_onehot",
    ]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"CSVに必要な列がありません: {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["datetime", "time"]).reset_index(drop=True)

    per_day, step_minutes = infer_step_minutes(df)

    has_time_component = (
        (df["datetime"].dt.hour != 0)
        | (df["datetime"].dt.minute != 0)
        | (df["datetime"].dt.second != 0)
    ).any()

    if has_time_component:
        df["timestamp"] = df["datetime"]
    else:
        df["timestamp"] = (
            df["datetime"]
            + pd.to_timedelta(df["time"] * step_minutes, unit="m")
        )

    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday

    df = df.dropna().reset_index(drop=True)
    df = pd.get_dummies(
        df,
        columns=["weekday", "month"],
        drop_first=True,
    )

    exog_features = [
        "temperature",
        "precipitation",
        "time",
        "day",
        "solar_mean",
        "restday",
        "holiday_onehot",
    ] + [
        column
        for column in df.columns
        if column.startswith("weekday_") or column.startswith("month_")
    ]

    target = "power_demand"

    # 全モデルに与える情報を統一:
    # TransformerFixed / iTransformer:
    #   past demand + exogenous variables
    # HybridHQ:
    #   past demand branch + same exogenous variables
    full_features = [target] + exog_features

    n_points = len(df)
    train_end = int(n_points * TRAIN_RATIO)
    val_end = int(n_points * (TRAIN_RATIO + VAL_RATIO))

    scaler_full = StandardScaler().fit(df.iloc[:train_end][full_features])
    scaler_exog = StandardScaler().fit(df.iloc[:train_end][exog_features])
    scaler_y = StandardScaler().fit(df.iloc[:train_end][[target]])

    full_points = scaler_full.transform(df[full_features])
    exog_points = scaler_exog.transform(df[exog_features])
    target_points = scaler_y.transform(df[[target]])

    X_full, X_exog, y_past, y_future = build_sequences(
        full_points=full_points,
        exog_points=exog_points,
        target_points=target_points,
    )

    starts = np.arange(len(y_future))
    y_start = starts + SEQ_LEN
    y_end = y_start + HORIZON - 1

    # target windowが境界をまたがないstrict split
    train_mask = y_end < train_end
    val_mask = (y_start >= train_end) & (y_end < val_end)
    test_mask = (y_start >= val_end) & (y_end < n_points)

    data = {
        "train": (
            X_full[train_mask],
            X_exog[train_mask],
            y_past[train_mask],
            y_future[train_mask],
        ),
        "val": (
            X_full[val_mask],
            X_exog[val_mask],
            y_past[val_mask],
            y_future[val_mask],
        ),
        # testは探索には使用しません。件数確認用だけ保持します。
        "test": (
            X_full[test_mask],
            X_exog[test_mask],
            y_past[test_mask],
            y_future[test_mask],
        ),
        "full_dim": int(X_full.shape[2]),
        "exog_dim": int(X_exog.shape[2]),
        "full_features": full_features,
        "exog_features": exog_features,
        "per_day": int(per_day),
        "step_minutes": int(step_minutes),
        "has_time_component": bool(has_time_component),
        "train_end": int(train_end),
        "val_end": int(val_end),
    }

    train_n = len(data["train"][3])
    val_n = len(data["val"][3])
    test_n = len(data["test"][3])

    if min(train_n, val_n, test_n) <= 0:
        raise ValueError(
            f"sequence数が不正です: train={train_n}, val={val_n}, test={test_n}"
        )

    print(
        f"[DATA] train/val/test sequences = {train_n}/{val_n}/{test_n}"
    )
    print(
        f"[DATA] full_dim={data['full_dim']}, exog_dim={data['exog_dim']}"
    )
    print(
        "[DATA] feature policy = past demand + weather + calendar "
        "(explicit lag_1/24/48 is not added)"
    )

    return data


# ============================================================
# Common layers
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        max_len: int = 5000,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        encoding[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            encoding[:, 1::2] = torch.cos(position * div_term)
        else:
            encoding[:, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            x + self.encoding[:, :x.size(1), :]
        )


# ============================================================
# TransformerFixed
# ============================================================

class TransformerFixed(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        self.query_embed = nn.Parameter(
            torch.randn(HORIZON, d_model)
        )
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        memory = self.encoder(
            self.positional_encoding(
                self.input_projection(x)
            )
        )

        query = self.query_embed.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

        decoded = self.decoder(
            tgt=query,
            memory=memory,
        )

        return self.output_projection(decoded).squeeze(-1)


# ============================================================
# iTransformer
# ============================================================

class VariableEmbedding(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        input_dim: int,
    ):
        super().__init__()
        self.projection = nn.Linear(seq_len, d_model)
        self.variable_embedding = nn.Parameter(
            torch.randn(1, input_dim, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x + self.variable_embedding[:, :x.size(1)]


class iTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        nhead: int,
        dropout: float,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        transformed = self.feed_forward(x)
        return self.norm2(x + self.dropout(transformed))


class iTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.embedding = VariableEmbedding(
            seq_len=SEQ_LEN,
            d_model=d_model,
            input_dim=input_dim,
        )

        self.blocks = nn.Sequential(*[
            iTransformerBlock(
                d_model=d_model,
                hidden_dim=hidden_dim,
                nhead=nhead,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(d_model, HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.blocks(x)
        return self.head(x.mean(dim=1))


# ============================================================
# HybridHQ Full model
# ============================================================

class TemporalEmbeddingLinear(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
    ):
        super().__init__()
        self.projection = nn.Linear(1, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.activation(
                self.projection(x)
            )
        )


class VariableAxisTransformer(nn.Module):
    def __init__(
        self,
        exog_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.projection = nn.Linear(SEQ_LEN, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        x = self.positional_encoding(x)
        return self.encoder(x)


class HybridHQ(nn.Module):
    def __init__(
        self,
        exog_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.d_model = d_model

        self.temporal_embedding = TemporalEmbeddingLinear(
            d_model=d_model,
            dropout=dropout,
        )
        self.temporal_positional_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
        )

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=num_layers,
        )

        self.variable_encoder = VariableAxisTransformer(
            exog_dim=exog_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.horizon_queries = nn.Parameter(
            torch.randn(1, HORIZON, d_model)
        )

        if HYBRID_HQ_COND_MODE == "shared":
            self.query_condition = nn.Linear(
                d_model,
                d_model,
            )
        else:
            self.query_condition = nn.Linear(
                d_model,
                HORIZON * d_model,
            )

        self.cross_time = nn.MultiheadAttention(
            d_model,
            nhead,
            batch_first=True,
            dropout=dropout,
        )
        self.cross_variable = nn.MultiheadAttention(
            d_model,
            nhead,
            batch_first=True,
            dropout=dropout,
        )

        self.norm_time = nn.LayerNorm(d_model)
        self.norm_variable = nn.LayerNorm(d_model)

        self.branch_ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

        self.norm_time_output = nn.LayerNorm(d_model)
        self.norm_variable_output = nn.LayerNorm(d_model)

        self.film_time_to_variable = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * d_model),
        )
        self.film_variable_to_time = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * d_model),
        )

        self.fusion_projection = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.fusion_ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm_fusion = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, 1)
        self.residual_projection = nn.Linear(SEQ_LEN, HORIZON)

    @staticmethod
    def transform_gamma(gamma: torch.Tensor) -> torch.Tensor:
        if HYBRID_FILM_USE_TANH:
            return torch.tanh(gamma)
        return torch.sigmoid(gamma)

    def forward(
        self,
        x_exog: torch.Tensor,
        y_past: torch.Tensor,
    ) -> torch.Tensor:
        temporal_memory = self.temporal_encoder(
            self.temporal_positional_encoding(
                self.temporal_embedding(y_past)
            )
        )

        variable_memory = self.variable_encoder(x_exog)

        batch_size = y_past.size(0)
        query = self.horizon_queries.expand(
            batch_size,
            -1,
            -1,
        ).clone()

        condition_source = temporal_memory[:, -1, :]

        if HYBRID_HQ_COND_MODE == "shared":
            condition = (
                self.query_condition(condition_source)
                .unsqueeze(1)
                .expand(batch_size, HORIZON, -1)
            )
        else:
            condition = self.query_condition(
                condition_source
            ).view(batch_size, HORIZON, self.d_model)

        query = query + condition

        time_attended, _ = self.cross_time(
            query,
            temporal_memory,
            temporal_memory,
        )
        variable_attended, _ = self.cross_variable(
            query,
            variable_memory,
            variable_memory,
        )

        time_representation = self.norm_time(
            query + time_attended
        )
        variable_representation = self.norm_variable(
            query + variable_attended
        )

        if HYBRID_USE_Q_SKIP:
            time_representation = time_representation + query
            variable_representation = variable_representation + query

        time_feature = self.norm_time_output(
            time_representation
            + self.branch_ffn(time_representation)
        )
        variable_feature = self.norm_variable_output(
            variable_representation
            + self.branch_ffn(variable_representation)
        )

        time_to_variable = self.film_time_to_variable(
            time_feature
        )
        gamma_tv_raw, beta_tv = time_to_variable.chunk(
            2,
            dim=-1,
        )
        gamma_tv = self.transform_gamma(gamma_tv_raw)
        variable_modulated = (
            variable_feature * (1.0 + gamma_tv) + beta_tv
        )

        variable_to_time = self.film_variable_to_time(
            variable_feature
        )
        gamma_vt_raw, beta_vt = variable_to_time.chunk(
            2,
            dim=-1,
        )
        gamma_vt = self.transform_gamma(gamma_vt_raw)
        time_modulated = (
            time_feature * (1.0 + gamma_vt) + beta_vt
        )

        fused = self.fusion_projection(
            torch.cat(
                [time_modulated, variable_modulated],
                dim=-1,
            )
        )
        fused = self.norm_fusion(
            fused + self.fusion_ffn(fused)
        )

        prediction_delta = self.head(fused).squeeze(-1)
        residual = self.residual_projection(
            y_past.squeeze(-1)
        )

        return residual + prediction_delta


# ============================================================
# Search spaces
# ============================================================

def suggest_common_training_params(
    trial: optuna.Trial,
) -> Dict[str, object]:
    return {
        "dropout": trial.suggest_float(
            "dropout",
            0.0,
            0.30,
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size",
            [32, 64, 128],
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-5,
            1e-3,
            log=True,
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            1e-7,
            1e-2,
            log=True,
        ),
        "optimizer_name": trial.suggest_categorical(
            "optimizer_name",
            ["Adam", "AdamW"],
        ),
    }


def suggest_attention_dimensions(
    trial: optuna.Trial,
) -> Tuple[int, int]:
    # 無効な d_model % nhead を防ぐためペアで選択
    pair = trial.suggest_categorical(
        "d_model_nhead",
        [
            "64_4",
            "64_8",
            "128_4",
            "128_8",
            "128_16",
            "256_4",
            "256_8",
            "256_16",
        ],
    )

    d_model, nhead = map(int, pair.split("_"))
    return d_model, nhead


def suggest_model_params(
    model_name: str,
    trial: optuna.Trial,
) -> Dict[str, object]:
    params = suggest_common_training_params(trial)
    d_model, nhead = suggest_attention_dimensions(trial)

    params["d_model"] = d_model
    params["nhead"] = nhead

    if model_name == "TransformerFixed":
        params.update({
            "num_encoder_layers": trial.suggest_int(
                "num_encoder_layers",
                1,
                4,
            ),
            "num_decoder_layers": trial.suggest_int(
                "num_decoder_layers",
                1,
                4,
            ),
            "dim_feedforward": trial.suggest_categorical(
                "dim_feedforward",
                [256, 512, 1024],
            ),
        })

    elif model_name == "iTransformer":
        params.update({
            "num_layers": trial.suggest_int(
                "num_layers",
                1,
                4,
            ),
            "hidden_dim": trial.suggest_categorical(
                "hidden_dim",
                [256, 512, 1024],
            ),
        })

    elif model_name == "HybridHQ":
        params.update({
            "num_layers": trial.suggest_int(
                "num_layers",
                1,
                3,
            ),
            "hidden_dim": trial.suggest_categorical(
                "hidden_dim",
                [256, 512, 1024],
            ),
        })

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return params


# ============================================================
# Build model / optimizer
# ============================================================

def build_model(
    model_name: str,
    data: Dict[str, object],
    params: Dict[str, object],
) -> nn.Module:
    if model_name == "TransformerFixed":
        return TransformerFixed(
            input_dim=int(data["full_dim"]),
            d_model=int(params["d_model"]),
            nhead=int(params["nhead"]),
            num_encoder_layers=int(params["num_encoder_layers"]),
            num_decoder_layers=int(params["num_decoder_layers"]),
            dim_feedforward=int(params["dim_feedforward"]),
            dropout=float(params["dropout"]),
        )

    if model_name == "iTransformer":
        return iTransformer(
            input_dim=int(data["full_dim"]),
            d_model=int(params["d_model"]),
            nhead=int(params["nhead"]),
            num_layers=int(params["num_layers"]),
            hidden_dim=int(params["hidden_dim"]),
            dropout=float(params["dropout"]),
        )

    if model_name == "HybridHQ":
        return HybridHQ(
            exog_dim=int(data["exog_dim"]),
            d_model=int(params["d_model"]),
            nhead=int(params["nhead"]),
            num_layers=int(params["num_layers"]),
            hidden_dim=int(params["hidden_dim"]),
            dropout=float(params["dropout"]),
        )

    raise ValueError(f"Unknown model: {model_name}")


def build_optimizer(
    model: nn.Module,
    params: Dict[str, object],
):
    optimizer_class = getattr(
        optim,
        str(params["optimizer_name"]),
    )

    return optimizer_class(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )


def forward_model(
    model_name: str,
    model: nn.Module,
    batch,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X_full, X_exog, y_past, y_future = batch

    X_full = X_full.to(device, non_blocking=True)
    X_exog = X_exog.to(device, non_blocking=True)
    y_past = y_past.to(device, non_blocking=True)
    y_future = y_future.to(device, non_blocking=True)

    if model_name == "TransformerFixed":
        prediction = model(X_full)
    elif model_name == "iTransformer":
        prediction = model(X_full)
    elif model_name == "HybridHQ":
        prediction = model(X_exog, y_past)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return prediction, y_future


# ============================================================
# Objective
# ============================================================

def create_objective(
    model_name: str,
    data: Dict[str, object],
):
    train_dataset = SequenceDataset(*data["train"])
    val_dataset = SequenceDataset(*data["val"])

    criterion = nn.MSELoss()

    def objective(trial: optuna.Trial) -> float:
        # すべてのtrialを同じseed・shuffle順で比較
        seed_everything(TUNING_SEED)

        params = suggest_model_params(
            model_name=model_name,
            trial=trial,
        )

        train_loader = make_loader(
            train_dataset,
            batch_size=int(params["batch_size"]),
            shuffle=True,
        )
        val_loader = make_loader(
            val_dataset,
            batch_size=int(params["batch_size"]),
            shuffle=False,
        )

        model = build_model(
            model_name=model_name,
            data=data,
            params=params,
        ).to(device)

        optimizer = build_optimizer(
            model=model,
            params=params,
        )

        best_validation = float("inf")
        bad_epochs = 0

        try:
            for epoch in range(1, MAX_EPOCHS + 1):
                model.train()

                for batch in train_loader:
                    optimizer.zero_grad(set_to_none=True)

                    prediction, target = forward_model(
                        model_name=model_name,
                        model=model,
                        batch=batch,
                    )

                    loss = criterion(prediction, target)

                    if model_name == "HybridHQ" and HYBRID_HQ_L2_WEIGHT > 0:
                        loss = loss + (
                            HYBRID_HQ_L2_WEIGHT
                            * model.horizon_queries.pow(2).mean()
                        )

                    if not torch.isfinite(loss):
                        raise optuna.TrialPruned(
                            "Non-finite training loss"
                        )

                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        MAX_GRAD_NORM,
                    )
                    optimizer.step()

                model.eval()
                validation_losses = []

                with torch.no_grad():
                    for batch in val_loader:
                        prediction, target = forward_model(
                            model_name=model_name,
                            model=model,
                            batch=batch,
                        )
                        validation_losses.append(
                            float(
                                criterion(
                                    prediction,
                                    target,
                                ).item()
                            )
                        )

                validation_loss = float(
                    np.mean(validation_losses)
                )

                trial.report(
                    validation_loss,
                    step=epoch,
                )

                if trial.should_prune():
                    raise optuna.TrialPruned()

                if validation_loss < best_validation:
                    best_validation = validation_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                    if bad_epochs >= EARLY_PATIENCE:
                        break

            trial.set_user_attr(
                "best_validation_loss",
                float(best_validation),
            )
            trial.set_user_attr(
                "tuning_seed",
                int(TUNING_SEED),
            )
            trial.set_user_attr(
                "dataset",
                "all",
            )
            trial.set_user_attr(
                "feature_policy",
                "past_demand+weather+calendar",
            )

            return float(best_validation)

        finally:
            del model
            del optimizer
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return objective


# ============================================================
# Study runner
# ============================================================

def count_existing_trials(study: optuna.Study) -> int:
    # COMPLETE/PRUNED/FAILを使用済みbudgetとして数える
    return len([
        trial
        for trial in study.trials
        if trial.state in {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
        }
    ])


def normalize_best_params(
    params: Dict[str, object],
) -> Dict[str, object]:
    params = dict(params)

    pair = params.pop("d_model_nhead", None)

    if pair is not None:
        d_model, nhead = map(int, str(pair).split("_"))
        params["d_model"] = d_model
        params["nhead"] = nhead

    return params


def run_model_search(
    model_name: str,
    data: Dict[str, object],
) -> Dict[str, object]:
    database_path = OUTPUT_DIR / f"{model_name}_optuna.db"
    storage = f"sqlite:///{database_path.as_posix()}"

    sampler = optuna.samplers.TPESampler(
        seed=OPTUNA_SAMPLER_SEED,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=PRUNER_STARTUP_TRIALS,
        n_warmup_steps=PRUNER_WARMUP_EPOCHS,
    )

    study = optuna.create_study(
        study_name=(
            f"{model_name}_all_seed{TUNING_SEED}_"
            f"equal_budget"
        ),
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    used_trials = count_existing_trials(study)
    remaining_trials = max(
        0,
        TOTAL_TRIALS_PER_MODEL - used_trials,
    )

    print("\n" + "=" * 110)
    print(f"MODEL             : {model_name}")
    print(f"TUNING DATA       : all")
    print(f"TUNING SEED       : {TUNING_SEED}")
    print(f"TOTAL TRIAL BUDGET: {TOTAL_TRIALS_PER_MODEL}")
    print(f"EXISTING TRIALS   : {used_trials}")
    print(f"REMAINING TRIALS  : {remaining_trials}")
    print(f"MAX EPOCHS        : {MAX_EPOCHS}")
    print(f"PATIENCE          : {EARLY_PATIENCE}")
    print(f"OBJECTIVE         : validation MSE")
    print("=" * 110)

    if remaining_trials > 0:
        study.optimize(
            create_objective(
                model_name=model_name,
                data=data,
            ),
            n_trials=remaining_trials,
            gc_after_trial=True,
            show_progress_bar=True,
        )
    else:
        print("既に総trial budgetに達しているため追加探索を行いません。")

    best_params = normalize_best_params(
        study.best_params
    )

    best_json = OUTPUT_DIR / f"{model_name}_best_params_all_seed42.json"
    with best_json.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "model": model_name,
                "tuning_dataset": "all",
                "tuning_seed": TUNING_SEED,
                "total_trial_budget": TOTAL_TRIALS_PER_MODEL,
                "max_epochs": MAX_EPOCHS,
                "patience": EARLY_PATIENCE,
                "objective": "validation_mse_standardized",
                "feature_policy": "past_demand+weather+calendar",
                "best_value": float(study.best_value),
                "best_trial_number": int(study.best_trial.number),
                "best_params": best_params,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    trials_csv = OUTPUT_DIR / f"{model_name}_optuna_trials.csv"
    study.trials_dataframe().to_csv(
        trials_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"BEST VALUE : {study.best_value:.8f}")
    print(f"BEST TRIAL : {study.best_trial.number}")
    print(f"BEST PARAMS: {best_params}")
    print(f"JSON       : {best_json}")
    print(f"TRIAL CSV  : {trials_csv}")

    return {
        "model": model_name,
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_params_json": str(best_json),
        "trials_csv": str(trials_csv),
        **best_params,
    }


def run_search(
    model_names: List[str] = None,
) -> None:
    if model_names is None:
        model_names = MODELS_TO_TUNE

    allowed_models = {
        "TransformerFixed",
        "iTransformer",
        "HybridHQ",
    }

    unknown = [
        model_name
        for model_name in model_names
        if model_name not in allowed_models
    ]

    if unknown:
        raise ValueError(f"Unknown models: {unknown}")

    print(f"Using device: {device}")
    print(f"Models      : {model_names}")

    data = prepare_all_season_data()
    summaries = []

    for model_name in model_names:
        summaries.append(
            run_model_search(
                model_name=model_name,
                data=data,
            )
        )

    summary_df = pd.DataFrame(summaries)
    summary_path = OUTPUT_DIR / "THREE_MODELS_optuna_best_summary.csv"

    if summary_path.exists():
        old = pd.read_csv(summary_path)
        old = old[
            ~old["model"].isin(summary_df["model"])
        ]
        summary_df = pd.concat(
            [old, summary_df],
            ignore_index=True,
        )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 110)
    print("OPTUNA SEARCH FINISHED")
    print("=" * 110)
    print(summary_df.to_string(index=False))
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    run_search()
