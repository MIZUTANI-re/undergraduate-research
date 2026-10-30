# ============================================================
# PREE 2026 Reviewer Comment 1
# Equal-budget Optuna + multi-seed ablation study core
#
# Variants
#   full_bifilm   : HQ ON  / bidirectional FiLM ON / residual ON
#   simple_concat : HQ ON  / bidirectional FiLM OFF / residual ON
#   no_hq         : HQ OFF / bidirectional FiLM ON / residual ON
#   no_residual   : HQ ON  / bidirectional FiLM ON / residual OFF
#
# Search protocol (default)
#   dataset       : all-season only
#   tuning seed   : 42
#   split         : strict warm 80/10/10
#   input/output  : 72 h -> 24 h one-shot
#   objective     : standardized validation MSE
#   budget        : 30 trials / variant
#   max epoch     : 80
#   patience      : 10
#   sampler       : TPE(seed=42)
#   pruner        : MedianPruner
#   optimizer     : Adam / AdamW
#   test set      : never used during HP selection
#
# Final evaluation (default)
#   best HP fixed across all/spring/summer/autumn/winter
#   seeds = 42, 123, 2026
#   RMSE (MW), MAPE (%), mean +/- std
# ============================================================

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "Optunaがありません。仮想環境で `pip install optuna` を実行してください。"
    ) from exc


# ============================================================
# Fixed research settings
# ============================================================
SEQ_LEN = 72
HORIZON = 24
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

CNN_MODE = "linear"
HQ_COND_MODE = "shared"
USE_Q_SKIP = True
FILM_USE_TANH = True
HQ_L2_WEIGHT = 0.0
HQ_NOISE_STD = 0.0
MAX_GRAD_NORM = 1.0

PRUNER_STARTUP_TRIALS = 5
PRUNER_WARMUP_EPOCHS = 10

DEFAULT_BASE_DIR = Path(__file__).resolve().parents[2]  # project root (clean_code/)
DEFAULT_SEASONS = ["all", "spring", "summer", "autumn", "winter"]
DEFAULT_SEEDS = [42, 123, 2026]

VARIANTS = ("full_bifilm", "simple_concat", "no_hq", "no_residual")
VARIANT_LABELS = {
    "full_bifilm": "Full: HQ + bidirectional FiLM + linear residual",
    "simple_concat": "Ablation: HQ + simple concatenation + linear residual",
    "no_hq": "Ablation: no horizon queries + bidirectional FiLM + linear residual",
    "no_residual": "Ablation: HQ + bidirectional FiLM + no linear residual",
}


def variant_flags(variant: str) -> Dict[str, int]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    return {
        "film_01": int(variant != "simple_concat"),
        "hq_01": int(variant != "no_hq"),
        "residual_01": int(variant != "no_residual"),
    }


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
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=generator if shuffle else None,
    )


# ============================================================
# Dataset
# ============================================================
class TimeSeriesDatasetHQ(Dataset):
    def __init__(self, X_exog: np.ndarray, Y_past: np.ndarray, Y_future: np.ndarray):
        self.X_exog = torch.tensor(X_exog, dtype=torch.float32)
        self.Y_past = torch.tensor(Y_past, dtype=torch.float32)
        self.Y_future = torch.tensor(Y_future, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.Y_future)

    def __getitem__(self, index: int):
        return self.X_exog[index], self.Y_past[index], self.Y_future[index]


# ============================================================
# Model components
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class TemporalEmbeddingLinear(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VariableAxisTransformer(nn.Module):
    """iTransformer-style variable-token encoder."""

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.projection = nn.Linear(seq_len, d_model)
        self.position = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, seq, variables) -> (B, variables, seq) -> (B, variables, d)
        x = x.permute(0, 2, 1)
        return self.encoder(self.position(self.projection(x)))


class AblationHybridTransformer(nn.Module):
    """Controlled HybridHQ ablation model."""

    def __init__(
        self,
        variant: str,
        seq_len: int,
        exog_dim: int,
        horizon: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        del exog_dim  # Linear(seq_len -> d_model) does not require it explicitly.

        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        self.variant = variant
        self.seq_len = seq_len
        self.horizon = horizon
        self.d_model = d_model
        self.use_hq = variant != "no_hq"
        self.use_film = variant != "simple_concat"
        self.use_residual = variant != "no_residual"

        # Temporal branch: past demand only.
        self.temporal_embedding = TemporalEmbeddingLinear(d_model, dropout)
        self.temporal_position = PositionalEncoding(d_model, dropout)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=num_layers
        )

        # Variable branch: weather + calendar variables.
        self.variable_encoder = VariableAxisTransformer(
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.branch_ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm_time_output = nn.LayerNorm(d_model)
        self.norm_variable_output = nn.LayerNorm(d_model)

        if self.use_hq:
            self.horizon_queries = nn.Parameter(torch.randn(1, horizon, d_model))
            if HQ_COND_MODE == "per_h":
                self.query_condition_h = nn.Linear(d_model, horizon * d_model)
            else:
                self.query_condition = nn.Linear(d_model, d_model)

            self.cross_time = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )
            self.cross_variable = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )
            self.norm_time = nn.LayerNorm(d_model)
            self.norm_variable = nn.LayerNorm(d_model)

        if self.use_film:
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

        if self.use_hq:
            self.hq_head = nn.Linear(d_model, 1)
        else:
            # No HQ: one pooled global representation -> 24 one-shot outputs.
            self.no_hq_head = nn.Linear(d_model, horizon)

        if self.use_residual:
            self.residual_projection = nn.Linear(seq_len, horizon)

    @staticmethod
    def _gamma_transform(gamma_raw: torch.Tensor) -> torch.Tensor:
        if FILM_USE_TANH:
            return torch.tanh(gamma_raw)
        return torch.sigmoid(gamma_raw)

    @staticmethod
    def _gamma_to_alpha(gamma: torch.Tensor) -> torch.Tensor:
        if FILM_USE_TANH:
            return ((gamma + 1.0) * 0.5).mean(dim=-1)
        return gamma.mean(dim=-1)

    def _fuse(
        self,
        time_feature: torch.Tensor,
        variable_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        auxiliary: Dict[str, torch.Tensor] = {}

        if self.use_film:
            tv = self.film_time_to_variable(time_feature)
            gamma_tv_raw, beta_tv = tv.chunk(2, dim=-1)
            gamma_tv = self._gamma_transform(gamma_tv_raw)
            variable_modulated = variable_feature * (1.0 + gamma_tv) + beta_tv

            vt = self.film_variable_to_time(variable_feature)
            gamma_vt_raw, beta_vt = vt.chunk(2, dim=-1)
            gamma_vt = self._gamma_transform(gamma_vt_raw)
            time_modulated = time_feature * (1.0 + gamma_vt) + beta_vt

            auxiliary["alpha_tv"] = self._gamma_to_alpha(gamma_tv)
            auxiliary["alpha_vt"] = self._gamma_to_alpha(gamma_vt)
        else:
            # Simple concatenation: no gamma/beta and no cross-modulation.
            time_modulated = time_feature
            variable_modulated = variable_feature

        fused = self.fusion_projection(
            torch.cat([time_modulated, variable_modulated], dim=-1)
        )
        fused = self.norm_fusion(fused + self.fusion_ffn(fused))
        return fused, auxiliary

    def forward(
        self,
        x_exog: torch.Tensor,
        y_past: torch.Tensor,
        return_aux: bool = False,
    ):
        temporal_memory = self.temporal_encoder(
            self.temporal_position(self.temporal_embedding(y_past))
        )
        variable_memory = self.variable_encoder(x_exog)
        batch_size = y_past.size(0)

        if self.use_hq:
            query = self.horizon_queries.expand(batch_size, -1, -1).clone()
            condition_source = temporal_memory[:, -1, :]

            if HQ_COND_MODE == "per_h":
                condition = self.query_condition_h(condition_source).view(
                    batch_size, self.horizon, self.d_model
                )
            else:
                condition = self.query_condition(condition_source).unsqueeze(1).expand(
                    batch_size, self.horizon, self.d_model
                )
            query = query + condition

            if self.training and HQ_NOISE_STD > 0:
                query = query + torch.randn_like(query) * HQ_NOISE_STD

            time_attended, _ = self.cross_time(
                query, temporal_memory, temporal_memory
            )
            variable_attended, _ = self.cross_variable(
                query, variable_memory, variable_memory
            )

            time_representation = self.norm_time(query + time_attended)
            variable_representation = self.norm_variable(query + variable_attended)

            if USE_Q_SKIP:
                time_representation = time_representation + query
                variable_representation = variable_representation + query

            time_feature = self.norm_time_output(
                time_representation + self.branch_ffn(time_representation)
            )
            variable_feature = self.norm_variable_output(
                variable_representation + self.branch_ffn(variable_representation)
            )
            fused, auxiliary = self._fuse(time_feature, variable_feature)
            prediction_delta = self.hq_head(fused).squeeze(-1)
        else:
            # HQ is completely removed: no trainable horizon tokens and no query cross-attention.
            time_representation = temporal_memory[:, -1, :]
            variable_representation = variable_memory.mean(dim=1)
            time_feature = self.norm_time_output(
                time_representation + self.branch_ffn(time_representation)
            )
            variable_feature = self.norm_variable_output(
                variable_representation + self.branch_ffn(variable_representation)
            )
            fused, auxiliary = self._fuse(time_feature, variable_feature)
            prediction_delta = self.no_hq_head(fused)

        if self.use_residual:
            baseline = self.residual_projection(y_past.squeeze(-1))
            prediction = baseline + prediction_delta
        else:
            prediction = prediction_delta

        if return_aux:
            return prediction, auxiliary
        return prediction


# ============================================================
# Data preparation
# ============================================================
def infer_step_minutes(df: pd.DataFrame) -> Tuple[int, int]:
    per_day = int(
        df.groupby(df["datetime"].dt.date)["time"].nunique().mode().iloc[0]
    )
    if 1440 % per_day == 0:
        step_minutes = 1440 // per_day
    else:
        step_minutes = 60 if int(df["time"].max()) <= 23 else 30
    return per_day, step_minutes


def build_timestamp(df: pd.DataFrame) -> Tuple[pd.Series, bool, int, int]:
    datetime_series = pd.to_datetime(df["datetime"])
    has_time_component = bool(
        (
            (datetime_series.dt.hour != 0)
            | (datetime_series.dt.minute != 0)
            | (datetime_series.dt.second != 0)
        ).any()
    )
    per_day, step_minutes = infer_step_minutes(df)
    if has_time_component:
        timestamp = datetime_series
    else:
        timestamp = datetime_series + pd.to_timedelta(
            df["time"] * step_minutes, unit="m"
        )
    return timestamp, has_time_component, per_day, step_minutes


def make_sequences_hq(
    exog: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_exog: List[np.ndarray] = []
    Y_past: List[np.ndarray] = []
    Y_future: List[np.ndarray] = []

    for index in range(len(y) - seq_len - horizon + 1):
        X_exog.append(exog[index : index + seq_len])
        Y_past.append(y[index : index + seq_len])
        Y_future.append(
            y[index + seq_len : index + seq_len + horizon].reshape(-1)
        )

    return np.asarray(X_exog), np.asarray(Y_past), np.asarray(Y_future)


def prepare_data(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)
    required_columns = {
        "datetime",
        "time",
        "temperature",
        "precipitation",
        "solar_mean",
        "restday",
        "holiday_onehot",
        "power_demand",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {file_path}: {sorted(missing)}")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["datetime", "time"]).reset_index(drop=True)
    (
        df["timestamp"],
        has_time_component,
        per_day,
        step_minutes,
    ) = build_timestamp(df)

    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday
    df = pd.get_dummies(
        df.dropna(), columns=["weekday", "month"], drop_first=True
    ).reset_index(drop=True)

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
        if column.startswith(("weekday_", "month_"))
    ]

    target = "power_demand"
    n_rows = len(df)
    train_end = int(n_rows * TRAIN_RATIO)
    val_end = int(n_rows * (TRAIN_RATIO + VAL_RATIO))

    scaler_x = StandardScaler().fit(df.iloc[:train_end][exog_features])
    scaler_y = StandardScaler().fit(df.iloc[:train_end][[target]])

    X_all = scaler_x.transform(df[exog_features])
    y_all = scaler_y.transform(df[[target]])
    X_seq, Yp_seq, Yf_seq = make_sequences_hq(
        X_all, y_all, SEQ_LEN, HORIZON
    )

    # Strict warm split: every target window belongs fully to its split.
    train_i_end = train_end - SEQ_LEN - HORIZON + 1
    val_i_start = train_end - SEQ_LEN
    val_i_end = val_end - SEQ_LEN - HORIZON + 1
    test_i_start = val_end - SEQ_LEN

    if train_i_end <= 0 or val_i_end <= val_i_start or test_i_start < 0:
        raise ValueError("Invalid strict-warm split indices.")

    return {
        "train": (
            X_seq[:train_i_end],
            Yp_seq[:train_i_end],
            Yf_seq[:train_i_end],
        ),
        "val": (
            X_seq[val_i_start:val_i_end],
            Yp_seq[val_i_start:val_i_end],
            Yf_seq[val_i_start:val_i_end],
        ),
        "test": (
            X_seq[test_i_start:],
            Yp_seq[test_i_start:],
            Yf_seq[test_i_start:],
        ),
        "scaler_y": scaler_y,
        "exog_dim": int(X_seq.shape[2]),
        "timestamps": df["timestamp"].reset_index(drop=True),
        "times": df["time"].reset_index(drop=True),
        "test_start_i": int(test_i_start),
        "per_day": int(per_day),
        "step_minutes": int(step_minutes),
        "has_time_component": bool(has_time_component),
        "feature_names": exog_features,
        "split_info": {
            "train_i_end": int(train_i_end),
            "val_i_start": int(val_i_start),
            "val_i_end": int(val_i_end),
            "test_i_start": int(test_i_start),
        },
    }


def build_data_files(base_dir: Path) -> Dict[str, Path]:
    return {
        "all": base_dir / "data" / "demand_holiday_with_solar_mean.csv",
        "spring": base_dir / "data" / "demand_holiday_with_solar_mean_spring.csv",
        "summer": base_dir / "data" / "demand_holiday_with_solar_mean_summer.csv",
        "autumn": base_dir / "data" / "demand_holiday_with_solar_mean_autumn.csv",
        "winter": base_dir / "data" / "demand_holiday_with_solar_mean_winter.csv",
    }


# ============================================================
# Search space / builders
# ============================================================
def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
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
    return {
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "hidden_dim": trial.suggest_categorical(
            "hidden_dim", [256, 512, 1024]
        ),
        "dropout": trial.suggest_float("dropout", 0.0, 0.30),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 1e-3, log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-7, 1e-2, log=True
        ),
        "optimizer_name": trial.suggest_categorical(
            "optimizer_name", ["Adam", "AdamW"]
        ),
    }


def normalize_best_params(params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(params)
    pair = params.pop("d_model_nhead", None)
    if pair is not None:
        d_model, nhead = map(int, str(pair).split("_"))
        params["d_model"] = d_model
        params["nhead"] = nhead
    return params


def build_model(
    variant: str,
    exog_dim: int,
    params: Dict[str, Any],
) -> AblationHybridTransformer:
    return AblationHybridTransformer(
        variant=variant,
        seq_len=SEQ_LEN,
        exog_dim=exog_dim,
        horizon=HORIZON,
        d_model=int(params["d_model"]),
        nhead=int(params["nhead"]),
        num_layers=int(params["num_layers"]),
        hidden_dim=int(params["hidden_dim"]),
        dropout=float(params["dropout"]),
    )


def build_optimizer(model: nn.Module, params: Dict[str, Any]):
    optimizer_class = getattr(optim, str(params["optimizer_name"]))
    return optimizer_class(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )


# ============================================================
# Optuna search
# ============================================================
def create_objective(
    variant: str,
    data: Dict[str, Any],
    device: torch.device,
    tuning_seed: int,
    max_epochs: int,
    patience: int,
):
    train_dataset = TimeSeriesDatasetHQ(*data["train"])
    val_dataset = TimeSeriesDatasetHQ(*data["val"])
    criterion = nn.MSELoss()

    def objective(trial: optuna.Trial) -> float:
        seed_everything(tuning_seed)
        params = suggest_params(trial)
        train_loader = make_loader(
            train_dataset,
            int(params["batch_size"]),
            True,
            tuning_seed,
            device,
        )
        val_loader = make_loader(
            val_dataset,
            int(params["batch_size"]),
            False,
            tuning_seed,
            device,
        )
        model = build_model(variant, int(data["exog_dim"]), params).to(device)
        optimizer = build_optimizer(model, params)
        best_validation = float("inf")
        bad_epochs = 0

        try:
            for epoch in range(1, max_epochs + 1):
                model.train()
                for x_exog, y_past, y_future in train_loader:
                    x_exog = x_exog.to(device, non_blocking=True)
                    y_past = y_past.to(device, non_blocking=True)
                    y_future = y_future.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(x_exog, y_past)
                    loss = criterion(prediction, y_future)
                    if HQ_L2_WEIGHT > 0 and model.use_hq:
                        loss = loss + HQ_L2_WEIGHT * model.horizon_queries.pow(2).mean()
                    if not torch.isfinite(loss):
                        raise optuna.TrialPruned("Non-finite training loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

                model.eval()
                validation_losses: List[float] = []
                with torch.no_grad():
                    for x_exog, y_past, y_future in val_loader:
                        x_exog = x_exog.to(device, non_blocking=True)
                        y_past = y_past.to(device, non_blocking=True)
                        y_future = y_future.to(device, non_blocking=True)
                        prediction = model(x_exog, y_past)
                        validation_losses.append(
                            float(criterion(prediction, y_future).item())
                        )

                validation_loss = float(np.mean(validation_losses))
                trial.report(validation_loss, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if validation_loss < best_validation:
                    best_validation = validation_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        break

            flags = variant_flags(variant)
            trial.set_user_attr("best_validation_loss", best_validation)
            trial.set_user_attr("variant", variant)
            trial.set_user_attr("tuning_seed", tuning_seed)
            trial.set_user_attr("dataset", "all")
            trial.set_user_attr("feature_policy", "past_demand+weather+calendar")
            for key, value in flags.items():
                trial.set_user_attr(key, value)
            return best_validation
        finally:
            del model
            del optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return objective


def count_existing_trials(study: optuna.Study) -> int:
    counted_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    return len([trial for trial in study.trials if trial.state in counted_states])


def run_search(
    variant: str,
    base_dir: Path,
    output_root: Path,
    device: torch.device,
    total_trials: int,
    tuning_seed: int,
    sampler_seed: int,
    max_epochs: int,
    patience: int,
) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")

    output_root.mkdir(parents=True, exist_ok=True)
    data = prepare_data(build_data_files(base_dir)["all"])

    database_path = output_root / f"{variant}_optuna.db"
    storage = f"sqlite:///{database_path.as_posix()}"
    study = optuna.create_study(
        study_name=f"{variant}_all_seed{tuning_seed}_equal_budget",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=PRUNER_STARTUP_TRIALS,
            n_warmup_steps=PRUNER_WARMUP_EPOCHS,
        ),
        storage=storage,
        load_if_exists=True,
    )

    used_trials = count_existing_trials(study)
    remaining_trials = max(0, int(total_trials) - used_trials)

    print("\n" + "=" * 100)
    print(f"VARIANT            : {variant}")
    print(f"MEANING            : {VARIANT_LABELS[variant]}")
    print(f"TUNING DATA        : all-season")
    print(f"TUNING SEED        : {tuning_seed}")
    print(f"TOTAL TRIAL TARGET : {total_trials}")
    print(f"EXISTING TRIALS    : {used_trials}")
    print(f"REMAINING TRIALS   : {remaining_trials}")
    print(f"MAX EPOCHS         : {max_epochs}")
    print(f"PATIENCE           : {patience}")
    print("OBJECTIVE          : standardized validation MSE")
    print("=" * 100)

    if remaining_trials > 0:
        study.optimize(
            create_objective(
                variant=variant,
                data=data,
                device=device,
                tuning_seed=tuning_seed,
                max_epochs=max_epochs,
                patience=patience,
            ),
            n_trials=remaining_trials,
            gc_after_trial=True,
            show_progress_bar=True,
        )
    else:
        print("既に指定した総trial数へ到達しているため、追加探索しません。")

    if not any(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    ):
        raise RuntimeError("No completed Optuna trial is available.")

    best_params = normalize_best_params(study.best_params)
    flags = variant_flags(variant)
    best_json = output_root / f"{variant}_best_params_all_seed{tuning_seed}.json"
    payload = {
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        **flags,
        "tuning_dataset": "all",
        "tuning_seed": tuning_seed,
        "sampler_seed": sampler_seed,
        "total_trial_budget": total_trials,
        "max_epochs": max_epochs,
        "patience": patience,
        "objective": "validation_mse_standardized",
        "split": "strict_warm_80_10_10",
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,
        "feature_policy": "past_demand+weather+calendar",
        "test_used_for_hp_selection": False,
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_params": best_params,
    }
    best_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    trials_csv = output_root / f"{variant}_optuna_trials.csv"
    study.trials_dataframe().to_csv(
        trials_csv, index=False, encoding="utf-8-sig"
    )

    print(f"BEST VALUE : {study.best_value:.8f}")
    print(f"BEST TRIAL : {study.best_trial.number}")
    print(f"BEST PARAMS: {best_params}")
    print(f"JSON       : {best_json}")
    print(f"TRIAL CSV  : {trials_csv}")
    return best_json


# ============================================================
# Final evaluation helpers
# ============================================================
def inverse_standardized(scaler_y: StandardScaler, values: np.ndarray) -> np.ndarray:
    return values * float(scaler_y.scale_[0]) + float(scaler_y.mean_[0])


def safe_load_state_dict(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def horizon_metrics(trues: np.ndarray, predictions: np.ndarray) -> Dict[str, np.ndarray]:
    difference = predictions - trues
    rmse_per_h = np.sqrt(np.mean(difference ** 2, axis=0))
    denominator = np.clip(np.abs(trues), 1e-8, None)
    mape_per_h = np.mean(np.abs(difference) / denominator, axis=0) * 100.0
    return {"rmse_per_h": rmse_per_h, "mape_per_h": mape_per_h}


def save_prediction_csv(
    path: Path,
    variant: str,
    season: str,
    seed: int,
    predictions: np.ndarray,
    trues: np.ndarray,
    timestamps: pd.Series,
    times: pd.Series,
    test_start_i: int,
    best_val_loss: float,
    params: Dict[str, Any],
) -> None:
    rows: List[List[Any]] = []
    for sample_index in range(len(predictions)):
        sequence_index = test_start_i + sample_index
        timestamp_slice = timestamps.iloc[
            sequence_index + SEQ_LEN : sequence_index + SEQ_LEN + HORIZON
        ].to_numpy()
        time_slice = times.iloc[
            sequence_index + SEQ_LEN : sequence_index + SEQ_LEN + HORIZON
        ].to_numpy()

        for horizon_index in range(HORIZON):
            timestamp = pd.to_datetime(timestamp_slice[horizon_index])
            rows.append(
                [
                    timestamp,
                    timestamp.date().isoformat(),
                    int(time_slice[horizon_index]),
                    int(timestamp.hour),
                    int(timestamp.minute),
                    horizon_index + 1,
                    float(trues[sample_index, horizon_index]),
                    float(predictions[sample_index, horizon_index]),
                ]
            )

    output = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "date",
            "time",
            "hour",
            "minute",
            "horizon",
            "y_true",
            "y_pred",
        ],
    )
    output["variant"] = variant
    output["season"] = season
    output["seed"] = seed
    output["best_val_loss_standardized"] = best_val_loss
    for key, value in variant_flags(variant).items():
        output[key] = value
    for key, value in params.items():
        output[key] = value
    output.to_csv(path, index=False, encoding="utf-8-sig")


def train_and_evaluate_one(
    variant: str,
    season: str,
    seed: int,
    data_file: Path,
    params: Dict[str, Any],
    output_root: Path,
    device: torch.device,
    max_epochs: int,
    patience: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    seed_everything(seed)
    data = prepare_data(data_file)
    train_dataset = TimeSeriesDatasetHQ(*data["train"])
    val_dataset = TimeSeriesDatasetHQ(*data["val"])
    test_dataset = TimeSeriesDatasetHQ(*data["test"])

    batch_size = int(params["batch_size"])
    train_loader = make_loader(train_dataset, batch_size, True, seed, device)
    val_loader = make_loader(val_dataset, batch_size, False, seed, device)
    test_loader = make_loader(test_dataset, batch_size, False, seed, device)

    model = build_model(variant, int(data["exog_dim"]), params).to(device)
    optimizer = build_optimizer(model, params)
    criterion = nn.MSELoss()

    variant_root = output_root / "final" / variant
    model_dir = variant_root / "models" / f"seed_{seed}"
    pred_dir = variant_root / "predictions" / f"seed_{seed}"
    history_dir = variant_root / "histories" / f"seed_{seed}"
    summary_dir = variant_root / "summaries"
    for directory in [model_dir, pred_dir, history_dir, summary_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    run_name = f"{variant}_{season}_seed{seed}"
    model_path = model_dir / f"{run_name}.pth"
    prediction_path = pred_dir / f"{run_name}.csv"
    history_path = history_dir / f"{run_name}.csv"

    best_val_loss = float("inf")
    bad_epochs = 0
    history_rows: List[Dict[str, float]] = []

    print("\n" + "-" * 100)
    print(f"FINAL | variant={variant} | season={season} | seed={seed}")
    print(f"params={params}")
    print("-" * 100)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: List[float] = []
        for x_exog, y_past, y_future in train_loader:
            x_exog = x_exog.to(device, non_blocking=True)
            y_past = y_past.to(device, non_blocking=True)
            y_future = y_future.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_exog, y_past)
            loss = criterion(prediction, y_future)
            if HQ_L2_WEIGHT > 0 and model.use_hq:
                loss = loss + HQ_L2_WEIGHT * model.horizon_queries.pow(2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for x_exog, y_past, y_future in val_loader:
                x_exog = x_exog.to(device, non_blocking=True)
                y_past = y_past.to(device, non_blocking=True)
                y_future = y_future.to(device, non_blocking=True)
                prediction = model(x_exog, y_past)
                val_losses.append(float(criterion(prediction, y_future).item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history_rows.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), model_path)
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | train={train_loss:.6f} | "
                f"val={val_loss:.6f} | best={best_val_loss:.6f}"
            )
        if bad_epochs >= patience:
            print(f"Early stop @ epoch={epoch}")
            break

    pd.DataFrame(history_rows).to_csv(
        history_path, index=False, encoding="utf-8-sig"
    )
    model.load_state_dict(safe_load_state_dict(model_path, device))
    model.eval()

    prediction_batches: List[np.ndarray] = []
    true_batches: List[np.ndarray] = []
    with torch.no_grad():
        for x_exog, y_past, y_future in test_loader:
            x_exog = x_exog.to(device, non_blocking=True)
            y_past = y_past.to(device, non_blocking=True)
            prediction = model(x_exog, y_past)
            prediction_batches.append(prediction.cpu().numpy())
            true_batches.append(y_future.numpy())

    predictions_std = np.concatenate(prediction_batches, axis=0)
    trues_std = np.concatenate(true_batches, axis=0)
    scaler_y: StandardScaler = data["scaler_y"]
    predictions = inverse_standardized(scaler_y, predictions_std)
    trues = inverse_standardized(scaler_y, trues_std)

    rmse = float(
        np.sqrt(mean_squared_error(trues.reshape(-1), predictions.reshape(-1)))
    )
    mape = float(
        mean_absolute_percentage_error(
            trues.reshape(-1), predictions.reshape(-1)
        )
        * 100.0
    )

    save_prediction_csv(
        path=prediction_path,
        variant=variant,
        season=season,
        seed=seed,
        predictions=predictions,
        trues=trues,
        timestamps=data["timestamps"],
        times=data["times"],
        test_start_i=int(data["test_start_i"]),
        best_val_loss=best_val_loss,
        params=params,
    )

    metrics = horizon_metrics(trues, predictions)
    horizon_df = pd.DataFrame(
        {
            "variant": variant,
            "season": season,
            "seed": seed,
            "horizon": np.arange(1, HORIZON + 1),
            "rmse_mw": metrics["rmse_per_h"],
            "mape_percent": metrics["mape_per_h"],
        }
    )

    result = {
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        **variant_flags(variant),
        "season": season,
        "seed": seed,
        "rmse_mw": rmse,
        "mape_percent": mape,
        "best_val_loss_standardized": best_val_loss,
        "epochs_completed": len(history_rows),
        "model_path": str(model_path),
        "prediction_csv": str(prediction_path),
        "history_csv": str(history_path),
        **params,
    }
    print(f"TEST | RMSE={rmse:.4f} MW | MAPE={mape:.4f}%")
    return result, horizon_df


def load_best_params(best_json: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not best_json.exists():
        raise FileNotFoundError(
            f"Best-parameter JSON not found: {best_json}\n"
            "先に --mode search を実行してください。"
        )
    metadata = json.loads(best_json.read_text(encoding="utf-8"))
    return dict(metadata["best_params"]), metadata


def run_final_evaluation(
    variant: str,
    base_dir: Path,
    output_root: Path,
    device: torch.device,
    seeds: Sequence[int],
    seasons: Sequence[str],
    max_epochs: int,
    patience: int,
    best_json: Optional[Path] = None,
    tuning_seed: int = 42,
) -> Tuple[Path, Path]:
    if best_json is None:
        best_json = output_root / f"{variant}_best_params_all_seed{tuning_seed}.json"
    params, search_metadata = load_best_params(best_json)
    data_files = build_data_files(base_dir)

    all_results: List[Dict[str, Any]] = []
    all_horizon: List[pd.DataFrame] = []
    for seed in seeds:
        for season in seasons:
            result, horizon_df = train_and_evaluate_one(
                variant=variant,
                season=season,
                seed=int(seed),
                data_file=data_files[season],
                params=params,
                output_root=output_root,
                device=device,
                max_epochs=max_epochs,
                patience=patience,
            )
            result["hp_json"] = str(best_json)
            result["hp_tuning_seed"] = search_metadata.get("tuning_seed", tuning_seed)
            result["hp_trial_budget"] = search_metadata.get("total_trial_budget")
            all_results.append(result)
            all_horizon.append(horizon_df)

    variant_summary_dir = output_root / "final" / variant / "summaries"
    variant_summary_dir.mkdir(parents=True, exist_ok=True)
    each_run = pd.DataFrame(all_results)
    each_run_path = variant_summary_dir / f"{variant}_multiseed_each_run.csv"
    each_run.to_csv(each_run_path, index=False, encoding="utf-8-sig")

    aggregate = (
        each_run.groupby("season", sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            rmse_mean=("rmse_mw", "mean"),
            rmse_std=("rmse_mw", "std"),
            mape_mean=("mape_percent", "mean"),
            mape_std=("mape_percent", "std"),
        )
        .reset_index()
    )
    aggregate["variant"] = variant
    aggregate["RMSE_mean_std_MW"] = aggregate.apply(
        lambda row: f"{row.rmse_mean:.3f} ± {row.rmse_std:.3f}", axis=1
    )
    aggregate["MAPE_mean_std_percent"] = aggregate.apply(
        lambda row: f"{row.mape_mean:.3f} ± {row.mape_std:.3f}", axis=1
    )
    aggregate["rmse_cv_percent"] = (
        aggregate["rmse_std"] / aggregate["rmse_mean"] * 100.0
    )
    aggregate["mape_cv_percent"] = (
        aggregate["mape_std"] / aggregate["mape_mean"] * 100.0
    )
    aggregate_path = variant_summary_dir / f"{variant}_multiseed_mean_std.csv"
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")

    paper_table = aggregate[
        [
            "variant",
            "season",
            "n_seeds",
            "RMSE_mean_std_MW",
            "MAPE_mean_std_percent",
            "rmse_cv_percent",
            "mape_cv_percent",
        ]
    ]
    paper_table.to_csv(
        variant_summary_dir / f"{variant}_multiseed_paper_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    horizon_each = pd.concat(all_horizon, ignore_index=True)
    horizon_each.to_csv(
        variant_summary_dir / f"{variant}_horizon_each_run.csv",
        index=False,
        encoding="utf-8-sig",
    )
    horizon_aggregate = (
        horizon_each.groupby(["season", "horizon"], sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            rmse_mean=("rmse_mw", "mean"),
            rmse_std=("rmse_mw", "std"),
            mape_mean=("mape_percent", "mean"),
            mape_std=("mape_percent", "std"),
        )
        .reset_index()
    )
    horizon_aggregate["variant"] = variant
    horizon_aggregate.to_csv(
        variant_summary_dir / f"{variant}_horizon_mean_std.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 100)
    print(f"FINAL EVALUATION FINISHED: {variant}")
    print(f"Each run : {each_run_path}")
    print(f"Mean/std : {aggregate_path}")
    print(aggregate[["season", "RMSE_mean_std_MW", "MAPE_mean_std_percent"]].to_string(index=False))
    print("=" * 100)
    return each_run_path, aggregate_path


# ============================================================
# CLI
# ============================================================
def resolve_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but it is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Equal-budget Optuna and multi-seed HybridHQ ablation"
    )
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument(
        "--mode",
        choices=["search", "final", "search_and_final"],
        default="search",
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--tuning-seed", type=int, default=42)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--search-epochs", type=int, default=80)
    parser.add_argument("--search-patience", type=int, default=10)
    parser.add_argument("--final-epochs", type=int, default=200)
    parser.add_argument("--final-patience", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--seasons", nargs="+", choices=DEFAULT_SEASONS, default=DEFAULT_SEASONS
    )
    parser.add_argument("--best-json", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    output_root = args.output_root or (
        args.base_dir / "result" / "result_csv" / "ablation_optuna_equal_budget_seed42"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Output root: {output_root}")

    best_json = args.best_json
    if args.mode in {"search", "search_and_final"}:
        best_json = run_search(
            variant=args.variant,
            base_dir=args.base_dir,
            output_root=output_root,
            device=device,
            total_trials=args.n_trials,
            tuning_seed=args.tuning_seed,
            sampler_seed=args.sampler_seed,
            max_epochs=args.search_epochs,
            patience=args.search_patience,
        )

    if args.mode in {"final", "search_and_final"}:
        run_final_evaluation(
            variant=args.variant,
            base_dir=args.base_dir,
            output_root=output_root,
            device=device,
            seeds=args.seeds,
            seasons=args.seasons,
            max_epochs=args.final_epochs,
            patience=args.final_patience,
            best_json=best_json,
            tuning_seed=args.tuning_seed,
        )


if __name__ == "__main__":
    main()
