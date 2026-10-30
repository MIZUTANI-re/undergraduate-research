# ============================================================
# LSTM one-shot load forecasting
# - all / spring / summer / autumn / winter
# - warm-start STRICT split
# - Optuna tuning:
#     "all_only"  : allデータで探索し、同じHPを全季節へ固定（公平比較向け・推奨）
#     "by_season" : 各季節で個別に探索
#     "off"       : FIXED_PARAMSを使用
# - 3 random seeds: 42, 123, 2026
# - mean ± std / horizon-wise metrics / prediction CSV
#
# 必要:
#   pip install optuna
#
# 実行:
#   python LSTM_optuna_multiseed_all_seasons.py
# ============================================================

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import matplotlib.pyplot as plt

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    optuna = None
    TrialState = None


# ============================================================
# ここを設定
# ============================================================

SEEDS = [42, 123, 2026]

SEASONS_TO_RUN = [
    "all",
    "spring",
    "summer",
    "autumn",
    "winter",
]

# "all_only" / "by_season" / "off"
#
# 公平比較では "all_only" 推奨:
# allデータでOptuna探索した1種類のHPを全季節に固定します。
OPTUNA_MODE = "all_only"

# Optunaを新規に回すか
# Falseの場合は保存済みbest_params JSONを使用します。
RUN_OPTUNA_SEARCH = True

N_TRIALS = 30
OPTUNA_TIMEOUT_SECONDS = None
OPTUNA_SEED = 42

# Optuna中の学習設定
OPTUNA_MAX_EPOCHS = 80
OPTUNA_PATIENCE = 10

# 最終3seed評価
RUN_FINAL_EVALUATION = True
FINAL_MAX_EPOCHS = 200
FINAL_PATIENCE = 20

SEQUENCE_MODE = "warm"   # "warm" / "cold"
STRICT_SPLIT = True
SORT_BY_TIMESTAMP = False

SEQ_LEN = 72
HORIZON = 24

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

# 入力特徴量はiTransformerとの比較を意識し、
# power_demand + weather + calendarを使用します。
USE_POWER_DEMAND_AS_INPUT = True

BASE_DIR = Path(__file__).resolve().parents[2]  # project root (clean_code/)
DATA_DIR = BASE_DIR / "data"

DATA_FILES = {
    "all": DATA_DIR / "demand_holiday_with_solar_mean.csv",
    "spring": DATA_DIR / "demand_holiday_with_solar_mean_spring.csv",
    "summer": DATA_DIR / "demand_holiday_with_solar_mean_summer.csv",
    "autumn": DATA_DIR / "demand_holiday_with_solar_mean_autumn.csv",
    "winter": DATA_DIR / "demand_holiday_with_solar_mean_winter.csv",
}

RESULT_DIR = BASE_DIR / "result" / "result_csv"
MODEL_DIR = BASE_DIR / "result" / "learning_saver" / "LSTM_multiseed"
FIG_DIR = BASE_DIR / "picture" / "LSTM_multiseed"
PRED_DIR = RESULT_DIR / "pred_csv_LSTM_multiseed"
OPTUNA_DIR = RESULT_DIR / "optuna_LSTM"

for d in [RESULT_DIR, MODEL_DIR, FIG_DIR, PRED_DIR, OPTUNA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EACH_RUN_PATH = RESULT_DIR / "LSTM_multiseed_each_run.csv"
MEAN_STD_PATH = RESULT_DIR / "LSTM_multiseed_mean_std.csv"
PAPER_TABLE_PATH = RESULT_DIR / "LSTM_multiseed_paper_table.csv"
HORIZON_EACH_RUN_PATH = RESULT_DIR / "LSTM_multiseed_horizon_each_run.csv"
HORIZON_MEAN_STD_PATH = RESULT_DIR / "LSTM_multiseed_horizon_mean_std.csv"

# OPTUNA_MODE="off"のときに使用
FIXED_PARAMS = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.1,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "head_hidden": 128,
    "optimizer_name": "AdamW",
}


# ============================================================
# Seed / device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset / model
# ============================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMOneShot(nn.Module):
    """
    入力:
        x: (B, seq_len, input_dim)
    出力:
        y: (B, horizon)

    単方向LSTMの最終時刻表現から24時間をone-shot予測します。
    """
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        horizon: int,
        head_hidden: int,
    ):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=False,
        )

        self.norm = nn.LayerNorm(hidden_size)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.norm(last)
        return self.head(last)


# ============================================================
# Data helpers
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
        tmax = int(df["time"].max())
        step_minutes = 60 if tmax <= 23 else 30

    return per_day, step_minutes


def build_timestamp(
    df: pd.DataFrame,
) -> Tuple[pd.Series, bool, int, int]:
    dt = pd.to_datetime(df["datetime"])

    has_time_component = (
        (dt.dt.hour != 0)
        | (dt.dt.minute != 0)
        | (dt.dt.second != 0)
    ).any()

    per_day, step_minutes = infer_step_minutes(df)

    if has_time_component:
        timestamp = dt
    else:
        timestamp = dt + pd.to_timedelta(
            df["time"] * step_minutes,
            unit="m",
        )

    return (
        timestamp,
        bool(has_time_component),
        int(per_day),
        int(step_minutes),
    )


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []

    for i in range(len(X) - seq_len - horizon + 1):
        Xs.append(X[i:i + seq_len])
        ys.append(
            y[i + seq_len:i + seq_len + horizon].reshape(-1)
        )

    return np.asarray(Xs), np.asarray(ys)


def split_sequences_by_target_window(
    X_all: np.ndarray,
    y_all: np.ndarray,
    n_raw: int,
    tr: int,
    va: int,
    seq_len: int,
    horizon: int,
):
    starts = np.arange(len(y_all))

    y_start = starts + seq_len
    y_end = y_start + horizon - 1

    mask_train = y_end < tr
    mask_val = (y_start >= tr) & (y_end < va)
    mask_test = (y_start >= va) & (y_end < n_raw)

    return (
        (
            X_all[mask_train],
            y_all[mask_train],
            starts[mask_train],
        ),
        (
            X_all[mask_val],
            y_all[mask_val],
            starts[mask_val],
        ),
        (
            X_all[mask_test],
            y_all[mask_test],
            starts[mask_test],
        ),
    )


def inverse_1col(
    scaler_y: StandardScaler,
    values_2d: np.ndarray,
) -> np.ndarray:
    return (
        values_2d * float(scaler_y.scale_[0])
        + float(scaler_y.mean_[0])
    )


def prepare_data(
    file_path: Path,
    sequence_mode: str = "warm",
) -> Dict[str, object]:
    if not file_path.exists():
        raise FileNotFoundError(f"CSV not found: {file_path}")

    df = pd.read_csv(file_path)

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
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{file_path} に必要な列がありません: {missing}"
        )

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["datetime", "time"]).reset_index(drop=True)

    (
        df["timestamp"],
        has_time_component,
        per_day,
        step_minutes,
    ) = build_timestamp(df)

    if SORT_BY_TIMESTAMP:
        df = df.sort_values("timestamp").reset_index(drop=True)

    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday

    df = df.dropna().reset_index(drop=True)
    df = pd.get_dummies(
        df,
        columns=["weekday", "month"],
        drop_first=True,
    )

    features = [
        "temperature",
        "precipitation",
        "time",
        "day",
        "solar_mean",
        "restday",
        "holiday_onehot",
    ]

    if USE_POWER_DEMAND_AS_INPUT:
        features = ["power_demand"] + features

    features += [
        c for c in df.columns
        if c.startswith("weekday_") or c.startswith("month_")
    ]

    target = "power_demand"

    n = len(df)
    tr = int(n * TRAIN_RATIO)
    va = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:tr].copy()
    val_df = df.iloc[tr:va].copy()
    test_df = df.iloc[va:].copy()

    scaler_X = StandardScaler().fit(train_df[features])
    scaler_y = StandardScaler().fit(train_df[[target]])

    X_train_points = scaler_X.transform(train_df[features])
    X_val_points = scaler_X.transform(val_df[features])
    X_test_points = scaler_X.transform(test_df[features])

    y_train_points = scaler_y.transform(train_df[[target]])
    y_val_points = scaler_y.transform(val_df[[target]])
    y_test_points = scaler_y.transform(test_df[[target]])

    if sequence_mode == "warm":
        X_points = np.concatenate(
            [X_train_points, X_val_points, X_test_points],
            axis=0,
        )
        y_points = np.concatenate(
            [y_train_points, y_val_points, y_test_points],
            axis=0,
        )

        all_ts = df["timestamp"].reset_index(drop=True)
        all_time = df["time"].reset_index(drop=True)

        X_all, y_all = create_sequences(
            X_points,
            y_points,
            SEQ_LEN,
            HORIZON,
        )

        if STRICT_SPLIT:
            (
                (Xtr, ytr, train_starts),
                (Xva, yva, val_starts),
                (Xte, yte, test_starts),
            ) = split_sequences_by_target_window(
                X_all=X_all,
                y_all=y_all,
                n_raw=len(X_points),
                tr=tr,
                va=va,
                seq_len=SEQ_LEN,
                horizon=HORIZON,
            )
        else:
            train_seq_end = (
                len(train_df) - SEQ_LEN - HORIZON + 1
            )
            val_seq_end = (
                len(train_df)
                + len(val_df)
                - SEQ_LEN
                - HORIZON
                + 1
            )

            Xtr = X_all[:train_seq_end]
            ytr = y_all[:train_seq_end]
            Xva = X_all[train_seq_end:val_seq_end]
            yva = y_all[train_seq_end:val_seq_end]
            Xte = X_all[val_seq_end:]
            yte = y_all[val_seq_end:]

            train_starts = np.arange(len(Xtr))
            val_starts = np.arange(
                train_seq_end,
                val_seq_end,
            )
            test_starts = np.arange(
                val_seq_end,
                val_seq_end + len(Xte),
            )

    elif sequence_mode == "cold":
        Xtr, ytr = create_sequences(
            X_train_points,
            y_train_points,
            SEQ_LEN,
            HORIZON,
        )
        Xva, yva = create_sequences(
            X_val_points,
            y_val_points,
            SEQ_LEN,
            HORIZON,
        )
        Xte, yte = create_sequences(
            X_test_points,
            y_test_points,
            SEQ_LEN,
            HORIZON,
        )

        all_ts = test_df["timestamp"].reset_index(drop=True)
        all_time = test_df["time"].reset_index(drop=True)

        train_starts = np.arange(len(Xtr))
        val_starts = np.arange(len(Xva))
        test_starts = np.arange(len(Xte))

    else:
        raise ValueError(
            f"SEQUENCE_MODE={sequence_mode!r} は無効です。"
        )

    if min(len(ytr), len(yva), len(yte)) == 0:
        raise ValueError(
            "train/val/testのいずれかのsequence数が0です。"
        )

    return {
        "Xtr": Xtr,
        "ytr": ytr,
        "Xva": Xva,
        "yva": yva,
        "Xte": Xte,
        "yte": yte,
        "scaler_y": scaler_y,
        "all_ts": all_ts,
        "all_time": all_time,
        "test_starts": test_starts,
        "features": features,
        "input_dim": int(Xtr.shape[2]),
        "per_day": per_day,
        "step_minutes": step_minutes,
        "has_time_component": has_time_component,
        "n_raw": n,
        "tr_point": tr,
        "va_point": va,
    }


# ============================================================
# Training helpers
# ============================================================

def build_optimizer(
    model: nn.Module,
    params: Dict[str, object],
):
    optimizer_name = str(params["optimizer_name"])

    optimizer_class = getattr(
        optim,
        optimizer_name,
        None,
    )

    if optimizer_class is None:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}"
        )

    kwargs = {
        "lr": float(params["learning_rate"]),
        "weight_decay": float(params["weight_decay"]),
    }

    return optimizer_class(model.parameters(), **kwargs)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    params: Dict[str, object],
    max_epochs: int,
    patience: int,
    save_path: Optional[Path] = None,
    trial=None,
) -> Tuple[float, int, int]:
    optimizer = build_optimizer(model, params)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_epoch = -1
    bad = 0
    last_epoch = 0

    best_state = None

    for epoch in range(1, max_epochs + 1):
        last_epoch = epoch

        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}"
                )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                val_pred = model(xb)
                val_loss = criterion(val_pred, yb)
                val_losses.append(float(val_loss.item()))

        current_val = (
            float(np.mean(val_losses))
            if val_losses
            else float("inf")
        )

        if trial is not None:
            trial.report(current_val, step=epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        if current_val < best_val:
            best_val = current_val
            best_epoch = epoch
            bad = 0

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            if save_path is not None:
                save_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                torch.save(best_state, save_path)
        else:
            bad += 1

            if bad >= patience:
                break

    if save_path is None and best_state is not None:
        model.load_state_dict(best_state)

    return best_val, best_epoch, last_epoch


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    scaler_y: StandardScaler,
):
    model.eval()

    preds_scaled = []
    trues_scaled = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)

            preds_scaled.append(
                model(xb).cpu().numpy()
            )
            trues_scaled.append(yb.numpy())

    preds_scaled = np.concatenate(preds_scaled, axis=0)
    trues_scaled = np.concatenate(trues_scaled, axis=0)

    preds = inverse_1col(scaler_y, preds_scaled)
    trues = inverse_1col(scaler_y, trues_scaled)

    rmse = float(
        np.sqrt(
            mean_squared_error(
                trues.reshape(-1),
                preds.reshape(-1),
            )
        )
    )

    mape = float(
        mean_absolute_percentage_error(
            trues.reshape(-1),
            preds.reshape(-1),
        )
        * 100.0
    )

    diff = preds - trues

    rmse_per_h = np.sqrt(
        np.mean(diff ** 2, axis=0)
    )

    denominator = np.clip(
        np.abs(trues),
        1e-8,
        None,
    )

    mape_per_h = (
        np.mean(
            np.abs(diff) / denominator,
            axis=0,
        )
        * 100.0
    )

    return {
        "preds": preds,
        "trues": trues,
        "rmse": rmse,
        "mape": mape,
        "rmse_per_horizon": rmse_per_h,
        "mape_per_horizon": mape_per_h,
    }


# ============================================================
# Optuna
# ============================================================

def suggest_params(trial) -> Dict[str, object]:
    num_layers = trial.suggest_int(
        "num_layers",
        1,
        3,
    )

    params = {
        "hidden_size": trial.suggest_categorical(
            "hidden_size",
            [32, 64, 128, 256],
        ),
        "num_layers": num_layers,
        "dropout": trial.suggest_float(
            "dropout",
            0.0,
            0.35,
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size",
            [32, 64, 128],
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-5,
            3e-3,
            log=True,
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            1e-7,
            1e-2,
            log=True,
        ),
        "head_hidden": trial.suggest_categorical(
            "head_hidden",
            [32, 64, 128, 256],
        ),
        "optimizer_name": trial.suggest_categorical(
            "optimizer_name",
            ["Adam", "AdamW"],
        ),
    }

    return params


def best_params_json_path(season: str) -> Path:
    if OPTUNA_MODE == "all_only":
        return OPTUNA_DIR / "LSTM_best_params_all.json"

    return OPTUNA_DIR / f"LSTM_best_params_{season}.json"


def run_optuna_for_season(
    season: str,
    file_path: Path,
) -> Dict[str, object]:
    if optuna is None:
        raise ImportError(
            "Optunaがインストールされていません。"
            " `pip install optuna` を実行してください。"
        )

    data = prepare_data(
        file_path,
        sequence_mode=SEQUENCE_MODE,
    )

    def objective(trial):
        params = suggest_params(trial)

        seed_everything(OPTUNA_SEED)

        train_loader = DataLoader(
            TimeSeriesDataset(
                data["Xtr"],
                data["ytr"],
            ),
            batch_size=int(params["batch_size"]),
            shuffle=True,
        )

        val_loader = DataLoader(
            TimeSeriesDataset(
                data["Xva"],
                data["yva"],
            ),
            batch_size=int(params["batch_size"]),
            shuffle=False,
        )

        model = LSTMOneShot(
            input_dim=int(data["input_dim"]),
            hidden_size=int(params["hidden_size"]),
            num_layers=int(params["num_layers"]),
            dropout=float(params["dropout"]),
            horizon=HORIZON,
            head_hidden=int(params["head_hidden"]),
        ).to(device)

        best_val, _, _ = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            params=params,
            max_epochs=OPTUNA_MAX_EPOCHS,
            patience=OPTUNA_PATIENCE,
            save_path=None,
            trial=trial,
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return float(best_val)

    storage_path = OPTUNA_DIR / f"LSTM_optuna_{season}.db"
    storage_url = f"sqlite:///{storage_path.as_posix()}"

    study_name = f"LSTM_{season}_strict{int(STRICT_SPLIT)}"

    sampler = optuna.samplers.TPESampler(
        seed=OPTUNA_SEED,
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True,
    )

    print("\n" + "=" * 100)
    print(
        f"OPTUNA START | season={season} "
        f"| trials={N_TRIALS}"
    )
    print("=" * 100)

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=OPTUNA_TIMEOUT_SECONDS,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    best_params = dict(study.best_params)

    best_params_path = best_params_json_path(season)
    with best_params_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            best_params,
            f,
            ensure_ascii=False,
            indent=2,
        )

    trials_df = study.trials_dataframe()
    trials_csv = (
        OPTUNA_DIR
        / f"LSTM_optuna_trials_{season}.csv"
    )
    trials_df.to_csv(
        trials_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Best value : {study.best_value:.8f}")
    print(f"Best params: {best_params}")
    print(f"Saved JSON : {best_params_path}")
    print(f"Saved trials: {trials_csv}")

    return best_params


def load_best_params(season: str) -> Dict[str, object]:
    if OPTUNA_MODE == "off":
        return dict(FIXED_PARAMS)

    path = best_params_json_path(season)

    if not path.exists():
        raise FileNotFoundError(
            f"保存済みOptunaパラメータがありません: {path}\n"
            "RUN_OPTUNA_SEARCH=Trueで先に探索してください。"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def prepare_all_best_params() -> Dict[str, Dict[str, object]]:
    if OPTUNA_MODE not in {
        "all_only",
        "by_season",
        "off",
    }:
        raise ValueError(
            f"OPTUNA_MODE={OPTUNA_MODE!r} は無効です。"
        )

    params_by_season = {}

    if OPTUNA_MODE == "all_only":
        if RUN_OPTUNA_SEARCH:
            all_params = run_optuna_for_season(
                "all",
                DATA_FILES["all"],
            )
        else:
            all_params = load_best_params("all")

        for season in SEASONS_TO_RUN:
            params_by_season[season] = dict(all_params)

    elif OPTUNA_MODE == "by_season":
        for season in SEASONS_TO_RUN:
            if RUN_OPTUNA_SEARCH:
                params = run_optuna_for_season(
                    season,
                    DATA_FILES[season],
                )
            else:
                params = load_best_params(season)

            params_by_season[season] = dict(params)

    else:
        for season in SEASONS_TO_RUN:
            params_by_season[season] = dict(FIXED_PARAMS)

    return params_by_season


# ============================================================
# Final evaluation
# ============================================================

def save_prediction_csv(
    output_path: Path,
    season: str,
    seed: int,
    data: Dict[str, object],
    params: Dict[str, object],
    evaluation: Dict[str, object],
    best_val: float,
    best_epoch: int,
):
    preds = evaluation["preds"]
    trues = evaluation["trues"]

    rows = []

    test_starts = np.asarray(
        data["test_starts"]
    )

    for sample_index in range(len(preds)):
        start_idx = int(test_starts[sample_index])

        ts_slice = data["all_ts"].iloc[
            start_idx + SEQ_LEN:
            start_idx + SEQ_LEN + HORIZON
        ].to_numpy()

        time_slice = data["all_time"].iloc[
            start_idx + SEQ_LEN:
            start_idx + SEQ_LEN + HORIZON
        ].to_numpy()

        for h in range(HORIZON):
            ts = pd.to_datetime(ts_slice[h])

            rows.append({
                "timestamp": ts,
                "date": ts.date().isoformat(),
                "time": int(time_slice[h]),
                "hour": int(ts.hour),
                "minute": int(ts.minute),
                "horizon": int(h + 1),
                "y_true": float(trues[sample_index, h]),
                "y_pred": float(preds[sample_index, h]),
            })

    output = pd.DataFrame(rows)

    output["model"] = "LSTM"
    output["seed"] = int(seed)
    output["season"] = season
    output["seq_len"] = SEQ_LEN
    output["pred_len"] = HORIZON
    output["seq_mode"] = SEQUENCE_MODE
    output["strict_split"] = int(STRICT_SPLIT)
    output["optuna_mode"] = OPTUNA_MODE
    output["best_val_loss"] = float(best_val)
    output["best_epoch"] = int(best_epoch)
    output["feature_set"] = (
        "power_demand+weather+calendar"
        if USE_POWER_DEMAND_AS_INPUT
        else "weather+calendar"
    )

    for key, value in params.items():
        output[key] = value

    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def run_final_one(
    season: str,
    seed: int,
    params: Dict[str, object],
) -> Dict[str, object]:
    seed_everything(seed)

    data = prepare_data(
        DATA_FILES[season],
        sequence_mode=SEQUENCE_MODE,
    )

    train_loader = DataLoader(
        TimeSeriesDataset(
            data["Xtr"],
            data["ytr"],
        ),
        batch_size=int(params["batch_size"]),
        shuffle=True,
    )

    val_loader = DataLoader(
        TimeSeriesDataset(
            data["Xva"],
            data["yva"],
        ),
        batch_size=int(params["batch_size"]),
        shuffle=False,
    )

    test_loader = DataLoader(
        TimeSeriesDataset(
            data["Xte"],
            data["yte"],
        ),
        batch_size=int(params["batch_size"]),
        shuffle=False,
    )

    model = LSTMOneShot(
        input_dim=int(data["input_dim"]),
        hidden_size=int(params["hidden_size"]),
        num_layers=int(params["num_layers"]),
        dropout=float(params["dropout"]),
        horizon=HORIZON,
        head_hidden=int(params["head_hidden"]),
    ).to(device)

    seed_model_dir = MODEL_DIR / f"seed_{seed}"
    seed_pred_dir = PRED_DIR / f"seed_{seed}"
    seed_fig_dir = FIG_DIR / f"seed_{seed}"

    for d in [
        seed_model_dir,
        seed_pred_dir,
        seed_fig_dir,
    ]:
        d.mkdir(
            parents=True,
            exist_ok=True,
        )

    run_name = (
        f"LSTM_{season}_{SEQUENCE_MODE}_"
        f"strict{int(STRICT_SPLIT)}_"
        f"{OPTUNA_MODE}_seed{seed}"
    )

    model_path = seed_model_dir / f"{run_name}.pth"
    pred_path = seed_pred_dir / f"{run_name}.csv"
    fig_path = seed_fig_dir / f"{run_name}.png"

    print("\n" + "=" * 100)
    print(
        f"FINAL RUN | model=LSTM | season={season} "
        f"| seed={seed} | optuna_mode={OPTUNA_MODE}"
    )
    print(f"params={params}")
    print(
        f"sequence train/val/test="
        f"{len(data['ytr'])}/"
        f"{len(data['yva'])}/"
        f"{len(data['yte'])}"
    )
    print("=" * 100)

    best_val, best_epoch, last_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        params=params,
        max_epochs=FINAL_MAX_EPOCHS,
        patience=FINAL_PATIENCE,
        save_path=model_path,
        trial=None,
    )

    state_dict = torch.load(
        model_path,
        map_location=device,
    )
    model.load_state_dict(state_dict)

    evaluation = evaluate_model(
        model=model,
        test_loader=test_loader,
        scaler_y=data["scaler_y"],
    )

    save_prediction_csv(
        output_path=pred_path,
        season=season,
        seed=seed,
        data=data,
        params=params,
        evaluation=evaluation,
        best_val=best_val,
        best_epoch=best_epoch,
    )

    plot_data = pd.read_csv(pred_path)
    plot_data["timestamp"] = pd.to_datetime(
        plot_data["timestamp"]
    )

    plot_data = (
        plot_data
        .groupby("timestamp", as_index=False)[
            ["y_true", "y_pred"]
        ]
        .mean()
        .sort_values("timestamp")
    )

    plt.figure(figsize=(12, 4))
    plt.plot(
        plot_data["timestamp"],
        plot_data["y_true"],
        label="true",
    )
    plt.plot(
        plot_data["timestamp"],
        plot_data["y_pred"],
        label="LSTM",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        fig_path,
        dpi=200,
    )
    plt.close()

    print(
        f"RESULT | RMSE={evaluation['rmse']:.4f} MW "
        f"| MAPE={evaluation['mape']:.4f}% "
        f"| best_epoch={best_epoch}"
    )
    print(f"Model: {model_path}")
    print(f"Pred : {pred_path}")
    print(f"Fig  : {fig_path}")

    result = {
        "model": "LSTM",
        "seed": int(seed),
        "season": season,
        "rmse": float(evaluation["rmse"]),
        "mape": float(evaluation["mape"]),
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "last_epoch": int(last_epoch),
        "optuna_mode": OPTUNA_MODE,
        "seq_mode": SEQUENCE_MODE,
        "strict_split": int(STRICT_SPLIT),
        "feature_set": (
            "power_demand+weather+calendar"
            if USE_POWER_DEMAND_AS_INPUT
            else "weather+calendar"
        ),
        "input_dim": int(data["input_dim"]),
        "per_day": int(data["per_day"]),
        "step_minutes": int(data["step_minutes"]),
        "has_time_component": bool(
            data["has_time_component"]
        ),
        "model_path": str(model_path),
        "pred_csv": str(pred_path),
        "fig_path": str(fig_path),
        "rmse_per_horizon": (
            evaluation["rmse_per_horizon"].tolist()
        ),
        "mape_per_horizon": (
            evaluation["mape_per_horizon"].tolist()
        ),
    }

    for key, value in params.items():
        result[key] = value

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# Aggregation
# ============================================================

def build_paper_table(
    aggregate_df: pd.DataFrame,
) -> pd.DataFrame:
    output = aggregate_df[[
        "model",
        "season",
        "n_seeds",
        "rmse_mean",
        "rmse_std",
        "mape_mean",
        "mape_std",
        "rmse_cv_percent",
        "mape_cv_percent",
    ]].copy()

    output["RMSE_mean_std_MW"] = output.apply(
        lambda row: (
            f"{row['rmse_mean']:.3f} ± "
            f"{row['rmse_std']:.3f}"
        ),
        axis=1,
    )

    output["MAPE_mean_std_percent"] = output.apply(
        lambda row: (
            f"{row['mape_mean']:.3f} ± "
            f"{row['mape_std']:.3f}"
        ),
        axis=1,
    )

    return output[[
        "model",
        "season",
        "n_seeds",
        "RMSE_mean_std_MW",
        "MAPE_mean_std_percent",
        "rmse_cv_percent",
        "mape_cv_percent",
    ]]


def save_aggregates(
    results: List[Dict[str, object]],
) -> None:
    df = pd.DataFrame(results)

    each_run_df = df.drop(
        columns=[
            "rmse_per_horizon",
            "mape_per_horizon",
        ],
        errors="ignore",
    )

    each_run_df.to_csv(
        EACH_RUN_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    aggregate_df = (
        df.groupby(
            ["model", "season"],
            sort=False,
        )
        .agg(
            n_seeds=("seed", "count"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            rmse_min=("rmse", "min"),
            rmse_max=("rmse", "max"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            mape_min=("mape", "min"),
            mape_max=("mape", "max"),
            best_val_loss_mean=(
                "best_val_loss",
                "mean",
            ),
            best_val_loss_std=(
                "best_val_loss",
                "std",
            ),
        )
        .reset_index()
    )

    numeric_columns = aggregate_df.select_dtypes(
        include=[np.number]
    ).columns

    aggregate_df[numeric_columns] = (
        aggregate_df[numeric_columns]
        .fillna(0.0)
    )

    aggregate_df["rmse_cv_percent"] = (
        aggregate_df["rmse_std"]
        / aggregate_df["rmse_mean"].replace(
            0,
            np.nan,
        )
        * 100.0
    )

    aggregate_df["mape_cv_percent"] = (
        aggregate_df["mape_std"]
        / aggregate_df["mape_mean"].replace(
            0,
            np.nan,
        )
        * 100.0
    )

    aggregate_df.to_csv(
        MEAN_STD_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    paper_table = build_paper_table(
        aggregate_df
    )

    paper_table.to_csv(
        PAPER_TABLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    horizon_rows = []

    for result in results:
        for h in range(HORIZON):
            horizon_rows.append({
                "model": result["model"],
                "seed": int(result["seed"]),
                "season": result["season"],
                "horizon": int(h + 1),
                "rmse": float(
                    result["rmse_per_horizon"][h]
                ),
                "mape": float(
                    result["mape_per_horizon"][h]
                ),
            })

    horizon_df = pd.DataFrame(horizon_rows)

    horizon_df.to_csv(
        HORIZON_EACH_RUN_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    horizon_aggregate_df = (
        horizon_df.groupby(
            ["model", "season", "horizon"],
            sort=False,
        )
        .agg(
            n_seeds=("seed", "count"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
        )
        .reset_index()
    )

    numeric_horizon_columns = (
        horizon_aggregate_df.select_dtypes(
            include=[np.number]
        ).columns
    )

    horizon_aggregate_df[
        numeric_horizon_columns
    ] = (
        horizon_aggregate_df[
            numeric_horizon_columns
        ]
        .fillna(0.0)
    )

    horizon_aggregate_df.to_csv(
        HORIZON_MEAN_STD_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 100)
    print("LSTM MULTI-SEED FINISHED")
    print("=" * 100)
    print(f"Each run       : {EACH_RUN_PATH}")
    print(f"Mean/std       : {MEAN_STD_PATH}")
    print(f"Paper table    : {PAPER_TABLE_PATH}")
    print(f"Horizon runs   : {HORIZON_EACH_RUN_PATH}")
    print(f"Horizon mean/std: {HORIZON_MEAN_STD_PATH}")
    print("\n")
    print(paper_table.to_string(index=False))


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(f"Using device : {device}")
    print(f"Seeds        : {SEEDS}")
    print(f"Seasons      : {SEASONS_TO_RUN}")
    print(f"OPTUNA_MODE  : {OPTUNA_MODE}")
    print(f"Run Optuna   : {RUN_OPTUNA_SEARCH}")
    print(f"Run final    : {RUN_FINAL_EVALUATION}")

    invalid_seasons = [
        season
        for season in SEASONS_TO_RUN
        if season not in DATA_FILES
    ]

    if invalid_seasons:
        raise ValueError(
            f"Unknown seasons: {invalid_seasons}"
        )

    params_by_season = prepare_all_best_params()

    print("\nFinal parameters")
    for season, params in params_by_season.items():
        print(f"  {season}: {params}")

    if not RUN_FINAL_EVALUATION:
        print(
            "\nOptuna探索のみ完了しました。"
            "最終3seed評価は実行していません。"
        )
        return

    results: List[Dict[str, object]] = []

    for seed in SEEDS:
        print("\n" + "#" * 100)
        print(f"START LSTM seed={seed}")
        print("#" * 100)

        for season in SEASONS_TO_RUN:
            result = run_final_one(
                season=season,
                seed=seed,
                params=params_by_season[season],
            )
            results.append(result)

    save_aggregates(results)


if __name__ == "__main__":
    main()
