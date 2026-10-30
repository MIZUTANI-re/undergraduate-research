# ============================================================
# ✅ iTransformer final multi-seed evaluation (equal-budget Optuna HP)
# ✅ 変更点:
#   - RUN_MODEの挙動修正（"all"→全期間 / "seasons"→季節別）
#   - PARAM_MODE追加: "unified"（単一HPで全季節） or "by_season"（季節別HP）
#   - STRICT_SPLIT追加: y(24点)がtrain/val/testで一切重ならない厳密split
#   - warmでもstrictになるように「y窓の所属」でシーケンスを割り当て
#   - pred CSV生成: test_start_i+idxの仮定をやめ、各サンプルのstart_idxを使う
# ============================================================

import os
import random
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt


# =========================
# ✅ ここだけ設定すればOK
# =========================
# 3-seed validation
SEEDS = [42, 123, 2026]

# all・春・夏・秋・冬をすべて実行
PARAM_MODE = "unified"      # "unified" or "by_season"
SEQUENCE_MODE = "warm"      # "warm" or "cold"
STRICT_SPLIT = True
SORT_BY_TIMESTAMP = False

SEQ_LEN = 72
HORIZON = 24

# split比率（時系列の前から順に）
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1  # testは残り

EPOCHS = 200
EARLY_PATIENCE = 20

BASE_DIR = Path(__file__).resolve().parents[2]  # project root (clean_code/)
DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "result" / "result_csv"
MODEL_DIR = BASE_DIR / "result" / "learning_saver" / "iTransformer_multiseed"
FIG_DIR = BASE_DIR / "picture" / "iTransformer_multiseed"
PRED_DIR = RESULT_DIR / "pred_csv_iTransformer_multiseed"

for d in [RESULT_DIR, MODEL_DIR, FIG_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EACH_RUN_PATH = RESULT_DIR / "iTransformer_multiseed_each_run.csv"
MEAN_STD_PATH = RESULT_DIR / "iTransformer_multiseed_mean_std.csv"
PAPER_TABLE_PATH = RESULT_DIR / "iTransformer_multiseed_paper_table.csv"
HORIZON_EACH_RUN_PATH = RESULT_DIR / "iTransformer_multiseed_horizon_each_run.csv"
HORIZON_MEAN_STD_PATH = RESULT_DIR / "iTransformer_multiseed_horizon_mean_std.csv"

# ---- 対象CSV ----
FILE_ALL = str(DATA_DIR / "demand_holiday_with_solar_mean.csv")
SEASONS_FILES = {
    "spring": str(DATA_DIR / "demand_holiday_with_solar_mean_spring.csv"),
    "summer": str(DATA_DIR / "demand_holiday_with_solar_mean_summer.csv"),
    "autumn": str(DATA_DIR / "demand_holiday_with_solar_mean_autumn.csv"),
    "winter": str(DATA_DIR / "demand_holiday_with_solar_mean_winter.csv"),
}

# ---- best params ----
BEST_PARAMS_BY_SEASON = {
    "all": {
        # Equal-budget Optuna:
        # dataset=all, seed=42, trials=30, best_trial=22
        "d_model": 64,
        "nhead": 8,
        "num_layers": 4,
        "hidden_dim": 512,
        "dropout_rate": 0.0881380854399652,
        "batch_size": 64,
        "learning_rate": 0.0002762600136428,
        "weight_decay": 2.9734224636618785e-05,
        "optimizer_name": "AdamW",
    },
}


# =========================
# Seed / Device
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Dataset
# =========================
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# =========================
# iTransformer
# =========================
class VariableEmbedding(nn.Module):
    def __init__(self, seq_len, d_model, in_dim):
        super().__init__()
        self.proj = nn.Linear(seq_len, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, in_dim, d_model))
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, in_dim, seq_len)
        x = self.proj(x)        # (B, in_dim, d_model)
        return x + self.pos_embedding[:, :x.size(1)]

class iTransformerBlock(nn.Module):
    def __init__(self, d_model, hidden_dim, nhead, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(a))
        f = self.ffn(x)
        return self.norm2(x + self.drop(f))

class iTransformer(nn.Module):
    def __init__(self, seq_len, in_dim, pred_len, d_model, hidden_dim, nhead, num_layers, dropout_rate):
        super().__init__()
        self.embed = VariableEmbedding(seq_len, d_model, in_dim)
        self.blocks = nn.Sequential(*[
            iTransformerBlock(d_model, hidden_dim, nhead, dropout_rate) for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x):
        x = self.embed(x)
        x = self.blocks(x)
        return self.head(x.mean(dim=1))


# =========================
# Helpers（TransformerFixed準拠）
# =========================
def create_sequences(X, y, seq_len, horizon):
    Xs, ys = [], []
    for i in range(len(X) - seq_len - horizon + 1):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len:i+seq_len+horizon].flatten())
    return np.array(Xs), np.array(ys)

def infer_step_minutes(df):
    per_day = int(df.groupby(df["datetime"].dt.date)["time"].nunique().mode().iloc[0])
    if 1440 % per_day == 0:
        step_minutes = 1440 // per_day
    else:
        tmax = int(df["time"].max())
        step_minutes = 60 if tmax <= 23 else 30
    return per_day, step_minutes

def build_timestamp(df):
    dt = pd.to_datetime(df["datetime"])
    has_time_component = ((dt.dt.hour != 0) | (dt.dt.minute != 0) | (dt.dt.second != 0)).any()

    per_day, step_minutes = infer_step_minutes(df)
    if has_time_component:
        ts = dt
    else:
        ts = dt + pd.to_timedelta(df["time"] * step_minutes, unit="m")
    return ts, has_time_component, per_day, step_minutes

def inverse_1col(scaler_y, y2d):
    return y2d * scaler_y.scale_[0] + scaler_y.mean_[0]


# =========================
# ✅ 厳密split（warmでもy窓で割り当て）
# =========================
def split_sequences_by_target_window(X_all, y_all, n_raw, tr, va, seq_len, horizon):
    """
    sequence start = i (0..n_seq-1)
    X uses [i, i+seq_len-1]
    y uses [i+seq_len, i+seq_len+horizon-1]
    strict:
      train: y_end < tr
      val  : y_start >= tr and y_end < va
      test : y_start >= va and y_end < n_raw
    """
    n_seq = len(y_all)
    starts = np.arange(n_seq)

    y_start = starts + seq_len
    y_end   = y_start + horizon - 1

    m_tr = (y_end < tr)
    m_va = (y_start >= tr) & (y_end < va)
    m_te = (y_start >= va) & (y_end < n_raw)

    Xtr, ytr, str_tr = X_all[m_tr], y_all[m_tr], starts[m_tr]
    Xva, yva, str_va = X_all[m_va], y_all[m_va], starts[m_va]
    Xte, yte, str_te = X_all[m_te], y_all[m_te], starts[m_te]

    return (Xtr, ytr, str_tr), (Xva, yva, str_va), (Xte, yte, str_te)


# =========================
# summary追記（固定列）
# =========================
def append_summary_row(summary_path: Path, row: dict):
    cols = [
        "season", "rmse", "mape", "best_val_loss",
        "d_model", "nhead", "num_layers", "hidden_dim", "dropout",
        "batch_size", "learning_rate",
        "seq_mode", "strict_split", "per_day", "step_minutes", "has_time_component",
        "param_mode",
        "model_path", "pred_csv", "fig_path"
    ]
    row = dict(row)
    if "dropout" not in row:
        row["dropout"] = float(row.get("dropout_rate", np.nan))

    df_row = pd.DataFrame([row], columns=cols)
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        new = pd.concat([old, df_row], ignore_index=True)
        for c in cols:
            if c not in new.columns:
                new[c] = np.nan
        new = new[cols]
        new.to_csv(summary_path, index=False, encoding="utf-8-sig")
    else:
        df_row.to_csv(summary_path, index=False, encoding="utf-8-sig")


# =========================
# run one dataset
# =========================
def run_one(season: str, file_path: str, seed: int) -> Dict[str, object]:
    seed_everything(seed)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"not found: {file_path}")

    # ✅ param取り出し（単一HP or 季節別）
    if PARAM_MODE == "unified":
        params = BEST_PARAMS_BY_SEASON["all"]
    else:
        params = BEST_PARAMS_BY_SEASON.get(season, BEST_PARAMS_BY_SEASON["all"])

    if params["d_model"] % params["nhead"] != 0:
        raise ValueError(f"d_model({params['d_model']}) % nhead({params['nhead']}) != 0")

    print("\n" + "="*100)
    print(f"✅ RUN | model=iTransformer | seed={seed} | season={season} | param_mode={PARAM_MODE} | seq_mode={SEQUENCE_MODE} | strict_split={STRICT_SPLIT}")
    print(f"   file={file_path}")
    print(f"   params={params}")
    print("="*100)

    # ---- load ----
    df = pd.read_csv(file_path)
    if "time" not in df.columns:
        raise ValueError("CSVに 'time' 列がありません。")

    df["datetime"] = pd.to_datetime(df["datetime"])
    # ✅ 最重要：date + time で順序固定
    df = df.sort_values(["datetime", "time"]).reset_index(drop=True)

    df["timestamp"], has_time_component, per_day, step_minutes = build_timestamp(df)
    if SORT_BY_TIMESTAMP:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # ✅ timestamp基準のカレンダー特徴
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday

    df = df.dropna().reset_index(drop=True)
    df = pd.get_dummies(df, columns=["weekday", "month"], drop_first=True)

    features = [
        "power_demand", "temperature", "precipitation", "time", "day",
        "solar_mean", "restday", "holiday_onehot",
    ] + [c for c in df.columns if c.startswith("weekday_") or c.startswith("month_")]
    target = "power_demand"

    # ---- split（時系列順のまま）----
    n = len(df)
    tr = int(n * TRAIN_RATIO)
    va = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:tr].copy()
    val_df   = df.iloc[tr:va].copy()
    test_df  = df.iloc[va:].copy()

    sx = StandardScaler().fit(train_df[features])
    sy = StandardScaler().fit(train_df[[target]])

    X_tr = sx.transform(train_df[features])
    X_va = sx.transform(val_df[features])
    X_te = sx.transform(test_df[features])

    y_tr = sy.transform(train_df[[target]])
    y_va = sy.transform(val_df[[target]])
    y_te = sy.transform(test_df[[target]])

    # ---- sequences ----
    if SEQUENCE_MODE == "warm":
        all_X = np.concatenate([X_tr, X_va, X_te], axis=0)
        all_y = np.concatenate([y_tr, y_va, y_te], axis=0)

        all_ts   = df["timestamp"].reset_index(drop=True)
        all_time = df["time"].reset_index(drop=True)

        X_all, y_all = create_sequences(all_X, all_y, SEQ_LEN, HORIZON)

        if STRICT_SPLIT:
            (Xtr, ytr, tr_starts), (Xva, yva, va_starts), (Xte, yte, te_starts) = split_sequences_by_target_window(
                X_all, y_all, n_raw=len(all_X), tr=tr, va=va, seq_len=SEQ_LEN, horizon=HORIZON
            )
        else:
            # 従来のゆるい切り方（参考として残す）
            train_seq_end = len(train_df) - SEQ_LEN - HORIZON + 1
            val_seq_end   = len(train_df) + len(val_df) - SEQ_LEN - HORIZON + 1
            Xtr, ytr = X_all[:train_seq_end], y_all[:train_seq_end]
            Xva, yva = X_all[train_seq_end:val_seq_end], y_all[train_seq_end:val_seq_end]
            Xte, yte = X_all[val_seq_end:], y_all[val_seq_end:]
            tr_starts = np.arange(len(Xtr))
            va_starts = np.arange(train_seq_end, val_seq_end)
            te_starts = np.arange(val_seq_end, val_seq_end + len(Xte))

    else:
        # cold: splitごとに個別作成（境界コンテキスト無し）※これ自体が厳密
        Xtr, ytr = create_sequences(X_tr, y_tr, SEQ_LEN, HORIZON)
        Xva, yva = create_sequences(X_va, y_va, SEQ_LEN, HORIZON)
        Xte, yte = create_sequences(X_te, y_te, SEQ_LEN, HORIZON)

        # timestamp参照はtest_df基準
        all_ts   = test_df["timestamp"].reset_index(drop=True)
        all_time = test_df["time"].reset_index(drop=True)

        tr_starts = np.arange(len(Xtr))
        va_starts = np.arange(len(Xva))
        te_starts = np.arange(len(Xte))

    print(f"[{SEQUENCE_MODE}] seq train/val/test = {len(ytr)}, {len(yva)}, {len(yte)}")
    print(f"[time-axis] per_day={per_day}, step_minutes={step_minutes}, has_time_component={has_time_component}")

    if len(ytr) == 0 or len(yva) == 0 or len(yte) == 0:
        raise ValueError("❌ シーケンス数が0です。データ長/SEQ_LEN/HORIZON/split比率を確認してください。")

    # ---- loaders ----
    train_loader = DataLoader(TimeSeriesDataset(Xtr, ytr), batch_size=int(params["batch_size"]), shuffle=True)
    val_loader   = DataLoader(TimeSeriesDataset(Xva, yva), batch_size=int(params["batch_size"]), shuffle=False)
    test_loader  = DataLoader(TimeSeriesDataset(Xte, yte), batch_size=int(params["batch_size"]), shuffle=False)

    # ---- model ----
    model = iTransformer(
        seq_len=SEQ_LEN, in_dim=Xtr.shape[2], pred_len=HORIZON,
        d_model=int(params["d_model"]),
        hidden_dim=int(params["hidden_dim"]),
        nhead=int(params["nhead"]),
        num_layers=int(params["num_layers"]),
        dropout_rate=float(params["dropout_rate"]),
    ).to(device)

    optimizer_cls = getattr(optim, str(params["optimizer_name"]))
    opt = optimizer_cls(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    loss_fn = nn.MSELoss()

    seed_model_dir = MODEL_DIR / f"seed_{seed}"
    seed_pred_dir = PRED_DIR / f"seed_{seed}"
    seed_fig_dir = FIG_DIR / f"seed_{seed}"
    for d in [seed_model_dir, seed_pred_dir, seed_fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    run_name = (
        f"iTransformer_{season}_{SEQUENCE_MODE}_"
        f"strict{int(STRICT_SPLIT)}_{PARAM_MODE}_seed{seed}"
    )
    model_path = seed_model_dir / f"{run_name}.pth"
    pred_csv = seed_pred_dir / f"{run_name}.csv"
    fig_path = seed_fig_dir / f"{run_name}.png"

    # ---- train ----
    best_val = float("inf")
    bad = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                vals.append(loss_fn(model(xb), yb).item())
        val_loss = float(np.mean(vals)) if vals else float("nan")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | val={val_loss:.6f} | best={best_val:.6f}")

        if np.isfinite(val_loss) and (val_loss < best_val):
            best_val = val_loss
            bad = 0
            torch.save(model.state_dict(), str(model_path))
        else:
            bad += 1
            if bad >= EARLY_PATIENCE:
                print(f"⏹ Early stop @ epoch={epoch}")
                break

    # ---- test ----
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()

    preds_s, trues_s = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds_s.append(model(xb).cpu().numpy())
            trues_s.append(yb.numpy())

    preds_s = np.concatenate(preds_s, axis=0)
    trues_s = np.concatenate(trues_s, axis=0)

    preds = inverse_1col(sy, preds_s)
    trues = inverse_1col(sy, trues_s)

    rmse = float(np.sqrt(mean_squared_error(trues.reshape(-1), preds.reshape(-1))))
    mape = float(mean_absolute_percentage_error(trues.reshape(-1), preds.reshape(-1)) * 100.0)
    print(f"✅ TEST | RMSE={rmse:.4f}, MAPE={mape:.4f}%")

    # horizon別指標
    diff = preds - trues
    rmse_per_h = np.sqrt(np.mean(diff ** 2, axis=0))
    denom = np.clip(np.abs(trues), 1e-8, None)
    mape_per_h = np.mean(np.abs(diff) / denom, axis=0) * 100.0

    # ---- pred CSV（TransformerFixedと同じ列）----
    out_rows = []

    # ✅ 各サンプルの開始indexを使う（strict/warmでもズレない）
    for s in range(len(preds)):
        start_idx = int(te_starts[s])

        if SEQUENCE_MODE == "warm":
            # all_ts/all_time は全期間df基準
            ts_slice   = all_ts.iloc[start_idx + SEQ_LEN : start_idx + SEQ_LEN + HORIZON].to_numpy()
            time_slice = all_time.iloc[start_idx + SEQ_LEN : start_idx + SEQ_LEN + HORIZON].to_numpy()
        else:
            # coldはtest_df基準（test内index）
            ts_slice   = all_ts.iloc[start_idx + SEQ_LEN : start_idx + SEQ_LEN + HORIZON].to_numpy()
            time_slice = all_time.iloc[start_idx + SEQ_LEN : start_idx + SEQ_LEN + HORIZON].to_numpy()

        for h in range(HORIZON):
            ts = pd.to_datetime(ts_slice[h])
            out_rows.append([
                ts,
                ts.date().isoformat(),
                int(time_slice[h]),
                int(ts.hour),
                int(ts.minute),
                int(h + 1),
                float(trues[s, h]),
                float(preds[s, h]),
            ])

    df_out = pd.DataFrame(out_rows, columns=[
        "timestamp", "date", "time", "hour", "minute", "horizon", "y_true", "y_pred"
    ])

    # メタ列
    df_out["model"] = "iTransformer"
    df_out["seed"] = int(seed)
    df_out["season"] = season
    df_out["seq_mode"] = SEQUENCE_MODE
    df_out["strict_split"] = int(STRICT_SPLIT)
    df_out["param_mode"] = PARAM_MODE
    for k, v in params.items():
        df_out[k] = v
    df_out["seq_len"] = SEQ_LEN
    df_out["pred_len"] = HORIZON

    df_out.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    print(f"📦 pred保存: {pred_csv}")

    # ---- fig ----
    df_plot = df_out.groupby("timestamp", as_index=False)[["y_true", "y_pred"]].mean().sort_values("timestamp")
    plt.figure(figsize=(12, 4))
    plt.plot(df_plot["timestamp"], df_plot["y_true"], label="true")
    plt.plot(df_plot["timestamp"], df_plot["y_pred"], label="pred")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"🖼️ 図保存: {fig_path}")

    # ---- one-run result ----
    summary_row = {
        "model": "iTransformer",
        "seed": int(seed),
        "season": season,
        "rmse": rmse,
        "mape": mape,
        "best_val_loss": float(best_val),
        "d_model": int(params["d_model"]),
        "nhead": int(params["nhead"]),
        "num_layers": int(params["num_layers"]),
        "hidden_dim": int(params["hidden_dim"]),
        "dropout": float(params["dropout_rate"]),
        "batch_size": int(params["batch_size"]),
        "learning_rate": float(params["learning_rate"]),
        "weight_decay": float(params["weight_decay"]),
        "optimizer": str(params["optimizer_name"]),
        "seq_mode": SEQUENCE_MODE,
        "strict_split": int(STRICT_SPLIT),
        "per_day": int(per_day),
        "step_minutes": int(step_minutes),
        "has_time_component": bool(has_time_component),
        "param_mode": PARAM_MODE,
        "feature_set": "power_demand+weather+calendar",
        "model_path": str(model_path),
        "pred_csv": str(pred_csv),
        "fig_path": str(fig_path),
        "rmse_per_horizon": rmse_per_h.tolist(),
        "mape_per_horizon": mape_per_h.tolist(),
    }
    return summary_row


def build_paper_table(aggregate_df: pd.DataFrame) -> pd.DataFrame:
    out = aggregate_df[[
        "model", "season", "n_seeds",
        "rmse_mean", "rmse_std",
        "mape_mean", "mape_std",
        "rmse_cv_percent", "mape_cv_percent",
    ]].copy()
    out["RMSE_mean_std_MW"] = out.apply(
        lambda r: f"{r['rmse_mean']:.3f} ± {r['rmse_std']:.3f}",
        axis=1,
    )
    out["MAPE_mean_std_percent"] = out.apply(
        lambda r: f"{r['mape_mean']:.3f} ± {r['mape_std']:.3f}",
        axis=1,
    )
    return out[[
        "model", "season", "n_seeds",
        "RMSE_mean_std_MW",
        "MAPE_mean_std_percent",
        "rmse_cv_percent",
        "mape_cv_percent",
    ]]


def main():
    print(f"Using device: {device}")
    print(f"Seeds: {SEEDS}")

    targets = {
        "all": FILE_ALL,
        **SEASONS_FILES,
    }

    results: List[Dict[str, object]] = []

    for seed in SEEDS:
        print("\\n" + "#" * 100)
        print(f"START iTransformer seed={seed}")
        print("#" * 100)

        for season, file_path in targets.items():
            results.append(run_one(season, file_path, seed))

    df = pd.DataFrame(results)

    # list列を除いた各runの表
    each_run_df = df.drop(
        columns=["rmse_per_horizon", "mape_per_horizon"],
        errors="ignore",
    )
    each_run_df.to_csv(
        EACH_RUN_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    aggregate_df = (
        df.groupby(["model", "season"], sort=False)
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
            best_val_loss_mean=("best_val_loss", "mean"),
            best_val_loss_std=("best_val_loss", "std"),
        )
        .reset_index()
        .fillna(0.0)
    )
    aggregate_df["rmse_cv_percent"] = (
        aggregate_df["rmse_std"]
        / aggregate_df["rmse_mean"].replace(0, np.nan)
        * 100.0
    )
    aggregate_df["mape_cv_percent"] = (
        aggregate_df["mape_std"]
        / aggregate_df["mape_mean"].replace(0, np.nan)
        * 100.0
    )
    aggregate_df.to_csv(
        MEAN_STD_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    paper_df = build_paper_table(aggregate_df)
    paper_df.to_csv(
        PAPER_TABLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # horizon別
    horizon_rows = []
    for result in results:
        for h in range(HORIZON):
            horizon_rows.append({
                "model": result["model"],
                "seed": result["seed"],
                "season": result["season"],
                "horizon": h + 1,
                "rmse": result["rmse_per_horizon"][h],
                "mape": result["mape_per_horizon"][h],
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
        .fillna(0.0)
    )
    horizon_aggregate_df.to_csv(
        HORIZON_MEAN_STD_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\\n" + "=" * 100)
    print("iTransformer MULTI-SEED FINISHED")
    print(f"Each run       : {EACH_RUN_PATH}")
    print(f"Mean/std       : {MEAN_STD_PATH}")
    print(f"Paper table    : {PAPER_TABLE_PATH}")
    print(f"Horizon runs   : {HORIZON_EACH_RUN_PATH}")
    print(f"Horizon mean/std: {HORIZON_MEAN_STD_PATH}")
    print("=" * 100)
    print(paper_df.to_string(index=False))


if __name__ == "__main__":
    main()
