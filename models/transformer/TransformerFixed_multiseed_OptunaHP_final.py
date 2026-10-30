#============================================================
# ✅ TransformerFixed final multi-seed evaluation (equal-budget Optuna HP)
# - warm-start（全期間でシーケンス作成）
# - ✅ strict split（評価汚染を避ける境界切り：train/val/test の target が混ざらない）
# - ✅ time特徴量 = 元CSVの "time" をそのまま使う（iTransformerと同じ）
# - ✅ timestamp生成（date+time刻み推定）→ CSV/図は timestamp 基準
# - ✅ scalerは train point のみ fit（リーク防止）
# - ✅ 保存先を iTransformer / Hybrid と統一
#   - model: learning_saver
#   - pred : result_csv
#   - fig  : picture
#   - summary: result_csv/TransformerFixed_summary.csv
# ============================================================

import os
import random
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import matplotlib.pyplot as plt


# -----------------------------
# 保存先（✅ iTransformerと同じ場所）
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # project root (clean_code/)
DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "result" / "result_csv"
MODEL_DIR = BASE_DIR / "result" / "learning_saver" / "TransformerFixed_multiseed"
FIG_DIR = BASE_DIR / "picture" / "TransformerFixed_multiseed"
PRED_DIR = RESULT_DIR / "pred_csv_TransformerFixed_multiseed"

for d in [RESULT_DIR, MODEL_DIR, FIG_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2026]

EACH_RUN_PATH = RESULT_DIR / "TransformerFixed_multiseed_each_run.csv"
MEAN_STD_PATH = RESULT_DIR / "TransformerFixed_multiseed_mean_std.csv"
PAPER_TABLE_PATH = RESULT_DIR / "TransformerFixed_multiseed_paper_table.csv"
HORIZON_EACH_RUN_PATH = RESULT_DIR / "TransformerFixed_multiseed_horizon_each_run.csv"
HORIZON_MEAN_STD_PATH = RESULT_DIR / "TransformerFixed_multiseed_horizon_mean_std.csv"


# -----------------------------
# 基本設定
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Dataset
# -----------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)   # (N, L, feat)
        self.y = torch.tensor(y, dtype=torch.float32)   # (N, H)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# -----------------------------
# Transformer Encoder-Decoder One-shot
# -----------------------------
class TransformerEncoderDecoderOneShot(nn.Module):
    def __init__(
        self,
        feature_size: int,
        target_length: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.target_length = target_length
        self.d_model = d_model

        self.input_projection = nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # decoderへのクエリ（学習パラメータ）: [H, d_model]
        self.query_embed = nn.Parameter(torch.randn(target_length, d_model))

        # one-shot出力: [B, H]（demand 1変数）
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        src: [B, L, feature_size]
        return: [B, H]
        """
        B, L, _ = src.shape
        x = self.input_projection(src)            # [B, L, d_model]
        x = self.pos_encoder(x)                   # [B, L, d_model]
        memory = self.transformer_encoder(x)      # [B, L, d_model]

        query = self.query_embed.unsqueeze(0).repeat(B, 1, 1)     # [B, H, d_model]
        out = self.transformer_decoder(tgt=query, memory=memory)  # [B, H, d_model]
        out = self.output_projection(out).squeeze(-1)             # [B, H]
        return out


# ============================================================
# ハイパーパラメータ利用モード
# - "unified_all": allデータで選んだ1種類のHPを全季節に使用
# - "seasonal"   : 従来どおり季節ごとのHPを使用
#
# 3モデルの公平な比較では "unified_all" を使用します。
# ============================================================
PARAM_MODE = "unified_all"

# ============================================================
# 固定ハイパーパラメータ（あなたのOptuna固定値）
# ============================================================
FIXED_PARAMS = {
    "all": {
        # Equal-budget Optuna:
        # dataset=all, seed=42, trials=30, best_trial=22
        "d_model": 256,
        "nhead": 16,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "dim_feedforward": 1024,
        "dropout": 0.0184519628461586,
        "batch_size": 32,
        "learning_rate": 2.742172327617993e-05,
        "weight_decay": 7.495531188539501e-05,
        "optimizer_name": "Adam",
    },
}


# -----------------------------
# Utility
# -----------------------------
def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    n = len(X)
    for i in range(n - seq_len - horizon + 1):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len : i + seq_len + horizon].flatten())
    return np.array(Xs), np.array(ys)

def inverse_1col(scaler_y: StandardScaler, y_scaled_2d: np.ndarray) -> np.ndarray:
    mean = scaler_y.mean_[0]
    scale = scaler_y.scale_[0]
    return y_scaled_2d * scale + mean

def append_summary_csv(summary_path: Path, row: Dict[str, object]) -> None:
    df_row = pd.DataFrame([row])
    if summary_path.exists():
        df_old = pd.read_csv(summary_path)
        df_new = pd.concat([df_old, df_row], ignore_index=True)
        df_new.to_csv(summary_path, index=False, encoding="utf-8-sig")
    else:
        df_row.to_csv(summary_path, index=False, encoding="utf-8-sig")


# -----------------------------
# config
# -----------------------------
@dataclass
class RunConfig:
    seq_len: int = 72
    horizon: int = 24
    max_epochs: int = 200
    patience: int = 30
    seed: int = 42
    model_name: str = "TransformerFixed"
    summary_name: str = "TransformerFixed_summary.csv"


# -----------------------------
# main per season
# -----------------------------
def process_season(season_name: str, file_path: str, cfg: RunConfig) -> Dict[str, object]:
    seed_everything(cfg.seed)
    if "all" not in FIXED_PARAMS:
        raise ValueError("FIXED_PARAMSに 'all' がありません。")

    if PARAM_MODE == "unified_all":
        # allデータで選んだHPを、spring/summer/autumn/winterにも固定使用
        params = FIXED_PARAMS["all"].copy()
        hp_source = "all"
    elif PARAM_MODE == "seasonal":
        # 従来の季節別HP
        params = FIXED_PARAMS[season_name].copy()
        hp_source = season_name
    else:
        raise ValueError(
            f"PARAM_MODE={PARAM_MODE!r} は無効です。"
            " 'unified_all' または 'seasonal' を指定してください。"
        )
    L = cfg.seq_len
    H = cfg.horizon

    print("\n" + "=" * 90)
    print(f"✅ [{season_name}] TransformerFixed warm-start STRICT | seed={cfg.seed} | file={file_path}")
    print(f"   seq_len={L}, horizon={H}, epochs={cfg.max_epochs}, patience={cfg.patience}")
    print(f"   PARAM_MODE={PARAM_MODE} | HP source={hp_source}")
    print("   hyperparams:", params)
    print("=" * 90)

    # ---- load ----
    df = pd.read_csv(file_path)

    required_cols = ["datetime", "time", "power_demand"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"CSVに必須列 '{c}' がありません。現在の列: {list(df.columns)}")

    df["datetime"] = pd.to_datetime(df["datetime"])

    # ✅ date+time で時系列順保証
    df = df.sort_values(["datetime", "time"]).reset_index(drop=True)

    # ✅ time刻み推定（24->60分, 48->30分, 96->15分…）
    per_day = int(df.groupby(df["datetime"].dt.date)["time"].nunique().mode().iloc[0])
    if 1440 % per_day == 0:
        step_minutes = 1440 // per_day
    else:
        tmax = int(df["time"].max())
        step_minutes = 60 if tmax <= 23 else 30

    # ✅ timestamp生成（datetimeが日付だけなら date + time*step_minutes）
    has_time_component = (
        (df["datetime"].dt.hour != 0) |
        (df["datetime"].dt.minute != 0) |
        (df["datetime"].dt.second != 0)
    ).any()

    if has_time_component:
        df["timestamp"] = df["datetime"]
    else:
        df["timestamp"] = df["datetime"] + pd.to_timedelta(df["time"] * step_minutes, unit="m")

    # ---- feature engineering（timestamp基準） ----
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday

    # 公平探索時と同じ入力情報に統一するため、
    # 明示的な lag_1 / lag_24 / lag_48 は追加しない。
    # 過去需要は72点の入力系列としてモデルへ与える。
    target = "power_demand"

    df = df.dropna().reset_index(drop=True)

    # ✅ 念のためtimestampで再ソート
    df = df.sort_values(["timestamp"]).reset_index(drop=True)

    # one-hot
    df = pd.get_dummies(df, columns=["weekday", "month"], drop_first=True)

    # 特徴量（✅ timeは元CSVのtime）
    base_features = [
        "power_demand",     # 直近需要（入力として使う）
        "temperature",
        "precipitation",
        "time",             # ✅ 元CSV time
        "day",
        "solar_mean",
        "restday",
        "holiday_onehot",
    ]
    oh_features = [c for c in df.columns if c.startswith("weekday_") or c.startswith("month_")]
    features = base_features + oh_features

    # ---- point split (80/10/10) ----
    n = len(df)
    tr = int(n * 0.8)
    va = int(n * 0.9)

    train_df = df.iloc[:tr].copy()
    val_df   = df.iloc[tr:va].copy()
    test_df  = df.iloc[va:].copy()

    # ✅ scalerは train point のみfit（リーク防止）
    scaler_X = StandardScaler().fit(train_df[features])
    scaler_y = StandardScaler().fit(train_df[[target]])

    X_tr = scaler_X.transform(train_df[features])
    X_va = scaler_X.transform(val_df[features])
    X_te = scaler_X.transform(test_df[features])

    y_tr = scaler_y.transform(train_df[[target]])
    y_va = scaler_y.transform(val_df[[target]])
    y_te = scaler_y.transform(test_df[[target]])

    # =========================================================
    # ✅ warm-start: 全期間concat → sequence作成
    # =========================================================
    all_X = np.concatenate([X_tr, X_va, X_te], axis=0)
    all_y = np.concatenate([y_tr, y_va, y_te], axis=0)

    # 時刻付け（all_X/all_yと1:1対応する point 系列）
    all_ts   = df["timestamp"].reset_index(drop=True)
    all_time = df["time"].reset_index(drop=True)

    X_seq_all, y_seq_all = make_sequences(all_X, all_y, L, H)

    # =========================================================
    # ✅ STRICT split（重要）
    # - train: target が train に完全に収まる
    # - val  : target が val に完全に収まる（入力は train を含んでOK）
    # - test : target が test に完全に収まる（入力は val を含んでOK）
    # =========================================================
    train_i_end  = tr - L - H + 1
    val_i_start  = tr - L
    val_i_end    = va - L - H + 1
    test_i_start = va - L

    if train_i_end <= 0 or val_i_end <= val_i_start or test_i_start < 0:
        raise ValueError("split indexが不正です。seq_len/horizonに対してデータが少なすぎます。")

    Xtr, ytr = X_seq_all[:train_i_end],            y_seq_all[:train_i_end]
    Xva, yva = X_seq_all[val_i_start:val_i_end],   y_seq_all[val_i_start:val_i_end]
    Xte, yte = X_seq_all[test_i_start:],           y_seq_all[test_i_start:]

    print(f"[warm-start strict] train_i_end={train_i_end}, val_i=[{val_i_start},{val_i_end}), test_i_start={test_i_start}")
    print(f"[warm-start strict] Xtr={Xtr.shape}, Xva={Xva.shape}, Xte={Xte.shape}")
    print(f"[time-axis] per_day={per_day}, step_minutes={step_minutes}, has_time_component={has_time_component}")

    train_loader = DataLoader(TimeSeriesDataset(Xtr, ytr), batch_size=int(params["batch_size"]), shuffle=True)
    val_loader   = DataLoader(TimeSeriesDataset(Xva, yva), batch_size=int(params["batch_size"]), shuffle=False)
    test_loader  = DataLoader(TimeSeriesDataset(Xte, yte), batch_size=int(params["batch_size"]), shuffle=False)

    # ---- model ----
    model = TransformerEncoderDecoderOneShot(
        feature_size=Xtr.shape[2],
        target_length=H,
        d_model=int(params["d_model"]),
        nhead=int(params["nhead"]),
        num_encoder_layers=int(params["num_encoder_layers"]),
        num_decoder_layers=int(params["num_decoder_layers"]),
        dim_feedforward=int(params["dim_feedforward"]),
        dropout=float(params["dropout"]),
    ).to(device)

    optimizer_cls = getattr(optim, str(params["optimizer_name"]))
    optimizer = optimizer_cls(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    criterion = nn.MSELoss()

    # ---- paths ----
    seed_model_dir = MODEL_DIR / f"seed_{cfg.seed}"
    seed_pred_dir = PRED_DIR / f"seed_{cfg.seed}"
    seed_fig_dir = FIG_DIR / f"seed_{cfg.seed}"
    for d in [seed_model_dir, seed_pred_dir, seed_fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    run_name = f"{cfg.model_name}_{season_name}_seed{cfg.seed}"
    best_model_path = seed_model_dir / f"{run_name}.pth"
    pred_csv_path = seed_pred_dir / f"{run_name}.csv"
    fig_path = seed_fig_dir / f"{run_name}.png"

    # ---- train with early stopping ----
    best_val = float("inf")
    best_epoch = -1
    bad = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        tr_losses = []

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                pred = model(Xb)
                va_losses.append(criterion(pred, yb).item())

        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va_loss = float(np.mean(va_losses)) if va_losses else float("nan")

        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            bad = 0
            torch.save(model.state_dict(), str(best_model_path))
        else:
            bad += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | train={tr_loss:.6f} | val={va_loss:.6f} | best_val={best_val:.6f} (best@{best_epoch})")

        if bad >= cfg.patience:
            print(f"⏹ Early stopping at epoch={epoch} (best@{best_epoch})")
            break

    # ---- test ----
    model.load_state_dict(torch.load(str(best_model_path), map_location=device))
    model.eval()

    preds_scaled = []
    trues_scaled = []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            pred = model(Xb).cpu().numpy()
            preds_scaled.append(pred)
            trues_scaled.append(yb.numpy())

    preds_scaled = np.concatenate(preds_scaled, axis=0)  # [N, H]
    trues_scaled = np.concatenate(trues_scaled, axis=0)  # [N, H]

    preds = inverse_1col(scaler_y, preds_scaled)
    trues = inverse_1col(scaler_y, trues_scaled)

    rmse = float(np.sqrt(mean_squared_error(trues.reshape(-1), preds.reshape(-1))))
    mape = float(mean_absolute_percentage_error(trues.reshape(-1), preds.reshape(-1)) * 100.0)

    print(f"✅ TEST | RMSE={rmse:.4f}, MAPE={mape:.4f}%")

    # horizon別指標
    diff = preds - trues
    rmse_per_h = np.sqrt(np.mean(diff ** 2, axis=0))
    denom = np.clip(np.abs(trues), 1e-8, None)
    mape_per_h = np.mean(np.abs(diff) / denom, axis=0) * 100.0

    # =========================================================
    # ✅ 予測CSV（timestamp/time/horizon付き）
    # =========================================================
    # test の sequence start index は test_i_start（strict split）
    out_rows = []
    for s in range(len(preds)):
        i = test_i_start + s  # sequence start (point index)

        ts_slice   = all_ts.iloc[i + L : i + L + H].to_numpy()
        time_slice = all_time.iloc[i + L : i + L + H].to_numpy()

        for h in range(H):
            ts = pd.to_datetime(ts_slice[h])
            out_rows.append([
                ts,                         # timestamp
                ts.date().isoformat(),      # date
                int(time_slice[h]),         # time（元CSV）
                int(ts.hour),               # hour
                int(ts.minute),             # minute
                int(h + 1),                 # horizon (1..H)
                float(trues[s, h]),         # y_true
                float(preds[s, h]),         # y_pred
            ])

    out = pd.DataFrame(out_rows, columns=[
        "timestamp", "date", "time", "hour", "minute", "horizon", "y_true", "y_pred"
    ])
    out["model"] = cfg.model_name
    out["seed"] = int(cfg.seed)
    out["season"] = season_name
    out["param_mode"] = PARAM_MODE
    out["hp_source"] = hp_source
    out["seq_len"] = int(L)
    out["pred_len"] = int(H)
    out["feature_set"] = "power_demand+weather+calendar"
    out.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
    print(f"📦 pred保存: {pred_csv_path}")

    # =========================================================
    # ✅ 図（重複timestampは平均して1本化）
    # =========================================================
    df_plot = out.groupby("timestamp", as_index=False)[["y_true", "y_pred"]].mean().sort_values("timestamp")
    plt.figure(figsize=(12, 4))
    plt.plot(df_plot["timestamp"], df_plot["y_true"], label="true")
    plt.plot(df_plot["timestamp"], df_plot["y_pred"], label="pred")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"🖼️ 図保存: {fig_path}")

    # =========================================================
    # one-run result
    # =========================================================
    summary_row = {
        "date_run": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": cfg.model_name,
        "seed": int(cfg.seed),
        "season": season_name,
        "param_mode": PARAM_MODE,
        "hp_source": hp_source,
        "file": os.path.basename(file_path),
        "seq_len": int(L),
        "horizon": int(H),
        "rmse": rmse,
        "mape": mape,
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "d_model": int(params["d_model"]),
        "nhead": int(params["nhead"]),
        "enc_layers": int(params["num_encoder_layers"]),
        "dec_layers": int(params["num_decoder_layers"]),
        "ff": int(params["dim_feedforward"]),
        "dropout": float(params["dropout"]),
        "batch_size": int(params["batch_size"]),
        "learning_rate": float(params["learning_rate"]),
        "weight_decay": float(params["weight_decay"]),
        "optimizer": str(params["optimizer_name"]),
        "feature_set": "power_demand+weather+calendar",
        "best_model_path": str(best_model_path),
        "pred_csv": str(pred_csv_path),
        "fig_path": str(fig_path),
        "per_day": int(per_day),
        "step_minutes": int(step_minutes),
        "has_time_component": bool(has_time_component),
        "train_i_end": int(train_i_end),
        "val_i_start": int(val_i_start),
        "val_i_end": int(val_i_end),
        "test_i_start": int(test_i_start),
        "rmse_per_horizon": rmse_per_h.tolist(),
        "mape_per_horizon": mape_per_h.tolist(),
    }

    print("=== OUTPUT SUMMARY ===")
    print(f"Model : {best_model_path}")
    print(f"CSV   : {pred_csv_path}")
    print(f"FIG   : {fig_path}")
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


# -----------------------------
# 実行
# -----------------------------
if __name__ == "__main__":
    seasons_files = {
        "all":    str(DATA_DIR / "demand_holiday_with_solar_mean.csv"),
        "spring": str(DATA_DIR / "demand_holiday_with_solar_mean_spring.csv"),
        "summer": str(DATA_DIR / "demand_holiday_with_solar_mean_summer.csv"),
        "autumn": str(DATA_DIR / "demand_holiday_with_solar_mean_autumn.csv"),
        "winter": str(DATA_DIR / "demand_holiday_with_solar_mean_winter.csv"),
    }

    missing = [k for k, p in seasons_files.items() if not os.path.exists(p)]
    if missing:
        print("\\n[ERROR] 次のseasonのCSVが見つかりません:")
        for key in missing:
            print(f"  - {key}: {seasons_files[key]}")
        raise FileNotFoundError("CSV path is invalid.")

    print(f"Using device: {device}")
    print(f"Seeds: {SEEDS}")
    print(f"PARAM_MODE: {PARAM_MODE}")
    print("Unified HP source: all")

    results: List[Dict[str, object]] = []

    for seed in SEEDS:
        print("\\n" + "#" * 100)
        print(f"START TransformerFixed seed={seed}")
        print("#" * 100)

        cfg = RunConfig(
            seq_len=72,
            horizon=24,
            max_epochs=200,
            patience=20,
            seed=seed,
            model_name="TransformerFixed",
            summary_name="unused.csv",
        )

        for season, path in seasons_files.items():
            results.append(process_season(season, path, cfg))

    df = pd.DataFrame(results)

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

    horizon_rows = []
    for result in results:
        for h in range(result["horizon"]):
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
    print("TransformerFixed MULTI-SEED FINISHED")
    print(f"Each run        : {EACH_RUN_PATH}")
    print(f"Mean/std        : {MEAN_STD_PATH}")
    print(f"Paper table     : {PAPER_TABLE_PATH}")
    print(f"Horizon runs    : {HORIZON_EACH_RUN_PATH}")
    print(f"Horizon mean/std: {HORIZON_MEAN_STD_PATH}")
    print("=" * 100)
    print(paper_df.to_string(index=False))