# ============================================================
# ✅ HybridHQ final multi-seed evaluation (equal-budget Optuna HP)
#    - HQあり（24本query）
#    - warm-start strict split（targetリーク防止）
#    - ✅ 並列FiLM融合（feature fusion）:
#         (1) t_feat から (gamma_tv,beta_tv) を生成し v_feat を変調 → v_mod
#             v_mod = v_feat * (1 + tanh(gamma_tv)) + beta_tv
#         (2) v_feat から (gamma_vt,beta_vt) を生成し t_feat を変調 → t_mod
#             t_mod = t_feat * (1 + tanh(gamma_vt)) + beta_vt
#         (3) fused = fuse_proj([t_mod, v_mod]) で統合（predict融合はしない）
#
#    - FiLM強度レポート:
#         alpha_tv(B,24) = mean_d( (tanh(gamma_tv)+1)/2 )  # t→v 変調強度
#         alpha_vt(B,24) = mean_d( (tanh(gamma_vt)+1)/2 )  # v→t 変調強度
#       ※「時間側採用率」ではなく「相互変調の強さ」
#
#    - ✅ 追加（難易度分析）:
#         horizon別RMSE/MAPE（h=1..24）を出力＆resultsに保存
#         early(1..6) vs late(19..24) の比較も出力＆保存
#
#    - pred CSV columns: timestamp,date,time,hour,minute,horizon,y_true,y_pred（比較用に固定）
#    - SEQUENCE_MODE: "warm" or "cold"
#    - CNN_MODE: "cnn" / "cnn1" / "linear"
#    - ✅ Multi-seed validation: SEEDS=[42,123,2026]
#    - seed別モデル・予測CSV保存
#    - 季節別 mean ± standard deviation 自動集計
# ============================================================

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


# =========================
# ✅ ここだけ触ればOK
# =========================
CNN_MODE = "linear"       # "cnn"=CNNあり(kernel=HP), "cnn1"=kernel=1, "linear"=CNN無し
SEQUENCE_MODE = "warm"    # "warm" or "cold"
SORT_BY_TIMESTAMP = False # 通常False（datetime+timeで並んでいる前提）

# HQの条件付けモード
#  - "shared": t_lastから作ったcondを全hに同じだけ足す（互換）
#  - "per_h" : t_lastからh別のcondを生成して足す（改善案）
HQ_COND_MODE = "shared"

# HQの安定化（必要なときだけ上げる）
HQ_L2_WEIGHT = 0.0        # 例: 1e-6 ～ 1e-4
HQ_NOISE_STD = 0.0        # 学習時にqueryへ微小ノイズ（例: 0.01）
USE_Q_SKIP   = True       # q（query）を後段にスキップして安定化

# ✅ FiLM設定
FILM_USE_TANH = True      # True推奨（downscaleも可能で安定）
# 並列FiLMのレポート（alpha_tv/alpha_vt）を保存する
FILM_REPORT_BIDIR = True

HP = {
    # Equal-budget Optuna:
    # dataset=all, seed=42, trials=30, best_trial=2
    "d_model": 128,
    "num_layers": 1,
    "hidden_dim": 1024,
    "dropout": 0.036611470453433645,
    "learning_rate": 3.292759134423613e-05,
    "weight_decay": 0.0002054051942538847,
    "batch_size": 128,
    "optimizer_name": "AdamW",
    "cnn_kernel": 3,
    "nhead": 4,
}

EPOCHS_FINAL = 200
EARLY_PATIENCE = 20
MAX_GRAD_NORM = 1.0

BASE_DIR = Path(__file__).resolve().parents[2]  # project root (clean_code/)
DATA_DIR = BASE_DIR / "data"
DATA_FILES = {
    "all":    DATA_DIR / "demand_holiday_with_solar_mean.csv",
    "spring": DATA_DIR / "demand_holiday_with_solar_mean_spring.csv",
    "summer": DATA_DIR / "demand_holiday_with_solar_mean_summer.csv",
    "autumn": DATA_DIR / "demand_holiday_with_solar_mean_autumn.csv",
    "winter": DATA_DIR / "demand_holiday_with_solar_mean_winter.csv",
}

# ============================================================
# ✅ Multi-seed settings
#   査読コメント「複数のrandom seed + standard deviation」への対応
# ============================================================
SEEDS = [42, 123, 2026]

# ============================================================
# ✅ 保存先を他の2モデルと統一
# ============================================================
RESULT_DIR = BASE_DIR / "result" / "result_csv"
MODEL_DIR = BASE_DIR / "result" / "learning_saver" / "HybridHQ_multiseed"
PRED_DIR = RESULT_DIR / "pred_csv_HybridHQ_multiseed"
SUMMARY_DIR = RESULT_DIR

for d in [RESULT_DIR, MODEL_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 72
HORIZON = 24

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_RUNS_PATH = RESULT_DIR / "HybridHQ_multiseed_each_run.csv"
AGGREGATE_PATH = RESULT_DIR / "HybridHQ_multiseed_mean_std.csv"
PAPER_TABLE_PATH = RESULT_DIR / "HybridHQ_multiseed_paper_table.csv"
HORIZON_ALL_PATH = RESULT_DIR / "HybridHQ_multiseed_horizon_each_run.csv"
HORIZON_AGG_PATH = RESULT_DIR / "HybridHQ_multiseed_horizon_mean_std.csv"


# =========================
# Seed
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


# =========================
# Dataset
# =========================
class TimeSeriesDatasetHQ(Dataset):
    def __init__(self, X_exog, Y_past, Y_future):
        self.X_exog = torch.tensor(X_exog, dtype=torch.float32)
        self.Y_past = torch.tensor(Y_past, dtype=torch.float32)
        self.Y_fut  = torch.tensor(Y_future, dtype=torch.float32)

    def __len__(self):
        return len(self.Y_fut)

    def __getitem__(self, idx):
        return self.X_exog[idx], self.Y_past[idx], self.Y_fut[idx]


# =========================
# Model Components
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class VariableAxisTransformer(nn.Module):
    """
    iTransformer風：tokens = variables(exog_dim)
    入力 (B, seq, exog_dim) -> (B, exog_dim, seq) にして
    seqをLinear(seq->d)で潰してから、変数同士のSelf-Attn
    """
    def __init__(self, seq_len, d_model, nhead, num_layers, hidden_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(seq_len, d_model)
        self.pos  = PositionalEncoding(d_model, dropout)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=num_layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, exog_dim, seq_len)
        x = self.proj(x)        # (B, exog_dim, d_model)
        x = self.pos(x)
        return self.enc(x)      # (B, exog_dim, d_model)


class TemporalEmbeddingCNN(nn.Module):
    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            1, d_model, kernel_size=kernel_size,
            padding=padding, padding_mode='replicate'
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,1,seq)
        x = self.conv(x)        # (B,d,seq)
        x = self.act(x)
        x = x.permute(0, 2, 1)  # (B,seq,d)
        return self.dropout(x)


class TemporalEmbeddingLinear(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(x)        # (B,seq,d)
        x = self.act(x)
        return self.dropout(x)


# ============================================================
# ✅ HQあり + Parallel (Bi-directional) FiLM Feature Fusion + Report
#   - t_feat, v_feat を両方向FiLMで相互変調してから融合
#   - predは融合後の fused からのみ生成（predict融合しない）
# ============================================================
class ImprovedIntegratedHQTransformer_ParallelFiLMFeatureFusionReport(nn.Module):
    def __init__(self, seq_len, exog_dim, horizon, d_model, nhead, num_layers, hidden_dim,
                 dropout, cnn_kernel, cnn_mode="cnn", hq_cond_mode="shared",
                 hq_noise_std=0.0, use_q_skip=True,
                 film_use_tanh=True):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.hq_cond_mode = hq_cond_mode
        self.hq_noise_std = float(hq_noise_std)
        self.use_q_skip = bool(use_q_skip)
        self.film_use_tanh = bool(film_use_tanh)

        # temporal branch
        if cnn_mode == "linear":
            self.temp_embed = TemporalEmbeddingLinear(d_model, dropout=dropout)
        else:
            k = 1 if cnn_mode == "cnn1" else cnn_kernel
            self.temp_embed = TemporalEmbeddingCNN(d_model, kernel_size=k, dropout=dropout)

        self.temp_pos = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, hidden_dim, dropout, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers)

        # variable branch
        self.var_axis = VariableAxisTransformer(seq_len, d_model, nhead, num_layers, hidden_dim, dropout)

        # HQ: 24 queries
        self.hq = nn.Parameter(torch.randn(1, horizon, d_model))

        # cond for q
        if hq_cond_mode == "per_h":
            self.q_cond_h = nn.Linear(d_model, horizon * d_model)  # (B,d)->(B,24*d)
        else:
            self.q_cond = nn.Linear(d_model, d_model)              # (B,d)->(B,d)

        # cross-attn
        self.cross_time = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_var  = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)

        # per-branch feature
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm_out_t = nn.LayerNorm(d_model)
        self.norm_out_v = nn.LayerNorm(d_model)

        # ✅ FiLM: t -> v
        self.film_tv = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * d_model),
        )
        # ✅ FiLM: v -> t
        self.film_vt = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * d_model),
        )

        # ✅ rich fusion: concat(t_mod, v_mod) then project back
        self.fuse_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # refine
        self.ffn_fuse = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm_fuse = nn.LayerNorm(d_model)

        # head（融合後に1本）
        self.head = nn.Linear(d_model, 1)

        # baseline residual
        self.residual_proj = nn.Linear(seq_len, horizon)

    def _gamma_transform(self, gamma_raw: torch.Tensor) -> torch.Tensor:
        # gamma in [-1,1] or [0,1]
        if self.film_use_tanh:
            return torch.tanh(gamma_raw)
        return torch.sigmoid(gamma_raw)

    def _gamma_to_alpha(self, gamma: torch.Tensor) -> torch.Tensor:
        # gamma: (B,24,d) -> alpha: (B,24) 0..1
        if self.film_use_tanh:
            return ((gamma + 1.0) * 0.5).mean(dim=-1)
        return gamma.mean(dim=-1)

    def forward(self, x_ex, y_p, return_alpha=False):
        # memories
        t_m = self.temporal_encoder(self.temp_pos(self.temp_embed(y_p)))  # (B,seq,d)
        v_m = self.var_axis(x_ex)                                         # (B,exog_dim,d)

        B = y_p.size(0)

        # build query q: (B,24,d)
        q = self.hq.expand(B, -1, -1).clone()
        cond_src = t_m[:, -1, :]  # (B,d)

        if self.hq_cond_mode == "per_h":
            cond = self.q_cond_h(cond_src).view(B, self.horizon, -1)      # (B,24,d)
        else:
            cond = self.q_cond(cond_src).unsqueeze(1).expand(B, self.horizon, -1)  # (B,24,d)

        q = q + cond

        # noise (train only)
        if self.training and self.hq_noise_std > 0:
            q = q + torch.randn_like(q) * self.hq_noise_std

        # cross-attn (parallel per horizon)
        qt, _ = self.cross_time(q, t_m, t_m)  # (B,24,d)
        qv, _ = self.cross_var (q, v_m, v_m)  # (B,24,d)

        # representations
        t_rep = self.norm_t(q + qt)
        v_rep = self.norm_v(q + qv)

        # optional q-skip for stability
        if self.use_q_skip:
            t_rep = t_rep + q
            v_rep = v_rep + q

        # per-branch features
        t_feat = self.norm_out_t(t_rep + self.ffn(t_rep))   # (B,24,d)
        v_feat = self.norm_out_v(v_rep + self.ffn(v_rep))   # (B,24,d)

        # =====================================================
        # ✅ Parallel FiLM (Bi-directional)
        #  (1) t -> v で v_feat を変調
        #  (2) v -> t で t_feat を変調
        # =====================================================
        # t -> v
        gb_tv = self.film_tv(t_feat)                         # (B,24,2d)
        gamma_tv_raw, beta_tv = gb_tv.chunk(2, dim=-1)
        gamma_tv = self._gamma_transform(gamma_tv_raw)
        v_mod = v_feat * (1.0 + gamma_tv) + beta_tv          # (B,24,d)

        # v -> t
        gb_vt = self.film_vt(v_feat)                         # (B,24,2d)
        gamma_vt_raw, beta_vt = gb_vt.chunk(2, dim=-1)
        gamma_vt = self._gamma_transform(gamma_vt_raw)
        t_mod = t_feat * (1.0 + gamma_vt) + beta_vt          # (B,24,d)

        # ✅ fusion (feature-level): concat then project
        fused = self.fuse_proj(torch.cat([t_mod, v_mod], dim=-1))  # (B,24,d)

        # refine
        fused = self.norm_fuse(fused + self.ffn_fuse(fused))       # (B,24,d)

        # prediction from fused feature
        pred_delta = self.head(fused).squeeze(-1)                  # (B,24)
        baseline = self.residual_proj(y_p.squeeze(-1))             # (B,24)
        y_pred = baseline + pred_delta

        if return_alpha:
            alpha_tv = self._gamma_to_alpha(gamma_tv)             # (B,24)
            alpha_vt = self._gamma_to_alpha(gamma_vt)             # (B,24)
            # 参考pred（デバッグ用）
            pred_t_ref = self.head(t_feat).squeeze(-1)            # (B,24)
            pred_v_ref = self.head(v_feat).squeeze(-1)            # (B,24)
            return y_pred, alpha_tv, alpha_vt, pred_t_ref, pred_v_ref

        return y_pred


# =========================
# Helpers: timestamp
# =========================
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


# =========================
# Data Prep
# =========================
def make_sequences_hq(exog, y, seq_len, horizon):
    X_ex, Y_past, Y_fut = [], [], []
    for i in range(len(y) - seq_len - horizon + 1):
        X_ex.append(exog[i:i+seq_len])
        Y_past.append(y[i:i+seq_len])
        Y_fut.append(y[i+seq_len:i+seq_len+horizon].reshape(-1))
    return np.array(X_ex), np.array(Y_past), np.array(Y_fut)


def prepare_data(f_path, sequence_mode="warm"):
    df = pd.read_csv(f_path)
    if "time" not in df.columns:
        raise ValueError("CSVに 'time' 列がありません。")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["datetime", "time"]).reset_index(drop=True)

    df["timestamp"], has_time_component, per_day, step_minutes = build_timestamp(df)
    if SORT_BY_TIMESTAMP:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # calendar features
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday

    df = pd.get_dummies(df.dropna(), columns=["weekday", "month"], drop_first=True).reset_index(drop=True)

    ex_f = ["temperature", "precipitation", "time", "day", "solar_mean", "restday", "holiday_onehot"] + \
           [c for c in df.columns if c.startswith(("weekday_", "month_"))]
    target = "power_demand"

    n = len(df)
    tr, va = int(n * 0.8), int(n * 0.9)

    # fit scalers on TRAIN only
    sx = StandardScaler().fit(df.iloc[:tr][ex_f])
    sy = StandardScaler().fit(df.iloc[:tr][[target]])

    if sequence_mode == "warm":
        # full series -> sequences
        X_all = sx.transform(df[ex_f])
        y_all = sy.transform(df[[target]])

        X_seq, Yp_seq, Yf_seq = make_sequences_hq(X_all, y_all, SEQ_LEN, HORIZON)

        # timestamp/time (point index)
        all_ts   = df["timestamp"].reset_index(drop=True)
        all_time = df["time"].reset_index(drop=True)

        # =========================================================
        # ✅ STRICT split
        #  train: targetがtrain内に完全に収まる
        #  val  : targetがval内に完全に収まる（入力はtrainを含んでOK）
        #  test : targetがtest内に完全に収まる（入力はvalを含んでOK）
        # =========================================================
        train_i_end  = tr - SEQ_LEN - HORIZON + 1
        val_i_start  = tr - SEQ_LEN
        val_i_end    = va - SEQ_LEN - HORIZON + 1
        test_i_start = va - SEQ_LEN

        if train_i_end <= 0 or val_i_end <= val_i_start or test_i_start < 0:
            raise ValueError("split indexが不正です。seq_len/horizonに対してデータが少なすぎます。")

        Xtr, Yptr, Yftr = X_seq[:train_i_end],          Yp_seq[:train_i_end],          Yf_seq[:train_i_end]
        Xva, Ypva, Yfva = X_seq[val_i_start:val_i_end], Yp_seq[val_i_start:val_i_end], Yf_seq[val_i_start:val_i_end]
        Xte, Ypte, Yfte = X_seq[test_i_start:],         Yp_seq[test_i_start:],         Yf_seq[test_i_start:]

        return (Xtr, Yptr, Yftr,
                Xva, Ypva, Yfva,
                Xte, Ypte, Yfte, sy, X_seq.shape[2],
                all_ts, all_time, test_i_start,
                per_day, step_minutes, has_time_component,
                {"train_i_end": train_i_end, "val_i_start": val_i_start, "val_i_end": val_i_end, "test_i_start": test_i_start})

    else:
        # cold: split毎にseq作成（境界コンテキスト無し）
        train_df = df.iloc[:tr].copy()
        val_df   = df.iloc[tr:va].copy()
        test_df  = df.iloc[va:].copy()

        X_tr = sx.transform(train_df[ex_f]); y_tr = sy.transform(train_df[[target]])
        X_va = sx.transform(val_df[ex_f]);   y_va = sy.transform(val_df[[target]])
        X_te = sx.transform(test_df[ex_f]);  y_te = sy.transform(test_df[[target]])

        Xtr, Yptr, Yftr = make_sequences_hq(X_tr, y_tr, SEQ_LEN, HORIZON)
        Xva, Ypva, Yfva = make_sequences_hq(X_va, y_va, SEQ_LEN, HORIZON)
        Xte, Ypte, Yfte = make_sequences_hq(X_te, y_te, SEQ_LEN, HORIZON)

        all_ts   = test_df["timestamp"].reset_index(drop=True)
        all_time = test_df["time"].reset_index(drop=True)
        test_start_i = 0

        return (Xtr, Yptr, Yftr,
                Xva, Ypva, Yfva,
                Xte, Ypte, Yfte, sy, Xtr.shape[2],
                all_ts, all_time, test_start_i,
                per_day, step_minutes, has_time_component,
                {"train_i_end": None, "val_i_start": None, "val_i_end": None, "test_i_start": None})


def inverse_1col_2d(scaler_y: StandardScaler, y2d: np.ndarray) -> np.ndarray:
    mu = float(scaler_y.mean_[0])
    sd = float(scaler_y.scale_[0])
    return y2d * sd + mu


# =========================
# Gate stats util（alphaの統計）
# =========================
def summarize_alpha(alpha_all: np.ndarray):
    flat = alpha_all.reshape(-1)
    q = np.quantile(flat, [0, 0.25, 0.5, 0.75, 1.0])
    per_h = alpha_all.mean(axis=0)  # (24,)
    return {
        "alpha_mean": float(flat.mean()),
        "alpha_std":  float(flat.std()),
        "alpha_min":  float(q[0]),
        "alpha_p25":  float(q[1]),
        "alpha_p50":  float(q[2]),
        "alpha_p75":  float(q[3]),
        "alpha_max":  float(q[4]),
        "alpha_per_h": per_h.astype(float),
    }


# =========================
# ✅ Horizon-wise error util（horizon別RMSE/MAPE）
# =========================
def summarize_horizon_errors(trues: np.ndarray, preds: np.ndarray):
    """
    trues/preds: (N, HORIZON) in original scale
    return: dict with rmse_per_h, mape_per_h, early/late stats, worst/best horizon
    """
    assert trues.shape == preds.shape
    H = trues.shape[1]

    diff = preds - trues  # (N,H)
    rmse_per_h = np.sqrt(np.mean(diff ** 2, axis=0))  # (H,)

    denom = np.clip(np.abs(trues), 1e-8, None)
    mape_per_h = np.mean(np.abs(diff) / denom, axis=0) * 100.0  # (H,)

    # early vs late (例: 1..6, 19..24)
    early_rmse_mean = float(rmse_per_h[:6].mean())
    late_rmse_mean  = float(rmse_per_h[-6:].mean())
    late_div_early  = float(late_rmse_mean / early_rmse_mean) if early_rmse_mean > 0 else float("nan")

    # worst/best horizon (1-indexed)
    worst_h_rmse = int(np.argmax(rmse_per_h) + 1)
    best_h_rmse  = int(np.argmin(rmse_per_h) + 1)
    worst_h_mape = int(np.argmax(mape_per_h) + 1)
    best_h_mape  = int(np.argmin(mape_per_h) + 1)

    return {
        "rmse_per_h": rmse_per_h.astype(float),
        "mape_per_h": mape_per_h.astype(float),

        "early_rmse_mean_1_6": early_rmse_mean,
        "late_rmse_mean_19_24": late_rmse_mean,
        "late_div_early_rmse": late_div_early,

        "worst_h_rmse": worst_h_rmse,
        "best_h_rmse": best_h_rmse,
        "worst_h_mape": worst_h_mape,
        "best_h_mape": best_h_mape,
    }


# =========================
# Save pred CSV（列固定）
# =========================
def save_pred_csv(pred_path: Path,
                  season: str,
                  preds: np.ndarray,
                  trues: np.ndarray,
                  all_ts: pd.Series,
                  all_time: pd.Series,
                  test_start_i: int,
                  best_val_loss: float,
                  extra_meta: dict = None):
    out_rows = []
    for s in range(len(preds)):
        i = test_start_i + s
        ts_slice   = all_ts.iloc[i + SEQ_LEN : i + SEQ_LEN + HORIZON].to_numpy()
        time_slice = all_time.iloc[i + SEQ_LEN : i + SEQ_LEN + HORIZON].to_numpy()

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

    # meta
    df_out["season"] = season
    df_out["cnn_mode"] = CNN_MODE
    df_out["seq_mode"] = SEQUENCE_MODE
    df_out["seq_len"]  = SEQ_LEN
    df_out["pred_len"] = HORIZON
    df_out["best_val_loss"] = float(best_val_loss)

    df_out["hq_cond_mode"] = HQ_COND_MODE
    df_out["hq_l2_weight"] = float(HQ_L2_WEIGHT)
    df_out["hq_noise_std"] = float(HQ_NOISE_STD)
    df_out["use_q_skip"]   = bool(USE_Q_SKIP)

    df_out["film_use_tanh"] = bool(FILM_USE_TANH)
    df_out["fusion_type"]   = "parallel_film_feature_fusion"  # ✅ predict融合じゃない

    for k, v in HP.items():
        df_out[k] = v

    if extra_meta:
        for k, v in extra_meta.items():
            df_out[k] = v

    df_out.to_csv(pred_path, index=False, encoding="utf-8-sig")


# =========================
# Safe load state_dict（PyTorchのweights_only対応）
# =========================
def safe_load_state_dict(path: Path, map_location):
    try:
        sd = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location=map_location)
    return sd


# =========================
# Train/Eval
# =========================
def run_final(season: str, seed: int):
    # 各runの直前にseedを再設定し、初期値・Dropout・DataLoader shuffleを統一管理
    seed_everything(seed)
    print(f"\n🚀 Backtest: {season} | seed={seed} | MODEL=HQ(Parallel-BiFiLM FeatureFusion) | CNN_MODE={CNN_MODE} | SEQ_MODE={SEQUENCE_MODE} | STRICT(warm)")

    (xt, ypt, yft,
     xv, ypv, yfv,
     xte, ypte, yfte, sy, ed,
     all_ts, all_time, test_start_i,
     per_day, step_minutes, has_time_component,
     split_info) = prepare_data(DATA_FILES[season], sequence_mode=SEQUENCE_MODE)

    print(f"[time-axis] per_day={per_day}, step_minutes={step_minutes}, has_time_component={has_time_component}")
    if SEQUENCE_MODE == "warm":
        print(f"[warm strict] train_i_end={split_info['train_i_end']}, val_i=[{split_info['val_i_start']},{split_info['val_i_end']}), test_i_start={split_info['test_i_start']}")
        print(f"[warm strict] Xtr={xt.shape}, Xva={xv.shape}, Xte={xte.shape}")

    dl_t  = DataLoader(TimeSeriesDatasetHQ(xt,  ypt,  yft),  batch_size=HP["batch_size"], shuffle=True)
    dl_v  = DataLoader(TimeSeriesDatasetHQ(xv,  ypv,  yfv),  batch_size=HP["batch_size"], shuffle=False)
    dl_te = DataLoader(TimeSeriesDatasetHQ(xte, ypte, yfte), batch_size=HP["batch_size"], shuffle=False)

    m = ImprovedIntegratedHQTransformer_ParallelFiLMFeatureFusionReport(
        SEQ_LEN, ed, HORIZON,
        HP["d_model"], HP["nhead"], HP["num_layers"], HP["hidden_dim"],
        HP["dropout"], HP["cnn_kernel"], cnn_mode=CNN_MODE,
        hq_cond_mode=HQ_COND_MODE,
        hq_noise_std=HQ_NOISE_STD,
        use_q_skip=USE_Q_SKIP,
        film_use_tanh=FILM_USE_TANH
    ).to(device)

    optimizer_cls = getattr(optim, str(HP["optimizer_name"]))
    opt = optimizer_cls(
        m.parameters(),
        lr=float(HP["learning_rate"]),
        weight_decay=float(HP["weight_decay"]),
    )
    crit = nn.MSELoss()

    best_v, bad = float("inf"), 0

    # seedごとにフォルダを分け、上書きを防止
    seed_model_dir = MODEL_DIR / f"seed_{seed}"
    seed_pred_dir = PRED_DIR / f"seed_{seed}"
    seed_model_dir.mkdir(parents=True, exist_ok=True)
    seed_pred_dir.mkdir(parents=True, exist_ok=True)

    run_name = (
        f"HybridHQ_ParallelBiFiLMFeatureFusion_STRICT_"
        f"{season}_{CNN_MODE}_{SEQUENCE_MODE}_seed{seed}"
    )
    m_p = seed_model_dir / f"{run_name}.pth"
    pred_csv = seed_pred_dir / f"{run_name}.csv"

    # ---- train ----
    for ep in range(1, EPOCHS_FINAL + 1):
        m.train()
        for x, yp, yf in dl_t:
            x, yp, yf = x.to(device), yp.to(device), yf.to(device)
            opt.zero_grad()

            pred = m(x, yp)
            loss = crit(pred, yf)

            # HQ L2 regularization (optional)
            if HQ_L2_WEIGHT > 0:
                loss = loss + (HQ_L2_WEIGHT * (m.hq.pow(2).mean()))

            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), MAX_GRAD_NORM)
            opt.step()

        # val
        m.eval()
        vals = []
        with torch.no_grad():
            for x, yp, yf in dl_v:
                x, yp, yf = x.to(device), yp.to(device), yf.to(device)
                v_pred = m(x, yp)
                vals.append(crit(v_pred, yf).item())
        v_l = float(np.mean(vals)) if vals else float("nan")

        if v_l < best_v:
            best_v, bad = v_l, 0
            torch.save(m.state_dict(), m_p)
        else:
            bad += 1
            if bad >= EARLY_PATIENCE:
                print(f"⏹ Early stop @ epoch={ep}")
                break

        if ep % 20 == 0 or ep == 1:
            print(f"  Epoch {ep:3d} | val_loss={v_l:.6f} | best={best_v:.6f}")

    # ---- test ----
    m.load_state_dict(safe_load_state_dict(m_p, map_location=device))
    m.eval()

    p_s, t_s, a_tv_s, a_vt_s = [], [], [], []
    with torch.no_grad():
        for x, yp, yf in dl_te:
            x, yp = x.to(device), yp.to(device)
            pred, alpha_tv, alpha_vt, pred_t_ref, pred_v_ref = m(x, yp, return_alpha=True)
            p_s.append(pred.cpu().numpy())
            t_s.append(yf.numpy())
            a_tv_s.append(alpha_tv.cpu().numpy())
            a_vt_s.append(alpha_vt.cpu().numpy())

    pred_std   = np.concatenate(p_s, axis=0)
    true_std   = np.concatenate(t_s, axis=0)
    alpha_tv_all = np.concatenate(a_tv_s, axis=0)  # (N,24)
    alpha_vt_all = np.concatenate(a_vt_s, axis=0)  # (N,24)

    preds = inverse_1col_2d(sy, pred_std)
    trues = inverse_1col_2d(sy, true_std)

    rmse = float(np.sqrt(mean_squared_error(trues.reshape(-1), preds.reshape(-1))))
    mape = float(mean_absolute_percentage_error(trues.reshape(-1), preds.reshape(-1)) * 100.0)

    # ✅ horizon別誤差（難易度）
    h_err = summarize_horizon_errors(trues, preds)

    gate_tv = summarize_alpha(alpha_tv_all)
    gate_vt = summarize_alpha(alpha_vt_all)

    # ✅ pred CSV保存（列固定）＋metaに両方向統計＋horizon難易度を付加
    save_pred_csv(
        pred_csv, season, preds, trues, all_ts, all_time, test_start_i, best_v,
        extra_meta={
            "seed": int(seed),
            "alpha_tv_mean": gate_tv["alpha_mean"],
            "alpha_tv_std":  gate_tv["alpha_std"],
            "alpha_tv_min":  gate_tv["alpha_min"],
            "alpha_tv_p25":  gate_tv["alpha_p25"],
            "alpha_tv_p50":  gate_tv["alpha_p50"],
            "alpha_tv_p75":  gate_tv["alpha_p75"],
            "alpha_tv_max":  gate_tv["alpha_max"],

            "alpha_vt_mean": gate_vt["alpha_mean"],
            "alpha_vt_std":  gate_vt["alpha_std"],
            "alpha_vt_min":  gate_vt["alpha_min"],
            "alpha_vt_p25":  gate_vt["alpha_p25"],
            "alpha_vt_p50":  gate_vt["alpha_p50"],
            "alpha_vt_p75":  gate_vt["alpha_p75"],
            "alpha_vt_max":  gate_vt["alpha_max"],

            # ✅ horizon難易度（スカラーだけ）
            "early_rmse_mean_1_6":  float(h_err["early_rmse_mean_1_6"]),
            "late_rmse_mean_19_24": float(h_err["late_rmse_mean_19_24"]),
            "late_div_early_rmse":  float(h_err["late_div_early_rmse"]),
            "worst_h_rmse": int(h_err["worst_h_rmse"]),
            "best_h_rmse":  int(h_err["best_h_rmse"]),
            "worst_h_mape": int(h_err["worst_h_mape"]),
            "best_h_mape":  int(h_err["best_h_mape"]),
        }
    )

    # ---- print report ----
    print(f"✅ RESULT | RMSE={rmse:.4f}, MAPE={mape:.4f}% | best_val(std)={best_v:.6f}")

    # ✅ horizon-wise error print
    print("\n📈 Horizon-wise error (TEST)  ※予測開始〜最後の難易度")
    print("  RMSE(h): " + " ".join([f"{i+1}:{h_err['rmse_per_h'][i]:.3f}" for i in range(HORIZON)]))
    print("  MAPE(h): " + " ".join([f"{i+1}:{h_err['mape_per_h'][i]:.3f}%" for i in range(HORIZON)]))
    print(f"  early RMSE mean (h=1..6)   = {h_err['early_rmse_mean_1_6']:.4f}")
    print(f"  late  RMSE mean (h=19..24) = {h_err['late_rmse_mean_19_24']:.4f}")
    print(f"  late/early (RMSE)          = {h_err['late_div_early_rmse']:.4f}")
    print(f"  worst_h(RMSE)={h_err['worst_h_rmse']} best_h(RMSE)={h_err['best_h_rmse']} | "
          f"worst_h(MAPE)={h_err['worst_h_mape']} best_h(MAPE)={h_err['best_h_mape']}")

    print("\n🧠 Parallel FiLM strength report")
    print("  [t→v] alpha_tv = mean_d((tanh(gamma_tv)+1)/2)")
    print(f"    mean={gate_tv['alpha_mean']:.4f} std={gate_tv['alpha_std']:.4f} "
          f"min={gate_tv['alpha_min']:.4f} p50={gate_tv['alpha_p50']:.4f} max={gate_tv['alpha_max']:.4f}")
    print("  [v→t] alpha_vt = mean_d((tanh(gamma_vt)+1)/2)")
    print(f"    mean={gate_vt['alpha_mean']:.4f} std={gate_vt['alpha_std']:.4f} "
          f"min={gate_vt['alpha_min']:.4f} p50={gate_vt['alpha_p50']:.4f} max={gate_vt['alpha_max']:.4f}")

    per_h_tv = gate_tv["alpha_per_h"]
    per_h_vt = gate_vt["alpha_per_h"]
    print("\n🧠 alpha_tv per horizon (1..24)")
    print("  " + " ".join([f"{i+1}:{per_h_tv[i]:.3f}" for i in range(len(per_h_tv))]))
    print("\n🧠 alpha_vt per horizon (1..24)")
    print("  " + " ".join([f"{i+1}:{per_h_vt[i]:.3f}" for i in range(len(per_h_vt))]))

    print(f"\n📦 pred保存: {pred_csv}")
    print(f"💾 model保存: {m_p}")

    return {
        "model": "HybridHQ",
        "seed": int(seed),
        "season": season,
        "cnn_mode": CNN_MODE,
        "seq_mode": SEQUENCE_MODE,
        "hq_cond_mode": HQ_COND_MODE,
        "hq_l2_weight": float(HQ_L2_WEIGHT),
        "hq_noise_std": float(HQ_NOISE_STD),
        "use_q_skip": bool(USE_Q_SKIP),

        "film_use_tanh": bool(FILM_USE_TANH),
        "fusion_type": "parallel_film_feature_fusion",

        "rmse": rmse,
        "mape": mape,
        "best_val_loss": float(best_v),

        # ✅ horizon別誤差（難易度）
        "rmse_per_horizon": h_err["rmse_per_h"].tolist(),
        "mape_per_horizon": h_err["mape_per_h"].tolist(),
        "early_rmse_mean_1_6":  float(h_err["early_rmse_mean_1_6"]),
        "late_rmse_mean_19_24": float(h_err["late_rmse_mean_19_24"]),
        "late_div_early_rmse":  float(h_err["late_div_early_rmse"]),
        "worst_h_rmse": int(h_err["worst_h_rmse"]),
        "best_h_rmse":  int(h_err["best_h_rmse"]),
        "worst_h_mape": int(h_err["worst_h_mape"]),
        "best_h_mape":  int(h_err["best_h_mape"]),

        "alpha_tv_mean": float(gate_tv["alpha_mean"]),
        "alpha_tv_std":  float(gate_tv["alpha_std"]),
        "alpha_tv_min":  float(gate_tv["alpha_min"]),
        "alpha_tv_p25":  float(gate_tv["alpha_p25"]),
        "alpha_tv_p50":  float(gate_tv["alpha_p50"]),
        "alpha_tv_p75":  float(gate_tv["alpha_p75"]),
        "alpha_tv_max":  float(gate_tv["alpha_max"]),
        "alpha_tv_per_horizon": gate_tv["alpha_per_h"].tolist(),

        "alpha_vt_mean": float(gate_vt["alpha_mean"]),
        "alpha_vt_std":  float(gate_vt["alpha_std"]),
        "alpha_vt_min":  float(gate_vt["alpha_min"]),
        "alpha_vt_p25":  float(gate_vt["alpha_p25"]),
        "alpha_vt_p50":  float(gate_vt["alpha_p50"]),
        "alpha_vt_p75":  float(gate_vt["alpha_p75"]),
        "alpha_vt_max":  float(gate_vt["alpha_max"]),
        "alpha_vt_per_horizon": gate_vt["alpha_per_h"].tolist(),

        "model_path": str(m_p),
        "pred_csv": str(pred_csv),

        "per_day": int(per_day),
        "step_minutes": int(step_minutes),
        "has_time_component": bool(has_time_component),
    }


# =========================
# Main: 3 seeds + mean/std aggregation
# =========================
if __name__ == "__main__":
    print("Using device:", device)
    print("Seeds:", SEEDS)
    print("Result directory:", RESULT_DIR)

    results = []

    # --------------------------------------------------------
    # 3 seeds × 5 datasets
    # --------------------------------------------------------
    for seed in SEEDS:
        print("\n" + "#" * 90)
        print(f"START SEED = {seed}")
        print("#" * 90)

        seed_results = []
        for season in DATA_FILES.keys():
            result = run_final(season, seed)
            results.append(result)
            seed_results.append(result)

        # seedごとのsummaryも保存
        seed_df = pd.DataFrame(seed_results)
        seed_show_df = seed_df.drop(columns=[
            "alpha_tv_per_horizon",
            "alpha_vt_per_horizon",
            "rmse_per_horizon",
            "mape_per_horizon",
        ], errors="ignore")

        seed_summary_path = SUMMARY_DIR / (
            f"seed_{seed}_summary_{CNN_MODE}_{SEQUENCE_MODE}.csv"
        )
        seed_show_df.to_csv(
            seed_summary_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"\n📝 Seed {seed} summary saved: {seed_summary_path}")

    # --------------------------------------------------------
    # 全seedの生結果
    # --------------------------------------------------------
    df = pd.DataFrame(results)

    show_df = df.drop(columns=[
        "alpha_tv_per_horizon",
        "alpha_vt_per_horizon",
        "rmse_per_horizon",
        "mape_per_horizon",
    ], errors="ignore")

    show_df.to_csv(
        ALL_RUNS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 季節別 mean / std
    # pandasのstdは標本標準偏差（ddof=1）
    # --------------------------------------------------------
    aggregate_df = (
        df.groupby("season", sort=False)
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

            alpha_tv_mean_across_seeds=("alpha_tv_mean", "mean"),
            alpha_tv_std_across_seeds=("alpha_tv_mean", "std"),
            alpha_vt_mean_across_seeds=("alpha_vt_mean", "mean"),
            alpha_vt_std_across_seeds=("alpha_vt_mean", "std"),
        )
        .reset_index()
    )

    aggregate_df["rmse_std"] = aggregate_df["rmse_std"].fillna(0.0)
    aggregate_df["mape_std"] = aggregate_df["mape_std"].fillna(0.0)
    aggregate_df["best_val_loss_std"] = aggregate_df[
        "best_val_loss_std"
    ].fillna(0.0)

    # ばらつきの相対量も確認できるようにCVを追加
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
        AGGREGATE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 論文へ貼りやすい形式
    # --------------------------------------------------------
    paper_table_df = aggregate_df[[
        "season",
        "n_seeds",
        "rmse_mean",
        "rmse_std",
        "mape_mean",
        "mape_std",
        "rmse_cv_percent",
        "mape_cv_percent",
    ]].copy()

    paper_table_df["RMSE_mean_std_MW"] = paper_table_df.apply(
        lambda row: f"{row['rmse_mean']:.3f} ± {row['rmse_std']:.3f}",
        axis=1,
    )
    paper_table_df["MAPE_mean_std_percent"] = paper_table_df.apply(
        lambda row: f"{row['mape_mean']:.3f} ± {row['mape_std']:.3f}",
        axis=1,
    )

    paper_table_df = paper_table_df[[
        "season",
        "n_seeds",
        "RMSE_mean_std_MW",
        "MAPE_mean_std_percent",
        "rmse_cv_percent",
        "mape_cv_percent",
    ]]

    paper_table_df.to_csv(
        PAPER_TABLE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # horizon別：各seedの結果を縦持ちで保存
    # --------------------------------------------------------
    horizon_rows = []

    for result in results:
        for h in range(HORIZON):
            horizon_rows.append({
                "seed": int(result["seed"]),
                "season": result["season"],
                "horizon": int(h + 1),
                "rmse": float(result["rmse_per_horizon"][h]),
                "mape_percent": float(result["mape_per_horizon"][h]),
                "alpha_tv": float(result["alpha_tv_per_horizon"][h]),
                "alpha_vt": float(result["alpha_vt_per_horizon"][h]),
            })

    horizon_df = pd.DataFrame(horizon_rows)
    horizon_df.to_csv(
        HORIZON_ALL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    horizon_aggregate_df = (
        horizon_df.groupby(
            ["season", "horizon"],
            sort=False,
        )
        .agg(
            n_seeds=("seed", "count"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mape_mean=("mape_percent", "mean"),
            mape_std=("mape_percent", "std"),
            alpha_tv_mean=("alpha_tv", "mean"),
            alpha_tv_std=("alpha_tv", "std"),
            alpha_vt_mean=("alpha_vt", "mean"),
            alpha_vt_std=("alpha_vt", "std"),
        )
        .reset_index()
        .fillna(0.0)
    )

    horizon_aggregate_df.to_csv(
        HORIZON_AGG_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------
    print("\n" + "=" * 100)
    print("ALL MULTI-SEED RUNS FINISHED")
    print("=" * 100)
    print(f"All runs             : {ALL_RUNS_PATH}")
    print(f"Mean/std by season   : {AGGREGATE_PATH}")
    print(f"Paper-ready table    : {PAPER_TABLE_PATH}")
    print(f"Horizon all seeds    : {HORIZON_ALL_PATH}")
    print(f"Horizon mean/std     : {HORIZON_AGG_PATH}")

    print("\n==================== EACH SEED RESULT ====================")
    print(
        show_df[[
            "seed",
            "season",
            "rmse",
            "mape",
            "best_val_loss",
            "model_path",
            "pred_csv",
        ]]
        .sort_values(["season", "seed"])
        .to_string(index=False)
    )

    print("\n==================== MEAN ± STD ====================")
    print(paper_table_df.to_string(index=False))

    print("\nInterpretation guide:")
    print("  - stdが小さいほど、seedによる結果のばらつきが小さい")
    print("  - CV(%) = std / mean × 100")
    print("  - 論文では通常、RMSE・MAPEを mean ± std で記載")
