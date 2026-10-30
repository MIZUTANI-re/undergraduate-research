#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=====================================================================
 季節別 時間帯平均 予測 vs 実績 プロット生成スクリプト
=====================================================================

特徴:
  - 春夏秋冬＋allの5季節を **縦軸統一** で並べて比較可能
  - 表示モデルを自由に選択 (A, B, C, D 形式で指定)
  - pred CSV から自動集計（hour列でgroupby → 平均）
  - 論文品質の高解像度PNG出力

使い方:
  # 全モデル表示（デフォルト）
  python plot_hourly_profiles.py

  # モデルを選んで表示
  python plot_hourly_profiles.py --models A C

  # 季節を選んで表示
  python plot_hourly_profiles.py --seasons summer winter

  # 縦軸の範囲を手動指定
  python plot_hourly_profiles.py --ymin 200 --ymax 450

  # 個別グラフのみ生成（季節ごとに1枚ずつ）
  python plot_hourly_profiles.py --individual-only

  # パネル図のみ生成
  python plot_hourly_profiles.py --panel-only
=====================================================================
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# パス設定
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]  # project root (clean_code/)
DATA_DIR = BASE_DIR / "data"
RESULT_CSV_DIR = BASE_DIR / "result" / "result_csv"
PICTURE_DIR = BASE_DIR / "picture"
OUTPUT_DIR = PICTURE_DIR / "hourly_profiles"

# ============================================================
# モデル定義（A, B, C, D）
#   label : グラフ凡例に表示される名前
#   color : 線の色
#   pred_dir / pred_pattern : pred CSVの探索パス
# ============================================================
MODEL_REGISTRY: Dict[str, dict] = {
    "A": {
        "label": "TransformerFixed",
        "color": "#2196F3",       # blue
        "marker": "s",
        "pred_dir": RESULT_CSV_DIR / "pred_csv_TransformerFixed_multiseed",
        "pred_pattern": "TransformerFixed_*_{season}_*_seed{seed}.csv",
    },
    "B": {
        "label": "iTransformer",
        "color": "#FF9800",       # orange
        "marker": "^",
        "pred_dir": RESULT_CSV_DIR / "pred_csv_iTransformer_multiseed",
        "pred_pattern": "iTransformer_{season}_*_seed{seed}.csv",
    },
    "C": {
        "label": "HybridHQ",
        "color": "#4CAF50",       # green
        "marker": "D",
        "pred_dir": RESULT_CSV_DIR / "pred_csv_HybridHQ_multiseed",
        "pred_pattern": "HybridHQ_*_{season}_*_seed{seed}.csv",
    },
    "D": {
        "label": "LSTM",
        "color": "#E91E63",       # pink
        "marker": "v",
        "pred_dir": RESULT_CSV_DIR / "pred_csv_LSTM_multiseed",
        "pred_pattern": "LSTM_*_{season}_*_seed{seed}.csv",
    },
}

# 旧形式 pred CSV（seed分離されていない古い形式にも対応）
LEGACY_PRED_DIRS: Dict[str, Dict[str, Path]] = {
    "iTransformer": {
        season: RESULT_CSV_DIR / f"pred_csv_iTransformer_multiseed" / f"seed_42"
        for season in ["all", "spring", "summer", "autumn", "winter"]
    },
    "HybridHQ": {
        season: RESULT_CSV_DIR / "pred_csv_hybrid_hq_parallel_film_feature_strict"
        for season in ["all", "spring", "summer", "autumn", "winter"]
    },
}

SEASON_ORDER = ["all", "spring", "summer", "autumn", "winter"]
SEASON_LABELS = {
    "all": "All Seasons",
    "spring": "Spring",
    "summer": "Summer",
    "autumn": "Autumn",
    "winter": "Winter",
}

DEFAULT_SEED = 42


# ============================================================
# データ読み込み
# ============================================================
def find_pred_csv(model_key: str, season: str, seed: int = DEFAULT_SEED) -> Optional[Path]:
    """pred CSVを探索して返す。見つからなければNone。"""
    info = MODEL_REGISTRY[model_key]
    pred_dir: Path = info["pred_dir"]

    # 1) seed_XX フォルダ内を探索
    seed_dir = pred_dir / f"seed_{seed}"
    if seed_dir.exists():
        pattern = info["pred_pattern"].format(season=season, seed=seed)
        matches = list(seed_dir.glob(pattern))
        if matches:
            return matches[0]

    # 2) pred_dir 直下を探索
    if pred_dir.exists():
        pattern = info["pred_pattern"].format(season=season, seed=seed)
        matches = list(pred_dir.glob(pattern))
        if matches:
            return matches[0]

    # 3) 旧形式を探索（iTransformer / HybridHQ の古い保存形式）
    label = info["label"]
    if label in LEGACY_PRED_DIRS:
        legacy = LEGACY_PRED_DIRS[label]
        if season in legacy:
            legacy_dir = legacy[season]
            if legacy_dir.exists():
                # season名を含むCSVを検索
                candidates = [
                    f for f in legacy_dir.glob("*.csv")
                    if season in f.stem.lower()
                ]
                if candidates:
                    return candidates[0]

    return None


def load_hourly_mean(csv_path: Path) -> pd.DataFrame:
    """pred CSVからhour別の平均 y_true / y_pred を計算。"""
    df = pd.read_csv(csv_path, usecols=["hour", "y_true", "y_pred"])
    grouped = df.groupby("hour").agg(
        y_true_mean=("y_true", "mean"),
        y_pred_mean=("y_pred", "mean"),
    ).reset_index()
    return grouped.sort_values("hour")


# ============================================================
# プロット
# ============================================================
def setup_style():
    """論文品質のmatplotlibスタイル設定。"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def compute_unified_ylim(
    all_data: Dict[str, Dict[str, pd.DataFrame]],
    margin: float = 0.05,
) -> Tuple[float, float]:
    """全季節・全モデルのデータから統一Y軸範囲を計算。"""
    all_values = []
    for season_data in all_data.values():
        for df in season_data.values():
            all_values.extend(df["y_true_mean"].tolist())
            all_values.extend(df["y_pred_mean"].tolist())

    if not all_values:
        return 200, 450

    vmin, vmax = min(all_values), max(all_values)
    span = vmax - vmin
    return vmin - span * margin, vmax + span * margin


def plot_single_season(
    ax: plt.Axes,
    season: str,
    model_data: Dict[str, pd.DataFrame],
    model_keys: List[str],
    ylim: Tuple[float, float],
    show_legend: bool = True,
):
    """1つのAxesに、ある季節のActual + 各モデル予測をプロット。"""
    hours = np.arange(24)

    # Actual（どのモデルのy_trueでも同じ→最初のを使う）
    first_key = next(iter(model_data))
    actual = model_data[first_key]["y_true_mean"].values
    ax.plot(hours, actual, color="#333333", linewidth=2.0,
            linestyle="-", marker="o", markersize=4,
            label="Actual", zorder=10)

    # 各モデルの予測
    for key in model_keys:
        if key not in model_data:
            continue
        info = MODEL_REGISTRY[key]
        pred = model_data[key]["y_pred_mean"].values
        ax.plot(hours, pred,
                color=info["color"],
                linewidth=1.8,
                linestyle="--",
                marker=info["marker"],
                markersize=4,
                alpha=0.85,
                label=f'{info["label"]} Pred',
                zorder=5)

    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(ylim)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Power Demand (MW)")
    ax.set_title(SEASON_LABELS.get(season, season), fontweight="bold")

    if show_legend:
        ax.legend(loc="upper left", framealpha=0.9)


def save_individual_plots(
    all_data: Dict[str, Dict[str, pd.DataFrame]],
    model_keys: List[str],
    ylim: Tuple[float, float],
    output_dir: Path,
):
    """各季節ごとに1枚ずつPNG保存。"""
    for season, model_data in all_data.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_single_season(ax, season, model_data, model_keys, ylim)
        fig.tight_layout()

        out_path = output_dir / f"hourly_mean_{season}.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # CSVも保存
    for season, model_data in all_data.items():
        rows = {}
        first_key = next(iter(model_data))
        rows["Actual"] = model_data[first_key]["y_true_mean"].values
        for key in model_keys:
            if key in model_data:
                label = MODEL_REGISTRY[key]["label"]
                rows[f"{label} Pred"] = model_data[key]["y_pred_mean"].values
        df_out = pd.DataFrame(rows, index=range(24))
        df_out.index.name = "hour"
        csv_path = output_dir / f"hourly_mean_{season}.csv"
        df_out.to_csv(csv_path, encoding="utf-8-sig")
        print(f"  Saved: {csv_path}")


def save_panel_plot(
    all_data: Dict[str, Dict[str, pd.DataFrame]],
    model_keys: List[str],
    ylim: Tuple[float, float],
    output_dir: Path,
    seasons: List[str],
):
    """全季節を1枚のパネル図にまとめて保存。"""
    n = len(seasons)
    if n <= 3:
        ncols, nrows = n, 1
    elif n <= 4:
        ncols, nrows = 2, 2
    else:
        ncols, nrows = 3, 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, season in enumerate(seasons):
        model_data = all_data.get(season, {})
        show_legend = (i == 0)
        plot_single_season(axes[i], season, model_data, model_keys, ylim, show_legend)

    # 余ったAxesを非表示
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    # 共通の凡例をfigの上部に
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels,
                   loc="upper center",
                   ncol=min(len(handles), 6),
                   bbox_to_anchor=(0.5, 1.02),
                   framealpha=0.9,
                   fontsize=10)
        # 個別の凡例を削除
        for ax in axes[:n]:
            legend = ax.get_legend()
            if legend:
                legend.remove()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    models_tag = "_".join(model_keys)
    out_path = output_dir / f"hourly_panel_{models_tag}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# メイン
# ============================================================
def main():
    setup_style()
    parser = argparse.ArgumentParser(
        description="季節別 時間帯平均 予測 vs 実績 プロット",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
モデルキー:
  A = TransformerFixed
  B = iTransformer
  C = HybridHQ
  D = LSTM

例:
  python plot_hourly_profiles.py --models B C
  python plot_hourly_profiles.py --models A B C D --seasons spring summer
  python plot_hourly_profiles.py --ymin 200 --ymax 420
        """,
    )
    parser.add_argument(
        "--models", nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
        help="表示するモデルのキー (A B C D)",
    )
    parser.add_argument(
        "--seasons", nargs="+",
        default=SEASON_ORDER,
        choices=SEASON_ORDER,
        help="表示する季節",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--individual-only", action="store_true",
                        help="個別グラフのみ出力")
    parser.add_argument("--panel-only", action="store_true",
                        help="パネル図のみ出力")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    all_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    found_models: List[str] = []

    print("=" * 80)
    print("Hourly Profile Plot Generator")
    print("=" * 80)
    print(f"Models requested: {[MODEL_REGISTRY[k]['label'] for k in args.models]}")
    print(f"Seasons: {args.seasons}")
    print(f"Seed: {args.seed}")
    print()

    for season in args.seasons:
        season_data: Dict[str, pd.DataFrame] = {}
        for model_key in args.models:
            csv_path = find_pred_csv(model_key, season, args.seed)
            if csv_path is None:
                print(f"  [SKIP] {MODEL_REGISTRY[model_key]['label']} / {season} - pred CSV not found")
                continue
            try:
                df = load_hourly_mean(csv_path)
                season_data[model_key] = df
                if model_key not in found_models:
                    found_models.append(model_key)
                print(f"  [OK]   {MODEL_REGISTRY[model_key]['label']} / {season} - {csv_path.name}")
            except Exception as e:
                print(f"  [ERR]  {MODEL_REGISTRY[model_key]['label']} / {season} - {e}")

        if season_data:
            all_data[season] = season_data

    if not all_data:
        print("\n[ERROR] データが見つかりませんでした。")
        print("pred CSVがまだ生成されていない可能性があります。")
        print("先に各モデルのスクリプトを実行してください。")
        return

    # Y軸範囲の決定
    auto_ymin, auto_ymax = compute_unified_ylim(all_data)
    ylim = (
        args.ymin if args.ymin is not None else auto_ymin,
        args.ymax if args.ymax is not None else auto_ymax,
    )
    print(f"\nUnified Y-axis: [{ylim[0]:.1f}, {ylim[1]:.1f}]")
    print(f"Models found: {[MODEL_REGISTRY[k]['label'] for k in found_models]}")
    print()

    # グラフ生成
    if not args.panel_only:
        print("--- Individual plots ---")
        save_individual_plots(all_data, found_models, ylim, OUTPUT_DIR)

    if not args.individual_only:
        print("\n--- Panel plot ---")
        save_panel_plot(all_data, found_models, ylim, OUTPUT_DIR, args.seasons)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
