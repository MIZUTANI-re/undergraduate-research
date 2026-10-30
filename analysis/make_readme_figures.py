"""
README 用の図を、論文の確定値（result/final/）から作る。

出力（figures/）:
    accuracy_by_season.png   論文 Table IV：4 モデルの RMSE と MAPE（3 seed 平均 ± 標準偏差）
    ablation.png             論文 Table V：部品を外したときの RMSE・MAPE の変化（Full HybridHQ との差）

実行:
    python analysis/make_readme_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
FIG = BASE / "figures"
SEASONS = ["spring", "summer", "autumn", "winter", "all"]
SEASON_JA = {"spring": "春", "summer": "夏", "autumn": "秋", "winter": "冬", "all": "通年"}
MODELS = [("TransformerFixed", "Transformer", "#2a78d6"), ("iTransformer", "iTransformer", "#eb6834"),
          ("HybridHQ", "HybridHQ（提案）", "#1baf7a"), ("LSTM", "LSTM", "#eda100")]
plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def read(p):
    return pd.read_csv(p, encoding="utf-8-sig")


def accuracy() -> None:
    d = read(BASE / "result/final/four_models/FOUR_MODELS_multiseed_mean_std.csv").set_index(["model", "season"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    x = np.arange(len(SEASONS))
    w = 0.2
    for ax, key, title, unit in ((axes[0], "rmse", "RMSE", "RMSE（10⁴ kW）"), (axes[1], "mape", "MAPE", "MAPE（%）")):
        for k, (m, label, c) in enumerate(MODELS):
            v = [d.loc[(m, s), f"{key}_mean"] for s in SEASONS]
            e = [d.loc[(m, s), f"{key}_std"] for s in SEASONS]
            ax.bar(x + (k - 1.5) * w, v, w * 0.92, yerr=e, color=c, label=label, capsize=2, error_kw={"lw": 0.8})
        for i, s in enumerate(SEASONS):
            best = min(MODELS, key=lambda mm: d.loc[(mm[0], s), f"{key}_mean"])
            k = MODELS.index(best)
            val = d.loc[(best[0], s), f"{key}_mean"]
            ax.text(i + (k - 1.5) * w, val * 0.5, "★", ha="center", va="center", color="white", fontsize=13)
        ax.set_xticks(x, [SEASON_JA[s] for s in SEASONS], fontsize=12)
        ax.set_ylabel(unit, fontsize=11)
        ax.set_title(f"{title}（3 seed 平均 ± 標準偏差、低いほど良い）", fontsize=13, fontweight="bold")
        ax.grid(axis="y", color="#e5e5e5")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.04))
    fig.text(0.01, -0.09, "★ = その季節で最良のモデル。数値は論文 Table IV と同じ（result/final/four_models/）。", fontsize=10, color="#555")
    fig.tight_layout()
    fig.savefig(FIG / "accuracy_by_season.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def ablation() -> None:
    d = read(BASE / "result/final/optuna_ablation/comparison_reuse_existing_full/ABLATION_test_mean_std.csv").set_index(["variant", "season"])
    variants = [("simple_concat", "FiLM を外す（単純結合）", "#2a78d6"), ("no_hq", "Horizon Query を外す", "#eb6834"),
                ("no_residual", "線形残差を外す", "#1baf7a")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    x = np.arange(len(SEASONS))
    w = 0.26
    for ax, key, title in ((axes[0], "rmse", "RMSE の変化（10⁴ kW）"), (axes[1], "mape", "MAPE の変化（pt）")):
        deltas = {v: [d.loc[(v, s), f"{key}_mean"] - d.loc[("full_bifilm", s), f"{key}_mean"] for s in SEASONS]
                  for v, _, _ in variants}
        allv = [x_ for vs in deltas.values() for x_ in vs]
        pad = 0.015 * (max(allv) - min(allv))
        for k, (v, label, c) in enumerate(variants):
            bars = ax.bar(x + (k - 1) * w, deltas[v], w * 0.92, color=c, label=label)
            for b, val in zip(bars, deltas[v]):
                ax.text(b.get_x() + b.get_width() / 2, val + (pad if val >= 0 else -pad), f"{val:+.2f}", ha="center",
                        va="bottom" if val >= 0 else "top", fontsize=8.5)
        ax.set_ylim(min(allv) - 12 * pad, max(allv) + 12 * pad)
        ax.axhline(0, color="#333", lw=1)
        ax.set_xticks(x, [SEASON_JA[s] for s in SEASONS], fontsize=12)
        ax.set_title(title + "　＋＝外すと悪化（その部品が効いている）", fontsize=12.5, fontweight="bold")
        ax.grid(axis="y", color="#e5e5e5")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.04))
    fig.text(0.01, -0.09, "Full HybridHQ との差。各条件が同じ 30 trial の予算で自分の HP を選んだ比較（論文 Table V）。",
             fontsize=10, color="#555")
    fig.tight_layout()
    fig.savefig(FIG / "ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    accuracy()
    ablation()
    print("saved", FIG)
