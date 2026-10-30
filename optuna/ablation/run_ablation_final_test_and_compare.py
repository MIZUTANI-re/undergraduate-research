from __future__ import annotations

"""
HybridHQ ablation: fixed best-HP final test evaluation and comparison.

Purpose
-------
1. Read each variant's own Optuna best-parameter JSON.
2. Keep those hyperparameters fixed.
3. Retrain each variant with seeds 42, 123, 2026.
4. Use validation only for early stopping/checkpoint selection.
5. Evaluate the selected checkpoint once on the untouched test split.
6. Report test RMSE (MW) and MAPE (%), mean ± std.
7. Export tables compatible with the existing four-model comparison format.

Required files in the same folder
---------------------------------
- ablation_optuna_core.py
- run_ablation_final_test_and_compare.py

Required Optuna outputs
-----------------------
<output-root>/
  full_bifilm_best_params_all_seed42.json
  simple_concat_best_params_all_seed42.json
  no_hq_best_params_all_seed42.json
  no_residual_best_params_all_seed42.json

Default output root
-------------------
<base-dir>/result/result_csv/ablation_optuna_equal_budget_seed42

Example
-------
& .\\venv_new\\Scripts\\python.exe .\\run_ablation_final_test_and_compare.py

Compare existing results only, without retraining:
& .\\venv_new\\Scripts\\python.exe .\\run_ablation_final_test_and_compare.py --only-compare
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

try:
    import ablation_optuna_core as core
except ImportError as exc:
    raise ImportError(
        "ablation_optuna_core.py をこのファイルと同じフォルダへ置いてください。"
    ) from exc


VARIANT_ORDER = [
    "full_bifilm",
    "simple_concat",
    "no_hq",
    "no_residual",
]

VARIANT_LABELS: Dict[str, str] = {
    "full_bifilm": "Full HybridHQ",
    "simple_concat": "No FiLM (Simple Concatenation)",
    "no_hq": "No Horizon Queries",
    "no_residual": "No Linear Residual",
}

COMPONENT_FLAGS: Dict[str, Dict[str, str]] = {
    "full_bifilm": {"FiLM": "Yes", "HQ": "Yes", "Residual": "Yes"},
    "simple_concat": {"FiLM": "No", "HQ": "Yes", "Residual": "Yes"},
    "no_hq": {"FiLM": "Yes", "HQ": "No", "Residual": "Yes"},
    "no_residual": {"FiLM": "Yes", "HQ": "Yes", "Residual": "No"},
}

SEASON_ORDER = ["all", "spring", "summer", "autumn", "winter"]
PAPER_SEASON_ORDER = ["spring", "summer", "autumn", "winter", "all"]
DEFAULT_SEEDS = [42, 123, 2026]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate four HybridHQ ablation variants using each variant's "
            "own saved Optuna best HP, then compare test RMSE and MAPE."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],  # project root (clean_code/)
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optuna/final output root. Default: "
            "<base-dir>/result_csv/ablation_optuna_equal_budget_seed42"
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        choices=SEASON_ORDER,
        default=SEASON_ORDER,
    )
    parser.add_argument("--tuning-seed", type=int, default=42)
    parser.add_argument("--final-epochs", type=int, default=200)
    parser.add_argument("--final-patience", type=int, default=20)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--only-compare",
        action="store_true",
        help="Skip training/test evaluation and aggregate existing final CSVs.",
    )
    return parser.parse_args()


def best_json_path(output_root: Path, variant: str, tuning_seed: int) -> Path:
    return output_root / f"{variant}_best_params_all_seed{tuning_seed}.json"


def final_each_run_path(output_root: Path, variant: str) -> Path:
    return (
        output_root
        / "final"
        / variant
        / "summaries"
        / f"{variant}_multiseed_each_run.csv"
    )


def load_and_validate_best_json(
    output_root: Path,
    variant: str,
    tuning_seed: int,
) -> tuple[Path, Dict[str, Any], Dict[str, Any]]:
    path = best_json_path(output_root, variant, tuning_seed)
    if not path.exists():
        raise FileNotFoundError(
            f"{variant} のbest HP JSONがありません:\n{path}\n"
            "先にOptuna探索を完了してください。"
        )

    metadata = json.loads(path.read_text(encoding="utf-8"))
    if "best_params" not in metadata:
        raise ValueError(
            f"{path} に 'best_params' がありません。"
        )

    params = core.normalize_best_params(dict(metadata["best_params"]))
    required = {
        "d_model",
        "nhead",
        "num_layers",
        "hidden_dim",
        "dropout",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "optimizer_name",
    }
    missing = sorted(required - set(params))
    if missing:
        raise ValueError(
            f"{path} のbest_paramsに必要な項目がありません: {missing}"
        )
    return path, params, metadata


def print_hp_table(
    hp_rows: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    hp_df = pd.DataFrame(hp_rows)
    hp_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    display_columns = [
        "variant_label",
        "best_validation_mse",
        "d_model",
        "nhead",
        "num_layers",
        "hidden_dim",
        "dropout",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "optimizer_name",
    ]
    available = [column for column in display_columns if column in hp_df.columns]
    print("\n" + "=" * 120)
    print("FIXED HYPERPARAMETERS USED FOR FINAL TEST EVALUATION")
    print("=" * 120)
    print(hp_df[available].to_string(index=False))
    print(f"\nSaved HP table: {output_path}")


def evaluate_all_variants(
    base_dir: Path,
    output_root: Path,
    seeds: Sequence[int],
    seasons: Sequence[str],
    tuning_seed: int,
    max_epochs: int,
    patience: int,
    device_name: str,
) -> None:
    device = core.resolve_device(device_name)
    print(f"Using device : {device}")
    print(f"Seeds        : {list(seeds)}")
    print(f"Seasons      : {list(seasons)}")
    print(f"Output root  : {output_root}")

    hp_rows: List[Dict[str, Any]] = []
    validated: Dict[str, tuple[Path, Dict[str, Any], Dict[str, Any]]] = {}

    for variant in VARIANT_ORDER:
        path, params, metadata = load_and_validate_best_json(
            output_root=output_root,
            variant=variant,
            tuning_seed=tuning_seed,
        )
        validated[variant] = (path, params, metadata)

        hp_rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                **COMPONENT_FLAGS[variant],
                "hp_json": str(path),
                "tuning_seed": metadata.get("tuning_seed", tuning_seed),
                "trial_budget": metadata.get("total_trial_budget"),
                "best_trial": metadata.get("best_trial"),
                "best_validation_mse": metadata.get("best_value"),
                **params,
            }
        )

    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    print_hp_table(
        hp_rows,
        comparison_dir / "ABLATION_best_hyperparameters.csv",
    )

    # Each variant is evaluated using its own fixed best HP JSON.
    for index, variant in enumerate(VARIANT_ORDER, start=1):
        path, _, _ = validated[variant]
        print("\n" + "#" * 120)
        print(
            f"[{index}/{len(VARIANT_ORDER)}] FINAL TEST EVALUATION: "
            f"{VARIANT_LABELS[variant]}"
        )
        print(f"Fixed HP JSON: {path}")
        print("#" * 120)

        core.run_final_evaluation(
            variant=variant,
            base_dir=base_dir,
            output_root=output_root,
            device=device,
            seeds=seeds,
            seasons=seasons,
            max_epochs=max_epochs,
            patience=patience,
            best_json=path,
            tuning_seed=tuning_seed,
        )


def normalize_each_run(path: Path, variant: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"最終test結果がありません:\n{path}\n"
            "先にfinal評価を実行してください。"
        )

    df = pd.read_csv(path)
    rename_map = {}
    if "rmse_mw" in df.columns and "rmse" not in df.columns:
        rename_map["rmse_mw"] = "rmse"
    if "mape_percent" in df.columns and "mape" not in df.columns:
        rename_map["mape_percent"] = "mape"
    df = df.rename(columns=rename_map)

    required = ["seed", "season", "rmse", "mape"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{path} に比較用列がありません: {missing}\n"
            f"現在の列: {list(df.columns)}"
        )

    out = df.copy()
    out["variant"] = variant
    out["variant_label"] = VARIANT_LABELS[variant]
    out["model"] = VARIANT_LABELS[variant]
    out["FiLM"] = COMPONENT_FLAGS[variant]["FiLM"]
    out["HQ"] = COMPONENT_FLAGS[variant]["HQ"]
    out["Residual"] = COMPONENT_FLAGS[variant]["Residual"]
    out["seed"] = pd.to_numeric(out["seed"], errors="raise").astype(int)
    out["season"] = out["season"].astype(str).str.lower().str.strip()
    out["rmse"] = pd.to_numeric(out["rmse"], errors="raise")
    out["mape"] = pd.to_numeric(out["mape"], errors="raise")
    out["source_file"] = str(path)
    return out


def validate_final_results(
    df: pd.DataFrame,
    expected_seeds: Sequence[int],
    expected_seasons: Sequence[str],
) -> None:
    problems: List[str] = []
    expected_seed_list = sorted(set(int(seed) for seed in expected_seeds))
    expected_season_list = list(expected_seasons)

    for variant in VARIANT_ORDER:
        group = df[df["variant"] == variant]
        actual_seeds = sorted(group["seed"].unique().tolist())
        actual_seasons = group["season"].unique().tolist()

        if actual_seeds != expected_seed_list:
            problems.append(
                f"{variant}: seeds expected={expected_seed_list}, "
                f"actual={actual_seeds}"
            )

        missing_seasons = [
            season
            for season in expected_season_list
            if season not in actual_seasons
        ]
        if missing_seasons:
            problems.append(
                f"{variant}: missing seasons={missing_seasons}"
            )

        duplicate = group.duplicated(
            subset=["seed", "season"],
            keep=False,
        )
        if duplicate.any():
            problems.append(
                f"{variant}: duplicate seed/season rows found"
            )

    if problems:
        raise ValueError(
            "Final result validation failed:\n- "
            + "\n- ".join(problems)
        )


def format_mean_std(mean_value: float, std_value: float) -> str:
    std_value = 0.0 if pd.isna(std_value) else float(std_value)
    return f"{float(mean_value):.3f} ± {std_value:.3f}"


def create_paper_wide_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        part = aggregate[aggregate["variant"] == variant].set_index("season")
        row: Dict[str, Any] = {
            "Variant": VARIANT_LABELS[variant],
            "FiLM": COMPONENT_FLAGS[variant]["FiLM"],
            "HQ": COMPONENT_FLAGS[variant]["HQ"],
            "Residual": COMPONENT_FLAGS[variant]["Residual"],
        }
        for season in PAPER_SEASON_ORDER:
            if season not in part.index:
                continue
            season_label = "All" if season == "all" else season.capitalize()
            values = part.loc[season]
            row[f"{season_label} RMSE (MW)"] = format_mean_std(
                values["rmse_mean"],
                values["rmse_std"],
            )
            row[f"{season_label} MAPE (%)"] = format_mean_std(
                values["mape_mean"],
                values["mape_std"],
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_and_compare(
    output_root: Path,
    seeds: Sequence[int],
    seasons: Sequence[str],
) -> None:
    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    all_runs = pd.concat(
        [
            normalize_each_run(
                final_each_run_path(output_root, variant),
                variant,
            )
            for variant in VARIANT_ORDER
        ],
        ignore_index=True,
    )
    validate_final_results(all_runs, seeds, seasons)

    all_runs["variant"] = pd.Categorical(
        all_runs["variant"],
        categories=VARIANT_ORDER,
        ordered=True,
    )
    all_runs["season"] = pd.Categorical(
        all_runs["season"],
        categories=SEASON_ORDER,
        ordered=True,
    )
    all_runs = all_runs.sort_values(
        ["season", "variant", "seed"]
    )

    # Complete final output with all available columns.
    each_run_path = comparison_dir / "ABLATION_test_each_run.csv"
    all_runs.to_csv(each_run_path, index=False, encoding="utf-8-sig")

    # Same core columns as compare_four_models_multiseed.py.
    compatible = all_runs[
        [
            "model",
            "seed",
            "season",
            "rmse",
            "mape",
            "source_file",
        ]
    ].copy()
    compatible_path = (
        comparison_dir
        / "ABLATION_test_each_run_four_model_compatible.csv"
    )
    compatible.to_csv(
        compatible_path,
        index=False,
        encoding="utf-8-sig",
    )

    aggregate = (
        all_runs.groupby(
            ["variant", "variant_label", "FiLM", "HQ", "Residual", "season"],
            observed=True,
            sort=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            rmse_min=("rmse", "min"),
            rmse_max=("rmse", "max"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            mape_min=("mape", "min"),
            mape_max=("mape", "max"),
        )
        .reset_index()
    )

    aggregate["rmse_std"] = aggregate["rmse_std"].fillna(0.0)
    aggregate["mape_std"] = aggregate["mape_std"].fillna(0.0)
    aggregate["RMSE_mean_std_MW"] = aggregate.apply(
        lambda row: format_mean_std(
            row["rmse_mean"],
            row["rmse_std"],
        ),
        axis=1,
    )
    aggregate["MAPE_mean_std_percent"] = aggregate.apply(
        lambda row: format_mean_std(
            row["mape_mean"],
            row["mape_std"],
        ),
        axis=1,
    )

    aggregate_path = comparison_dir / "ABLATION_test_mean_std.csv"
    aggregate.to_csv(
        aggregate_path,
        index=False,
        encoding="utf-8-sig",
    )

    paper_wide = create_paper_wide_table(aggregate)
    paper_path = comparison_dir / "ABLATION_test_paper_table_wide.csv"
    paper_wide.to_csv(
        paper_path,
        index=False,
        encoding="utf-8-sig",
    )

    rmse_pivot = (
        aggregate.pivot(
            index="season",
            columns="variant_label",
            values="RMSE_mean_std_MW",
        )
        .reindex(
            index=SEASON_ORDER,
            columns=[VARIANT_LABELS[v] for v in VARIANT_ORDER],
        )
    )
    mape_pivot = (
        aggregate.pivot(
            index="season",
            columns="variant_label",
            values="MAPE_mean_std_percent",
        )
        .reindex(
            index=SEASON_ORDER,
            columns=[VARIANT_LABELS[v] for v in VARIANT_ORDER],
        )
    )

    rmse_path = comparison_dir / "ABLATION_test_RMSE_mean_std_pivot.csv"
    mape_path = comparison_dir / "ABLATION_test_MAPE_mean_std_pivot.csv"
    rmse_pivot.to_csv(rmse_path, encoding="utf-8-sig")
    mape_pivot.to_csv(mape_path, encoding="utf-8-sig")

    # Delta = ablated variant - Full.
    # Positive values mean removing the component worsened the metric.
    full = (
        aggregate[aggregate["variant"] == "full_bifilm"]
        .set_index("season")[["rmse_mean", "mape_mean"]]
        .rename(
            columns={
                "rmse_mean": "full_rmse_mean",
                "mape_mean": "full_mape_mean",
            }
        )
    )
    delta = aggregate.merge(
        full,
        left_on="season",
        right_index=True,
        how="left",
    )
    delta["delta_rmse_vs_full_mw"] = (
        delta["rmse_mean"] - delta["full_rmse_mean"]
    )
    delta["delta_mape_vs_full_percent_point"] = (
        delta["mape_mean"] - delta["full_mape_mean"]
    )
    delta["rmse_change_vs_full_percent"] = (
        delta["delta_rmse_vs_full_mw"]
        / delta["full_rmse_mean"].replace(0, np.nan)
        * 100.0
    )
    delta["mape_change_vs_full_percent"] = (
        delta["delta_mape_vs_full_percent_point"]
        / delta["full_mape_mean"].replace(0, np.nan)
        * 100.0
    )
    delta_path = comparison_dir / "ABLATION_test_delta_vs_full.csv"
    delta.to_csv(delta_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 120)
    print("ABLATION TEST COMPARISON FINISHED")
    print("=" * 120)
    print(f"Each run          : {each_run_path}")
    print(f"Compatible format : {compatible_path}")
    print(f"Mean/std          : {aggregate_path}")
    print(f"Paper table       : {paper_path}")
    print(f"RMSE pivot        : {rmse_path}")
    print(f"MAPE pivot        : {mape_path}")
    print(f"Delta vs Full     : {delta_path}")

    print("\nTEST RMSE (MW), mean ± std")
    print(rmse_pivot.to_string())

    print("\nTEST MAPE (%), mean ± std")
    print(mape_pivot.to_string())

    print(
        "\nDelta interpretation: positive delta means the ablated "
        "variant is worse than Full, supporting the removed component."
    )


def main() -> None:
    args = parse_args()
    output_root = args.output_root or (
        args.base_dir
        / "result"
        / "result_csv"
        / "ablation_optuna_equal_budget_seed42"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.only_compare:
        evaluate_all_variants(
            base_dir=args.base_dir,
            output_root=output_root,
            seeds=args.seeds,
            seasons=args.seasons,
            tuning_seed=args.tuning_seed,
            max_epochs=args.final_epochs,
            patience=args.final_patience,
            device_name=args.device,
        )

    aggregate_and_compare(
        output_root=output_root,
        seeds=args.seeds,
        seasons=args.seasons,
    )


if __name__ == "__main__":
    main()
