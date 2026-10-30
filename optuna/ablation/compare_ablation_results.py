from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


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

COMPONENT_FLAGS = {
    "full_bifilm": {"FiLM": "Yes", "HQ": "Yes", "Residual": "Yes"},
    "simple_concat": {"FiLM": "No", "HQ": "Yes", "Residual": "Yes"},
    "no_hq": {"FiLM": "Yes", "HQ": "No", "Residual": "Yes"},
    "no_residual": {"FiLM": "Yes", "HQ": "Yes", "Residual": "No"},
}

SEASON_ORDER = ["all", "spring", "summer", "autumn", "winter"]
PAPER_SEASON_ORDER = ["spring", "summer", "autumn", "winter", "all"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the four HybridHQ ablation variants and create "
            "paper-ready mean ± std and delta-vs-full tables."
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
            "Root produced by ablation_optuna_core.py. Default: "
            "<base-dir>/result_csv/ablation_optuna_equal_budget_seed42"
        ),
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional explicit seed list, e.g. --expected-seeds 42 123 2026. "
            "If omitted, the common seed set is inferred automatically."
        ),
    )
    return parser.parse_args()


def result_path(root: Path, variant: str) -> Path:
    return (
        root
        / "final"
        / variant
        / "summaries"
        / f"{variant}_multiseed_each_run.csv"
    )


def normalize_variant(path: Path, variant: str) -> pd.DataFrame:
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
            f"{path} does not contain required columns {missing}. "
            f"Current columns: {list(df.columns)}"
        )

    out = df[required].copy()
    out["variant"] = variant
    out["variant_label"] = VARIANT_LABELS[variant]
    out["source_file"] = str(path)
    out["seed"] = pd.to_numeric(out["seed"], errors="raise").astype(int)
    out["season"] = out["season"].astype(str).str.lower().str.strip()
    out["rmse"] = pd.to_numeric(out["rmse"], errors="raise")
    out["mape"] = pd.to_numeric(out["mape"], errors="raise")

    flags = COMPONENT_FLAGS[variant]
    out["FiLM"] = flags["FiLM"]
    out["HQ"] = flags["HQ"]
    out["Residual"] = flags["Residual"]

    return out[
        [
            "variant",
            "variant_label",
            "FiLM",
            "HQ",
            "Residual",
            "seed",
            "season",
            "rmse",
            "mape",
            "source_file",
        ]
    ]


def validate_common_conditions(
    df: pd.DataFrame,
    expected_seeds: List[int] | None,
) -> List[int]:
    problems: List[str] = []
    inferred_seed_sets: Dict[str, List[int]] = {}

    for variant, group in df.groupby("variant", sort=False):
        seeds = sorted(group["seed"].unique().tolist())
        inferred_seed_sets[variant] = seeds

        if expected_seeds is not None and seeds != sorted(expected_seeds):
            problems.append(
                f"{variant}: expected seeds={sorted(expected_seeds)}, actual={seeds}"
            )

        missing_seasons = [
            season
            for season in SEASON_ORDER
            if season not in group["season"].unique().tolist()
        ]
        if missing_seasons:
            problems.append(f"{variant}: missing seasons={missing_seasons}")

        expected_rows = len(seeds) * len(SEASON_ORDER)
        if len(group) != expected_rows:
            problems.append(
                f"{variant}: expected {expected_rows} rows from seeds×seasons, "
                f"but found {len(group)}"
            )

    if expected_seeds is None:
        unique_sets = {tuple(value) for value in inferred_seed_sets.values()}
        if len(unique_sets) != 1:
            details = ", ".join(
                f"{variant}={seeds}"
                for variant, seeds in inferred_seed_sets.items()
            )
            problems.append(f"Seed sets differ across variants: {details}")

    duplicate_mask = df.duplicated(
        subset=["variant", "seed", "season"], keep=False
    )
    if duplicate_mask.any():
        duplicate_rows = df.loc[
            duplicate_mask, ["variant", "seed", "season"]
        ]
        problems.append(
            "Duplicate variant/seed/season rows:\n"
            + duplicate_rows.to_string(index=False)
        )

    unknown_seasons = sorted(
        set(df["season"].unique().tolist()) - set(SEASON_ORDER)
    )
    if unknown_seasons:
        problems.append(f"Unknown seasons found: {unknown_seasons}")

    if problems:
        raise ValueError("Ablation comparison validation failed:\n- " + "\n- ".join(problems))

    if expected_seeds is not None:
        return sorted(expected_seeds)
    return next(iter(inferred_seed_sets.values()))


def format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.3f} ± {std_value:.3f}"


def create_wide_paper_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANT_ORDER:
        variant_df = aggregate[aggregate["variant"] == variant].set_index("season")
        row = {
            "Variant": VARIANT_LABELS[variant],
            "FiLM": COMPONENT_FLAGS[variant]["FiLM"],
            "HQ": COMPONENT_FLAGS[variant]["HQ"],
            "Residual": COMPONENT_FLAGS[variant]["Residual"],
        }
        for season in PAPER_SEASON_ORDER:
            values = variant_df.loc[season]
            season_label = "All" if season == "all" else season.capitalize()
            row[f"{season_label} RMSE (MW)"] = format_mean_std(
                values["rmse_mean"], values["rmse_std"]
            )
            row[f"{season_label} MAPE (%)"] = format_mean_std(
                values["mape_mean"], values["mape_std"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def create_delta_vs_full(aggregate: pd.DataFrame) -> pd.DataFrame:
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

    comparison = aggregate.merge(full, left_on="season", right_index=True)
    comparison["delta_rmse_vs_full"] = (
        comparison["rmse_mean"] - comparison["full_rmse_mean"]
    )
    comparison["delta_mape_vs_full"] = (
        comparison["mape_mean"] - comparison["full_mape_mean"]
    )
    comparison["rmse_change_percent_vs_full"] = (
        comparison["delta_rmse_vs_full"]
        / comparison["full_rmse_mean"].replace(0, np.nan)
        * 100.0
    )
    comparison["mape_change_percent_vs_full"] = (
        comparison["delta_mape_vs_full"]
        / comparison["full_mape_mean"].replace(0, np.nan)
        * 100.0
    )

    # Positive delta means the ablated version is worse than Full,
    # supporting the usefulness of the removed component.
    comparison["interpretation"] = np.where(
        comparison["variant"] == "full_bifilm",
        "Reference",
        np.where(
            (comparison["delta_rmse_vs_full"] > 0)
            & (comparison["delta_mape_vs_full"] > 0),
            "Worse than Full in both metrics",
            np.where(
                (comparison["delta_rmse_vs_full"] < 0)
                & (comparison["delta_mape_vs_full"] < 0),
                "Better than Full in both metrics",
                "Mixed result",
            ),
        ),
    )

    return comparison[
        [
            "variant",
            "variant_label",
            "season",
            "rmse_mean",
            "mape_mean",
            "full_rmse_mean",
            "full_mape_mean",
            "delta_rmse_vs_full",
            "delta_mape_vs_full",
            "rmse_change_percent_vs_full",
            "mape_change_percent_vs_full",
            "interpretation",
        ]
    ]


def create_markdown_table(paper_wide: pd.DataFrame, output_path: Path) -> None:
    try:
        markdown = paper_wide.to_markdown(index=False)
    except ImportError:
        markdown = paper_wide.to_csv(index=False)
    output_path.write_text(markdown + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.output_root or (
        args.base_dir
        / "result"
        / "result_csv"
        / "ablation_optuna_equal_budget_seed42"
    )

    paths = {variant: result_path(root, variant) for variant in VARIANT_ORDER}
    for variant, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{VARIANT_LABELS[variant]} result file not found: {path}\n"
                "Run final evaluation first, for example:\n"
                "python run_all_ablation_optuna.py --mode final "
                "--seeds 42 123 2026"
            )

    all_runs = pd.concat(
        [normalize_variant(paths[variant], variant) for variant in VARIANT_ORDER],
        ignore_index=True,
    )
    common_seeds = validate_common_conditions(all_runs, args.expected_seeds)

    all_runs["variant"] = pd.Categorical(
        all_runs["variant"], categories=VARIANT_ORDER, ordered=True
    )
    all_runs["season"] = pd.Categorical(
        all_runs["season"], categories=SEASON_ORDER, ordered=True
    )
    all_runs = all_runs.sort_values(["season", "variant", "seed"])

    out_dir = root / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_runs.to_csv(
        out_dir / "ABLATION_multiseed_each_run.csv",
        index=False,
        encoding="utf-8-sig",
    )

    aggregate = (
        all_runs.groupby(["variant", "season"], observed=True, sort=False)
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
    aggregate["variant"] = aggregate["variant"].astype(str)
    aggregate["season"] = aggregate["season"].astype(str)
    aggregate["variant_label"] = aggregate["variant"].map(VARIANT_LABELS)
    aggregate["FiLM"] = aggregate["variant"].map(
        lambda value: COMPONENT_FLAGS[value]["FiLM"]
    )
    aggregate["HQ"] = aggregate["variant"].map(
        lambda value: COMPONENT_FLAGS[value]["HQ"]
    )
    aggregate["Residual"] = aggregate["variant"].map(
        lambda value: COMPONENT_FLAGS[value]["Residual"]
    )
    aggregate["RMSE_mean_std_MW"] = aggregate.apply(
        lambda row: format_mean_std(row["rmse_mean"], row["rmse_std"]),
        axis=1,
    )
    aggregate["MAPE_mean_std_percent"] = aggregate.apply(
        lambda row: format_mean_std(row["mape_mean"], row["mape_std"]),
        axis=1,
    )
    aggregate["rmse_cv_percent"] = (
        aggregate["rmse_std"]
        / aggregate["rmse_mean"].replace(0, np.nan)
        * 100.0
    )
    aggregate["mape_cv_percent"] = (
        aggregate["mape_std"]
        / aggregate["mape_mean"].replace(0, np.nan)
        * 100.0
    )

    aggregate.to_csv(
        out_dir / "ABLATION_multiseed_mean_std.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rmse_pivot = aggregate.pivot(
        index="season", columns="variant", values="RMSE_mean_std_MW"
    ).reindex(index=SEASON_ORDER, columns=VARIANT_ORDER)
    rmse_pivot.columns = [VARIANT_LABELS[column] for column in rmse_pivot.columns]
    rmse_pivot.to_csv(
        out_dir / "ABLATION_RMSE_mean_std_pivot.csv",
        encoding="utf-8-sig",
    )

    mape_pivot = aggregate.pivot(
        index="season", columns="variant", values="MAPE_mean_std_percent"
    ).reindex(index=SEASON_ORDER, columns=VARIANT_ORDER)
    mape_pivot.columns = [VARIANT_LABELS[column] for column in mape_pivot.columns]
    mape_pivot.to_csv(
        out_dir / "ABLATION_MAPE_mean_std_pivot.csv",
        encoding="utf-8-sig",
    )

    paper_wide = create_wide_paper_table(aggregate)
    paper_wide.to_csv(
        out_dir / "ABLATION_paper_table_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )
    create_markdown_table(
        paper_wide,
        out_dir / "ABLATION_paper_table_wide.md",
    )

    delta_vs_full = create_delta_vs_full(aggregate)
    delta_vs_full.to_csv(
        out_dir / "ABLATION_delta_vs_full.csv",
        index=False,
        encoding="utf-8-sig",
    )

    delta_rmse_pivot = delta_vs_full.pivot(
        index="season", columns="variant", values="delta_rmse_vs_full"
    ).reindex(index=SEASON_ORDER, columns=VARIANT_ORDER)
    delta_rmse_pivot.columns = [
        VARIANT_LABELS[column] for column in delta_rmse_pivot.columns
    ]
    delta_rmse_pivot.to_csv(
        out_dir / "ABLATION_delta_RMSE_vs_full_pivot.csv",
        encoding="utf-8-sig",
    )

    delta_mape_pivot = delta_vs_full.pivot(
        index="season", columns="variant", values="delta_mape_vs_full"
    ).reindex(index=SEASON_ORDER, columns=VARIANT_ORDER)
    delta_mape_pivot.columns = [
        VARIANT_LABELS[column] for column in delta_mape_pivot.columns
    ]
    delta_mape_pivot.to_csv(
        out_dir / "ABLATION_delta_MAPE_vs_full_pivot.csv",
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 110)
    print("HYBRIDHQ ABLATION COMPARISON FINISHED")
    print("=" * 110)
    print(f"Common seeds: {common_seeds}")
    for variant in VARIANT_ORDER:
        print(f"{VARIANT_LABELS[variant]:34s}: {paths[variant]}")
    print(f"Output directory: {out_dir}")

    print("\nRMSE mean ± std (MW)")
    print(rmse_pivot.to_string())
    print("\nMAPE mean ± std (%)")
    print(mape_pivot.to_string())

    print("\nDelta vs Full: positive values mean the ablated model is worse")
    print("RMSE delta (MW)")
    print(delta_rmse_pivot.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nMAPE delta (percentage points)")
    print(delta_mape_pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nPaper-ready table:")
    print(paper_wide.to_string(index=False))


if __name__ == "__main__":
    main()
