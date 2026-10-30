from __future__ import annotations

r"""
Evaluate only the three ablated variants and reuse the existing HybridHQ
3-seed test results as the Full-model row.

Full result reused from:
<base_dir>/result/result_csv/HybridHQ_multiseed_each_run.csv

Ablated variants evaluated:
- simple_concat
- no_hq
- no_residual
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

import ablation_optuna_core as core


VARIANTS = ["simple_concat", "no_hq", "no_residual"]
VARIANT_ORDER = ["full_bifilm", "simple_concat", "no_hq", "no_residual"]

LABELS = {
    "full_bifilm": "Full HybridHQ",
    "simple_concat": "No FiLM (Simple Concatenation)",
    "no_hq": "No Horizon Queries",
    "no_residual": "No Linear Residual",
}

FLAGS = {
    "full_bifilm": {"FiLM": "Yes", "HQ": "Yes", "Residual": "Yes"},
    "simple_concat": {"FiLM": "No", "HQ": "Yes", "Residual": "Yes"},
    "no_hq": {"FiLM": "Yes", "HQ": "No", "Residual": "Yes"},
    "no_residual": {"FiLM": "Yes", "HQ": "Yes", "Residual": "No"},
}

SEASONS = ["all", "spring", "summer", "autumn", "winter"]
PAPER_SEASONS = ["spring", "summer", "autumn", "winter", "all"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],  # project root (clean_code/)
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--full-result",
        type=Path,
        default=None,
        help="Existing HybridHQ_multiseed_each_run.csv",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2026],
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        choices=SEASONS,
        default=SEASONS,
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
        help="Do not train; only rebuild tables from existing CSV files.",
    )
    return parser.parse_args()


def best_json(output_root: Path, variant: str, tuning_seed: int) -> Path:
    return output_root / f"{variant}_best_params_all_seed{tuning_seed}.json"


def ablation_result_path(output_root: Path, variant: str) -> Path:
    return (
        output_root
        / "final"
        / variant
        / "summaries"
        / f"{variant}_multiseed_each_run.csv"
    )


def run_three_ablations(
    base_dir: Path,
    output_root: Path,
    seeds: Sequence[int],
    seasons: Sequence[str],
    tuning_seed: int,
    final_epochs: int,
    final_patience: int,
    device_name: str,
) -> None:
    device = core.resolve_device(device_name)
    print(f"Using device: {device}")

    for index, variant in enumerate(VARIANTS, start=1):
        hp_json = best_json(output_root, variant, tuning_seed)
        if not hp_json.exists():
            raise FileNotFoundError(
                f"Best HP JSON not found for {variant}: {hp_json}"
            )

        print("\n" + "#" * 100)
        print(f"[{index}/3] FINAL TEST: {LABELS[variant]}")
        print(f"Fixed HP: {hp_json}")
        print("#" * 100)

        core.run_final_evaluation(
            variant=variant,
            base_dir=base_dir,
            output_root=output_root,
            device=device,
            seeds=seeds,
            seasons=seasons,
            max_epochs=final_epochs,
            patience=final_patience,
            best_json=hp_json,
            tuning_seed=tuning_seed,
        )


def normalize_full(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Existing Full HybridHQ result not found: {path}")

    df = pd.read_csv(path)
    rename = {}
    if "rmse_mw" in df.columns and "rmse" not in df.columns:
        rename["rmse_mw"] = "rmse"
    if "mape_percent" in df.columns and "mape" not in df.columns:
        rename["mape_percent"] = "mape"
    df = df.rename(columns=rename)

    required = ["seed", "season", "rmse", "mape"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Full result is missing columns {missing}: {list(df.columns)}"
        )

    out = df[required].copy()
    out["variant"] = "full_bifilm"
    out["variant_label"] = LABELS["full_bifilm"]
    out["FiLM"] = "Yes"
    out["HQ"] = "Yes"
    out["Residual"] = "Yes"
    out["source_file"] = str(path)
    return out


def normalize_ablation(path: Path, variant: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Final test result not found for {variant}: {path}"
        )

    df = pd.read_csv(path)
    rename = {}
    if "rmse_mw" in df.columns:
        rename["rmse_mw"] = "rmse"
    if "mape_percent" in df.columns:
        rename["mape_percent"] = "mape"
    df = df.rename(columns=rename)

    required = ["seed", "season", "rmse", "mape"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{variant} result is missing columns {missing}: {list(df.columns)}"
        )

    out = df[required].copy()
    out["variant"] = variant
    out["variant_label"] = LABELS[variant]
    out["FiLM"] = FLAGS[variant]["FiLM"]
    out["HQ"] = FLAGS[variant]["HQ"]
    out["Residual"] = FLAGS[variant]["Residual"]
    out["source_file"] = str(path)
    return out


def validate(
    df: pd.DataFrame,
    expected_seeds: Sequence[int],
    expected_seasons: Sequence[str],
) -> None:
    problems: List[str] = []
    expected_seeds = sorted(set(int(x) for x in expected_seeds))

    for variant in VARIANT_ORDER:
        group = df[df["variant"] == variant]
        seeds = sorted(group["seed"].astype(int).unique().tolist())
        seasons = group["season"].astype(str).str.lower().unique().tolist()

        if seeds != expected_seeds:
            problems.append(
                f"{variant}: expected seeds={expected_seeds}, actual={seeds}"
            )

        missing = [s for s in expected_seasons if s not in seasons]
        if missing:
            problems.append(f"{variant}: missing seasons={missing}")

        if group.duplicated(["seed", "season"]).any():
            problems.append(f"{variant}: duplicate seed-season rows")

    if problems:
        raise ValueError(
            "Comparison condition mismatch:\n- " + "\n- ".join(problems)
        )


def fmt(mean: float, std: float) -> str:
    std = 0.0 if pd.isna(std) else float(std)
    return f"{float(mean):.3f} ± {std:.3f}"


def compare(
    output_root: Path,
    full_result: Path,
    seeds: Sequence[int],
    seasons: Sequence[str],
) -> None:
    frames = [normalize_full(full_result)]
    frames.extend(
        normalize_ablation(ablation_result_path(output_root, v), v)
        for v in VARIANTS
    )

    all_runs = pd.concat(frames, ignore_index=True)
    all_runs["seed"] = pd.to_numeric(all_runs["seed"], errors="raise").astype(int)
    all_runs["season"] = (
        all_runs["season"].astype(str).str.lower().str.strip()
    )
    all_runs["rmse"] = pd.to_numeric(all_runs["rmse"], errors="raise")
    all_runs["mape"] = pd.to_numeric(all_runs["mape"], errors="raise")

    validate(all_runs, seeds, seasons)

    comparison_dir = output_root / "comparison_reuse_existing_full"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    all_runs["variant"] = pd.Categorical(
        all_runs["variant"],
        categories=VARIANT_ORDER,
        ordered=True,
    )
    all_runs["season"] = pd.Categorical(
        all_runs["season"],
        categories=SEASONS,
        ordered=True,
    )
    all_runs = all_runs.sort_values(["season", "variant", "seed"])

    each_path = comparison_dir / "ABLATION_test_each_run.csv"
    all_runs.to_csv(each_path, index=False, encoding="utf-8-sig")

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
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
        )
        .reset_index()
    )
    aggregate["RMSE_mean_std_MW"] = aggregate.apply(
        lambda r: fmt(r["rmse_mean"], r["rmse_std"]),
        axis=1,
    )
    aggregate["MAPE_mean_std_percent"] = aggregate.apply(
        lambda r: fmt(r["mape_mean"], r["mape_std"]),
        axis=1,
    )

    mean_path = comparison_dir / "ABLATION_test_mean_std.csv"
    aggregate.to_csv(mean_path, index=False, encoding="utf-8-sig")

    paper_rows: List[Dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        part = aggregate[aggregate["variant"] == variant].set_index("season")
        row: Dict[str, Any] = {
            "Variant": LABELS[variant],
            "FiLM": FLAGS[variant]["FiLM"],
            "HQ": FLAGS[variant]["HQ"],
            "Residual": FLAGS[variant]["Residual"],
        }
        for season in PAPER_SEASONS:
            label = "All" if season == "all" else season.capitalize()
            row[f"{label} RMSE (MW)"] = part.loc[
                season, "RMSE_mean_std_MW"
            ]
            row[f"{label} MAPE (%)"] = part.loc[
                season, "MAPE_mean_std_percent"
            ]
        paper_rows.append(row)

    paper = pd.DataFrame(paper_rows)
    paper_path = comparison_dir / "ABLATION_test_paper_table_wide.csv"
    paper.to_csv(paper_path, index=False, encoding="utf-8-sig")

    rmse_pivot = (
        aggregate.pivot(
            index="season",
            columns="variant_label",
            values="RMSE_mean_std_MW",
        )
        .reindex(
            index=SEASONS,
            columns=[LABELS[v] for v in VARIANT_ORDER],
        )
    )
    mape_pivot = (
        aggregate.pivot(
            index="season",
            columns="variant_label",
            values="MAPE_mean_std_percent",
        )
        .reindex(
            index=SEASONS,
            columns=[LABELS[v] for v in VARIANT_ORDER],
        )
    )

    rmse_path = comparison_dir / "ABLATION_test_RMSE_pivot.csv"
    mape_path = comparison_dir / "ABLATION_test_MAPE_pivot.csv"
    rmse_pivot.to_csv(rmse_path, encoding="utf-8-sig")
    mape_pivot.to_csv(mape_path, encoding="utf-8-sig")

    full = (
        aggregate[aggregate["variant"] == "full_bifilm"]
        .set_index("season")[["rmse_mean", "mape_mean"]]
        .rename(
            columns={
                "rmse_mean": "full_rmse",
                "mape_mean": "full_mape",
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
        delta["rmse_mean"] - delta["full_rmse"]
    )
    delta["delta_mape_vs_full_percentage_point"] = (
        delta["mape_mean"] - delta["full_mape"]
    )
    delta_path = comparison_dir / "ABLATION_test_delta_vs_full.csv"
    delta.to_csv(delta_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 110)
    print("ABLATION COMPARISON FINISHED")
    print("Full row was reused from the existing HybridHQ test result.")
    print("=" * 110)
    print(f"Existing Full     : {full_result}")
    print(f"Each run         : {each_path}")
    print(f"Mean/std         : {mean_path}")
    print(f"Paper table      : {paper_path}")
    print(f"Delta vs Full    : {delta_path}")

    print("\nTEST RMSE (MW), mean ± std")
    print(rmse_pivot.to_string())
    print("\nTEST MAPE (%), mean ± std")
    print(mape_pivot.to_string())


def main() -> None:
    args = parse_args()

    output_root = args.output_root or (
        args.base_dir
        / "result"
        / "result_csv"
        / "ablation_optuna_equal_budget_seed42"
    )
    full_result = args.full_result or (
        args.base_dir
        / "result"
        / "result_csv"
        / "HybridHQ_multiseed_each_run.csv"
    )

    if not args.only_compare:
        run_three_ablations(
            base_dir=args.base_dir,
            output_root=output_root,
            seeds=args.seeds,
            seasons=args.seasons,
            tuning_seed=args.tuning_seed,
            final_epochs=args.final_epochs,
            final_patience=args.final_patience,
            device_name=args.device,
        )

    compare(
        output_root=output_root,
        full_result=full_result,
        seeds=args.seeds,
        seasons=args.seasons,
    )


if __name__ == "__main__":
    main()
