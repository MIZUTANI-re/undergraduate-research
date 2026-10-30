from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]  # project root (clean_code/)
RESULT_DIR = BASE_DIR / "result" / "result_csv"

OUTPUT_EACH_RUN = RESULT_DIR / "FOUR_MODELS_multiseed_each_run.csv"
OUTPUT_MEAN_STD = RESULT_DIR / "FOUR_MODELS_multiseed_mean_std.csv"
OUTPUT_PAPER_TABLE = RESULT_DIR / "FOUR_MODELS_multiseed_paper_table.csv"
OUTPUT_RMSE_PIVOT = RESULT_DIR / "FOUR_MODELS_RMSE_mean_std_pivot.csv"
OUTPUT_MAPE_PIVOT = RESULT_DIR / "FOUR_MODELS_MAPE_mean_std_pivot.csv"

SEASON_ORDER = ["all", "spring", "summer", "autumn", "winter"]
MODEL_ORDER = ["TransformerFixed", "iTransformer", "HybridHQ", "LSTM"]
EXPECTED_SEEDS = [42, 123, 2026]

def find_hybrid_file() -> Path:
    candidates = [
        RESULT_DIR / "HybridHQ_multiseed_each_run.csv",
        RESULT_DIR / "HybridHQ_all_seed_runs.csv",
        RESULT_DIR / "all_seed_runs_linear_warm.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list((BASE_DIR / "result").glob("**/all_seed_runs_*_warm.csv"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            "HybridHQ候補が複数見つかりました。find_hybrid_file()のcandidatesへ使用するファイルを明示してください。\n"
            + "\n".join(str(path) for path in matches)
        )
    raise FileNotFoundError("HybridHQの3seed結果が見つかりません。")

def normalize_result(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {}
    if "mape_percent" in df.columns and "mape" not in df.columns:
        rename_map["mape_percent"] = "mape"
    if "rmse_mw" in df.columns and "rmse" not in df.columns:
        rename_map["rmse_mw"] = "rmse"
    df = df.rename(columns=rename_map)

    required = ["seed", "season", "rmse", "mape"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} に比較用の列がありません: {missing}\n現在の列: {list(df.columns)}"
        )

    out = df[required].copy()
    out["model"] = model_name
    out["source_file"] = str(path)
    out["seed"] = pd.to_numeric(out["seed"], errors="raise").astype(int)
    out["season"] = out["season"].astype(str).str.lower().str.strip()
    out["rmse"] = pd.to_numeric(out["rmse"], errors="raise")
    out["mape"] = pd.to_numeric(out["mape"], errors="raise")
    return out[["model", "seed", "season", "rmse", "mape", "source_file"]]

def validate_common_conditions(df: pd.DataFrame) -> None:
    problems: List[str] = []
    for model, group in df.groupby("model"):
        seeds = sorted(group["seed"].unique().tolist())
        seasons = group["season"].unique().tolist()
        if seeds != EXPECTED_SEEDS:
            problems.append(
                f"{model}: seedが想定と異なります。expected={EXPECTED_SEEDS}, actual={seeds}"
            )
        missing_seasons = [s for s in SEASON_ORDER if s not in seasons]
        if missing_seasons:
            problems.append(f"{model}: season不足 {missing_seasons}")

    duplicate_mask = df.duplicated(
        subset=["model", "seed", "season"], keep=False
    )
    if duplicate_mask.any():
        duplicate_rows = df.loc[
            duplicate_mask, ["model", "seed", "season"]
        ]
        problems.append(
            "重複結果があります:\n" + duplicate_rows.to_string(index=False)
        )

    if problems:
        raise ValueError("4モデル比較の条件に問題があります:\n- " + "\n- ".join(problems))

def main() -> None:
    result_paths = {
        "TransformerFixed": RESULT_DIR / "TransformerFixed_multiseed_each_run.csv",
        "iTransformer": RESULT_DIR / "iTransformer_multiseed_each_run.csv",
        "HybridHQ": find_hybrid_file(),
        "LSTM": RESULT_DIR / "LSTM_multiseed_each_run.csv",
    }

    for model_name, path in result_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{model_name}の結果ファイルがありません: {path}")

    all_runs = pd.concat(
        [normalize_result(result_paths[m], m) for m in MODEL_ORDER],
        ignore_index=True,
    )
    validate_common_conditions(all_runs)

    all_runs["model"] = pd.Categorical(
        all_runs["model"], categories=MODEL_ORDER, ordered=True
    )
    all_runs["season"] = pd.Categorical(
        all_runs["season"], categories=SEASON_ORDER, ordered=True
    )
    all_runs = all_runs.sort_values(["season", "model", "seed"])
    all_runs.to_csv(OUTPUT_EACH_RUN, index=False, encoding="utf-8-sig")

    aggregate = (
        all_runs.groupby(["model", "season"], observed=True, sort=False)
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
        )
        .reset_index()
    )

    aggregate["model"] = aggregate["model"].astype(str)
    aggregate["season"] = aggregate["season"].astype(str)
    numeric_columns = aggregate.select_dtypes(include=[np.number]).columns
    aggregate[numeric_columns] = aggregate[numeric_columns].fillna(0.0)

    aggregate["rmse_cv_percent"] = (
        aggregate["rmse_std"] / aggregate["rmse_mean"].replace(0, np.nan) * 100.0
    )
    aggregate["mape_cv_percent"] = (
        aggregate["mape_std"] / aggregate["mape_mean"].replace(0, np.nan) * 100.0
    )
    aggregate[["rmse_cv_percent", "mape_cv_percent"]] = (
        aggregate[["rmse_cv_percent", "mape_cv_percent"]].fillna(0.0)
    )
    aggregate.to_csv(OUTPUT_MEAN_STD, index=False, encoding="utf-8-sig")

    paper = aggregate.copy()
    paper["RMSE_mean_std_MW"] = paper.apply(
        lambda row: f"{row['rmse_mean']:.3f} ± {row['rmse_std']:.3f}", axis=1
    )
    paper["MAPE_mean_std_percent"] = paper.apply(
        lambda row: f"{row['mape_mean']:.3f} ± {row['mape_std']:.3f}", axis=1
    )
    paper = paper[
        [
            "model", "season", "n_seeds",
            "RMSE_mean_std_MW", "MAPE_mean_std_percent",
            "rmse_cv_percent", "mape_cv_percent",
        ]
    ]
    paper.to_csv(OUTPUT_PAPER_TABLE, index=False, encoding="utf-8-sig")

    rmse_pivot = paper.pivot(
        index="season", columns="model", values="RMSE_mean_std_MW"
    ).reindex(index=SEASON_ORDER, columns=MODEL_ORDER)
    rmse_pivot.to_csv(OUTPUT_RMSE_PIVOT, encoding="utf-8-sig")

    mape_pivot = paper.pivot(
        index="season", columns="model", values="MAPE_mean_std_percent"
    ).reindex(index=SEASON_ORDER, columns=MODEL_ORDER)
    mape_pivot.to_csv(OUTPUT_MAPE_PIVOT, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("FOUR-MODEL COMPARISON FINISHED")
    print("=" * 100)
    for model_name in MODEL_ORDER:
        print(f"{model_name:17s}: {result_paths[model_name]}")
    print(f"Each run          : {OUTPUT_EACH_RUN}")
    print(f"Mean/std          : {OUTPUT_MEAN_STD}")
    print(f"Paper table       : {OUTPUT_PAPER_TABLE}")
    print(f"RMSE pivot        : {OUTPUT_RMSE_PIVOT}")
    print(f"MAPE pivot        : {OUTPUT_MAPE_PIVOT}")
    print("\nRMSE mean ± std")
    print(rmse_pivot.to_string())
    print("\nMAPE mean ± std")
    print(mape_pivot.to_string())

if __name__ == "__main__":
    main()
