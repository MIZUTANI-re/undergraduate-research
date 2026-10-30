from __future__ import annotations

r"""
Create final summer and winter prediction figures for the manuscript.

Default behavior
----------------
- Models:
    TransformerFixed (displayed as Transformer)
    iTransformer
    HybridHQ
    LSTM
- Seeds:
    42, 123, 2026
- Seasons:
    summer, winter
- Output:
    separate PNG and SVG figures for summer and winter
    CSV files containing the plotted hourly averages

Processing
----------
1. Read each model's final test prediction CSV for each seed.
2. If a timestamp is present, average duplicate forecasts targeting the
   same timestamp. This prevents timestamps with many overlapping forecast
   origins from being counted repeatedly.
3. Average the predictions across the three seeds.
4. Group the actual and predicted values by clock hour (0-23).
5. Plot the final hourly average daily profile.

Expected prediction CSV columns
-------------------------------
The script automatically recognizes common names, including:
- timestamp / datetime
- hour / time
- y_true / actual
- y_pred / prediction

The final ablation code uses:
timestamp, date, time, hour, minute, horizon, y_true, y_pred

Run
---
PowerShell:

& .\venv_new\Scripts\python.exe .\plot_final_summer_winter.py

Optional examples:

# Use only seed 42
& .\venv_new\Scripts\python.exe .\plot_final_summer_winter.py --seeds 42

# Change output directory
& .\venv_new\Scripts\python.exe .\plot_final_summer_winter.py `
  --output-dir <project_root>\paper_figures
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Default configuration
# -------------------------------------------------------------------------

DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1]  # project root (clean_code/)
DEFAULT_SEEDS = [42, 123, 2026]
DEFAULT_SEASONS = ["summer", "winter"]

# The first existing directory in each list is used.
MODEL_DIRECTORIES: Dict[str, List[str]] = {
    "TransformerFixed": [
        "pred_csv_TransformerFixed_multiseed",
        "pred_csv_Transformer_multiseed",
    ],
    "iTransformer": [
        "pred_csv_iTransformer_multiseed",
        "pred_csv_ITransformer_multiseed",
    ],
    "HybridHQ": [
        "pred_csv_HybridHQ_multiseed",
        "pred_csv_Hybrid_multiseed",
    ],
    "LSTM": [
        "pred_csv_LSTM_multiseed",
        "pred_csv_lstm_multiseed",
    ],
}

DISPLAY_NAMES: Dict[str, str] = {
    "TransformerFixed": "Transformer",
    "iTransformer": "iTransformer",
    "HybridHQ": "HybridHQ",
    "LSTM": "LSTM",
}

MODEL_ORDER = [
    "TransformerFixed",
    "iTransformer",
    "HybridHQ",
    "LSTM",
]

# Line styles and markers are specified so that the figure remains readable
# when printed in grayscale. Matplotlib's default color cycle is retained.
LINE_FORMATS = {
    "Actual": {"linestyle": "-", "marker": "o", "linewidth": 2.2},
    "TransformerFixed": {"linestyle": "--", "marker": "s", "linewidth": 1.6},
    "iTransformer": {"linestyle": "-.", "marker": "^", "linewidth": 1.6},
    "HybridHQ": {"linestyle": "-", "marker": "D", "linewidth": 1.9},
    "LSTM": {"linestyle": ":", "marker": "v", "linewidth": 1.6},
}

# Keep this label if the source load values use 10^4 kW, as in the current paper.
# Change to "Load (MW)" only when the CSV values themselves are confirmed to be MW.
DEFAULT_Y_LABEL = r"Demand ($10^4$ kW)"


# -------------------------------------------------------------------------
# Column handling
# -------------------------------------------------------------------------

COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "timestamp": (
        "timestamp",
        "datetime",
        "date_time",
        "target_timestamp",
        "target_datetime",
    ),
    "date": (
        "date",
        "target_date",
    ),
    "hour": (
        "hour",
        "target_hour",
        "hour_of_day",
        "time",
    ),
    "horizon": (
        "horizon",
        "forecast_horizon",
        "step",
        "lead_time",
    ),
    "y_true": (
        "y_true",
        "actual",
        "target",
        "observed",
        "true",
        "ground_truth",
    ),
    "y_pred": (
        "y_pred",
        "prediction",
        "predicted",
        "forecast",
        "pred",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create separate final summer and winter hourly-average "
            "prediction figures from multiseed test prediction CSV files."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Research project root.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Defaults to <base-dir>/result/result_csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <base-dir>/picture/final_prediction.",
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
        choices=["summer", "winter"],
        default=DEFAULT_SEASONS,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=MODEL_ORDER,
    )
    parser.add_argument(
        "--y-label",
        default=DEFAULT_Y_LABEL,
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=220.0,
        help="Minimum value for Y-axis (default: 220.0)",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=400.0,
        help="Maximum value for Y-axis (default: 400.0)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Skip a missing model or seed instead of stopping. "
            "For the final paper, leaving this off is recommended."
        ),
    )
    return parser.parse_args()


def normalize_column_name(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    return text


def find_column(
    normalized_columns: Dict[str, str],
    canonical_name: str,
    required: bool = True,
) -> Optional[str]:
    for alias in COLUMN_ALIASES[canonical_name]:
        normalized_alias = normalize_column_name(alias)
        if normalized_alias in normalized_columns:
            return normalized_columns[normalized_alias]

    if required:
        raise KeyError(
            f"Could not find the '{canonical_name}' column. "
            f"Available columns: {list(normalized_columns.values())}"
        )
    return None


# -------------------------------------------------------------------------
# File discovery
# -------------------------------------------------------------------------

def resolve_model_directory(
    result_dir: Path,
    model: str,
) -> Path:
    candidates = MODEL_DIRECTORIES[model]
    existing = [result_dir / name for name in candidates if (result_dir / name).exists()]

    if existing:
        return existing[0]

    raise FileNotFoundError(
        f"No prediction directory was found for {model}.\n"
        f"Checked:\n"
        + "\n".join(f"  - {result_dir / name}" for name in candidates)
    )


def is_prediction_csv(path: Path, season: str) -> bool:
    name = path.name.lower()
    if season.lower() not in name:
        return False

    excluded_terms = (
        "summary",
        "mean_std",
        "each_run",
        "horizon_mean",
        "horizon_each",
        "history",
        "loss",
        "optuna",
        "trial",
        "metric",
    )
    return not any(term in name for term in excluded_terms)


def find_prediction_file(
    model_dir: Path,
    model: str,
    season: str,
    seed: int,
) -> Path:
    seed_dir_candidates = [
        model_dir / f"seed_{seed}",
        model_dir / str(seed),
        model_dir,
    ]

    matches: List[Path] = []
    for directory in seed_dir_candidates:
        if not directory.exists():
            continue

        for path in directory.rglob("*.csv"):
            if not is_prediction_csv(path, season):
                continue

            lower_path = str(path).lower()
            seed_tokens = (
                f"seed_{seed}",
                f"seed{seed}",
                f"\\{seed}\\",
                f"/{seed}/",
            )

            # When looking directly under the model directory, require the
            # seed to appear somewhere in the path or filename.
            if directory == model_dir and not any(token in lower_path for token in seed_tokens):
                continue

            matches.append(path)

    matches = sorted(set(matches))

    if not matches:
        raise FileNotFoundError(
            f"No prediction CSV found for model={model}, "
            f"season={season}, seed={seed}\n"
            f"Model directory: {model_dir}"
        )

    if len(matches) == 1:
        return matches[0]

    # Prefer filenames that explicitly contain both the season and seed.
    scored: List[Tuple[int, Path]] = []
    for path in matches:
        name = path.name.lower()
        score = 0
        if season in name:
            score += 5
        if f"seed{seed}" in name or f"seed_{seed}" in name:
            score += 5
        if "strict" in name:
            score += 1
        if "warm" in name:
            score += 1
        score -= len(name) / 10000.0
        scored.append((score, path))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score = scored[0][0]
    best_matches = [path for score, path in scored if abs(score - best_score) < 1e-9]

    if len(best_matches) > 1:
        options = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(
            f"Multiple equally likely prediction CSVs were found for "
            f"{model}, {season}, seed {seed}:\n{options}\n"
            "Delete unused copies or specify a single prediction folder."
        )

    return best_matches[0]


# -------------------------------------------------------------------------
# Prediction CSV processing
# -------------------------------------------------------------------------

def read_prediction_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    normalized_columns = {
        normalize_column_name(column): column
        for column in df.columns
    }

    y_true_col = find_column(normalized_columns, "y_true", required=True)
    y_pred_col = find_column(normalized_columns, "y_pred", required=True)
    timestamp_col = find_column(
        normalized_columns,
        "timestamp",
        required=False,
    )
    date_col = find_column(
        normalized_columns,
        "date",
        required=False,
    )
    hour_col = find_column(
        normalized_columns,
        "hour",
        required=False,
    )

    output = pd.DataFrame()
    output["y_true"] = pd.to_numeric(df[y_true_col], errors="coerce")
    output["y_pred"] = pd.to_numeric(df[y_pred_col], errors="coerce")

    if timestamp_col is not None:
        output["timestamp"] = pd.to_datetime(
            df[timestamp_col],
            errors="coerce",
        )
    elif date_col is not None and hour_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        hours = pd.to_numeric(df[hour_col], errors="coerce")
        output["timestamp"] = dates + pd.to_timedelta(hours, unit="h")
    else:
        output["timestamp"] = pd.NaT

    if hour_col is not None:
        hours = pd.to_numeric(df[hour_col], errors="coerce")
        # Some files use 1-24 instead of 0-23.
        if hours.dropna().between(1, 24).all() and not (hours == 0).any():
            hours = hours % 24
        output["hour"] = hours
    elif output["timestamp"].notna().any():
        output["hour"] = output["timestamp"].dt.hour
    else:
        raise KeyError(
            f"{path} has neither a usable timestamp nor an hour column."
        )

    output = output.dropna(subset=["hour", "y_true", "y_pred"]).copy()
    output["hour"] = output["hour"].astype(int)

    invalid_hours = output.loc[
        ~output["hour"].between(0, 23),
        "hour",
    ].unique()
    if len(invalid_hours):
        raise ValueError(
            f"Invalid hour values in {path}: {invalid_hours.tolist()}"
        )

    return output


def collapse_duplicate_target_timestamps(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    A rolling 24-hour forecast can predict the same target timestamp from
    several forecast origins. Average those predictions first.
    """
    if df["timestamp"].notna().any():
        grouped = (
            df.dropna(subset=["timestamp"])
            .groupby(["timestamp", "hour"], as_index=False)
            .agg(
                y_true=("y_true", "mean"),
                y_pred=("y_pred", "mean"),
            )
        )
        return grouped

    # Fallback for already aggregated CSV files.
    return (
        df.groupby("hour", as_index=False)
        .agg(
            y_true=("y_true", "mean"),
            y_pred=("y_pred", "mean"),
        )
    )


def load_model_season(
    result_dir: Path,
    model: str,
    season: str,
    seeds: Sequence[int],
    allow_missing: bool,
) -> Tuple[pd.DataFrame, List[Path]]:
    model_dir = resolve_model_directory(result_dir, model)

    seed_frames: List[pd.DataFrame] = []
    source_files: List[Path] = []

    for seed in seeds:
        try:
            path = find_prediction_file(
                model_dir=model_dir,
                model=model,
                season=season,
                seed=seed,
            )
        except FileNotFoundError:
            if allow_missing:
                print(
                    f"WARNING: skipped missing result: "
                    f"{model}, {season}, seed={seed}"
                )
                continue
            raise

        print(
            f"READ | model={model:<18} | season={season:<6} | "
            f"seed={seed:<4} | {path}"
        )

        frame = read_prediction_csv(path)
        frame = collapse_duplicate_target_timestamps(frame)
        frame["seed"] = seed
        seed_frames.append(frame)
        source_files.append(path)

    if not seed_frames:
        raise RuntimeError(
            f"No usable prediction data for model={model}, season={season}"
        )

    combined = pd.concat(seed_frames, ignore_index=True)

    if combined["timestamp"].notna().any():
        # Average the same target timestamp across seeds.
        across_seeds = (
            combined.dropna(subset=["timestamp"])
            .groupby(["timestamp", "hour"], as_index=False)
            .agg(
                y_true=("y_true", "mean"),
                y_pred=("y_pred", "mean"),
            )
        )
        hourly = (
            across_seeds.groupby("hour", as_index=False)
            .agg(
                y_true=("y_true", "mean"),
                y_pred=("y_pred", "mean"),
                n_target_timestamps=("timestamp", "nunique"),
            )
        )
    else:
        hourly = (
            combined.groupby("hour", as_index=False)
            .agg(
                y_true=("y_true", "mean"),
                y_pred=("y_pred", "mean"),
                n_target_timestamps=("y_true", "size"),
            )
        )

    hourly = (
        hourly.set_index("hour")
        .reindex(range(24))
        .rename_axis("hour")
        .reset_index()
    )

    missing_hours = hourly.loc[
        hourly["y_pred"].isna(),
        "hour",
    ].tolist()
    if missing_hours:
        raise ValueError(
            f"Missing clock hours for model={model}, season={season}: "
            f"{missing_hours}"
        )

    return hourly, source_files


def build_plot_table(
    result_dir: Path,
    season: str,
    models: Sequence[str],
    seeds: Sequence[int],
    allow_missing: bool,
) -> Tuple[pd.DataFrame, Dict[str, List[Path]]]:
    plot_table = pd.DataFrame({"hour": np.arange(24)})
    source_map: Dict[str, List[Path]] = {}
    actual_candidates: List[pd.Series] = []

    loaded_models: List[str] = []

    for model in models:
        try:
            hourly, sources = load_model_season(
                result_dir=result_dir,
                model=model,
                season=season,
                seeds=seeds,
                allow_missing=allow_missing,
            )
        except (FileNotFoundError, RuntimeError):
            if allow_missing:
                print(f"WARNING: skipped model={model}, season={season}")
                continue
            raise

        loaded_models.append(model)
        source_map[model] = sources
        plot_table = plot_table.merge(
            hourly[["hour", "y_pred"]].rename(
                columns={"y_pred": model}
            ),
            on="hour",
            how="left",
        )
        actual_candidates.append(
            hourly.set_index("hour")["y_true"].reindex(range(24))
        )

    if not loaded_models:
        raise RuntimeError(f"No models were loaded for season={season}")

    actual_matrix = pd.concat(actual_candidates, axis=1)

    # The target values should be identical across model files. Small
    # differences can arise only from file rounding or slightly different
    # coverage at the boundaries.
    max_actual_spread = (
        actual_matrix.max(axis=1) - actual_matrix.min(axis=1)
    ).max()

    if pd.notna(max_actual_spread) and max_actual_spread > 1e-6:
        print(
            f"WARNING: actual hourly averages differ across model files "
            f"for {season}. Maximum spread={max_actual_spread:.8f}. "
            "The mean across model files will be used."
        )

    plot_table["Actual"] = actual_matrix.mean(axis=1).to_numpy()

    ordered_columns = ["hour", "Actual"] + loaded_models
    plot_table = plot_table[ordered_columns]

    return plot_table, source_map


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def make_figure(
    plot_table: pd.DataFrame,
    season: str,
    output_dir: Path,
    y_label: str,
    dpi: int,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
) -> None:
    season_display = season.capitalize()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # Actual first.
    actual_format = LINE_FORMATS["Actual"]
    ax.plot(
        plot_table["hour"],
        plot_table["Actual"],
        label="Actual",
        linestyle=actual_format["linestyle"],
        marker=actual_format["marker"],
        linewidth=actual_format["linewidth"],
        markersize=4.0,
        markevery=2,
    )

    for model in MODEL_ORDER:
        if model not in plot_table.columns:
            continue

        line_format = LINE_FORMATS[model]
        ax.plot(
            plot_table["hour"],
            plot_table[model],
            label=DISPLAY_NAMES[model],
            linestyle=line_format["linestyle"],
            marker=line_format["marker"],
            linewidth=line_format["linewidth"],
            markersize=3.5,
            markevery=2,
        )

    ax.set_title(
        f"Hourly Average Demand (Actual vs. Models)\n"
        f"Season: {season_display}"
    )
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(y_label)
    ax.set_xlim(0, 23)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.arange(0, 24, 1))
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(
        loc="best",
        frameon=True,
        fontsize=8,
    )

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"final_prediction_{season}.png"
    svg_path = output_dir / f"final_prediction_{season}.svg"

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        svg_path,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"SAVED PNG : {png_path}")
    print(f"SAVED SVG : {svg_path}")


def save_plot_data(
    plot_table: pd.DataFrame,
    source_map: Dict[str, List[Path]],
    season: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / f"final_prediction_{season}_hourly_values.csv"
    plot_table.to_csv(
        data_path,
        index=False,
        encoding="utf-8-sig",
    )

    source_rows: List[Dict[str, object]] = []
    for model, paths in source_map.items():
        for path in paths:
            source_rows.append(
                {
                    "season": season,
                    "model": model,
                    "display_name": DISPLAY_NAMES[model],
                    "source_csv": str(path),
                }
            )

    sources_path = output_dir / f"final_prediction_{season}_sources.csv"
    pd.DataFrame(source_rows).to_csv(
        sources_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"SAVED DATA: {data_path}")
    print(f"SAVED LIST: {sources_path}")


def main() -> None:
    args = parse_args()

    result_dir = args.result_dir or (
        args.base_dir / "result" / "result_csv"
    )
    output_dir = args.output_dir or (
        args.base_dir / "picture" / "final_prediction"
    )

    if not result_dir.exists():
        raise FileNotFoundError(
            f"Result directory does not exist: {result_dir}"
        )

    print("=" * 100)
    print("FINAL SUMMER/WINTER PREDICTION FIGURE GENERATOR")
    print("=" * 100)
    print(f"Result directory : {result_dir}")
    print(f"Output directory : {output_dir}")
    print(f"Models           : {args.models}")
    print(f"Seeds            : {args.seeds}")
    print(f"Seasons          : {args.seasons}")
    print(f"Y-axis range     : [{args.ymin}, {args.ymax}]")

    for season in args.seasons:
        print("\n" + "-" * 100)
        print(f"PROCESSING SEASON: {season.upper()}")
        print("-" * 100)

        plot_table, source_map = build_plot_table(
            result_dir=result_dir,
            season=season,
            models=args.models,
            seeds=args.seeds,
            allow_missing=args.allow_missing,
        )

        save_plot_data(
            plot_table=plot_table,
            source_map=source_map,
            season=season,
            output_dir=output_dir,
        )

        make_figure(
            plot_table=plot_table,
            season=season,
            output_dir=output_dir,
            y_label=args.y_label,
            dpi=args.dpi,
            ymin=args.ymin,
            ymax=args.ymax,
        )

        print("\nPLOTTED VALUES")
        print(plot_table.to_string(index=False))

    print("\n" + "=" * 100)
    print("FINISHED")
    print("=" * 100)


if __name__ == "__main__":
    main()
