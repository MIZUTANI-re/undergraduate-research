"""
通年のデータ（data/demand_holiday_with_solar_mean.csv）を、月で 4 つの季節に分ける。

    春 3〜5 月、夏 6〜8 月、秋 9〜11 月、冬 12〜2 月
    行の順番と列は通年のファイルのまま（季節の中で時系列順）

実行:
    python data_preparation/split_seasons.py            # data/ の季節別 4 ファイルを作り直す
    python data_preparation/split_seasons.py --check    # 作り直さずに、今の 4 ファイルと一致するかだけ確かめる
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ALL = BASE / "data" / "demand_holiday_with_solar_mean.csv"
SEASONS = {"spring": [3, 4, 5], "summer": [6, 7, 8], "autumn": [9, 10, 11], "winter": [12, 1, 2]}


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    month = pd.to_datetime(df.datetime).dt.month
    return {s: df[month.isin(m)].reset_index(drop=True) for s, m in SEASONS.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    df = pd.read_csv(ALL)
    for s, part in split(df).items():
        path = BASE / "data" / f"demand_holiday_with_solar_mean_{s}.csv"
        if args.check:
            same = part.equals(pd.read_csv(path))
            print(f"{'OK' if same else 'NG'}  {path.name}  {len(part):,} 行")
        else:
            part.to_csv(path, index=False)
            print(f"saved {path.name}  {len(part):,} 行")


if __name__ == "__main__":
    main()
