"""
data/ の CSV が、公開データから決まった手順で作られたものであることを確かめる。

確かめること:
    1. 形          期間 2016-04-01〜2024-02-29、毎日 0〜23 時がそろい、欠けや重複がない（69,384 行）
    2. 需要        power_demand が四国電力送配電「エリア需給実績」のエリア需要（10⁴ kW）と全行一致する
    3. カレンダー  weekday・restday・holiday_onehot が、日付と内閣府の祝日から make_calendar.py の規則で作れる
    4. 季節分割    季節別 4 ファイルが、通年のファイルを月で分けたもの（split_seasons.py）と完全に一致する
    5. 件数        論文 Table I のサンプル数と一致する
    6. 気象        値の範囲と欠けがないこと（元データの自動取得はできないため、範囲の確認のみ）

2 と 3 には公開データが必要（先に python data_preparation/sources.py で data/raw/ に取得する）。

実行:
    python data_preparation/sources.py          # 公開データの取得（初回だけ）
    python data_preparation/verify_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_calendar import calendar  # noqa: E402
from sources import RAW, read_demand, read_holidays  # noqa: E402
from split_seasons import ALL, BASE, split  # noqa: E402

TABLE_I = {"spring": 16920, "summer": 17664, "autumn": 17472, "winter": 17328, "all": 69384}


def main() -> None:
    df = pd.read_csv(ALL)
    df["date"] = pd.to_datetime(df.datetime)
    results = []

    # 1. 形
    days = pd.date_range("2016-04-01", "2024-02-29", freq="D")
    full = len(df) == len(days) * 24 and not df.duplicated(["datetime", "time"]).any() \
        and set(df.time) == set(range(24)) and df.date.min() == days[0] and df.date.max() == days[-1]
    results.append(("形（期間・0〜23 時・欠けなし）", full, f"{len(df):,} 行、{df.date.min():%Y-%m-%d}〜{df.date.max():%Y-%m-%d}"))

    # 2・3. 公開データとの照合
    if (RAW / "jukyu2016.xlsx").exists() and (RAW / "syukujitsu.csv").exists():
        dem = read_demand()
        m = df.merge(dem, on=["date", "time"], how="left")
        n_eq = int((m.demand == m.power_demand).sum())
        results.append(("需要 ＝ 四国電力送配電のエリア需要", n_eq == len(df), f"{n_eq:,} / {len(df):,} 行が一致"))
        cal = calendar(df.datetime, read_holidays())
        ok = {c: int((cal[c].values == df[c].values).sum()) for c in cal.columns}
        results.append(("カレンダーの列 ＝ 日付と内閣府の祝日から作った値", all(v == len(df) for v in ok.values()),
                        "、".join(f"{c} {v:,} 行一致" for c, v in ok.items())))
    else:
        results.append(("需要・カレンダーの照合", False, "data/raw/ に公開データがない（python data_preparation/sources.py で取得）"))

    # 4・5. 季節分割と件数
    parts = split(df.drop(columns="date"))
    for s, part in parts.items():
        f = pd.read_csv(BASE / "data" / f"demand_holiday_with_solar_mean_{s}.csv")
        results.append((f"季節分割：{s}", part.equals(f) and len(f) == TABLE_I[s], f"{len(f):,} 行（Table I：{TABLE_I[s]:,}）"))

    # 6. 気象
    w = df[["temperature", "precipitation", "solar_mean"]]
    rng = w.isna().sum().sum() == 0 and w.temperature.between(-15, 45).all() and (w.precipitation >= 0).all() \
        and (w.solar_mean >= 0).all()
    results.append(("気象（欠けなし・値の範囲）", rng, f"気温 {w.temperature.min():.1f}〜{w.temperature.max():.1f} ℃、"
                    f"降水量 {w.precipitation.min():.2f}〜{w.precipitation.max():.2f} mm、日射 {w.solar_mean.min():.2f}〜{w.solar_mean.max():.2f}"))

    print("| 確認 | 結果 | 内容 |\n|---|---|---|")
    for name, ok_, detail in results:
        print(f"| {name} | {'OK' if ok_ else 'NG'} | {detail} |")
    sys.exit(0 if all(r[1] for r in results) else 1)


if __name__ == "__main__":
    main()
