"""
カレンダーの列を、日付と内閣府の祝日から作る。

列の定義（data/demand_holiday_with_solar_mean.csv と同じ）:
    datetime        日付（YYYY-MM-DD）
    time            時刻（0〜23）
    weekday         曜日（月曜 = 0 … 日曜 = 6）
    restday         土曜・日曜のフラグ：土曜・日曜なら 1（祝日の平日は 0）
    holiday_onehot  休む日のフラグ：土曜・日曜・祝日（振替休日を含む、内閣府の CSV）なら 1

モデルの中では、さらに「日（day）」と「曜日・月の one-hot」を作って入力に加える（models/ の prepare_data）。
"""
from __future__ import annotations

import pandas as pd


def calendar(dates: pd.Series, holidays: set) -> pd.DataFrame:
    d = pd.to_datetime(dates)
    weekday = d.dt.weekday
    is_holiday = d.dt.date.map(lambda x: x in holidays)
    weekend = weekday >= 5
    return pd.DataFrame({"weekday": weekday.astype(int),
                         "restday": weekend.astype(int),
                         "holiday_onehot": (weekend | is_holiday).astype(int)})
