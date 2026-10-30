"""
公開データの取得と読み込み（data_preparation/ の共通部分）。

元データ（すべて公開データ）:
    電力需要   四国電力送配電「エリア需給実績」 年度別 Excel（1 時間値、2016〜2023 年度）
               https://www.yonden.co.jp/nw/supply_demand/data_download.html
               利用条件 https://www.yonden.co.jp/nw/supply_demand/site_info.html
    祝日       内閣府「国民の祝日」CSV
               https://www8.cao.go.jp/chosei/shukujitsu/gaiyou.html
    気象       気象庁「過去の気象データ・ダウンロード」（香川・徳島・愛媛・高知の観測値）
               https://www.data.jma.go.jp/risk/obsdl/
               ※ 画面から地点・項目を選んで取得する方式のため、自動取得のスクリプトはない（README を参照）

取得したファイルは data/raw/ に置く（git には入れない）。
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
YONDEN = "https://www.yonden.co.jp/nw/supply_demand/jukyu{year}.xlsx"
CAO_HOLIDAYS = "https://www8.cao.go.jp/chosei/shukujitsu/syukujitsu.csv"
FISCAL_YEARS = range(2016, 2024)  # 2016 年度（2016-04）〜 2023 年度（2024-03）


def fetch(overwrite: bool = False) -> None:
    """四国電力送配電の年度別 Excel と内閣府の祝日 CSV を data/raw/ に保存する"""
    from urllib.request import Request, urlopen

    RAW.mkdir(parents=True, exist_ok=True)
    targets = [(YONDEN.format(year=y), RAW / f"jukyu{y}.xlsx") for y in FISCAL_YEARS]
    targets.append((CAO_HOLIDAYS, RAW / "syukujitsu.csv"))
    for url, dst in targets:
        if dst.exists() and dst.stat().st_size > 1000 and not overwrite:
            print("skip", dst.name)
            continue
        try:
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as r:
                content = r.read()
        except OSError as e:  # 環境によっては証明書の検証で失敗する（内閣府のサイトなど）
            print(f"FAIL {dst.name}: {e}\n     ブラウザか curl で {url} を取得し、{dst} に保存してください")
            continue
        dst.write_bytes(content)
        print("ok  ", dst.name, f"{len(content) / 1000:.0f} KB")
        time.sleep(1)  # サーバーへの負荷を抑える


def read_demand(raw_dir: Path = RAW) -> pd.DataFrame:
    """年度別 Excel から 1 時間ごとのエリア需要（10⁴ kW）を読む。列：date（日付）, time（0〜23 時）, demand"""
    frames = []
    for y in FISCAL_YEARS:
        x = pd.read_excel(raw_dir / f"jukyu{y}.xlsx", header=None)
        head = x.index[x.iloc[:, 0].astype(str).str.strip() == "DATE"][0]
        body = x.iloc[head + 1:, :3]
        body.columns = ["DATE", "TIME", "demand"]
        body = body[pd.to_datetime(body.DATE, errors="coerce").notna()].copy()
        body["date"] = pd.to_datetime(body.DATE).dt.normalize()
        body["time"] = body.TIME.astype(str).str.split(":").str[0].astype(int)
        body["demand"] = pd.to_numeric(body.demand)
        frames.append(body[["date", "time", "demand"]])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["date", "time"])


def read_holidays(raw_dir: Path = RAW) -> set:
    """内閣府の祝日 CSV（Shift_JIS）から、祝日・振替休日の日付の集合を返す"""
    raw = (raw_dir / "syukujitsu.csv").read_bytes()
    h = pd.read_csv(io.BytesIO(raw), encoding="cp932", header=0, names=["date", "name"])
    return set(pd.to_datetime(h.date, format="%Y/%m/%d").dt.date)


if __name__ == "__main__":
    fetch()
