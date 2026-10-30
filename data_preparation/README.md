# data_preparation/ — データの出どころの確認

`data/` の CSV が公開データから作られたことを、自分の手元で確かめるためのスクリプトです。列ごとの出どころと加工の説明は、トップの [README](../README.md#データの出どころと作り方) にあります。

| ファイル | 役割 |
|---|---|
| `sources.py` | 公開データを `data/raw/` に取得する（四国電力送配電の年度別 Excel 8 つ、内閣府の祝日 CSV）。読み込みの関数もここにある |
| `make_calendar.py` | 日付と祝日から `weekday`・`restday`・`holiday_onehot` を作る規則 |
| `split_seasons.py` | 通年のファイルを月で 4 季節に分ける（`--check` で今のファイルと一致するかだけ確かめる） |
| `verify_dataset.py` | 形・需要・カレンダー・季節分割・件数・気象の範囲を確かめる |

```bash
python data_preparation/sources.py
```

```bash
python data_preparation/verify_dataset.py
```

- Excel の読み込みに `openpyxl` が必要です（`requirements.txt` に入っています）。
- 環境によっては、内閣府のサイトの証明書の検証で取得に失敗します。そのときは表示される URL をブラウザか curl で取得し、`data/raw/syukujitsu.csv` に保存してください。
- 気象の列は、気象庁のサイトが画面から選んで取得する方式のため、自動では取得しません。
