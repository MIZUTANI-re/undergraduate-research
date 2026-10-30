# 夏季・冬季の最終予測図作成コード

## 出力内容

残差図は作成せず、以下を作成します。

- `final_prediction_summer.png`
- `final_prediction_summer.svg`
- `final_prediction_winter.png`
- `final_prediction_winter.svg`
- 各図に使用した0〜23時の数値CSV
- 読み込んだ予測CSVの一覧

表示する線は次です。

- Actual
- Transformer
- iTransformer
- HybridHQ
- LSTM

標準ではseed 42、123、2026を平均します。

## 配置

```text
<project_root>\
└─ plot_final_summer_winter.py
```

## 実行

```powershell
cd <project_root>

& .\venv_new\Scripts\python.exe `
  .\plot_final_summer_winter.py
```

## 出力先

```text
<project_root>\paper_figures\final_prediction\
```

## 予測CSVの探索先

コードは、`result_csv`以下の次のフォルダを探します。

```text
pred_csv_TransformerFixed_multiseed
pred_csv_iTransformer_multiseed
pred_csv_HybridHQ_multiseed
pred_csv_LSTM_multiseed
```

各フォルダ内の、

```text
seed_42
seed_123
seed_2026
```

から、ファイル名に`summer`または`winter`を含む予測CSVを読み込みます。

## 1つのseedだけを使用する場合

```powershell
& .\venv_new\Scripts\python.exe `
  .\plot_final_summer_winter.py `
  --seeds 42
```

## 縦軸の単位

標準では現在の論文に合わせて、

```text
Demand (10^4 kW)
```

にしています。

予測CSVの値がMWへ変換済みであることを確認した場合は、

```powershell
& .\venv_new\Scripts\python.exe `
  .\plot_final_summer_winter.py `
  --y-label "Load (MW)"
```

としてください。

## モデルを減らす場合

例えばActual、iTransformer、HybridHQだけにする場合は、

```powershell
& .\venv_new\Scripts\python.exe `
  .\plot_final_summer_winter.py `
  --models iTransformer HybridHQ
```

## 注意

最終論文用では、欠損モデルを見落とさないため、通常は`--allow-missing`を付けないでください。
