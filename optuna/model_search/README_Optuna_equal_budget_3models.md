# TransformerFixed / iTransformer / HybridHQ
# all-season・seed=42・同一探索予算 Optuna

## 公平性の方針

3モデルで次を統一しています。

- 探索データ: all-seasonのみ
- 探索seed: 42
- split: strict warm 80/10/10
- 入力履歴: 72時間
- 出力: 24時間 one-shot
- 目的関数: 標準化後のvalidation MSE
- trial数: 各30 trial
- 最大epoch: 80
- Early Stopping: patience 10
- sampler: TPE、seed 42
- pruner: MedianPruner
- optimizer候補: Adam / AdamW
- testデータはHP選定に使用しない
- 入力情報: 過去需要 + 気象 + カレンダー

モデル固有の層数やhidden sizeは、それぞれの構造に対応した探索範囲です。
「同じ調整労力」は同じtrial数・学習予算・評価手順を意味し、
異なる構造に同一の層数や同一の数値HPを強制することではありません。

## TransformerFixedの特徴量について

既存コードにあった明示的な lag_1 / lag_24 / lag_48 は、
この公平探索版では追加していません。

TransformerFixedとiTransformerは
`power_demand + weather + calendar` の時系列を入力します。

HybridHQは、同じ情報を
- temporal branch: 過去需要
- variable branch: weather + calendar
に分けて入力します。

## インストール

```powershell
pip install optuna
```

## 3モデルを一括探索

4ファイルを同じフォルダへ置きます。

```powershell
python Optuna_search_3models_all_seed42_equal_budget.py
```

## 個別探索

```powershell
python Optuna_search_TransformerFixed_all_seed42.py
python Optuna_search_iTransformer_all_seed42.py
python Optuna_search_HybridHQ_all_seed42.py
```

## 保存先

```text
<project_root>\result_csv\optuna_equal_budget_seed42\
├─ TransformerFixed_best_params_all_seed42.json
├─ iTransformer_best_params_all_seed42.json
├─ HybridHQ_best_params_all_seed42.json
├─ TransformerFixed_optuna_trials.csv
├─ iTransformer_optuna_trials.csv
├─ HybridHQ_optuna_trials.csv
├─ THREE_MODELS_optuna_best_summary.csv
└─ 各モデルのSQLite DB
```

SQLiteが残っている場合、総trial数が30になるまで不足分だけ追加します。
すでに30 trialある場合は、勝手にさらに30 trial追加しません。

## 探索後

各JSONの `best_params` を、それぞれの最終3seed評価コードへ設定します。
その後、同じHPをall・spring・summer・autumn・winterに固定して
seed 42 / 123 / 2026で最終評価します。
