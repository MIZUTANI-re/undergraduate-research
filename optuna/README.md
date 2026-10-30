# optuna/ — ハイパーパラメータ探索

| フォルダ | 目的 | 結果（確定値） |
|---|---|---|
| `model_search/` | 4モデルを **同一予算**（30 trial・通年・seed 42・最大80 epoch・patience 10）で探索 | `result/final/optuna_model_search/` |
| `ablation/` | HybridHQ の部品（FiLM / HQ / Linear Residual）を外した4条件を、各条件同じ予算で探索 → 3 seed で最終評価 | `result/final/optuna_ablation/` |

## model_search/

| ファイル | 内容 |
|---|---|
| `Optuna_search_3models_all_seed42_equal_budget.py` | Transformer / iTransformer / HybridHQ の探索本体（3モデル一括） |
| `Optuna_search_{TransformerFixed,iTransformer,HybridHQ}_all_seed42.py` | 1モデルだけ探索する入口 |
| `LSTM_optuna_multiseed_all_seasons.py` | LSTM の探索 + 最終評価（`OPTUNA_MODE="all_only"`） |
| `README_Optuna_equal_budget_3models.md` | 公平性の方針と使い方 |

SQLite DB が残っていれば、総 trial 数が指定値になるまで不足分だけ追加されます。

## ablation/

使い方は `README_アブレーションOptuna実行方法.md`（探索）と `README_ablation_final_test.md`（最終評価）を参照。

| 条件 | HQ | 双方向FiLM | Linear Residual |
|---|:-:|:-:|:-:|
| `01_Full_BiFiLM_HQ_Residual` | ✔ | ✔ | ✔ |
| `02_NoFiLM_SimpleConcat_HQ_Residual` | ✔ | 単純結合 | ✔ |
| `03_BiFiLM_NoHQ_Residual` | – | ✔ | ✔ |
| `04_BiFiLM_HQ_NoResidual` | ✔ | ✔ | – |

選ばれた HP と結果はトップの [README](../README.md#5-実験の条件) を参照。
