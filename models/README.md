# models/ — モデル本体（論文で使った最新版）

各フォルダには論文で使った最新版 1 本だけを置いています（3 seed：42 / 123 / 2026、最大200 epoch、patience 20、Optuna の best HP）。

| フォルダ | スクリプト |
|---|---|
| `transformer/` | `TransformerFixed_multiseed_OptunaHP_final.py` |
| `itransformer/` | `iTransformer_multiseed_OptunaHP_final.py` |
| `hybrid/`（提案） | `HybridHQ_multiseed_OptunaHP_final.py` |
| `lstm/` | `LSTM_multiseed_OptunaHP_final.py` |

- HP は各スクリプト冒頭の `FIXED_PARAMS` / `BEST_PARAMS_BY_SEASON` / `HP` に、Optuna の結果（`result/final/optuna_model_search/`）が書き込まれています。
- 出力先：`result/result_csv/`（集計・予測CSV）、`result/learning_saver/`（重み）、`picture/`（学習曲線などの図）。

構造の図はトップの [README](../README.md#4-モデル) を参照してください。
