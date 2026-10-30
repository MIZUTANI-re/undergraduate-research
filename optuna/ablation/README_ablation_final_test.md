# HybridHQアブレーション：固定HPによる最終test評価

## 何をするコードか

各アブレーション条件について、Optuna探索で保存済みの**各モデル固有のbest HP JSON**を読み込みます。

- Full HybridHQ
- No FiLM（Simple Concatenation）
- No Horizon Queries
- No Linear Residual

各HPを固定したまま、以下を実施します。

- seed: 42, 123, 2026
- dataset: all, spring, summer, autumn, winter
- trainで学習
- validationでearly stoppingとbest checkpoint選択
- 最後に未使用testデータで1回だけ評価
- test RMSE（MW）とMAPE（%）を保存
- 3seedのmean ± stdを作成
- Fullとの差分を作成

## 必要な配置

次の2ファイルを同じフォルダへ置いてください。

```text
<project_root>\
├─ ablation_optuna_core.py
└─ run_ablation_final_test_and_compare.py
```

Optuna探索後のJSONが次に存在する必要があります。

```text
<project_root>\result_csv\ablation_optuna_equal_budget_seed42\
├─ full_bifilm_best_params_all_seed42.json
├─ simple_concat_best_params_all_seed42.json
├─ no_hq_best_params_all_seed42.json
└─ no_residual_best_params_all_seed42.json
```

## 実行

PowerShell:

```powershell
cd <project_root>

& .\venv_new\Scripts\python.exe `
  .\run_ablation_final_test_and_compare.py
```

1行:

```powershell
& .\venv_new\Scripts\python.exe .\run_ablation_final_test_and_compare.py
```

既に最終評価が完了していて、表だけ再作成する場合:

```powershell
& .\venv_new\Scripts\python.exe .\run_ablation_final_test_and_compare.py --only-compare
```

## 出力

```text
result_csv\ablation_optuna_equal_budget_seed42\comparison\
├─ ABLATION_best_hyperparameters.csv
├─ ABLATION_test_each_run.csv
├─ ABLATION_test_each_run_four_model_compatible.csv
├─ ABLATION_test_mean_std.csv
├─ ABLATION_test_paper_table_wide.csv
├─ ABLATION_test_RMSE_mean_std_pivot.csv
├─ ABLATION_test_MAPE_mean_std_pivot.csv
└─ ABLATION_test_delta_vs_full.csv
```

`ABLATION_test_each_run_four_model_compatible.csv`は、既存の4モデル比較と同じ主要列です。

```text
model, seed, season, rmse, mape, source_file
```

## 評価指標

- RMSE: MW
- MAPE: %

Optuna順位ではなく、最終的なtest RMSE・MAPEでアブレーションの効果を判断してください。
