# PREE 2026 アブレーション実験コード

## 4つのモデル

| ファイル | HQ | 双方向FiLM | Linear Residual |
|---|---:|---:|---:|
| `01_Full_BiFiLM_HQ_Residual_Optuna.py` | あり | あり | あり |
| `02_NoFiLM_SimpleConcat_HQ_Residual_Optuna.py` | あり | なし（単純結合） | あり |
| `03_BiFiLM_NoHQ_Residual_Optuna.py` | なし | あり | あり |
| `04_BiFiLM_HQ_NoResidual_Optuna.py` | あり | あり | なし |

全ファイルを同じフォルダへ置いて実行してください。

## 1. Optuna探索

各モデルをall-season・seed 42・30 trialsで探索します。

```powershell
python 01_Full_BiFiLM_HQ_Residual_Optuna.py --mode search --n-trials 30
python 02_NoFiLM_SimpleConcat_HQ_Residual_Optuna.py --mode search --n-trials 30
python 03_BiFiLM_NoHQ_Residual_Optuna.py --mode search --n-trials 30
python 04_BiFiLM_HQ_NoResidual_Optuna.py --mode search --n-trials 30
```

4モデル一括：

```powershell
python run_all_ablation_optuna.py --mode search --n-trials 30
```

SQLite DBが残るため、後からtrial総数を50へ増やす場合は次だけです。

```powershell
python run_all_ablation_optuna.py --mode search --n-trials 50
```

既存30 trialに不足する20 trialだけ追加されます。

## 2. 最終3seed評価

各モデル自身のbest HPを固定し、全期間・4季節を3seedで評価します。

```powershell
python run_all_ablation_optuna.py --mode final --seeds 42 123 2026
```

seedを5個へ増やす場合：

```powershell
python run_all_ablation_optuna.py --mode final --seeds 42 123 2026 777 3407
```

注意：このコマンドは指定seedを最初から実行します。既存結果を自動スキップする仕様ではありません。

## 3. 探索と最終評価を続けて実行

```powershell
python run_all_ablation_optuna.py --mode search_and_final --n-trials 30 --seeds 42 123 2026
```

## 4. 動作確認

まず1 trial、最終学習2 epoch、all-seasonのみで確認できます。

```powershell
python 02_NoFiLM_SimpleConcat_HQ_Residual_Optuna.py --mode search_and_final --n-trials 1 --search-epochs 2 --search-patience 2 --final-epochs 2 --final-patience 2 --seasons all --seeds 42
```

動作確認用DBと本番DBを混ぜたくない場合は、別の保存先を指定してください。

```powershell
python 02_NoFiLM_SimpleConcat_HQ_Residual_Optuna.py --mode search_and_final --n-trials 1 --search-epochs 2 --final-epochs 2 --seasons all --seeds 42 --output-root "<project_root>\result_csv\ablation_test"
```

## 5. 結果の比較表

4モデルの最終評価後：

```powershell
python compare_ablation_results.py
```

## 保存先

```text
<project_root>\result_csv\ablation_optuna_equal_budget_seed42\
├─ full_bifilm_optuna.db
├─ simple_concat_optuna.db
├─ no_hq_optuna.db
├─ no_residual_optuna.db
├─ 各best_params JSON
├─ 各Optuna trial CSV
└─ final\
   ├─ full_bifilm\
   ├─ simple_concat\
   ├─ no_hq\
   └─ no_residual\
```

## 公平性

4条件で以下を統一しています。

- all-seasonのみでHP探索
- tuning seed 42
- strict warm 80/10/10
- 72時間入力、24時間one-shot出力
- standardized validation MSE
- 同じtrial数
- 最大80 epoch、patience 10
- TPE sampler seed 42
- MedianPruner
- Adam / AdamW
- testデータをHP選定に使用しない
- 過去需要 + 気象 + カレンダー

### アブレーションの解釈上の注意

このコードは各変種を同じOptuna予算で独立調整します。そのため、「各構造が公平に調整された場合の性能」を比較できます。

構成要素だけの純粋な影響を確認する補足実験としては、Fullモデルのbest HPを全変種へ固定する比較も有効です。論文では、どちらの方針を採用したか明記してください。
