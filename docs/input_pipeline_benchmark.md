# テーブル取得・動画デコーダの比較

```bash
cd /fsx/koshi.makihara/openpi
BENCH_OUT="outputs/input_variants_$(date +%Y%m%d_%H%M%S)"
sbatch scripts/slurm_benchmark_input_variants.sh --output "$BENCH_OUT"
```

既定では元の640×480動画を使い、batch=256 / workers=64で次の4条件を順番に測ります。

| table_query_mode | video_backend | 内容 |
|---|---|---|
| legacy | pyav | 現行方式 |
| grouped | pyav | テーブル取得のみ改善 |
| legacy | torchcodec | 動画デコーダのみ変更 |
| grouped | torchcodec | 両方を変更 |

- H200×8、CPU 96、Slurm上限48時間。各条件のタイムアウトは初期化込み8時間。
- 各条件でモデルとloaderを作り直し、同じseed・同じサンプル順を使用します。
- 128 stepウォームアップ後、256 stepを計測します。先読みで隠れた待ち時間を捕捉するため、計測はworker数の4倍以上を必須にしています。
- 計測前に実データ16サンプル（先頭・末尾を含む。小規模データでは重複を除く）で出力を検証します。
- テーブル取得方式間は完全一致を必須にします。デコーダ間の画像差は数値で報告し、画像以外は完全一致を必須にします。
- 読み込みエラーを別サンプルへの置換で隠さず、その条件を失敗として記録します。
- チェックポイントやW&Bの出力は作りません。通常学習の既定方式はlegacyのままです。

## 結果

```bash
cat "$BENCH_OUT/results.csv"
cat "$BENCH_OUT/validation.json"
```

- `samples_per_second`: 待ち時間込みの処理量。大きいほど速い。
- `data_wait_seconds`: データ取得・GPUへの転送完了待ちの平均。小さいほど良い。
- `compute_seconds`: データ準備完了後からJAX計算完了までの平均。
- `step_max_seconds` / `data_wait_max_seconds`: 周期的な大きな停止も確認する。
- 条件ごとの`.steps.csv`: ウォームアップを含む各stepの時間。
- 条件ごとの`.log`: 初期化状況・各stepの時間・失敗理由。計測中はこちらで進捗を確認できる。

中央値やp95だけではまれな長時間停止を見落とします。平均処理量と最大待ち時間を確認してください。
このベンチマークは毎stepでJAXを同期するため、非同期の通常学習ログと速度を直接比較するのではなく、同じベンチマーク内の条件を比較します。

ファイルキャッシュや実行順序の影響を確認したい場合は、`--repeats 2`を追加します。2周目は逆順で実行します。
8条件のタイムアウト合計に合わせる場合は、`sbatch --time=96:00:00 ... --repeats 2`を使います。

条件を変える例：

```bash
sbatch scripts/slurm_benchmark_input_variants.sh --batches 512 --workers 64
```

実データ全体での検証とGPU計測は、上記のジョブを投入した際に実行されます。
