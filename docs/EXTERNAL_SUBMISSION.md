# 外部SSH GPUホストでのtest提出物生成

## 実行境界

GitLab Runnerはvalidation推論と評価だけを担当します。公式`test.jsonl`を入力に
するモデル推論はGitLab CIに置かず、別管理のLinuxホストへSSH接続した後に
DockerとNVIDIA GPUを使って実行します。

```text
GitLab GPU Runner
  └─ validation smoke / approval / shard / merge / score
       └─ 人がartifactを確認
            └─ CPU-only selection job
                 └─ reports/selection.json
                      └─ SSH/SCP（利用者が管理）
                           └─ 外部Linuxホスト
                                └─ Docker --gpus all
                                     └─ test shard / merge / format check
```

GitLabのCI/CD variableへSSH秘密鍵、接続先、SSH passwordを登録する必要は
ありません。接続とmanifest転送は利用者の端末から行い、接続先の
`authorized_keys`やDocker/NVIDIA設定もprojectリポジトリの管理対象外です。
現在checked-inされている外部runnerは、標準ベースラインの
Qwen3.5-27B MTP GGUF専用です。別モデルを選抜する場合は、そのruntimeを固定
したDockerfile、launcher、manifest検証テストを追加してからtestを実行します。

## 外部ホストの前提

- Linux x86_64
- NVIDIA Driver
- NVIDIA Container Toolkitを設定済みのDocker Engine
- GitとPython 3.11以上
- モデルとdatasetを取得できるネットワーク（必要ならDocker用proxy）
- 選抜モデルを実行できるGPUメモリ

最初に次を確認します。

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## GitLabからのhandoff

1. `experiment:model:qwen3.5-27b-mtp-thinking`を起動します。
2. 5行smoke artifactを確認し、approval後に全validationを完了させます。
3. 完全なvalidation予測、quality report、Macro/Micro F1を確認します。
4. 同じcommitの`selection:model:qwen3.5-27b-mtp-thinking`を手動実行します。
   relation-aware 20候補版では、完全な20候補validation Pipeline URLを
   `VALIDATION_PIPELINE_URL`へ指定して
   `selection:model:qwen3.5-27b-mtp-relation-aware-20`を実行します。
5. job artifactの`reports/selection.json`を取得します。
6. repository cloneとmanifestを通常のSSH/SCP運用で外部ホストへ転送します。

例:

```bash
scp selection.json submission-host:/srv/akbc-task/reports/selection.json
ssh submission-host
cd /srv/akbc-task
git fetch origin
git checkout <selection.jsonのcommit_sha>
bash scripts/submission/run-qwen27b-mtp-docker.sh reports/selection.json
```

relation-aware 20候補版は専用wrapperを使用します。

```bash
bash scripts/submission/run-qwen27b-mtp-relation-aware-20-docker.sh \
  reports/selection.json
```

launcherは次を満たさない場合、モデルを開始する前に失敗します。

- checkoutに未commit差分がない
- checkoutのcommitとmanifestの`commit_sha`が一致する
- model key、config path、config SHA-256、dataset commitが一致する
- manifest、config、launcherの候補数が一致する（20候補版は20）
- 実行環境名とdigest固定Docker imageが一致する
- DockerとGPUを利用できる

## 出力とresume

モデルweightは`.cache/submission-huggingface`、datasetは
`.cache/submission-dataset`へ保存されます。test推論は50行単位で逐次実行し、
候補JSONLが残っていれば`--resume`で再利用します。

完了時の主なファイル:

- `outputs/qwen3.5-27b-mtp-thinking-test.jsonl`: Codabench提出用
- `outputs/qwen3.5-27b-mtp-thinking-candidates-test.jsonl`: 分析・resume用
- `reports/qwen3.5-27b-mtp-thinking-test-quality.json`: 形式と空予測率
- `reports/qwen3.5-27b-mtp-thinking-test-metrics.json`: 統合metrics
- `reports/qwen3.5-27b-mtp-thinking-test.jsonl.sha256`: 提出物digest
- `reports/selection.json`: GitLabから受け取った選抜記録

20候補版では同じ構成で
`outputs/qwen3.5-27b-mtp-relation-aware-20-test.jsonl`を生成します。
候補生成は5候補版の約4倍になるため、途中停止時は同じlauncherを再実行して
既存candidate shardを`--resume`で再利用します。

統合処理は公式入力との行数、順序、`SubjectEntity`、`Relation`、出力keyを
照合します。`test.jsonl`を評価器へ渡す処理はありません。
