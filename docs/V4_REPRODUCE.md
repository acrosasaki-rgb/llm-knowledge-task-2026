# v4 予測の再現手順

v4 (`outputs/screening/mistral-small-24b-test-predictions-v4.jsonl`, sha256 先頭
`e1a491a9ed543515`) は 7 つの候補プールからビルダースクリプトで決定的に合成される。
val 公式スコア: All macro-F1 0.633(hasArea 0.800 / hasCapacity 0.280 /
company 0.785 / city 0.480 / borders 0.981 / award 0.126)。

## 1. 合成(決定的、プールがあれば完全再現)

```bash
python scripts/h100-bf16/build_predictions_v3.py \
  --base outputs/screening/mistral-small-24b-test-predictions.jsonl \
  --base-candidates outputs/screening/mistral-small-24b-test-candidates.jsonl \
  --area-primary outputs/screening/mistral-area-rq3-test-candidates.jsonl \
  --area-secondary outputs/screening/mistral-small-24b-test-candidates.jsonl \
  --area-arbiter outputs/screening/llama31-8b-area-test-candidates.jsonl \
  --cap-candidates outputs/screening/mistral-cap-grounding-test-candidates.jsonl \
  --cap-secondary outputs/screening/mistral-cap-step-test-candidates.jsonl \
  --cap-arbiter outputs/screening/llama31-8b-cap-test-candidates.jsonl \
  --output outputs/screening/mistral-small-24b-test-predictions-v4.jsonl
```

val 版は `--base/--base-candidates` を val 側ファイルに、各プールを対応する
val プール(`*-test-` なし)に差し替えるだけ。公式スコアは
`python .cache/dataset2026-latest/evaluate.py -p <predictions> -g data/val.jsonl`。

集約ロジック(スクリプト内に実装):
- hasArea: rq3 プール dominant_cluster(主)⊕ base プール unit_equivalence(副)
  ⊕ Llama モード裁定(±5% 不一致時のみ、どちらとも不一致なら主に退避)
- hasCapacity: P1083 プール dominant(主)⊕ cap-step プール dominant(副)⊕ Llama 裁定
- company: base プールを frequency 0.5 + empty_majority 8 で再集約
- 他リレーション: base 予測をそのまま維持(award は unanimous-no フィルタ適用済み)

## 2. プールの再生成(統計的再現; GPU 非決定性でビット同一は非保証、精度は ±1 行)

全プール共通: 8×A100 ホストで `scripts/h100-bf16/run-model-screening-docker.sh` を
リポジトリルートから実行。config はコミット済み(seed 42, temp 0.6, 20 候補/行)。

Mistral 系(3 本, GGUF: `unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF`
@`b750ec2299225e492f1bd27cab88a0a595fa848f`, part1
`Mistral-Small-3.2-24B-Instruct-2506-BF16.gguf`):

| プール | AKBC_MODEL_KEY | AKBC_CONFIG | AKBC_SUBJECTS |
|---|---|---|---|
| rq3 test | mistral-area-rq3-test | configs/experiment-mistral-small-24b-bf16-area-rq3.yaml | TESTREL:hasArea |
| cap-grounding test | mistral-cap-grounding-test | configs/experiment-mistral-small-24b-bf16-cap-grounding.yaml | ALLTESTCAP |
| cap-step test | mistral-cap-step-test | configs/experiment-mistral-small-24b-bf16-cap-step.yaml | TESTREL:hasCapacity |

Llama 系(2 本, GGUF: `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`
@`bf5b95e96dac0462e2a09145ec66cae9a3f12067`, part1
`Meta-Llama-3.1-8B-Instruct-Q8_0.gguf`, 追加 env:
`AKBC_GGUF_MIN_GIB=7 AKBC_REASONING_FLAGS="--reasoning off"`):

| プール | AKBC_MODEL_KEY | AKBC_CONFIG | AKBC_SUBJECTS |
|---|---|---|---|
| Llama area test | llama31-8b-area-test | configs/experiment-mistral-small-24b-bf16-screen.yaml | TESTREL:hasArea |
| Llama cap test | llama31-8b-cap-test | configs/experiment-mistral-small-24b-bf16-screen.yaml | TESTREL:hasCapacity |

base の test 予測/候補(477 行)は 2026-08-13 の
`configs/experiment-mistral-small-24b-bf16-test.yaml` 実行の成果物
(`mistral-small-24b-test-{predictions,candidates}.jsonl`)。award の
unanimous-no フィルタは `scripts/h100-bf16/apply_award_verify.py` を test 票
(`qwen3.5-27b-bf16-award-verify-test-votes.jsonl`)で適用したもの。

## 3. 再現性の根拠

- 合成は乱数なしの純関数(同一プール → バイト同一の出力)。
- 生成側はシードを変えても結果が ±1 行で安定することを実測済み
  (REAP アーム seed42=78 / seed20260814=78-79、Llama 裁定は両シードで 80)。
- 総パラメータ 24B + 8B ≈ 31.8B ≤ 32B(コンペ規約内)。

## 4. v5 / v6 への拡張(最終提出 = v6, test All 0.6097)

- **v5**(company 略語畳み込み): 追加プールなし。`build_predictions_v3.py` の
  `company_aggregate` が、頻度しきい値判定の前に大文字小文字の変種を統合し、
  共起する長い形の頭文字を綴る略語(NYSE → New York Stock Exchange)を長い形へ
  畳み込む。合成コマンドは v4 と同一。
- **v6**(capacity 第 2 裁定者): 8 本目のプール
  `ministral-8b-cap-test-candidates.jsonl` を追加生成し
  (GGUF: `bartowski/Ministral-8B-Instruct-2410-GGUF`
  @`b4e2ea74eb4eecb178aa88d482c3126b34ad0157`, part1
  `Ministral-8B-Instruct-2410-Q8_0.gguf`, env: `AKBC_GGUF_MIN_GIB=7
  AKBC_REASONING_FLAGS="--reasoning off"`, config は screen 用,
  `AKBC_SUBJECTS=TESTREL:hasCapacity`)、v4 の合成コマンドに
  `--cap-arbiter2 outputs/screening/ministral-8b-cap-test-candidates.jsonl`
  を追加する。capacity の裁定は副アーム [cap-step, base] × 裁定者
  [Llama, Ministral] に拡張される。
- 検証済みハッシュ: v5 `268c6260…` / v6 `180fd4d3…`(いずれも sha256 先頭、
  再合成でバイト同一を確認済み)。
- v7(city 2 段×プール統合)は test で非移転(−1 行)のため**最終構成に含めない**。
  材料(`mistral-city-profile*-candidates.jsonl`, `mistral-city2*-candidates.jsonl`)は
  論文の移転分類の再現用に保持する。
