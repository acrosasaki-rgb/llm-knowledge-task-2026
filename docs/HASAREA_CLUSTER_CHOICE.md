# hasArea cluster-choice experiment

## Hypothesis

Twenty independent numeric answers often contain several internally consistent
groups. A raw median or largest-cluster selector cannot recover a correct minority
group. This experiment compresses the 20 answers into numeric clusters, then asks
the same model to solve a multiple-choice question over the cluster representatives.

The experiment isolates this selection step from the preceding metadata experiment:

- reasoning is disabled;
- the initial generation settings are unchanged from the 20-candidate baseline;
- no unit, scope, or reference-year metadata is supplied to the selector;
- the first 50-row smoke reuses the exact candidates from Job 580;
- a row with only one numeric cluster skips the model-selection call.

## Flow

1. Parse the first numeric value from each of the 20 candidate answers.
2. While values remain, find the observed value whose relative five-percent
   neighborhood contains the most remaining values.
3. Emit that neighborhood as one cluster and remove its members. Ties prefer the
   tighter cluster and then the cluster nearer the overall median.
4. Use each cluster median as a multiple-choice value. The model sees only the
   subject and the flat list of numeric values; it is not told about clustering,
   candidate counts, support, range, or the dominant cluster.
5. If there is one cluster, return its median without another model call.
6. If there are multiple clusters, ask Qwen3.5-27B to select exactly one supplied
   value for the subject's total geographic area in square kilometers (or a lake's
   water-surface area).
7. If the response is not exactly one listed representative, fall back to the
   dominant cluster median.

On the saved first-50 candidates, 19 rows have one cluster and therefore skip the
selection call. The other 31 rows require one selection call. The median number of
choices is 3, with a range of 1 to 18.

## Prompt policy

The selector is told only to choose the subject's requested area from a numbered
list of values and copy one value into the JSON response. It is not allowed to
invent a new value or perform a unit conversion. Cluster support, range, and
dominance remain in `FinalSelection` for analysis but are never included in the
model prompt.

## Comparable smoke outputs

The 50-row smoke produces four predictions from the exact same 20 candidates:

- raw median;
- dominant five-percent cluster median;
- the previous metadata-judge output from Job 580;
- cluster-choice output.

The smoke is followed by a manual approval job. The remaining 50 validation rows
do not start until the smoke artifacts have been reviewed. No test-split inference
is present in this pipeline.

## Result with cluster evidence in the prompt (superseded)

The first-50 smoke ran in child Pipeline 60, Job 621. The reranking phase took
30.70 seconds, made 31 selection calls, skipped 19 single-cluster rows, used no
fallbacks, and peaked at 17.26 GiB VRAM.

| Selector | Correct rows | Macro F1 |
|---|---:|---:|
| Raw median | 24/50 | 0.48 |
| Dominant five-percent cluster | 25/50 | 0.50 |
| Metadata judge | 25/50 | 0.50 |
| Cluster-choice judge | 25/50 | 0.50 |

The cluster-choice judge selected the dominant cluster on 48/50 rows. It changed
only two already-wrong rows and produced no binary-score improvement or regression:

- Leros: dominant 19.25, selected 13.77, gold 54.052;
- Lake Texcoco: dominant 2200, selected 220, gold 2000.

Both changes increased numeric relative error even though they remain equivalent
under the binary five-percent score. The correct-row set is therefore identical to
the dominant-cluster selector.

Five dominant-wrong rows offered a correct cluster representative, but the model
selected the dominant wrong cluster in every case:

| Subject | Gold | Dominant choice (support) | Correct choice (support) |
|---|---:|---:|---:|
| Vygozero | 1250 | 244 (3) | 1200 (1), 1280 (1) |
| Rùm | 104 | 37.65 (14) | 102.12 (3) |
| Hashima Island | 0.063 | 0.63 (13) | 0.063 (3) |
| Lítla Dímun | 0.8 | 1.485 (4) | 0.8 (1) |
| Vasilyevsky Island | 10.9 | 43.1 (4) | 10.7 (1) |

Two more rows previously known to contain a directly correct raw candidate did not
retain a correct cluster representative. In Corfu and Lake Texcoco, the greedy
five-percent neighborhood absorbed correct candidates into a wider group whose
median fell outside the official tolerance. Thus only 5 of the 7 recoverable
minority-candidate cases remained recoverable after cluster compression.

This smoke provides no evidence that exposing clustered choices improves hasArea.
The support counts appear to reinforce the dominant wrong fact, and using only a
cluster median can erase a sparse correct value. The remaining 50 rows stay behind
the manual approval gate.

This run did not implement the intended flat-choice condition because it exposed
support, range, dominance, and the origin of the choices to the model. It is retained
only as an ablation result and must not be treated as the flat-choice result.

## Flat-choice result

The intended flat-choice condition ran in child Pipeline 63, Job 663, using the
same Job 580 candidates. The model prompt contained only the subject, requested
area scope, and a numbered list of numeric values. The reranking phase took 20.53
seconds, made 31 selection calls, skipped 19 single-choice rows, used no fallbacks,
and peaked at 17.26 GiB VRAM.

| Selector | Correct rows | Macro F1 |
|---|---:|---:|
| Raw median | 24/50 | 0.48 |
| Dominant five-percent cluster | 25/50 | 0.50 |
| Metadata judge | 25/50 | 0.50 |
| Flat cluster choices | 20/50 | 0.40 |

Removing cluster evidence made the selector much more willing to leave the
dominant value: it selected a non-dominant option on 19/50 rows, compared with
2/50 when support and dominance were shown. Those 19 changes produced zero
improvements, five regressions, and fourteen wrong-to-wrong changes.

| Regression | Gold | Dominant correct choice | Flat selected choice |
|---|---:|---:|---:|
| Icaria | 255.32 | 252.8 | 203.98 |
| Folegandros | 32.216 | 32 | 12.28 |
| Margarita Island | 1020 | 1021 | 960 |
| Vancouver Island | 32134 | 31285 | 13385 |
| Lewis and Harris | 2086 | 2179 | 1841 |

The same five dominant-wrong rows still offered a correct minority cluster
representative, but the flat selector recovered none. It chose a different wrong
option for Vygozero, Lítla Dímun, and Vasilyevsky Island, and retained the dominant
wrong option for Rùm and Hashima Island.

Therefore the intended flat multiple-choice formulation performs worse than both
the dominant-cluster selector and the evidence-bearing choice prompt. The result
suggests that Qwen3.5-27B does not reliably recognize the correct area fact from
the value list alone; removing consensus evidence increases arbitrary wrong-choice
selection without recovering correct minority facts. The remaining 50 rows stay
behind the manual approval gate.
