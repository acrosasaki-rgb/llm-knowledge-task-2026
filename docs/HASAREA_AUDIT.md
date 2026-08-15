# hasArea conversation audit

## Baseline observation

The non-thinking Qwen3.5-27B run generates 20 independent numeric candidates
and returns their median. It scores 44/100 on validation. Of 2,000 generated
candidates, 923 are within the official five-percent numeric tolerance. Correct
candidates are the majority in 44 rows, tied in one row, and the minority in 55
rows. Thirty-one rows contain no correct candidate at all.

A post-hoc largest-five-percent-cluster selector improves the same candidate
artifact from 44 to 49 correct rows. An oracle that chooses a correct candidate
whenever one exists reaches 69 rows, showing that aggregation can recover some
but not all of the gap.

## Experiment

Issue #13 adds one audit turn after every initial `hasArea` answer. The initial
prompt, five-shot selection, generation seed, sampling parameters, and 20 raw
candidates remain identical to the baseline. The follow-up keeps the same
conversation and asks for six structured fields:

- original value;
- underlying unit;
- area scope;
- reference year;
- value normalized to square kilometers;
- whether the candidate should be retained.

Supported square-mile, hectare, and acre values are converted to square
kilometers only when the conversion is numerically consistent. Candidates are
excluded when the audit identifies a land-only value, administrative subarea,
metropolitan area, historical area, unknown scope or unit, value mismatch,
conversion mismatch, or `keep=false`. The final prediction is the median of
the accepted normalized values. If every audit is rejected, it falls back to
the original 20-candidate median.

The validation-only child pipeline first runs offsets 0--49, which are the
first 50 `hasArea` rows. A manual artifact review gate blocks offsets 50--99.
After approval, the two ordered shards are merged against an exact 100-row
input slice. No test inference, selection manifest, or deployment is defined.

## First-50 smoke result

GPU job 541 completed offsets 0--49 in 1,956.6 seconds with 17.26 GiB
peak CUDA memory. All 1,000 initial answers and all 1,000 audit turns finished
without a token-limit or empty-answer failure.

The model returned the six requested fields in their documented order but
usually omitted the `field=` labels. The original parser therefore reported
997 parse failures even though the positional values were present. The parser
now accepts both the labeled representation and an exact six-string positional
representation. Reprocessing the saved raw audit text parses all 1,000 turns;
no inference output was repaired or regenerated for the measurements below.

With unknown or invalid reference years rejected, 890 candidates are accepted
and 110 are rejected. One row has no accepted audit and uses the documented raw
median fallback. The acceptance rates are:

- 460/471 (97.7%) for initially correct candidates;
- 430/529 (81.3%) for initially wrong candidates.

This raises correctness among retained candidates only from 47.1% to 51.7%.
The audit claims 28 values are hectares and 20 are square miles. Only two
hectare and three square-mile audits survive all consistency checks, and none
of those five normalized values is correct. Thus the audit does not recover a
single row through unit conversion in this shard.

The first 50 raw candidate rows reproduce the baseline exactly for 48 rows.
Hopen differs only at one candidate (`1080` versus `1090`) and Folegandros only
at one candidate (`32.02` versus `32.01`); neither difference changes an
aggregation result. The comparable row scores are:

| Selector | Correct rows | Macro F1 |
| --- | ---: | ---: |
| Raw 20-candidate median | 24/50 | 0.48 |
| Conversation-audited median | 24/50 | 0.48 |
| Post-hoc largest 5% cluster | 25/50 | 0.50 |

The audited median changes the numeric prediction on nine rows but produces no
improvement and no regression under the official five-percent tolerance.
Rejecting unknown reference years separately also remains 24/50.

The dominant failure is self-confirmation rather than output formatting. For
example, every Hopen candidate is wrong, yet all 20 audit turns label the
previous value as square kilometers, total geographic area, current/general,
and `keep=true`. Leros similarly retains all 20 values even though its median
describes the wrong area fact. Hashima Island contains three correct `0.063`
candidates, but the audit also retains the much more frequent factor-of-ten
`0.63` candidates. A follow-up from the same conversation therefore filters
some obvious anomalies without reliably distinguishing a plausible wrong fact
from the target fact.

The manual approval gate remains closed. Given no gain on the smoke shard and
the one extra generation per candidate, running offsets 50--99 is not
recommended unless the goal is to confirm the negative result on all 100 rows.
