# hasArea unit and magnitude plausibility experiment

## Goal

Test whether human-like unit and decimal-place checks can recover minority area
answers without repeating the failed unrestricted multiple-choice selection.
The first 50-row smoke reuses the exact 20 candidates from Job 580.

## Independent evidence

The model makes two new estimates before it sees any candidate values:

1. A scale probe classifies the subject as island, lake, region, other, or unknown,
   and selects one decade-wide area bin from below 0.01 km2 through at least
   100,000 km2. It is explicitly prohibited from returning an exact area.
2. A separate dimension probe estimates only maximum length, maximum width, and
   footprint shape. It is explicitly prohibited from stating or calculating area.

Code converts dimensions into a broad plausible area interval using fixed shape
fill ranges. The two calls use independent conversations and seeds, and neither
prompt contains candidate values, candidate counts, clusters, or prior answers.

## Mechanical hypotheses

Every individual raw numeric candidate produces four hypotheses:

- the value is already square kilometers;
- the value is square miles and is multiplied by 2.589988110336;
- the value is hectares and is multiplied by 0.01;
- the value is acres and is multiplied by 0.0040468564224.

Five-percent clusters are used only to calculate support and the conservative
default; unit conversion never replaces an observed value with a cluster median.
Power-of-ten relations among original raw clusters are recorded as decimal-place
conflicts. The experiment does not invent arbitrary shifted values: direction is
chosen only among observed clusters, using candidate-blind scale and dimension
evidence.

## Conservative selection

The dominant direct square-kilometer cluster remains the default. A deterministic
selector measures log-distance from each hypothesis to the estimated scale and/or
dimension interval. It overrides the default only when the proposed hypothesis
improves the evidence score by at least 0.15 log10 units after a small unit-conversion
penalty. The model never receives a final multiple-choice question.

The smoke evaluates six selectors from the same two model calls:

- scale with direct values only;
- scale with unit expansion;
- dimensions with direct values only;
- dimensions with unit expansion;
- scale plus dimensions with direct values only;
- scale plus dimensions with unit expansion.

Raw median and dominant-cluster predictions are included as controls. All parsed
estimates, raw responses, mechanical hypotheses, decimal relations, component
scores, and override decisions are saved in the candidate artifact.

## Validation gate

Only the first 50 validation rows run initially. A manual approval job blocks the
remaining 50 rows. The pipeline contains no test inference, selection manifest, or
deployment.

## Result

The corrected raw-candidate expansion ran in child Pipeline 68, Job 730. The two
candidate-blind probes took 88.25 seconds for 50 rows and peaked at 17.25 GiB VRAM.
Scale estimates parsed on 49/50 rows and dimension estimates on 43/50 rows.

| Selector | Correct rows | Macro F1 |
|---|---:|---:|
| Raw median | 24/50 | 0.48 |
| Dominant cluster | 25/50 | 0.50 |
| Scale, direct values | 20/50 | 0.40 |
| Scale, unit expansion | 9/50 | 0.18 |
| Dimensions, direct values | 24/50 | 0.48 |
| Dimensions, unit expansion | 23/50 | 0.46 |
| Scale + dimensions, direct values | 22/50 | 0.44 |
| Scale + dimensions, unit expansion | 12/50 | 0.24 |

### Evidence quality

The official value fell inside the predicted scale bin on only 13/49 parsed rows.
It fell inside the dimension-derived interval on only 16/43 parsed rows. The scale
probe made decisive order-of-magnitude errors on important cases: it classified
Vygozero and Misool as 10--100 km2 and Imrali as below 0.01 km2. The dimension probe
was more useful but still inaccurate; for example, it estimated Tanna as 20 by 12
km, producing an interval of 60--192 km2 for a 555 km2 island.

The direct dimension selector was the only variant with a useful recovery. It
changed ten rows versus the dominant cluster, improving Hashima Island from 0.63
to 0.063, regressing Tanna and Icaria, and changing seven wrong answers to other
wrong answers. Its net score was one row below the dominant selector.

### Unit expansion

Expanding each raw value does increase the oracle candidate set. Ten dominant-wrong
rows contain a correct direct or converted hypothesis after expansion. The newly
recoverable unit cases include:

- Leros: 19.9 square miles -> 51.54 km2 (gold 54.052);
- Misool: 800 square miles -> 2071.99 km2 (gold 2034);
- Imrali: 4.9 square miles -> 12.69 km2 (gold 13.32).

However, none of the three selectors recovered a new correct row from unit
expansion. Relative to their direct-only counterpart, scale plus units caused zero
improvements and eleven regressions; dimensions plus units caused zero improvements
and one regression; combined evidence plus units caused zero improvements and ten
regressions. The scale selector chose acre on 21/50 rows, showing that mechanically
plausible conversions become abundant false alternatives without independent unit
evidence.

Hashima illustrates the interaction. Its dimension estimate (0.8 by 0.4 km,
elongated) supported the correct direct value 0.063. Adding units then treated 0.063
as square miles and selected 0.163 km2, destroying the recovery. In Imrali, the
dimension interval correctly covered the converted 12.69 km2, but another direct
candidate also fell inside the broad interval and the conservative unit penalty
kept the wrong direct value.

The combined experiment therefore does not improve hasArea. Candidate-blind
dimensions contain a small useful signal for decimal direction, but candidate-blind
scale recall is unreliable, and applying every unit conversion without evidence of
the original unit is actively harmful. Future unit work should first detect a
specific cross-candidate conversion relationship or obtain independent evidence of
the source unit; conversion expansion alone should not authorize an override. The
remaining 50 rows stay behind the manual approval gate.
