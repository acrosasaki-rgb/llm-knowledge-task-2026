# Qwen3.5-27B MTP Reasoning ablation

## Validation results

The comparison uses the pinned Qwen3.5-27B MTP Q4_K_M model, the same
five-shot policy, candidate-specific deterministic seeds, and the official
478-row validation split. The non-thinking five-candidate result was obtained
by taking candidates 0 through 4 from the completed 20-candidate artifact and
reapplying the same aggregation policy. No additional model inference was used.

| Configuration | Candidates | Macro F1 | Micro F1 |
|---|---:|---:|---:|
| non-thinking, award threshold 0.2 | 5 | 0.4986 | 0.2804 |
| relation-aware mixed reasoning, award threshold 0.2 | 5 | 0.5273 | 0.2074 |
| non-thinking, award threshold 0.4 | 20 | 0.5052 | 0.2765 |

The mixed-reasoning configuration enables thinking only for `hasArea`,
`countryLandBordersCountry`, and `companyTradesAtStockExchange`. Its fair
five-candidate relation-level comparison is:

| Relation | Non-thinking | Reasoning | Delta |
|---|---:|---:|---:|
| hasArea | 0.470 | 0.610 | +0.140 |
| countryLandBordersCountry | 0.979 | 0.992 | +0.013 |
| companyTradesAtStockExchange | 0.693 | 0.682 | -0.011 |

Reasoning has a clear positive effect only for `hasArea` in this experiment.
It is nearly neutral for land borders and slightly harmful for stock exchange.
The mixed-reasoning run took 42,295 aggregate shard-seconds for five candidates;
the non-thinking run took 7,063 seconds for twenty candidates. This is about
24 times slower per candidate. Candidate count must therefore be held constant
when attributing score changes to reasoning.

The operationally strongest artifact-only hybrid replaces only the `hasArea`
rows in the non-thinking 20-candidate prediction with the reasoning
five-candidate predictions. It scores Macro F1 0.5407 and Micro F1 0.2894, but
this hybrid is a deployment comparison rather than a controlled reasoning
ablation.

## awardWonBy follow-up

The non-thinking 20-candidate run scores Macro F1 0.1089 on `awardWonBy`.
196 of its 200 award candidates reach the 128-token limit. Issue #11 tests a
controlled follow-up that keeps the same model, no-thinking mode, 20 candidates,
128-token limit, sampling settings, and all non-award behavior. It changes only
the award prompts: each candidate covers one non-overlapping award-date period,
then normalized recipient names are unioned in chronological candidate order.
The targeted smoke covers validation offsets 200 through 209, which are all ten
`awardWonBy` rows, so the category effect can be measured before rerunning the
468 unchanged rows.

### Era-decomposition result

Pipeline 48, smoke job 451, completed all ten award rows with no reasoning and
20 non-overlapping time periods. The direct union result is:

| Award aggregation | Macro P | Macro R | Macro F1 | Micro F1 | Avg. predictions |
|---|---:|---:|---:|---:|---:|
| original non-thinking frequency | 0.3650 | 0.0786 | 0.1089 | 0.0680 | 13.1 |
| era decomposition, union | 0.2199 | 0.1748 | 0.1685 | 0.1427 | 112.0 |

Era decomposition improves award Macro F1 by 0.0596 and more than doubles
Macro recall. Eight of ten award rows improve; the two regressions are the Yale
honorary doctorate and the Order of the Aztec Eagle. The strongest individual
gain is Turing Award, from F1 0.3061 to 0.6081.

The response-length problem is substantially reduced: token-limit candidates
fall from 196/200 (98%) to 60/200 (30%), with zero parse failures. Direct union,
however, retains every one-off hallucination and increases the average output
from 13.1 to 112 entities.

Replacing only the ten award rows in the completed 478-row non-thinking
prediction changes overall Macro F1 from 0.5052 to 0.5064 and Micro F1 from
0.2765 to 0.2728. Requiring an entity to appear in at least two era candidates
is a better balanced post-processing option: award Macro F1 0.1266, award Micro
F1 0.0990, overall Macro F1 0.5055, and overall Micro F1 0.2834. This variant
uses the same generated candidates and requires no additional inference.

## Reverse chronological conversation follow-up

Issue #12 tests whether award recipients are easier to retrieve when the model
can associate adjacent periods in one conversation. It keeps reasoning off and
uses the same total of 20 generations, split into two independent chains of ten
turns. Each chain starts with 2020--2026 and moves backward to before 1900. A
turn keeps the previous assistant answers in context, asks only for its current
period, and tells the model not to repeat recipients already returned in that
chain.

The primary aggregation accepts a recipient when it appears in either chain's
answer for the same period (1-of-2), then unions the ten periods. The stricter
2-of-2 result can be computed from the same candidate artifact without further
inference. This is a package-level hypothesis test rather than a perfectly
isolated conversation ablation: the earlier experiment used 20 independent,
finer-grained periods, while this experiment uses ten periods repeated across
two conversational chains.

### Reverse-conversation result

Pipeline 51, smoke job 495, completed all ten award rows. The two chains used
no reasoning and generated 20 answers in total. Results from the generated
candidate artifact are:

| Award aggregation | Macro P | Macro R | Macro F1 | Micro F1 | Avg. predictions |
|---|---:|---:|---:|---:|---:|
| original non-thinking frequency | 0.3650 | 0.0786 | 0.1089 | 0.0680 | 13.1 |
| 20 independent periods, union | 0.2199 | 0.1748 | 0.1685 | 0.1427 | 112.0 |
| two reverse conversations, 1-of-2 | 0.2253 | 0.1722 | 0.1551 | 0.1412 | 85.1 |
| two reverse conversations, 2-of-2 | 0.4002 | 0.0834 | 0.1134 | 0.0761 | 19.7 |

The 1-of-2 conversation result improves award Macro F1 by 0.0462 over the
original baseline, but trails the independent-period union by 0.0134. It
improves seven of ten award rows over the baseline and four of ten over the
independent-period run. Turing Award is the clearest positive case, improving
from 0.6081 with independent periods to 0.6750 with conversation context.

Conversation context did not consistently produce new period-specific recall.
Of 2,038 entity occurrences emitted after the first turn in each chain, 1,242
(60.9%) had already appeared earlier in that chain despite the explicit
no-repeat instruction. This propagates earlier guesses into older periods. The
run has no parse failures, but 96/200 candidates (48%) reach the 128-token cap,
compared with 60/200 (30%) in the independent-period run.

Replacing the ten award rows in the completed 478-row baseline yields overall
Macro F1 0.5061 and Micro F1 0.2821 for conversation 1-of-2. The corresponding
independent-period values are 0.5064 and 0.2728, while the original baseline is
0.5052 and 0.2765. Thus conversation 1-of-2 gives the best overall Micro F1 of
these direct-union variants by returning fewer false positives, but does not
support a general claim that conversational backward retrieval improves award
Macro F1. The comparison remains package-level because period granularity also
changed from 20 independent periods to ten repeated periods.
