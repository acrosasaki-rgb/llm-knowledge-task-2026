# Think-content sampling for hasArea (#27)

## Question

The BF16 20-candidate pool answers *what* the model samples but not *why*.
This experiment regenerates a stratified 12-row hasArea sample with the full
raw output (thinking included) persisted, so correct and wrong candidates can
be compared at the reasoning level.

## Method

- `save_raw_text: true` (new `ModelConfig` option) stores the complete model
  output in each candidate's diagnostics for both backends.
- 12 rows stratified by the baseline pool: 2 near-all-correct rows, 5
  selection-failure rows (a correct candidate exists but aggregation loses),
  5 pool-miss rows (no correct candidate among 20).
  Subjects: `scripts/h100-bf16/think-sample-subjects.json`.
- Candidate seeds derive from `(seed, subject, relation, candidate_index)`,
  so the run reproduces the baseline pool's sampling for the selected rows.
  Everything else matches the BF16 20-candidate baseline. Runs offline on a
  rented GPU via `scripts/h100-bf16/run-think-sample-docker.sh`.

## Findings (240 candidates)

- Correct and wrong candidates share the same reasoning skeleton
  (analyze → retrieve → verify). The decisive event is the **first recalled
  anchor value**; verification loops never dislodge it.
- Hashima Island (gold 0.063): all 11 correct candidates recalled
  "6.3 hectares" and converted; wrong candidates recalled "0.63 km²"
  directly — the km² surface form is the rarest representation in sources
  (Wikipedia prose states sub-km² areas in ha/m²).
- Wrong candidates carry **more** confidence markers than correct ones:
  source mentions 6.9 vs 4.2 per candidate, "Wait" loops 9.5 vs 8.3.
  Citations are decorative: different candidates attribute different numbers
  to the same "Wikipedia infobox" (Corfu: 593.3 vs 573.55), and wrong-anchor
  candidates fabricate consistent-looking conversion checks.
- Thinking-budget exhaustion (68% of hasArea candidates) is a symptom of
  oscillation between two remembered values (Lake Texcoco: 2,000 vs 2,200
  for ~2,000 tokens), not a cause of errors: within rows containing both
  truncated and finished candidates, their accuracy is identical (mean
  difference −0.006 over 93 rows), and aggregating only finished candidates
  scores worse (62 → 59).
- Referent/scope mismatches exist where the model knows the gold value but
  picks another interpretation (Basque Country: the Euskal Herria figure
  appears in the thinking, the autonomous-community figure is answered).

## Implication

Post-hoc judges that read the reasoning are systematically misled — wrong
candidates look more confident. Improvements must target the recall path
(elicitation), not selection; see #28/#29 for the follow-up prompts.
