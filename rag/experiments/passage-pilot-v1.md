# Passage-selection pilot v1

Status: preparation retry submitted on 2026-11-17. Initial GPU job **3727830**
failed with exit 1 after 10m02s, before creating context outputs or making any ELM
calls: importing the generator solely for its prompt required `dotenv`, absent
from the GPU environment. The runner now reads the literal prompt through AST
without importing API-only dependencies; a regression test covers that path.
The five selection/runner tests pass locally. Replacement GPU job **3728068**
was submitted, and existing CPU job **3727835** was updated to depend on its
success rather than duplicated. The first retry check shows the GPU job waiting
for resources. Submission is not a completed result; no new answer-score result
is available yet. The sample, algorithms, budget and model settings are unchanged.

## Fixed question and controls

Does selecting original-text passages from the same five Qwen-ranked parent
records change answer quality or input cost relative to full records and a
simple prefix-budget control?

The [frozen protocol](passage-pilot-v1.json) reuses the 24 queries from the
previously fixed zero-review-v1 diagnostic sample. This score-conditioned
sample includes zero and positive controls; it is not representative validation.
No dev data is used. The run stops after these 24 queries, with no automatic
budget search, model search or full-split expansion.

- `full5`: original full text of the five frozen Qwen-4B matched parent records.
- `prefix768`: the same parent order, existing prefix packing, maximum 768
  complete-chat input tokens. The final admitted record can be truncated.
- `passages768`: sentence-window selection from those same five parent records,
  with the same 768-token complete-chat input cap.

Generate fresh answers in all three arms. Keep the EuroLLM generator, system/user
instruction and temperature 0.1 unchanged. Freeze a shuffled per-query arm order.
Use the unchanged public evaluator, JSON mode and one judgement per answer.
The first two queries per arm are smoke checks; their judge cache is reused.
There are 72 planned answers and 72 planned successful judgements, with ordinary
API retries possible. All of these are new pilot observations, not extra copies
added to the old experiment.

The 768-token cap is fixed before new generation/judging. Prior full-context
input lengths on this sample range from 533 to 2,500 tokens, median 1,219.5;
21/24 exceed the cap. This describes input size, not new answer quality.
Equal caps do not guarantee equal realised token counts.

## How text is cut and selected

1. Detect conservative sentence/paragraph boundaries, preserving original
   character offsets. Protect common abbreviations such as `Dr.` and `U.S.`;
   decimal points without following whitespace are not sentence boundaries.
   The rule is heuristic and has no downloaded language model.
2. Make windows of three consecutive sentences with one-sentence overlap:
   sentences 1-3, 3-5, 5-7, etc. The final window can be shorter. A document
   with at most three sentences becomes one window. Do not cut a sentence to
   force a window to fit.
3. Score each question/window pair using pinned Qwen3-Reranker-4B, BF16,
   batch size 8, the existing official relevance instruction and descending
   yes-minus-no logits. The complete scorer input cap is 2,048 Qwen tokens;
   reject oversize windows rather than silently truncate them. This scorer
   input limit is separate from the generator's 768-token context cap.
4. Consider windows by descending score with stable original-order ties. Add
   a window only if the complete rendered generator chat stays within budget;
   skip nonfitting windows and try the remaining windows.
5. Merge overlapping selected ranges from each parent. Keep parent order and
   original within-parent text order in the final context. Mark omitted gaps
   with `[...]`. Save source IDs, character ranges and all window scores.

Overlap retains neighbouring text and can help preserve a nearby condition;
it does not establish logical completeness or prevent jurisdiction/role errors.
The scorer estimates relevance, not whether each rule applies to the question.
The selector reads no qrels, gold text lists or judge results. The five parent
records are fixed, although budgeted contexts may include fewer parent IDs.
Selected-ID recall alone cannot measure whether their useful spans survive.

## Execution, checks and outputs

[Selection code](../extensions/passage_selection.py),
[runner](../extensions/run_passage_pilot.py),
[tests](../tests/test_passage_selection.py),
[GPU job](../scripts/icf_passage_prepare.sbatch),
[CPU job](../scripts/icf_passage_answers.sbatch).

Local checks passed: four unit tests for sentence offsets/abbreviations,
negation overlap, merge/deduplication, complete-prompt budgeting and nonfitting
window handling. A corpus-level preflight covers all 120 parent records,
producing 495 windows (7-46 per query), all exact original-text substrings.
Uploaded code and existing shared interfaces have matching local/remote hashes.

Remote run directory: `rag/results/passage-pilot-20261117-v1/`.
Logs: `rag/results/passage-prep-3727830.log` and
`rag/results/passage-answers-3727835.log`.

GPU preparation verifies model/tokenizer/input hashes, scoring direction and
batching, then saves window scores and immutable prompt plans. It releases the
GPU before the CPU job calls ELM. Generation checks nonempty answers, normal
stop completions and server/local token-count agreement; failures stop further
calls for inspection. Resume requires identical sources and frozen artifacts.

Report three paired contrasts: prefix minus full, passages minus full, and
passages minus prefix. Include dimension means, normalised score, all-zero
counts, paired wins/losses/ties and query-bootstrap intervals (2,000 resamples,
seed 20261117). Inspect preserved facts and qualifications alongside judge
scores. Measure actual input/output tokens, added passage-scoring cost and
generation latency. Attributed cached-parent-reranking plus new selection and
generation is not measured audio-to-answer latency.

The pilot has single stochastic generations/judgements and a score-conditioned
sample. Its results diagnose mechanisms and implementation behaviour; they
cannot establish a full-benchmark or held-out quality gain.
