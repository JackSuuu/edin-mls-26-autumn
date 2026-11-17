# Qwen integration into the report (2026-11-17)

## Scope

This revision changes the report, not experimental methods or scores. No model
inference, answer generation, judge calls or dev reruns were performed.
The existing Qwen artifacts were pulled at `85a7f10`.

The report now follows:

1. Dense candidates contain relevant evidence beyond the initial cutoff.
2. Initial five-arm practice/dev comparisons do not detect an answer-score gain;
   MiniLM candidate-pool and L12 controls do not solve its selection problem.
3. Matched-visible-text Qwen retrieval improves early ranking.
4. A fresh within-run answer comparison observes a judge-score gain at similar
   average input length.
5. Reranking adds substantial cost; attributed stage costs are not a measured
   audio-to-answer latency.

Qwen remains a practice-only exploratory extension. Old dev results do not
validate it. Checkpoint changes do not isolate parameter count or establish a
single failure mechanism. No new experiment or optional ASR Try This was added.

## Comparators and ranking-cache boundary

The initial and Qwen phases use different frozen ranking artifacts:

- Initial: `rag/results/selection-rankings-practice-full/rankings.jsonl`.
- Qwen: `rag/results/diagnostic-retrieval-3713758/rankings.jsonl`.

Direct inspection finds identical query IDs/transcripts on all 350 queries,
but full candidate orders differ on 268, candidate sets on 18, and ordered
top-five lists on 3. Thus equal aggregate dense recall is not proof of identical
cross-phase inputs. Qwen's own dense/MiniLM and model-size controls share its
frozen inputs; the report does not pair Qwen answers with initial answers.

Use the fresh Qwen-answer-run dense score **0.337381**, never the initial
**0.326667**, against Qwen's **0.381905**.

## Evidence mapping

- Initial result table and paired appendix:
  `rag/results/selection-run-full/comparison.json`,
  `rag/results/dev-validation/comparison.json`.
- Qwen retrieval table:
  `rag/results/qwen06-retrieval-3724137/full/summary.json`,
  `rag/results/qwen4-retrieval-3724174/full/summary.json`.
  MiniLM timing in the table is the 4B run's replayed batch-8 timing, not the
  older batch-32 L6/L12 comparison.
- Size and longer-input contrasts:
  `rag/results/qwen4-retrieval-3724174/paired-size-comparison.json`.
- Fresh answer/cost table and paired score contrast:
  `rag/results/qwen4-answers-3724290/comparison.json`.
  Generation uses **medians**, attributed selection-plus-generation uses
  **means**; these are labelled separately. Full original selected records are
  used by the generator, while prefix matching applies to reranker scoring only.
- Remaining zero-score counts:
  `rag/results/qwen4-answers-3724290/answer-evaluation-audit.json`.
- Protocols:
  `rag/experiments/qwen06-retrieval.md`,
  `rag/experiments/qwen4-answer-evaluation.md`.

## Validation

- Reran both Qwen retrieval **offline audits** against all saved rankings and
  qrels: passed, including stable score ordering, candidates and all six metrics.
- Reran the answer **offline audit**: passed, including frozen prompts, candidate
  selection, cache keys, raw judge responses, normalised means, paired intervals
  and change counts. No calls to generation/judge services.
- All **25 tracked repository tests** pass. Five additional local, uncommitted
  query-normalization tests also pass; they are not part of this revision.
- The four ASR selected questions remain verbatim and their measurements are
  unchanged.
- All cited `\path{...}` evidence files exist and are tracked.
- Tectonic compilation and course PDF checker pass; the main body is six pages.
  The official style, geometry, review mode and base font settings are unchanged.
