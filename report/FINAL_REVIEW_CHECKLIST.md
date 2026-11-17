# Final review: automated checks versus author sign-off

## This revision

- Adds the two-point quality/cost figure from the **fresh paired** practice
  answer experiment; no earlier dense score is mixed in.
- Reports +13.2% mean judge score versus +126.6% **attributed** stage cost
  (+6.022 seconds), not measured audio-to-answer latency or a utility ratio.
- Discusses interactive voice versus offline QA as conditional use cases, not
  measured user preferences or a universal accept/reject threshold.
- Does not put Qwen 0.6B on an answer-quality frontier: its downstream answers
  have not been evaluated.
- Does not run any new inference, generation, judge or dev experiment.

## Automated checks (not a substitute for the authors)

- Recompute plotted means/relative changes from
  `rag/results/qwen4-answers-3724290/comparison.json`.
- Preserve the complete 0–1 score axis. Do not invent arm-level/timing error bars
  from a paired score-difference interval.
- Record source/figure hashes in `rag/results/report-quality-cost/summary.json`.
- Run plot arithmetic tests, tracked RAG tests, compilation and the course PDF
  checker; inspect the plotted figure and PDF table layout.
- Check citation-key resolution, `[?]`, the selected question quotations,
  evidence paths, official style integrity and the six-page body limit.

Latest outcome (2026-11-17): 25 tracked RAG tests and 4 new plot-arithmetic
tests passed. PDF checker PASS, main body six pages, no `[?]`; the figure,
tables and displayed question answers were visually inspected by the agent.
All six citation keys resolve and all 51 listed evidence paths are tracked.
These outcomes do not mark any author-sign-off checkbox below as complete.

## Required author review — still outstanding

- [ ] Both authors read the final PDF, including equations and table/figure
      captions, and accept the numerical interpretations and limitations.
- [ ] Confirm citations genuinely support the nearby statements and the
      bibliography accurately describes each source.
- [ ] Confirm Group ID, names and UUNs.
- [ ] Confirm the AI declaration covers all actual assistance. The current text
      names the OpenAI assistance documented in this record; independently
      confirm any earlier tools, including Cursor or ELM used for authoring,
      and add them if applicable. ELM generation/judging as an experimental
      component is separately described in the methodology.
- [ ] Check anonymous-submission packaging for identifying local paths in logs,
      metadata and supplemental artifacts. Do not assume a GitHub checkout is
      already an anonymous submission bundle.
- [ ] Review the two selected evidence sets and judge limitations; do not
      interpret the score as human-verified correctness or a held-out guarantee.

## Optional P2: genuinely new held-out validation

No such experiment is run in this revision. Before authorizing one:

1. Identify new, unused queries not involved in selecting Qwen, rewriting rules,
   or examining failures. The old inspected dev set is not a fresh untouched set.
2. Freeze Qwen-4B matched, dense5, generator/prompt, metrics and evaluation plan
   before seeing new results.
3. Confirm data access, qrels/reference evidence, GPU and API/judge budget.
4. Evaluate all chosen queries, retaining regressions, with no score-driven tuning.

If that data/resource access is unavailable, retain the explicit practice-only
exploratory limitation rather than relabeling old data as independent validation.
