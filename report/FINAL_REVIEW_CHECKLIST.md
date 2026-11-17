# Final review: automated checks versus author sign-off

## This revision

- Adds separate practice/dev quality-cost panels from **fresh paired** answer
  experiments; no old dense score or pooled split comparison is substituted.
- Reports practice +13.2% mean judge score versus +126.6% **attributed** stage cost
  (+6.022 seconds), and dev means +18.1% versus +124.4% cost. Dev's answer interval
  includes zero; none of these are accuracy gains, measured audio-to-answer
  latency, or a utility ratio.
- Discusses interactive voice versus offline QA as conditional use cases, not
  measured user preferences or a universal accept/reject threshold.
- Does not put Qwen 0.6B on an answer-quality frontier: its downstream answers
  have not been evaluated.
- Incorporates the explicitly authorized dev run completed on 2026-11-17,
  with no score-based rerun/retuning. Updating the report makes no new calls.

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

Before the dev integration, 25 tracked RAG tests and 4 plot-arithmetic tests
passed, with a six-page/PASS PDF. The new revision adds dev-split, input-integrity
and negative-interval endpoint tests and reruns the checks. None of these
automated outcomes marks any author-sign-off checkbox below as complete.

Latest dev-integration checks: **31 tracked RAG tests + 5 plot tests passed**;
both dev offline audits were rerun without model/API calls. PDF checker PASS,
main body six pages, no `[?]`, all 58 evidence paths tracked. Fresh split-specific
table values and the preserved initial table were matched against JSON; the
figure, tables and dev uncertainty wording were visually inspected by the agent.
For author review: fresh results Table 1 on page 3, dev result paragraph on
page 4, practice/dev figure on page 5, Discussion on page 6.

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

A frozen configuration has now been checked on the old inspected dev set.
It is not genuinely new held-out data: retrieval improves, but the downstream
answer interval is [-0.0007, 0.1127]. No rerun was used to seek significance.
For a genuinely new independent test, still:

1. Identify new, unused queries not involved in selecting Qwen, rewriting rules,
   or examining failures. The old inspected dev set is not a fresh untouched set.
2. Freeze Qwen-4B matched, dense5, generator/prompt, metrics and evaluation plan
   before seeing new results.
3. Confirm data access, qrels/reference evidence, GPU and API/judge budget.
4. Evaluate all chosen queries, retaining regressions, with no score-driven tuning.

If that data/resource access is unavailable, retain the explicit inspected-dev
qualification rather than relabeling old data as independent validation.
