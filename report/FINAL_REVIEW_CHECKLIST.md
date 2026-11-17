# Final review: automated checks versus author sign-off

## This revision

- Reframes the discussion around controlled explanations, evidence-use versus
  measurement limitations, and an explicit stopping boundary rather than further
  configuration search.
- Adds an exploratory, unblinded AI-assisted sample of 20/121 practice Qwen
  judge-zero cases. Six judge-text conflicts are flags, not corrected scores.
  Scope blending motivates a hypothesis; no snippet intervention is claimed.
- Adds a reproducible vector system architecture diagram (Figure 1), showing
  GPU/CPU/remote components, cached RAG inputs and offline-only evaluation.
- Moves all four selected ASR answers unchanged into Appendix A, linked from
  the introduction; core ASR results remain in the six-page main body.
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

Previous dev-integration checks: **31 tracked RAG tests + 5 plot tests passed**;
both dev offline audits were rerun without model/API calls. PDF checker PASS,
main body six pages, no `[?]`, all 58 evidence paths tracked. Fresh split-specific
table values and the preserved initial table were matched against JSON; the
figure, tables and dev uncertainty wording were visually inspected by the agent.
Previous architecture revision: **31 tracked RAG tests + 11 report tests passed**
(5 plot arithmetic and 6 architecture tests). Compilation and PDF checker PASS:
six-page main body, twelve pages total, no `[?]`. All 62 evidence paths exist
and are staged/tracked; citation keys resolve including the ASR input file.
Diagram implementation/source/PDF/PNG hashes match. The selected-answer input
exactly matches the former main-body block. Official style is unchanged and
review mode is retained. No models, generation or judging were run for this update.

Architecture-revision page pointers (superseded below): Figure 1 on page 2, fresh results Table 1 on
page 3, dev result paragraph on page 4, quality-cost Figure 2 and Discussion
start on page 5, References on page 6, all four selected answers in Appendix A
on page 7. Appendix starts on a fresh page. The agent visually inspected the
title/architecture, fresh table, quality-cost figure and selected-answer layout;
this is not independent author sign-off.

### Latest error-analysis integration

- **35 RAG tests** passed (31 previously tracked + 4 offline diagnostic tests);
  **18 report tests** passed (including 4 diagnostic-evidence checks and 3
  supplied-SVG integration checks).
- Diagnostic generation is deterministic across repeated offline runs; input
  and script hashes match. Verbatim quotations are checked against named sources.
  This verifies provenance, not the correctness of AI interpretations.
- Compilation and course PDF checker PASS: six-page main body, fourteen pages
  total, References start on page 6 and continue on page 7. No `[?]`;
  all 69 evidence paths exist and are included in this revision, including the
  new diagnostic artifacts. Diagnostic results and SVG conversion provenance are
  explicitly included despite broad result-directory ignore rules.
- Official style, review mode and four selected ASR answers are unchanged.
  The dev answer interval remains `[-0.0007, 0.1127]`. Saved answers/judgements
  were not modified. No inference, generation or judging was run.
- Current author-review locations: architecture Figure 1 page 2, fresh answer
  Table 1 page 3, quality-cost Figure 2 and offline diagnostic page 5, Discussion
  page 6, ASR selected answers Appendix A page 8, error analysis Appendix H pages
  12–13, AI declaration/group details page 14.
- The agent visually inspected the diagnostic/discussion and Appendix H layout.
  Authors still need to check the three proposed follow-up cases and six judge
  conflict flags. No author-review checkbox is completed by these automated checks.
- Pulled `07fdab9` with the group's latest no-title architecture SVG and replaced Figure 1 with
  a vector PDF converted from it. The supplied SVG is byte-for-byte unchanged;
  all 12 arrowheads survive conversion, no raster image is embedded, and repeated
  conversions produce identical artifacts in the recorded converter environment.
  `svg-summary.json` records source/converter/output hashes and versions.
  The rendered page 2 was visually inspected; body remains six pages/PASS.

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
