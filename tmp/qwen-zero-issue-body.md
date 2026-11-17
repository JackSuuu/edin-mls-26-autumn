## Problem and target

Qwen-4B reranking improves retrieval and downstream answer scores, but the retained 350-query practice run still contains **121 all-zero answers (34.57%)**. We need to diagnose and eliminate avoidable zero-score failures while preserving the complete benchmark.

An all-zero answer receives 0 for correctness, grounding and completeness from the existing public judge. We must establish whether each failure comes from the system, insufficient reference evidence, or the judge before choosing a fix. Zero remaining failures is the target; the current evidence does not establish that every zero is avoidable.

## Frozen baseline

See the [completed answer report](https://github.com/JackSuuu/edin-mls-26-autumn/blob/main/rag/experiments/qwen4-answer-evaluation.md), [paired comparison](https://github.com/JackSuuu/edin-mls-26-autumn/blob/main/rag/results/qwen4-answers-3724290/comparison.json), and [offline audit](https://github.com/JackSuuu/edin-mls-26-autumn/blob/main/rag/results/qwen4-answers-3724290/answer-evaluation-audit.json). Retained job: `3724290`; 350 answers and 350 judgements per arm.

- Dense5 to Qwen-4B top-5: normalised answer score **0.337381 to 0.381905**, paired difference **+0.044524**, query-bootstrap 95% interval **[+0.011190, +0.078810]**.
- Correctness: **1.425714 to 1.585714**; grounding: **1.448571 to 1.648571**; completeness: **1.174286 to 1.348571**, each on a 0-to-4 scale.
- All-zero answers: **138 to 121**. There are **53 zero-to-positive** and **36 positive-to-zero** transitions, so regressions also need inspection.
- Of the 121 Qwen zeros, **62** have no positive qrel in the selected five records, **29** have partial positive-qrel coverage, and **30** have full positive-qrel coverage. These labels describe document coverage, not whether the selected text contains a sufficient answer.
- The audit verifies all 700 answers, prompt hashes, record selections, judge cache keys, raw judgements and aggregate calculations. One judge explanation (`q_9a684db4ff009865`) reverses candidate and reference content; that establishes an explanation error, not a replacement score or a population error rate.
- The gain is modest in absolute score and has a cost: attributed selection-plus-generation time rises from **4.757 to 10.779 seconds/query**. These are combined measurements from separate stages, not measured audio-to-answer latency.

## Investigation and experiments

- [ ] Preserve an inventory of all 121 Qwen zeros, including query ID, transcript, selected records, gold records, candidate answer, raw judge explanation and paired dense5 outcome. The [existing frozen diagnostic data](https://github.com/JackSuuu/edin-mls-26-autumn/tree/main/rag/results/qwen4-answers-3724290/zero-review-v1) retains all 350 questions and provides a 24-question sample with 48 paired review packets. It is awaiting manual review.
- [ ] Review the fixed sample with method labels and judge scores hidden: six questions from each zero-coverage stratum and six positive controls. Record supported answer content, unsupported claims, missed qualifications and apparent judge errors. Separate the review key until the review is complete. This balanced sample cannot estimate population error rates.
- [ ] Extend the review to the remaining zero cases and the 36 positive-to-zero regressions. Allow multiple causes per question: ASR loss, retrieval miss, incomplete evidence, evidence-use failure, unsupported generation, and reference/judge problems. Keep uncertain cases explicit.
- [ ] Run a fixed-sample gold-context diagnostic with the same generator and judge. Supplying labelled relevant documents tests whether better supplied evidence helps; it is diagnostic only, uses gold information unavailable at inference, and is not a strict quality upper bound.
- [ ] Where transcription may change the answer, compare the ASR transcript against the available original question/audio and test a corrected-query control on a fixed subset. This connects ASR errors to retrieval and answer failures without assuming transcription is the main cause.
- [ ] Choose one fix supported by the reviewed failures and freeze its configuration before evaluation. Candidates include query-related evidence extraction, support-constrained answer generation, or retrieval/query changes if the corresponding failures justify them. Implement changes in the inference path without accessing qrels or judge labels at inference.
- [ ] Evaluate the selected change on all 350 practice questions using the unchanged public evaluator. Report all-zero counts and rates, zero-to-positive and positive-to-zero transitions, all three score dimensions, normalised mean score, paired confidence intervals, retrieval metrics, token usage, and measured stage latency. Keep development diagnostics separate from claims of generalisation.

## Acceptance criteria

- [ ] Every original zero has a recorded diagnosis or an explicit unresolved status with supporting evidence.
- [ ] Publish the attempted fix, configuration, raw outputs, paired analysis and cost. Demonstrate fewer than 121 zeros under the same full-set protocol before claiming a reduction, and report any score or latency regressions.
- [ ] Track residual avoidable failures until resolved. Claim elimination only if none remain under the stated evaluation protocol; document unanswerable/reference/judge limitations separately.
- [ ] Retain all 350 questions and original scores. Keep manual judgements and alternative-judge results separate from the official benchmark. Do not remove zero-score questions, replace their reference answers, or tune on withheld evaluation data.

Moving the mean dimensions toward 2 or 3 is a research target, not a guaranteed outcome. The first deliverable is evidence explaining the remaining zeros and a controlled test of a fix.
