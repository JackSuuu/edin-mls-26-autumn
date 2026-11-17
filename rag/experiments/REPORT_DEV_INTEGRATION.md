# Latest dev result integration for author review (2026-11-17)

## Read these first

- `report/latex_template/report.pdf`: updated report.
- `rag/experiments/QWEN_DEV_VALIDATION.md`: frozen protocol, environment,
  execution and completed result.
- `rag/results/qwen4-dev-validation-20261117-v1/protocol.json`: input/source
  hashes, authorization, fixed model and API settings.
- `rag/results/qwen4-dev-validation-20261117-v1/status.json`: all stages complete.

## Main result and exact source fields

Let `D = rag/results/qwen4-dev-validation-20261117-v1`.

| Quantity | Value | File / JSON field |
|---|---:|---|
| Dense Recall@5 | 0.3794974358974359 | `D/retrieval/summary.json`, `arms.dense.Recall@5` |
| Qwen Recall@5 | 0.4568043956043956 | same, `arms.qwen_matched.Recall@5` |
| Recall difference | 0.07730695970695971 | same, `paired_vs_dense.qwen_matched.Recall@5.mean_delta` |
| Recall 95% interval | [0.02003362637362637, 0.1338521978021978] | same contrast, `bootstrap_95pct_interval` |
| Fresh dense answer score | 0.31333333333333335 | `D/answer-evaluation/comparison.json`, `arms.dense5.mean_normalised_answer_score` |
| Qwen answer score | 0.3700000000000000 | same, `arms.qwen4_matched5.mean_normalised_answer_score` |
| Answer-score difference | 0.056666666666666664 | same, `paired_qwen4_minus_dense.normalised_answer_score.mean_delta` |
| Answer 95% interval | [-0.0006666666666666612, 0.11266666666666666] | same contrast, `query_bootstrap_95pct_interval` |

**Do not round the dev lower bound to zero and then claim significance.**
The displayed interval is **[-0.0007, 0.1127]**, including zero.
Forty answer scores improve, thirty worsen and fifty-five tie.

The interpretation is deliberately asymmetric:

- Retrieval improvement is supported on the inspected dev set.
- Mean answer-score improvement is directionally consistent with practice,
  but this dev interval does not exclude no improvement.
- This is frozen-config validation on previously inspected dev, not an
  untouched independent test.
- No configuration, evaluation rule or output was changed to obtain significance.

## Fresh controls, coverage and cost

Practice fresh Dense5/Qwen scores remain 0.337381/0.381905 with paired interval
[0.011190, 0.078810]. Use each split's own fresh dense comparator; do not use
the initial five-arm dense scores or pool the splits to conceal uncertainty.

Practice contributes 350 distinct questions; dev contributes 125 distinct
questions, with disjoint IDs. Each has two fresh answers and two judgements:
**475 questions, 950 answers and 950 judgements**. Smoke observations are reused,
not extra sample entries.

Main-table fields are:

- Score: `mean_normalised_answer_score`.
- Tokens: `mean_server_input_tokens`.
- Generation ms: `median_generation_ms`.
- Attributed seconds: `mean_attributed_selection_plus_generation_ms / 1000`.

Dev dense/Qwen values are tokens 1209.048/1245.16, generation medians
4621.710826/4988.398030 ms, and attributed means 4.572712/10.263044 s.
These combined costs are not measured audio-to-answer latency.

## Review layout

- Main fresh-result table now compares both splits, with distinct fresh controls.
- Initial five-arm table remains intact in the Appendix.
- Quality/cost figure has separate practice/dev panels, shared 0–1 score scales,
  and paired intervals. No individual-arm/timing error bars are invented.
- Discussion and conclusion distinguish replicated retrieval from uncertain
  dev answer gains. The AI declaration now includes the authorized dev run.
- `report/FINAL_REVIEW_CHECKLIST.md` remains the authors' sign-off checklist;
  automated checks do not claim that author review is already complete.

The dev runner/code, full frozen run artifacts and both offline audits accompany
the report for independent checking. No model weights, credentials, unrelated
blind-review keys or unrelated local experiments are included.

## Safe offline recheck (no inference or API calls)

With the course corpus/qrels installed and the documented environment:

```bash
PYTHONPATH=rag ~/mls-qwen-dev-env/bin/python - <<'PY'
from pathlib import Path
from extensions.run_qwen_dev_validation import audit_retrieval, offline_audit
r = Path("rag/results/qwen4-dev-validation-20261117-v1")
audit_retrieval(r / "retrieval", r / "inputs/dev_rankings.jsonl")
offline_audit(r / "answer-evaluation", r / "retrieval")
print("Both offline audits passed; no model/API calls")
PY
```

Do not run the orchestration command merely to inspect results: that command
requires an authorization flag precisely because it can make new API calls.
