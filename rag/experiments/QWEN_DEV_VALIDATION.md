# Frozen Qwen-4B validation on previously inspected dev

## Authorization and interpretation

On 2026-11-17 the user explicitly authorized setting up the environment and
launching fresh dev validation, superseding the earlier prohibition on new dev
generation/judging **for this frozen experiment only**.

The 125 dev questions were previously examined for old methods. This experiment
is therefore **frozen-configuration validation on previously inspected dev**,
not a newly collected or untouched independent held-out test. Do not tune models,
prompts, cutoffs or evaluation rules from these results.

## Fixed protocol

- All 125 dev IDs and saved ASR transcripts from `rag/experiments/dev125.jsonl`.
- Freeze the existing top-100 dense candidate lists from
  `rag/results/bm25-compare-dev/dense_rankings.jsonl`; no new ASR or query rewrite.
- Selector: `Qwen/Qwen3-Reranker-4B`, revision
  `22e683669bc0f0bd69640a1354a6d0aebcfeede5`, BF16, batch 8, SDPA.
- Only the already selected **matched** Qwen arm versus dense; do not introduce
  a new model or native-long-context comparison.
- Match the MiniLM-visible original-character document prefix at 512 pair
  tokens, use the same official Qwen instruction/template and yes-minus-no
  scoring, with stable dense tie ordering. Forbid further Qwen truncation.
- Pass full text of the selected five records to the unchanged EuroLLM
  generator/prompt, with temperature 0.1. No context truncation; the existing
  preflight ceiling is retained. Shuffle per-question arm order deterministically.
- Generate **fresh dense5 and qwen4_matched5 answers**, 125 per arm. Do not copy
  old dev/practice answers or judge caches into the new run.
- Same unchanged `evaluate_answers.py`, JSON mode, EuroLLM judge, one judgement
  per answer and existing public-dev qrel evidence reconstruction.
- Two query pairs first check nonempty/normal-stop generation and valid JSON
  judging. No threshold is imposed on smoke scores or retrieval gains.
  Successful smoke judgements are cached and reused in full evaluation.
- Planned logical totals: 250 generated answers and 250 judgements; bounded API
  retries may cause more HTTP requests. No repeated experimental runs or
  result-dependent early stopping.
- Recall@5/10/20/100, MRR@10, NDCG@10; paired answer scores and query-bootstrap
  95% intervals (2,000 resamples, seed 20261116). Preserve regressions.
- Fresh generation timing and tokens; rerank cost remains stage-attributed,
  not a measured audio-to-answer latency.

## Environment

Local host: fuji3; selected physical GPU: 1 (RTX A6000, idle at setup).

`~/mls-qwen-dev-env` is a new venv that reuses the unmodified baseline conda
environment's GPU packages via `--system-site-packages`:

- Python 3.11.18
- PyTorch 2.10.0+cu128
- Transformers 5.0.0
- NumPy 2.4.6
- OpenAI client 2.8.1 and python-dotenv 1.2.1 installed in the new venv

The existing `~/mls-rag-env` had newer GPU libraries; it was not downgraded.
Pinned Qwen weights were downloaded into `rag/models/qwen3-reranker-4b/` and
hashed into `experiment_manifest.json`. Inference runs offline after download.
ELM credentials are read from the existing ignored `rag/.env`; their values
are not printed, copied to snapshots or stored in the protocol.

## Launch

```bash
CUDA_VISIBLE_DEVICES=1 ~/mls-qwen-dev-env/bin/python -u \
  rag/extensions/run_qwen_dev_validation.py \
  --run rag/results/qwen4-dev-validation-20261117-v1 \
  --authorize-dev-api-calls \
  > rag/results/qwen4-dev-validation-20261117-v1.driver.log 2>&1
```

The orchestration runs sequentially:

1. Retrieval implementation smoke (2 dev queries).
2. Full 125-query matched reranking.
3. Offline retrieval integrity/metric audit.
4. Frozen fresh answer plans.
5. Two-query-per-arm generation and JSON judge smoke.
6. Remaining generation and judging, reusing only this new run's smoke cache.
7. Paired summary and independent read-back of prompts, caches, raw scores,
   query coverage and bootstrap intervals.

## Status, safety and resume

Run-local `protocol.json` freezes inputs, model files, source hashes and relevant
package versions before inference/API calls. Frozen input copies and source
snapshots are retained. `status.json` records PID, stage, timestamps and completed
stages; a run lock prevents concurrent launch into the same directory.

Final artifacts:

- `retrieval/summary.json`, `retrieval/offline-audit.json`
- `answer-evaluation/comparison.json`, `answer-evaluation/offline-audit.json`
- Full scores, rankings, new answers, judge cache/raw outputs and frozen plans

If a generation/judge stage fails, preserve all files. `--resume` requires the
identical protocol/source/environment and skips completed stages; answers/cache
are resumed strictly. A partial retrieval directory is not silently overwritten
or restarted, because the retrieval engine does not implement incremental
resume; inspect it before choosing a fresh output directory. Do not change
methods to make a validation score improve.

The report is not updated automatically. Only successful full coverage and both
offline audits justify labelling the run complete; failure/status messages are
not validation results.

## Completed result (2026-11-17)

Run `qwen4-dev-validation-20261117-v1` completed all stages, including both
offline audits, from 14:05:32 to 14:47:46 UTC. No result-dependent rerun or
configuration change was made.

### Retrieval: improvement replicated on dev

| Arm | Recall@5 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|
| Dense | 0.379497 | 0.459708 | 0.398342 |
| Qwen 4B matched | 0.456804 | 0.543873 | 0.482064 |

Paired differences with unadjusted 95% query-bootstrap intervals:

- Recall@5: **+0.077307 [0.020034, 0.133852]**;
  30 queries improve, 11 worsen, 84 tie.
- MRR@10: **+0.084165 [0.023992, 0.143974]**.
- NDCG@10: **+0.083721 [0.039617, 0.125742]**.

All preserve candidate Recall@100 at 0.724426. Median/mean Qwen reranking time
is 5753.85/5620.75 ms per query.

### Fresh paired answers: positive mean, interval includes zero

| Arm | Answer score | Mean server input tokens | Median generation ms | Mean attributed selection + generation s |
|---|---:|---:|---:|---:|
| Fresh dev dense5 | 0.313333 | 1209.048 | 4621.71 | 4.572712 |
| Qwen 4B matched5 | 0.370000 | 1245.160 | 4988.40 | 10.263044 |

The mean answer-score difference is **+0.056667**, with 95% paired interval
**[-0.000667, 0.112667]**: 40 improve, 30 worsen and 55 tie.
This interval **includes zero**. Its positive mean is directionally consistent
with practice, but this dev run does not exclude no mean answer-score improvement.
Do not round the small negative lower endpoint away or call the result significant.

Correctness/grounding/completeness differences on the 0–4 scale are respectively:

- +0.224, interval [-0.008, 0.448].
- +0.224, interval [-0.032, 0.472].
- +0.232, interval [0.000, 0.464].

These also do not exclude zero. Use this run's fresh dense answer score, not the
old dev dense score or the fresh practice dense score.

### Integrity and interpretation

Each arm has all 125 fresh answers and all 125 judgements. Retrieval candidates,
source prefixes, stable ordering and all metric means passed offline checks.
Answer prompts, selected records, judge cache keys, raw scores, paired means and
intervals passed the independent read-back. A post-completion check confirmed
that source/input hashes stayed frozen and raw means/medians match the summary.

The supported conclusion is: **retrieval gains replicate on the inspected dev
set; downstream mean scores move in the same direction, but answer-quality
evidence remains uncertain at the stated interval level.** This is still not
untouched independent validation. Do not run repeated generations/judges merely
to obtain a positive interval. The additional attributed stage cost remains
substantial, and is not measured audio-to-answer latency.

Results are in the run's `retrieval/summary.json`,
`answer-evaluation/comparison.json` and their `offline-audit.json` files.
The report is not automatically rewritten from these results.
