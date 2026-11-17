# Diagnostic 02: locating the evidence-selection opportunity

Date: 2026-11-12. Status: completed retrieval diagnostic, not a reranking or answer-generation experiment.

## Question and controls

How often does the original top-100 candidate pool contain positive-qrel evidence omitted by top five, and what candidate-pool ceiling limits reranking? Use all 350 practice queries with frozen ASR transcripts. Reuse enc_a, original exact dense retrieval, CPU execution, float16 encoder parameters and float32 vectors. No ASR inference, ELM generation/judging or dev evaluation was performed.

## Execution and reproducibility

ICF job **3713758**, Teaching partition, node `opencast.inf.ed.ac.uk`, 2 CPUs, 8 GB host memory, no GPU. Slurm reports COMPLETED, exit 0:0, elapsed 00:07:29. Log starts at 01:14:10 UTC and ends at 01:21:18 UTC. Shared-storage waits dominated initial loading; elapsed job time is not per-query latency. Batch MaxRSS was 806968K (host memory).

Code: [collector](../extensions/collect_retrieval_diagnostics.py), [batch script](../scripts/icf_retrieval_diagnostics.sbatch). Output: [summary](../results/diagnostic-retrieval-3713758/summary.json), [rankings](../results/diagnostic-retrieval-3713758/rankings.jsonl), [35 evidence packets](../results/diagnostic-retrieval-3713758/diagnostic_packets.jsonl), [configuration and input hashes](../results/diagnostic-retrieval-3713758/run_config.json), [job log](../results/retrieval-diagnostic-3713758.log), [Slurm accounting](../results/diagnostic-retrieval-3713758/slurm-accounting.txt).

Environment: Python 3.11.18, torch 2.10.0+cpu, NumPy 2.4.6, sentence-transformers 6.3.0; encoder parameters float16. Original core revision `c0b7a7fdc1dd7751dea7dfe0d8f0f88d0469a617`, with the collector supplied separately and hashed.

Validation passed: 350 unique query IDs, 100 unique ranked record IDs per query, 35 diagnostic packets, and all four downloaded result hashes matching remote. The six aggregate metrics reproduce the old baseline at its recorded six-decimal precision. All five saved answers have exactly matching ordered top-five retrieval results. We cannot prove identity of previously unsaved full rankings from aggregate agreement alone.

## Observations

- Macro Recall@5: **0.4213372501**.
- Macro Recall@10: **0.4795938982**.
- Macro Recall@20: **0.5785236034**.
- Macro Recall@100: **0.7698559060**.
- MRR@10: **0.4887460317**.
- NDCG@10: **0.4204605116**.

**197/350 queries (56.29%)** have at least one additional positive-qrel document at ranks 6–100. This includes **101/350 (28.86%)** with zero positive hits at top five but at least one at top 100. These are opportunities for selection improvements, not counts of queries whose answers will necessarily improve.

**42/350 (12.00%)** have no positive-qrel document in top 100. A reranker restricted to this pool cannot introduce a positive-qrel document for these queries. This does not prove the pool contains no useful unjudged evidence.

**95/350 (27.14%)** already retrieve all their positive-qrel documents in top five. Remaining **112/350 (32.00%)** retrieve some but not all at top five. These groups partition the dataset together with the 143 queries having no positive hit at five (101 recoverable within 100 plus 42 without a hit at 100).

An offline qrel-aware ideal selector choosing at most five documents from the top-100 pool has macro Recall@5 ceiling **0.7644166151**. For each query, count its positive documents in the pool, cap that count at five, divide by the query's total positive-qrel count, then average over 350 queries. This differs from Recall@100 because some queries have more than five positive documents. It is a reference ceiling for this metric and pool, using unavailable-at-inference labels; it is not a deployed method or an answer-quality ceiling. A token budget could constrain selection further.

## What the original cases reveal now

- Sales tax: positive-qrel documents at ranks **4 and 13**. More labelled evidence is available beyond top five; applicability still requires case inspection.
- Opening stock price: sole positive qrel at rank **5**. Its low judge score cannot be explained by failure to retrieve that positive document.
- Credit-card fund availability: sole positive qrel at rank **14**. This is a concrete candidate-selection opportunity; even a top-20 control would include it. Whether the generator then answers correctly remains untested.
- Negative P/E: sole positive qrel at rank **2**. The evidence/candidate/judge distinction from Diagnostic 01 remains essential.
- Vehicle insurance: positive documents at ranks **1 and 12**, with a third outside top 100. Candidate-limited reranking cannot retrieve all three.

## Interpretation and next comparison

The aggregate recall gap now has per-query evidence: a substantial subset has omitted labelled material inside the candidate pool. This justifies testing a reranker, while the 42 no-hit queries establish a clear pool limitation. It does not establish that ranking is the principal cause of poor final answers, or that a learned reranker beats increasing context.

Next, inspect the fixed 30-case practice packets and generate matched answers for that exploratory sample. Keep original top-five, add a modest larger-context control (top 20 is motivated by the depth curve and rank-14 case), and compare reranking from a fixed candidate pool. Fix token counting and overflow policy before the budget-matched runs. Include selection overhead and generation latency. Review answer support against actual input evidence as well as qrels references; preserve judge scores and separately record discrepancies.

Do not call this reranking success, a quality/cost improvement, or a representative answer-error measurement. The fixed 30 have not yet had answers generated. No further broad literature survey is required to complete this next test; consult targeted papers if the packet review exposes a specific unresolved mechanism.
