# Diagnostic 01: audit evidence before choosing an intervention

Date: 2026-11-12. Status: offline audit and expanded retrieval completed. See [Diagnostic 02 results](diagnostic-02.md). No new model inference or ELM requests in the completed offline stage; Diagnostic 02 re-ran CPU query encoding only.

## Purpose

Determine whether the existing examples support investigating evidence selection, and whether their judge explanations can be trusted. Do not infer population failure rates from the first five manifest queries. Do not choose an intervention based only on their scores.

## Completed procedure

Ran [audit_baseline.py](../extensions/audit_baseline.py) against the original course asset archive and frozen job 3713067 outputs:

```bash
python rag/extensions/audit_baseline.py --output rag/results/diagnostic-01
```

The script reads practice corpus/qrels/manifest, reconstructs the current baseline prompt for each saved answer, attaches the complete retrieved and reference texts, and records SHA-256 hashes. All five saved generation transcripts exactly match the retrieval transcript cache. Prompt reconstruction describes the current source; it is not a captured API request. No original audio was audited, so no ASR accuracy claim follows.

Outputs: [complete evidence review](../results/diagnostic-01/evidence_review.md), [machine-readable packets](../results/diagnostic-01/evidence_packets.jsonl), [summary and provenance](../results/diagnostic-01/summary.json), and [fixed additional sample](../results/diagnostic-01/next_practice_sample.jsonl).

## Case findings: qualitative assistant audit, not independent human ground truth

1. **Sales tax — q_7e9a0f2716c017d2.** One of two positive qrels appears at rank 4 (Recall@5 = 0.5). Retrieved rank 1 addresses a business owner and single-owner LLC; the answer applies this framing to a question about retail purchases. The judge says Maryland and single-owner LLC conditions are in the question, but neither occurs in the saved transcript. Maryland appears in a reference document. The text supports scope conflation in the answer and erroneous question attribution in the judge explanation. It does not establish a dataset-wide mechanism or a corrected numeric score.

2. **Opening stock price — q_ea0be00eda6ea088.** The single positive qrel is at rank 5 (Recall@5 = 1). The candidate includes a numbered Dutch Auction point. The judge says that the candidate does not mention Dutch auction: a directly checkable contradiction. The transcript is broad, while the positive qrel focuses on IPO opening. Other retrieved documents discuss regular trading-day opening. This also warrants checking question/reference scope, rather than labelling every unjudged record irrelevant.

3. **Credit-card fund availability — q_9cd791a970ab648f.** No positive qrel occurs in top 5 (Recall@5 = 0). The reference discusses depositing credit-card checks and bank hold policy. Retrieved texts concern statement cycles, credit availability, cash advances and payment timing; the answer describes billing cycles as the availability timeline. There is a topic/transaction mismatch relative to the reference. The saved transcript does not explicitly mention checks, so query ambiguity must remain a competing explanation. We cannot yet tell whether the positive reference is inside top 100.

4. **Negative P/E — q_20e98c45013f6cc6.** The single positive qrel is at rank 2 (Recall@5 = 1). The judge says reference evidence does not explicitly address bankruptcy risk. The reference explicitly states that negative P/E may or may not indicate imminent bankruptcy and describes balance-sheet factors. Thus this particular objection is contradicted by the supplied reference. Other incompleteness claims require separate judgement; we do not replace the score with an assumed perfect score.

5. **Adding a vehicle — q_e3f1cef46917a2a5.** One of three positive qrels occurs at rank 1 (Recall@5 = 1/3). That document describes a 30-day notification window at a different insurer, not a verified Geico-specific rule. The answer qualifies the absence of Geico-specific automatic-coverage details. The judge criticises a candidate mention of a small service charge, but that phrase occurs in a reference document and is absent from the answer. Its additional criticism about omission of the 30-day window requires attention to the different-insurer qualification.

These are text/evidence comparisons, not current financial or insurance advice. No external factual correctness claims were investigated.

## Consequences for the experiment

- Both retrieval misses and judge-explanation errors are present in the existing examples. A single explanation such as "low recall causes all low scores" is already inadequate.
- Four of five explanations have at least one concrete attribution/contradiction issue identified above. This describes this selected subset, not a judge error rate, and does not imply that all four numeric scores are wrong.
- Recall@5 = 1 on two examples does not certify sufficient context or a fully correct answer.
- Missing qrels are useful leads; unjudged records must not automatically be classified as irrelevant or harmful.
- No causal intervention has run yet. We have not demonstrated a reranking benefit, position effect, or quality/cost improvement.

## Fixed next diagnostic sample

Thirty practice query IDs were selected before inspecting their retrieval or answer outcomes. Exclude the five previously inspected IDs, sort the other IDs by SHA256(seed + ':' + query_id), and take the first 30. Seed: `2026-11-12-evidence-audit-v1`. The sample is exploratory; it is not a confirmatory test set. Keep dev untouched while selecting configurations.

Review the 30 cases with overlapping labels: missing positive candidate, selection miss, insufficient evidence, scope/condition mismatch, ambiguity in saved query, generation misuse, judge discrepancy, and undetermined. Keep supporting passages for each label. Do not force all cases into one cause. These 30 currently have frozen transcripts only; answers have not been generated or evaluated.

## Next executable experiment: recover per-query retrieval evidence

The old baseline saved aggregate metrics and transcripts, but did not preserve full top-100 rankings. [collect_retrieval_diagnostics.py](../extensions/collect_retrieval_diagnostics.py) therefore re-encodes all 350 cached practice transcripts on CPU and saves top-100 IDs, scores, positive-qrel ranks, metrics and diagnostic timings. It imports no ASR worker and makes no ELM requests. It saves full-text top-five/reference packets for the five old and 30 new diagnostic cases. It compares ordered top-five IDs against the five saved answers to expose reproduction drift. If they differ, investigate before claiming a matched baseline comparison.

Run in the established ICF RAG environment using a scheduler CPU allocation. Choose a fresh output directory:

```bash
source "$HOME/mls-rag-env/bin/activate"
cd "$HOME/edin-mls-26-autumn"
srun -p Teaching --cpus-per-task=2 --mem=8G --time=00:30:00 \
  python -u rag/extensions/collect_retrieval_diagnostics.py \
  --transcript-cache rag/results/icf-3713067/practice_transcripts.jsonl \
  --sample rag/results/diagnostic-01/next_practice_sample.jsonl \
  --saved-answers rag/results/icf-3713067/sample_answers.jsonl \
  --output rag/results/diagnostic-02-retrieval
```

Local diagnostic environment has NumPy but no PyTorch or sentence-transformers. ICF shared SSH access was restored. Uploads temporarily blocked on Lustre I/O, then completed. Remote SHA-256 hashes matched local files for the collector (`1a8578ac927de3c68b3759f1fecdf05243d7da41c6df72713d4c05bb21e83f15`), fixed sample (`edab085f2e45771cd1ad67890d90fd55cb50067f56cb255f7093666a5327b3fb`) and batch script (`10164f4a55ac5967730ae4364449261e029b6ec68df754224dbc2708a1529e67`).

Submitted [icf_retrieval_diagnostics.sbatch](../scripts/icf_retrieval_diagnostics.sbatch) at 01:13:22 on 2026-11-12 as job **3713758**, Teaching partition, 2 CPUs, 8 GB host memory, 30-minute limit, no GPU. It completed successfully (exit 0:0) on `opencast`, with elapsed time 00:07:29 including slow shared-storage loading. [Summary](../results/diagnostic-retrieval-3713758/summary.json), [350 unique ranking rows](../results/diagnostic-retrieval-3713758/rankings.jsonl) and the [completion log](../results/retrieval-diagnostic-3713758.log) were downloaded and validated. All four result-file hashes match remote originals; 35 diagnostic packets cover the existing five and fixed next 30. Six aggregate metrics reproduce baseline to the six decimals originally reported, and all five saved ordered top-five lists match.

## Main comparisons after diagnosis

Keep original dense top-five; add a limited larger-context control; compare a chosen reranker on a fixed larger candidate pool; then match evidence-token budgets using a documented tokenizer and explicit overflow/truncation policy. Keep the question, frozen transcripts, generator, prompt framework and evaluation settings matched. Report reranker truncation, selected IDs, actual prompt lengths, input/output usage when returned, reranking latency, generation latency and total downstream latency. Candidate-collection timings from the one-pass diagnostic are not steady-state benchmark results.

Use qrel-based ideal selection only as a labelled offline diagnostic, never as a deployable method. Select extra order/condition perturbations only if observed failures justify them. Freeze method selection before dev evaluation. Preserve original judge scores alongside any subsequent audit; repeated judging does not by itself remove systematic bias.
