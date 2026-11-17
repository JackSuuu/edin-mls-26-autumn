# Evidence selection pilot: frozen protocol

Status: completed. See [results and interpretation](selection-pilot-results.md). All 150 valid answers and 150 uniform JSON-mode judgements have matched coverage and verified hashes; default-text partial judgements and the invalid initial token-counting attempt are excluded.

Written 2026-11-12 before inspecting new generated answers. This is exploratory practice development, not a preregistered confirmatory study. The existing first five cases informed the question; the additional 30 were fixed in Diagnostic 01 before their retrieval/answer outcomes were inspected. Dev remains unused.

## Intervention and comparisons

All configurations reuse the same 30 transcripts and the same saved dense top-100 candidates from ICF job 3713758. Selection code does not load qrels. Preserve the original prompt text, ELM model and temperature 0.1.

- `dense5`: original rank order, five full documents; exact baseline prompt.
- `dense20`: original rank order, twenty full documents; larger-context control.
- `rerank5`: rerank 100 candidates and use five full documents.
- `dense_budget`: original rank order within 4096 local chat-input tokens.
- `rerank_budget`: reranked order within the same 4096-token cap.

Budget applies to the complete locally serialized chat input: fixed system message, question, document labels/text and assistant generation prefix. It is a cap, not a guarantee of equal actual server tokens or equal document count. Documents are admitted in rank order; the final overflowing document is cut to a feasible original-character prefix and selection stops. No snippets from later-ranked documents are packed into remaining space. Included/original character counts identify truncation. Do not present this as trained compression or optimal packing.

Use the official EuroLLM tokenizer. Count actual usage returned by ELM separately; inspect local/server discrepancies during the smoke run. Unbudgeted controls are preflight-checked against a 28672 local-token input ceiling; do not silently truncate them. No completion limit is added to the original request. Save non-stop finish reasons and stop for inspection rather than treat a truncated response as normal.

## Models and reproducibility

Reranker: [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. It is an existing English MS MARCO passage-ranking model, chosen as a small CPU-compatible starting point, not a new algorithm or a claim of optimality on this corpus. Use raw classification logits, descending order, stable original-rank tie breaking, batch size 16, max pair length 512, truncate document only. Record untruncated query/document pair lengths and truncation counts. Long-document truncation is a material limitation.

Tokenizer: [utter-project/EuroLLM-22B-Instruct-2512](https://huggingface.co/utter-project/EuroLLM-22B-Instruct-2512), revision `8c6ed5c8d18527d9daed994f645032b86c53f0bb`. Only tokenizer/config assets are downloaded, not generator weights. Save SHA-256 hashes of assets, source, frozen rankings, corpus and sample. The ELM server model revision is outside our control; retain response model IDs and timestamps/logs.

Warm up reranker once, exclude warmup/model loading from per-query inference times. Diagnostic untruncated-length counting is outside the measured reranking section. Context-build time includes token counting and truncation selection. Both reranked arms reuse identical scores, with the same observed reranking cost attributed to each hypothetical pipeline. These are single-pass exploratory latency observations; they do not establish steady-state throughput or a stable speedup. Cached retrieval/ASR are excluded from selection-plus-generation latency.

## Execution stages

Initial preparation/smoke job: **3713776**, Teaching, CPU-only, node `opencast`, submitted 2026-11-12. Source files matched local SHA-256 before submission. Local RAG unit suite: 8/8 pass; selection suite on compute node: 5/5 pass. Initial warm CPU reranking observations are about 19–28 seconds per 100-candidate query on this allocation; these are pilot measurements rather than a warmed repeated benchmark.

**Invalidated smoke attempt:** the initial counter took `len(apply_chat_template(...))`, but Transformers 5 returned a mapping (`input_ids`, `attention_mask`), so all counts incorrectly became 2. A stop was requested when this appeared in generated-output logs; accounting shows job 3713776 had already completed its 15-call smoke stage (exit 0:0, elapsed 00:20:44). Its prompts and generated answers must never enter method comparisons. The counter now extracts the IDs from mappings or list outputs; a regression test covers both forms and preparation calibrates that longer text produces more tokens. The RAG test suite now has 9 passing tests. Rebuild every prompt and answer in a fresh directory. The 30 completed reranker score lists do not depend on generator token counting; they can be reused only with matching input/model/settings hashes, retaining the original inference times and recording provenance.

Corrected smoke job: **3713795**, running on `saxa` CPU allocation (no GPU requested), reusing score lists computed on `opencast`. Quality comparisons use identical frozen scores/candidates and are unaffected by not recomputing scores. Cost provenance is mixed: original reranker times are from `opencast`, context-building from `saxa`, and generation uses remote ELM. The sum is an attributed cost estimate, not a measured single-node end-to-end latency. Require a later matched-node benchmark before a strong performance conclusion.

Corrected smoke completed: 150 valid context plans, 15 normal-stop completions. Verified all 30 `dense5` prompts exactly match the original builder, all budget plans respect 4096 local input tokens, and all 15 actual ELM prompt-token counts exactly match local counting. All five arms cover the same first three queries. Continuation/evaluation submitted as **3713799** against the same frozen plan, for the remaining 135 generations and 150 single judgements. No sample/method change was made after reading smoke answers.

Job 3713799 generated all 150 answers with normal `stop` completions; every actual ELM input-token count matches local counting and there are zero server budget overruns. It then failed during default-text judging: the fourth `dense5` query returned malformed JSON on all three attempts. Three valid judgements were cached, with no missing case replaced by zero. The previous judge output protocol is retained as the default. An optional `--json-mode` requests `response_format={"type":"json_object"}` while keeping the same question, rubric, model, temperature and strict score parser. Its cache namespace and summary metadata distinguish it from default-text evaluation. Raw returned content is logged for audit. Test the formerly failing query once, then evaluate all five arms uniformly into new `judge_json_*` directories if supported; do not mix the first three old-mode scores into the new comparison. This is a documented evaluation-protocol change, not a claim of unchanged original judging. Local RAG test suite: 12/12 pass, including default-request equivalence and strict parser checks.

Separate implementation check **3713800**, `opencast`, completed exit 0:0 in 00:01:36. The related Berlin passage scored above the unrelated passage, and 30 cached scores independently reproduced within maximum absolute difference 0.0000012. This checks scoring direction/data mapping; it does not establish retrieval quality on the coursework corpus.

Structured evaluation job **3713808**: the formerly failing query passed a one-call JSON-mode probe. All five arms are being evaluated uniformly with JSON-object output. Generation files are not modified. Judge scores remain provisional because targeted checks already find content errors: for DJIA, the explanation says reference evidence does not discuss long-term dividend effects, although the reference explicitly does; for credit-score use, it treats the anti-card opinion as the sole position despite other references recommending disciplined use; for CAGR, it claims the references establish an acronym rather than an initialism, overlooking a reference explicitly stating the latter. These are targeted assistant audits of supplied text, not representative human error-rate estimates or current financial advice. Do not silently replace their numeric scores.

1. Run tests, prepare all 150 prompts (30 queries times five arms), and save reranked candidates. No generation during preparation.
2. Generate all five configurations for the first three fixed queries (15 calls), with deterministic per-query shuffled arm order. Inspect budget accounting, finish reasons, grounding and API behavior.
3. Resume the same frozen plan for the remaining 27 queries (135 calls) if smoke checks pass. Preserve generated rows on failure; strict plan/source/endpoint fingerprints guard resume. An API call interrupted before its result is saved may need repetition; exactly-once billing is not guaranteed.
4. Evaluate each 30-answer arm with the unchanged public judge, one judgement per answer initially (150 calls). Label this as exploratory and audit major disagreements against question, actual input and reference evidence. Do not interpret tiny judge differences as clear wins; the earlier five cases already show explanation errors.

The first three fixed queries test ask-price order execution, DJIA divisor/dividends and credit-card use for credit score. Before generating answers, evidence review found: the ask-price query's sole positive qrel is outside top 100; the DJIA query already has a positive qrel at rank one but its second reference discusses scrip dividends; the credit-score query has zero positive qrels at top five despite unjudged documents explicitly advising small purchases, full repayment and avoiding high utilisation. These examples reinforce that qrel hits are not a complete judgement of usefulness. These are comparisons of supplied texts, not verification of current financial advice. Do not use these observations to change the fixed sample after seeing outcomes.

Record selected IDs/order, exact prompt and hash, local input length, server usage, output finish reason, per-stage elapsed time, and per-query answer scores. Analyse paired changes on identical queries, regressions as well as improvements, and qrel-based selection coverage. A truncated document ID hit does not certify that the relevant span survived.

## Implementation

[Evidence selection](../extensions/evidence_selection.py), [experiment runner](../extensions/run_selection_experiment.py), [asset preparation](../extensions/prepare_selection_assets.py), [pilot batch script](../scripts/icf_selection_pilot.sbatch), [tests](../tests/test_evidence_selection.py).

The original [ELM generator](../src/generator.py) retains identical request arguments and system text; it now also returns usage, response-model and finish-reason metadata. Original ASR, retrieval and baseline context-builder behavior is unchanged.
