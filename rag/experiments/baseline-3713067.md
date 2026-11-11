# RAG baseline: ICF job 3713067

## Run configuration

We ran the baseline on 11 November 2026 at code revision `c0b7a7fdc1dd7751dea7dfe0d8f0f88d0469a617`. ICF allocated one NVIDIA RTX A6000 (sm_86), two CPUs and 24 GB host memory on landonia11 in the Teaching partition, with a two-hour limit. The job completed with exit code `0:0` in 45 minutes 18 seconds. This duration includes compilation, tests and initialisation; we did not measure steady-state throughput.

The [Slurm accounting record](../results/icf-3713067/slurm-accounting.txt) and [job log](../results/icf-3713067/icf-baseline-3713067.log) provide execution evidence. The [manifest](baseline-3713067-manifest.json) records SHA-256 hashes for the shared result files, resource archive and core source. Source hashes describe the local files when we assembled the record; the saved remote core-source diff is empty.

We used `zai-org/GLM-ASR-Nano-2512` at revision `61ba4e0b3309b6656edea3e93e419f7bd5c61957`. Retrieval evaluation and answer generation used `ASR_CUDA=all`. A separate check of the first audio query with `none` and `all` produced matching transcripts and ordered top-five document IDs. That comparison covers one query.

The ASR environment used Python 3.11, PyTorch 2.10.0+cu128 and Transformers 5.0.0. The RAG environment used PyTorch 2.10.0+cpu, Transformers 5.0.0, sentence-transformers 6.3.0, NumPy 2.4.6 and OpenAI SDK 3.49.0. Full dependency snapshots are in [asr-packages.txt](../results/icf-3713067/asr-packages.txt) and [rag-packages.txt](../results/icf-3713067/rag-packages.txt). We did not save separate operating-system, driver or nvcc version outputs.

We encoded queries one at a time with enc_a on CPU. The encoder returned normalised 384-dimensional vectors, which the application converted to NumPy float32. The log reports CPU float16 model weights; we retained that default. Exact retrieval used NumPy dot products against 24,000 normalised enc_a document vectors. We used no ANN index or reranker.

For answer generation we concatenated the top five complete records in rank order, without deduplication, summarisation or an explicit token budget. We called University ELM's `utter-project/EuroLLM-22B-Instruct-2512` at temperature 0.1. The [generator](../src/generator.py) and [context builder](../src/context_builder.py) contain the prompts at the recorded code revision. We did not record a server model revision, a generation seed or token usage, so future calls may produce different answers.

## Evaluation

The course data contains 24,000 records, 350 practice audio queries and 125 dev audio queries. We evaluated practice only. Its qrels contain 793 positive query-document pairs, all with relevance 1.0. The evaluator treats unjudged documents as relevance zero.

Retrieval evaluation covers all 350 practice queries. For each query we retrieved 100 records, calculated Recall@5/10/20/100, MRR@10 and NDCG@10, then took the arithmetic mean across queries. The definitions are in [metrics.py](../src/metrics.py). The [transcript cache](../results/icf-3713067/practice_transcripts.jsonl) preserves the ASR outputs for downstream comparisons.

We generated answers for the first five queries in manifest order, without random sampling. This command transcribed audio again in a separate ASR process; it did not read the retrieval transcript cache. The [candidate file](../results/icf-3713067/sample_answers.jsonl) stores these transcripts and answers.

The judge used the same EuroLLM model, one judgement per query, the candidate ASR transcript as the question and qrels-derived corpus evidence. The evaluator version was `v0.3-public-answer-judge`, with rubric `rag-answer-rubric-v0.1`. It assigns correctness, grounding and completeness scores from 0 to 4 and divides their sum by 12 for the normalised answer score.

## Results

The [retrieval log](../results/icf-3713067/retrieval-practice.log) reports:

```text
Recall@5    0.421337
Recall@10   0.479594
Recall@20   0.578524
Recall@100  0.769856
MRR@10      0.488746
NDCG@10     0.420461
```

The [answer summary](../results/icf-3713067/sample_judge/answer_judge_summary.json) reports:

```text
Mean correctness                 1.6 / 4
Mean grounding                   2.0 / 4
Mean completeness                1.2 / 4
Mean normalised answer score     0.4
Coverage                         5 / 350 (1.43%)
Judgements per query             1
```

The five query IDs are:

```text
q_7e9a0f2716c017d2
q_ea0be00eda6ea088
q_9cd791a970ab648f
q_20e98c45013f6cc6
q_e3f1cef46917a2a5
```

The [setup checks](../results/icf-3713067/setup.log) passed. RoPE, reductions, tiled GEMM and attention reported ALL PASS in the [CUDA tests](../results/icf-3713067/cuda-tests.log). These checks cover the supplied test cases; we did not measure ASR accuracy over the full dataset.

## Interpretation and limitations

Recall@100 exceeds Recall@5 by 0.348519. We can investigate whether reranking or context selection moves relevant candidates into the five records used for generation. We have not tested that hypothesis. Reranking a fixed set of 100 candidates cannot recover documents outside that set.

The answer score describes a five-query diagnostic subset. In the first [judge explanation](../results/icf-3713067/sample_judge/answer_judge_per_query.jsonl), the judge attributed Maryland and single-owner LLC conditions to a question whose saved transcript contained neither. The candidate answer also mixed business-owner tax information into a consumer question. We need wider answer evaluation and human checks of judge explanations before drawing conclusions about overall answer quality.

We did not measure warmed-up repeated timings, latency distributions, peak GPU memory, token costs or confidence intervals. Slurm batch MaxRSS of 7340748K describes host memory. The reference query ran before the CUDA query, so cold-start differences prevent a controlled speedup claim.

## Evaluation commands

Run from the repository root on allocated compute resources after installing the course assets and ASR model. Set `RAG_ASR_PYTHON` to the ASR interpreter and activate the separate RAG environment. Build and check the CUDA kernels with `make -C hw1-asr test`. The key evaluation commands were:

```bash
export ASR_CUDA=all
python -u rag/scripts/evaluate_rag.py --device cpu --split practice --transcript-cache rag/results/icf-3713067/practice_transcripts.jsonl
python -u rag/scripts/generate_answers.py --device cpu --split practice --limit 5 --output rag/results/icf-3713067/sample_answers.jsonl
python rag/scripts/evaluate_answers.py --split practice --candidates rag/results/icf-3713067/sample_answers.jsonl --allow-subset --dry-run
python -u rag/scripts/evaluate_answers.py --split practice --candidates rag/results/icf-3713067/sample_answers.jsonl --allow-subset --out-dir rag/results/icf-3713067/sample_judge
```

The recorded outputs exist. Use a new experiment directory for a fresh run; generation rejects existing candidate files. To compare downstream RAG methods, preserve this run and reuse its transcripts. Record candidate depth, final context size, prompts, encoder and judge settings for each variant. Save per-query rankings and timings, and measure the costs relevant to the investigation under matched resources. Reserve dev evaluation for an independent check after choosing the method.
