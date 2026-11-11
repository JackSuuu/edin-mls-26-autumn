# RAG: answering audio queries with retrieved evidence

This part of the coursework connects your ASR system to a document corpus and
the University ELM service. For each audio query, the system transcribes the
recording, retrieves relevant records and generates an answer from that evidence.

You start with a working RAG baseline. Your investigation changes a component
and compares its quality and system cost with that baseline.

Before starting, follow [the ASR guide](../hw1-asr/README.md) to set up the
`mls-asr` environment, download the ASR model and check the PyTorch reference.
The setup steps below run on the head node. Audio inference runs on a GPU node.
Run commands from the **course repository root**, where `hw1-asr/` and `rag/`
are both visible.

## What is in this folder

| Path | What you use it for |
|---|---|
| `src/asr_adapter.py`, `src/asr_worker.py` | Call ASR in its own environment and reuse its loaded model |
| `src/encoder.py`, `src/retriever.py` | Encode the transcript and find relevant corpus records |
| `src/context_builder.py`, `src/generator.py` | Assemble evidence and request an ELM answer |
| `src/rag.py` | Run the complete audio-to-answer pipeline and record timings |
| `scripts/` | Install data, check setup, run queries and evaluate results |
| `data/rag_corpus/` | Course corpus, audio queries, qrels and embeddings |
| `models/enc_a/` | The supplied local query encoder |
| `tests/` | Tests for the ASR adapter and audio interface |
| `extensions/` | Optional home for investigation-specific code |

The shared [report resources](../report/README.md) live in the repository's
top-level `report/` folder.

## Step 1: create the RAG environment, once

Keep the ASR environment as supplied. RAG uses a separate environment so its
dependencies can be installed independently:

```bash
conda activate mls-asr
export RAG_ASR_PYTHON="$(python -c 'import sys; print(sys.executable)')"
python -m venv "$HOME/mls-rag-env"
source "$HOME/mls-rag-env/bin/activate"
python -m pip install -r rag/requirements.txt
cp rag/.env.example rag/.env
```

`RAG_ASR_PYTHON` points to the ASR environment's interpreter. RAG starts that
interpreter in a separate process; it does not reinstall the ASR dependencies.

Apply for your own ELM API key and add it to `rag/.env`:

```text
ELM_API_KEY=your_key_here
```

The file also provides the course endpoint and model settings. Leave those
settings at their supplied values for your first run. Keep your key local.

## Step 2: install the course data, once

The repository root contains [`rag-assets.zip`](../rag-assets.zip), stored with
**Git LFS**. The archive contains the corpus, query audio, public relevance
judgements, enc_a/enc_b corpus embeddings and the enc_a model.

Install [Git LFS](https://git-lfs.com/) on the machine where you cloned the
repository. From the repository root, download the archive contents:

```bash
git lfs install
git lfs pull --include="rag-assets.zip"
```

Run these commands for an existing checkout too. If Git LFS was already installed
before cloning, Git normally downloads the archive during the clone. If you
downloaded the source as a ZIP, use a Git clone with Git LFS for these steps.

Extract the archive, then install its contents into the RAG module:

```bash
mkdir -p "$HOME/mls-rag-assets"
unzip rag-assets.zip -d "$HOME/mls-rag-assets"
python rag/scripts/install_assets.py --from-dir "$HOME/mls-rag-assets"
```

After extraction, `mls-rag-assets/` contains `data/` and `models/`. A successful
installation reports `Course assets verified` and copies the files into `rag/`.
Running the same command again verifies existing files and skips matching ones.

You now have 24,000 corpus records, 350 practice audio queries and 125 dev audio
queries. Each split has public qrels that identify relevant records. The
embeddings directory includes enc_a and enc_b corpus vectors with separate ID
mappings. The first baseline uses enc_a for both queries and documents.

## Step 3: start a GPU session, each time

Get a GPU allocation using the instructions in the ASR guide. In that GPU shell,
restore the ASR interpreter path, activate RAG and build the ASR library:

```bash
conda activate mls-asr
export RAG_ASR_PYTHON="$(python -c 'import sys; print(sys.executable)')"
source "$HOME/mls-rag-env/bin/activate"
export PATH=/opt/cuda-12.8.0/bin:$PATH
cd ~/edin-mls-26-autumn
make -C hw1-asr lib
python rag/scripts/check_setup.py
```

Use your own checkout path if it differs from `~/edin-mls-26-autumn`.
The checker should report PASS for the corpus, audio manifest, embeddings,
enc_a model, ELM configuration and ASR source path. It checks setup without
sending ELM requests; the next step checks actual inference.

## Step 4: run one query

First check the complete path using the supplied PyTorch ASR reference:

```bash
export ASR_CUDA=none
python rag/scripts/run_audio_query.py --device cpu
```

This selects the first practice query from the manifest. The output includes
the query ID, ASR transcript, retrieved record IDs, evidence, answer and timings.
Read the evidence alongside the answer: are its claims supported?

To select a query yourself, use `--query-id` with an ID from the manifest, or
use `--audio` with a WAV path. `--split dev` selects from the dev split.

After your CUDA kernels pass the ASR tests, use them in the same RAG pipeline:

```bash
export ASR_CUDA=all
python rag/scripts/run_audio_query.py --device cpu
```

`ASR_CUDA` also accepts the task groups described in the ASR guide. The adapter
loads ASR once per command and reuses it for repeated queries. The example keeps
enc_a on CPU to leave GPU memory for ASR. Single-query and answer-generation
inference timings exclude ASR model loading.

## Step 5: evaluate retrieval

```bash
python rag/scripts/evaluate_rag.py --device cpu --split practice --transcript-cache rag/.cache/practice_transcripts.jsonl
```

The script prints per-query progress and averages Recall@5/10/20/100, MRR@10 and
NDCG@10. The transcript cache stores outputs produced by ASR. Reuse the same
transcripts to compare downstream variants, and record this choice in your
experiment. Use a new cache when comparing ASR implementations.

## Step 6: generate and evaluate answers

Begin with five queries so you can inspect the outputs before evaluating a
whole split:

```bash
python rag/scripts/generate_answers.py --device cpu --split practice --limit 5 --output rag/results/sample_answers.jsonl
python rag/scripts/evaluate_answers.py --split practice --candidates rag/results/sample_answers.jsonl --allow-subset --dry-run
python rag/scripts/evaluate_answers.py --split practice --candidates rag/results/sample_answers.jsonl --allow-subset --out-dir rag/results/sample_judge
```

The dry run checks candidate IDs and coverage without calling ELM. The final
command evaluates correctness, grounding and completeness, each on a 0–4 scale.
Its normalised score is their sum divided by 12. Per-query results and a summary
appear in the output directory. Generation and judging make ELM requests, so
plan how many queries and repeated judgements your experiment needs.

For a full-split run, remove generation `--limit` and evaluation `--allow-subset`,
and choose a new candidate filename and output directory. Set `ELM_JUDGE_MODEL`
in `.env` for a separate judge model, or use `--repeats` for repeated judging.

The public evaluator judges the ASR transcript as the question, using evidence
reconstructed from qrels and the corpus. An ASR error can change that question.
Report these downstream scores together with retrieval quality, query coverage,
judge settings and the system costs relevant to your investigation.

## Your investigation

Choose a precise systems question and compare a technical change with the
baseline under controlled conditions. For example, vary retrieval depth to
study evidence coverage, answer quality and context cost, or compare an ANN
index with exact retrieval, including search latency, memory and build time.
Explain the mechanism behind the result, including negative results.

Other directions include batching, reranking, context selection, dynamic updates
and representation migration. A language-model replacement alone is insufficient
as a systems investigation. You may change `src/` or use `extensions/` as needed.

The enc_b corpus vectors provide another representation for experiments. Its
query encoder is not bundled. Equal vector dimensions do not align the spaces;
define compatible query encoding or a justified mapping when using enc_b.

## When something goes wrong

| What you see | What to check |
|---|---|
| Corpus, embeddings or model files missing | Complete Step 2 using the course asset archive |
| `git: 'lfs' is not a git command` | Install Git LFS on that machine, then repeat Step 2 |
| `unzip` reports that `rag-assets.zip` is not a ZIP archive | It may still be a small LFS pointer file. Run `git lfs install` and `git lfs pull --include="rag-assets.zip"` from the repository root |
| ELM key missing | Add your own key to `rag/.env` |
| ELM authentication/model request error | Check the key, endpoint and selected model in `.env` |
| ASR source path missing | Use the complete course checkout, or set `RAG_ASR_ROOT` to its `hw1-asr/` directory |
| ASR worker cannot import torch/transformers | Set `RAG_ASR_PYTHON` to the interpreter inside `mls-asr` |
| No CUDA GPU or ASR library missing | Run on a GPU node and build the library there, following the ASR guide |
| Transcription fails with selected kernels | Run the corresponding ASR tests and compare with `ASR_CUDA=none` |
| Candidate file already exists | Choose a new output filename for the new experiment |
| Missing candidate queries | Finish the split, or label a diagnostic subset with `--allow-subset` |

## Report and tests

The recorded ICF baseline is documented in
[baseline-3713067.md](experiments/baseline-3713067.md), including configuration,
evaluation coverage, results and reproduction evidence. Preserve its transcript
cache and results when comparing alternative RAG methods.

Follow [the shared report guide](../report/README.md). The main body is limited
to six pages, excluding references and optional appendices. Keep essential
arguments and results in the main body; reviewers are not obliged to read
appendices. Use the supplied template and explain your methodology and results.

```bash
python -m unittest discover -s rag/tests -v
```

These tests check the adapter and audio protocol without a GPU. The ASR guide's
CUDA tests and real query runs check model inference on your GPU node.
