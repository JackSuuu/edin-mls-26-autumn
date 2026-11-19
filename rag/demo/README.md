# Audio RAG browser demo

The terminal starts each request; the browser shows the latest request. The
server only reads the event file and binds to `127.0.0.1`. It does not start
model calls. The dashboard shows stage updates, not streaming answer tokens.

On the GPU host, from the repository root, start the server in terminal 1:

```bash
python3 rag/demo/server.py --events rag/results/video-demo/events.jsonl --port 18765
```

If the browser is on another machine, open a local terminal there and keep
this SSH tunnel running (substitute your SSH host alias if needed):

```bash
ssh -N -L 18765:127.0.0.1:18765 jacksu@fuji3
```

Open <http://127.0.0.1:18765>. In terminal 2 on the GPU host, run exactly one
request at a time. Use the site's GPU allocation and the existing ASR and
Qwen environments/model caches. Live runs use `rag/.env` and call ELM.

```bash
cd ~/edin-mls-26-autumn
export RAG_ASR_PYTHON="$HOME/miniconda/envs/mls-asr/bin/python"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

"$HOME/mls-qwen-dev-env/bin/python" -u rag/demo/run.py \
  --events rag/results/video-demo/events.jsonl \
  --query-id q_7e9a0f2716c017d2 --method dense --judge

"$HOME/mls-qwen-dev-env/bin/python" -u rag/demo/run.py \
  --events rag/results/video-demo/events.jsonl \
  --query-id q_7e9a0f2716c017d2 --method qwen --judge
```

Run `make -C hw1-asr lib` in the configured ASR/CUDA environment first if
the compiled kernels are missing. To use your own recording, replace
`--query-id …` with `--audio /path/to/recording.wav` (16 kHz mono PCM16).
Remove `--judge` for your own WAV or for answer-only mode. For course queries,
`--judge` makes a separate ELM call against the labelled reference evidence
and shows the three rubric scores, their normalised sum, and the explanation.
Answer latency excludes model loading and judging. A single automatic score
or demo request is not a benchmark.

## Recorded fallback

No GPU or ELM call is needed, but the extracted course corpus and the recorded
result file are required. The browser labels this as **Recorded replay**:

```bash
python3 -u rag/demo/run.py --events rag/results/video-demo/events.jsonl \
  --method qwen --replay rag/results/paired-audio-cost-20261118-233916/rows.jsonl
```
