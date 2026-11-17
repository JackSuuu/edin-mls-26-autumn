#!/usr/bin/env bash
#SBATCH --job-name=mls-selection-judge
#SBATCH --partition=Teaching
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=rag/results/selection-judge-%j.log
set -euo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/htang2/toolchain-20251006/toolchain.rc
export PATH="$HOME/mls-rag-env/bin:$PATH"
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
cd "$HOME/edin-mls-26-autumn"
RUN_DIR=rag/results/selection-pilot-3713795
date -Is
hostname
python -m unittest discover -s rag/tests -p test_judge_protocol.py -v
# Probe the fourth dense5 query, which failed under default text response mode.
python -c 'import json,pathlib; p=pathlib.Path("rag/results/selection-pilot-3713795"); r=[json.loads(s) for s in (p/"answers/dense5.jsonl").read_text().splitlines()][3]; f=(p/"json-mode-probe.jsonl").open("x"); f.write(json.dumps(r)+"\n"); f.close()'
python -u rag/scripts/evaluate_answers.py --split practice \
  --candidates "$RUN_DIR/json-mode-probe.jsonl" --allow-subset --json-mode --api-retries 1 \
  --out-dir "$RUN_DIR/judge_json_probe"
echo JSON_MODE_PROBE_PASSED
for ARM in dense5 dense20 rerank5 dense_budget rerank_budget; do
  python -u rag/scripts/evaluate_answers.py --split practice \
    --candidates "$RUN_DIR/answers/$ARM.jsonl" --allow-subset --json-mode \
    --out-dir "$RUN_DIR/judge_json_$ARM"
done
python -u rag/extensions/summarize_selection_experiment.py \
  --run "$RUN_DIR" --judge-prefix judge_json_ --output "$RUN_DIR/comparison.json"
date -Is
echo SELECTION_PILOT_COMPLETE
