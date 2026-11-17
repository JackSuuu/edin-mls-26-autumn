#!/usr/bin/env bash
#SBATCH --job-name=mls-passage-answers
#SBATCH --partition=Teaching
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=rag/results/passage-answers-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
cd "$HOME/edin-mls-26-autumn"
PASSAGE_RUN_DIR="${1:?Supply frozen prepared run directory}"
PASSAGE_PYTHON="$HOME/mls-rag-env/bin/python"
date -Is
hostname
RESUME=()
if [[ -f "$PASSAGE_RUN_DIR/answers/generation_config.json" ]]; then RESUME=(--resume); fi
"$PASSAGE_PYTHON" -u rag/extensions/run_passage_pilot.py generate --run "$PASSAGE_RUN_DIR" --max-new-queries 2 "${RESUME[@]}"
"$PASSAGE_PYTHON" -u rag/extensions/run_passage_pilot.py smoke --run "$PASSAGE_RUN_DIR"
for ARM in full5 prefix768 passages768; do
  "$PASSAGE_PYTHON" -u rag/scripts/evaluate_answers.py --split practice --allow-subset --dry-run \
    --candidates "$PASSAGE_RUN_DIR/answers/smoke_$ARM.jsonl"
  "$PASSAGE_PYTHON" -u rag/scripts/evaluate_answers.py --split practice --allow-subset --json-mode --api-retries 1 \
    --model utter-project/EuroLLM-22B-Instruct-2512 --candidates "$PASSAGE_RUN_DIR/answers/smoke_$ARM.jsonl" \
    --out-dir "$PASSAGE_RUN_DIR/judge_json_$ARM"
done
echo PASSAGE_SMOKE_PASSED
"$PASSAGE_PYTHON" -u rag/extensions/run_passage_pilot.py generate --run "$PASSAGE_RUN_DIR" --resume
for ARM in full5 prefix768 passages768; do
  "$PASSAGE_PYTHON" -u rag/scripts/evaluate_answers.py --split practice --allow-subset --json-mode \
    --model utter-project/EuroLLM-22B-Instruct-2512 --candidates "$PASSAGE_RUN_DIR/answers/$ARM.jsonl" \
    --out-dir "$PASSAGE_RUN_DIR/judge_json_$ARM"
done
"$PASSAGE_PYTHON" -u rag/extensions/run_passage_pilot.py summarise --run "$PASSAGE_RUN_DIR"
date -Is
