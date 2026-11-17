#!/usr/bin/env bash
#SBATCH --job-name=mls-selection-eval
#SBATCH --partition=Teaching
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=rag/results/selection-eval-%j.log
set -euo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/htang2/toolchain-20251006/toolchain.rc
export PATH="$HOME/mls-rag-env/bin:$PATH"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
cd "$HOME/edin-mls-26-autumn"
RUN_DIR="${1:?Provide existing selection-pilot run directory}"
case "$RUN_DIR" in rag/results/selection-pilot-[0-9]*) ;; *) exit 2 ;; esac
test -f "$RUN_DIR/plan/prepared.json"
date -Is
hostname
python -u rag/extensions/run_selection_experiment.py generate \
  --plan "$RUN_DIR/plan" --output "$RUN_DIR/answers" --resume
# Validate every arm before spending any judge calls.
for ARM in dense5 dense20 rerank5 dense_budget rerank_budget; do
  python -u rag/scripts/evaluate_answers.py --split practice \
    --candidates "$RUN_DIR/answers/$ARM.jsonl" --allow-subset \
    --out-dir "$RUN_DIR/judge_$ARM" --dry-run
done
# One call per query/arm. Arm-wise judging order is a limitation to report.
for ARM in dense5 dense20 rerank5 dense_budget rerank_budget; do
  python -u rag/scripts/evaluate_answers.py --split practice \
    --candidates "$RUN_DIR/answers/$ARM.jsonl" --allow-subset \
    --out-dir "$RUN_DIR/judge_$ARM"
done
python -u rag/extensions/summarize_selection_experiment.py \
  --run "$RUN_DIR" --output "$RUN_DIR/comparison.json"
date -Is
echo SELECTION_PILOT_COMPLETE
