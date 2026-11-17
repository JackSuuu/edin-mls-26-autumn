#!/usr/bin/env bash
#SBATCH --job-name=mls-selection-pilot
#SBATCH --partition=Teaching
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=rag/results/selection-pilot-%j.log
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
RUN_DIR="rag/results/selection-pilot-$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"
date -Is
hostname
git rev-parse HEAD
python -m unittest discover -s rag/tests -p test_evidence_selection.py -v
EXTRA=()
if [[ -n "${1:-}" ]]; then
  EXTRA=(--rerank-cache "$1")
fi
python -u rag/extensions/run_selection_experiment.py prepare \
  --sample rag/results/diagnostic-01/next_practice_sample.jsonl \
  --rankings rag/results/diagnostic-retrieval-3713758/rankings.jsonl \
  --output "$RUN_DIR/plan" "${EXTRA[@]}"
python -u rag/extensions/run_selection_experiment.py generate \
  --plan "$RUN_DIR/plan" --output "$RUN_DIR/answers" --max-new-queries 3
date -Is
echo SELECTION_SMOKE_COMPLETE
