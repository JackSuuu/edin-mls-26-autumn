#!/usr/bin/env bash
#SBATCH --job-name=mls-rerank-check
#SBATCH --partition=Teaching
#SBATCH --nodelist=opencast
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=rag/results/rerank-check-%j.log
set -euo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/htang2/toolchain-20251006/toolchain.rc
export PATH="$HOME/mls-rag-env/bin:$PATH"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
cd "$HOME/edin-mls-26-autumn"
date -Is
hostname
python -u rag/extensions/validate_reranker.py \
  --plan rag/results/selection-pilot-3713795/plan \
  --output rag/results/selection-pilot-3713795/reranker-validation.json
date -Is
echo RERANKER_VALIDATION_COMPLETE
