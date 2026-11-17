#!/usr/bin/env bash
#SBATCH --job-name=mls-rag-diagnostic
#SBATCH --partition=Teaching
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=rag/results/retrieval-diagnostic-%j.log
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
date -Is
hostname
git rev-parse HEAD
python -u rag/extensions/collect_retrieval_diagnostics.py \
  --transcript-cache rag/results/icf-3713067/practice_transcripts.jsonl \
  --sample rag/results/diagnostic-01/next_practice_sample.jsonl \
  --saved-answers rag/results/icf-3713067/sample_answers.jsonl \
  --output "rag/results/diagnostic-retrieval-$SLURM_JOB_ID"
date -Is
echo RETRIEVAL_DIAGNOSTICS_COMPLETE
