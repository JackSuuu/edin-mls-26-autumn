#!/usr/bin/env bash
#SBATCH --job-name=mls-passage-prep
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=00:45:00
#SBATCH --output=rag/results/passage-prep-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
cd "$HOME/edin-mls-26-autumn"
PASSAGE_RUN_DIR="${1:?Supply unique pilot run directory}"
date -Is
hostname
"$HOME/mls-asr-env/bin/python" -m unittest discover -s rag/tests -p test_passage_selection.py -v
"$HOME/mls-asr-env/bin/python" -u rag/extensions/run_passage_pilot.py prepare --run "$PASSAGE_RUN_DIR"
date -Is
