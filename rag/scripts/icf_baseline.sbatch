#!/usr/bin/env bash
#SBATCH --job-name=mls-rag-baseline
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=rag/results/icf-baseline-%j.log
set -euo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/htang2/toolchain-20251006/toolchain.rc
export RAG_ASR_PYTHON="$HOME/mls-asr-env/bin/python"
export PATH="$HOME/mls-rag-env/bin:$PATH"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
cd "$HOME/edin-mls-26-autumn"
RUN_DIR="rag/results/icf-$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"
date -Is
hostname
nvidia-smi -L
git rev-parse HEAD
"$RAG_ASR_PYTHON" -c 'import torch,transformers; print("ASR:",torch.__version__,transformers.__version__,"GPU:",torch.cuda.get_device_name(0))'
python -c 'import torch,transformers; print("RAG:",torch.__version__,transformers.__version__)'
"$RAG_ASR_PYTHON" -m pip freeze > "$RUN_DIR/asr-packages.txt"
python -m pip freeze > "$RUN_DIR/rag-packages.txt"
make -C hw1-asr test 2>&1 | tee "$RUN_DIR/cuda-tests.log"
python rag/scripts/check_setup.py 2>&1 | tee "$RUN_DIR/setup.log"
ASR_CUDA=none python -u rag/scripts/run_audio_query.py --device cpu > "$RUN_DIR/reference-query.json"
ASR_CUDA=all python -u rag/scripts/run_audio_query.py --device cpu > "$RUN_DIR/cuda-query.json"
export ASR_CUDA=all
python -u rag/scripts/evaluate_rag.py --device cpu --split practice --transcript-cache "$RUN_DIR/practice_transcripts.jsonl" 2>&1 | tee "$RUN_DIR/retrieval-practice.log"
python -u rag/scripts/generate_answers.py --device cpu --split practice --limit 5 --output "$RUN_DIR/sample_answers.jsonl"
python rag/scripts/evaluate_answers.py --split practice --candidates "$RUN_DIR/sample_answers.jsonl" --allow-subset --dry-run
python -u rag/scripts/evaluate_answers.py --split practice --candidates "$RUN_DIR/sample_answers.jsonl" --allow-subset --out-dir "$RUN_DIR/sample_judge"
date -Is
echo 'ICF_BASELINE_COMPLETE'
