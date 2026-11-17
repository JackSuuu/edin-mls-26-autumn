#!/usr/bin/env bash
# Run on the ICF login node. Downloads only; no model inference here.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd "$HOME/edin-mls-26-autumn"
ASR_ENV="$HOME/mls-asr-env"
RAG_ENV="$HOME/mls-rag-env"
if [[ ! -x "$ASR_ENV/bin/python" ]]; then
  /opt/conda/bin/conda create -y -p "$ASR_ENV" --override-channels -c conda-forge python=3.11
fi
"$ASR_ENV/bin/python" -m pip install --no-cache-dir --no-compile torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
"$ASR_ENV/bin/python" -m pip install --no-cache-dir --no-compile transformers==5.0.0 numpy safetensors huggingface_hub
"$ASR_ENV/bin/python" -c 'from huggingface_hub import snapshot_download; print(snapshot_download("zai-org/GLM-ASR-Nano-2512"))'
if [[ ! -x "$RAG_ENV/bin/python" ]]; then
  "$ASR_ENV/bin/python" -m venv "$RAG_ENV"
fi
"$RAG_ENV/bin/python" -m pip install --no-cache-dir --no-compile torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
printf 'torch==2.10.0\ntransformers==5.0.0\n' > "$RAG_ENV/course-constraints.txt"
"$RAG_ENV/bin/python" -m pip install --no-cache-dir --no-compile -c "$RAG_ENV/course-constraints.txt" -r rag/requirements.txt
mkdir -p "$HOME/mls-rag-assets"
unzip -q -n rag-assets.zip -d "$HOME/mls-rag-assets"
"$RAG_ENV/bin/python" rag/scripts/install_assets.py --from-dir "$HOME/mls-rag-assets"
"$RAG_ENV/bin/python" -m unittest discover -s rag/tests -v
echo 'ICF_PREPARATION_COMPLETE'
