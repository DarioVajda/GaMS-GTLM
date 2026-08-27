#!/usr/bin/env bash
# Inner entry for run_pipeline.sbatch -- runs INSIDE the pyxis container.
#
# The Blackwell hosts are Ubuntu 24.04 / python3.12 and the project venv is built
# against 3.10, so a bare `srun` there dies on imports.  The container is a
# py3.10 base, which is what lets the venv resolve -- and what lets the four CPU
# stages run here at all rather than on the three Ubuntu 22.04 nodes they are
# pinned to when submitted one at a time.
#
#   in_container.sh [args for `python -m pipeline`]
set -uo pipefail

ROOT=/shared/workspace/povejmo/gams_gtlm
export PATH=/shared/workspace/povejmo/graph_model/.venv/bin:$PATH
export HF_HOME=/shared/workspace/povejmo/huggingface_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# No forks happen after a tokenizer is loaded in the parent, and the store build
# tokenizes 3.5M unique strings, so the fast path is worth having there.  The
# extraction children set this to false for themselves.
export TOKENIZERS_PARALLELISM=true

echo "[inner] host=$(hostname)  python=$(python -V 2>&1)"
echo "[inner] gpus=${CUDA_VISIBLE_DEVICES:-<unset>}  cpus=${SLURM_CPUS_PER_TASK:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

cd "$ROOT/data"
exec python -m pipeline "$@"
