#!/bin/bash
#
# Fine-tune GTLM (gemma-3-1b-it backbone, SPD + magnetic bias) on a subset of the
# generated QA types.
#
#     sbatch train/run_train.sbatch [EXTRA ARGS ...]
#     sbatch train/run_train.sbatch --max-items 24 --max-steps 3   # smoke test
#     sbatch train/run_train.sbatch --no-spd --no-magnetic         # control arm
#
# For a grid rather than one run, use the sweep runner instead -- it takes the
# same flags from a JSONC file and submits one job per configuration:
#
#     .venv/bin/python -m sweep train train/configs/spd_magnetic.jsonc
#
# ana/aga are the A100 nodes running Ubuntu 22.04 / python3.10, which is what the
# venv is built against -- so this needs no container, unlike the B200/B300 hosts.
#
#SBATCH --partition=frida
#SBATCH --nodelist=ana
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --job-name=sl_qa
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/sl_qa_%j.out

set -euo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
export PATH="$ROOT/.venv/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$ROOT"

python -m train \
    --types T3,T4,T9,T10 \
    --model-name google/gemma-3-1b-it \
    "$@"

echo "end: $(date)"
