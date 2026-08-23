#!/bin/bash
#
# Fine-tune GTLM (gemma-3-1b-it backbone, SPD + magnetic bias) on a subset of the
# generated QA types.
#
#     sbatch train/run_train.sbatch [EXTRA ARGS ...]
#     sbatch train/run_train.sbatch --no-spd --no-magnetic         # control arm
#
# For the plumbing checks and the evaluator's acceptance test, use
# train/run_smoke.sbatch instead.  For a grid rather than one run, use the sweep
# runner -- it takes the same flags from a JSONC file and submits one job per
# configuration:
#
#     .venv/bin/python -m sweep train train/configs/arms_v2.jsonc
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

# No --types: the run is over all 19 of them.  The four-type subset this job used
# to pin was a v1 convenience, and a score over four types is not the corpus's.
python -m train \
    --model-name google/gemma-3-1b-it \
    "$@"

echo "end: $(date)"
