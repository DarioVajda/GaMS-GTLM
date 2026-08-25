#!/bin/bash
#
# Fine-tune ONE configuration, on the same B200 + container stack the arms use.
#
#     sbatch train/slurm/run_train.sbatch [EXTRA ARGS ...]
#     sbatch train/slurm/run_train.sbatch --no-spd --no-magnetic         # control arm
#     sbatch train/slurm/run_train.sbatch --data-root data/datasets/balls/v2_clean_noretrieval \
#            --plain-llm --no-spd --no-magnetic --batch-size 16 --accumulation-steps 1
#
# For the plumbing checks and the evaluator's acceptance tests use
# `train/slurm/run_preflight.sbatch`.  For the study itself use the sweep runner, which
# takes the same flags from a JSONC file and submits one job per configuration:
#
#     .venv/bin/python -m sweep train train/configs/arms_v3.jsonc
#
# **Same hardware as the sweep, deliberately.** A one-off run on a different node
# type produces numbers that look comparable to the arms and are not; and the
# venv is python3.10 while the Blackwell hosts are 3.12, so the pyxis container
# is what makes the interpreter resolve at all.  The venv python is invoked
# DIRECTLY -- PYTHONPATH would leave the image's dist-packages on the path and
# its torchvision then loads against our torch (job 130961).
#
#SBATCH --partition=frida
#SBATCH --account=povejmo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_BRD:B200
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --job-name=sl_qa
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/sl_qa_%j.out

set -uo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
CONTAINER=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh

echo "host: $(hostname)  start: $(date)"

# No --types: the run is over all of them.  The four-type subset this job used to
# pin was a v1 convenience, and a score over four types is not the corpus's.
# `bash -c` rather than the interpreter directly, because `-m train` resolves
# against the working directory and srun inherits whatever sbatch was called from.
srun --container-image="$CONTAINER" --container-mounts=/shared:/shared \
     env HOME="$HOME" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
         TORCHINDUCTOR_CACHE_DIR="$ROOT/.inductor_cache/run_train" \
     bash -c "cd '$ROOT' && '$ROOT/.venv/bin/python' -m train \
              --model-name google/gemma-3-1b-it $*"

echo "srun exit: $?"
echo "end: $(date)"
