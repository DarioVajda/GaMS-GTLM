#!/bin/bash
#
# Verify the DDP path BEFORE any of it is trusted with a study's GPU-hours.
#
#     sbatch train/slurm/run_ddp_check.sbatch
#
# Both world sizes run in ONE job, on ONE node, against the SAME GPUs -- that is
# the whole point.  Splitting them across two allocations would compare two
# cards as well as two rank counts, and a difference could not be attributed.
#
# Four stages:
#   1  world=1 evaluation over a 96-item spread          -> /tmp/ddp_w1.jsonl
#   2  world=4 evaluation over the identical slice       -> /tmp/ddp_w4.jsonl
#   3  the two are diffed item by item, and the effective-batch factorisation
#      table is asserted for every rank count
#   4  a 20-step TRAINING smoke at world=4, which is what proves the reducer
#      does not deadlock on this model (LoRA + graph-bias + gradient
#      checkpointing) and that the derived schedule is rank-invariant
#
# See `train/checks/check_ddp.py` for what exactness is claimed and why.
#
#SBATCH --partition=frida
#SBATCH --account=lkm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --constraint='GPU_BRD:B200|GPU_BRD:B300|GPU_BRD:H100'
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --job-name=ddp_check
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/ddp_check_%j.out

set -uo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
CONTAINER=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh
OUT=/shared/workspace/povejmo/gams_gtlm/train/results/ddp_check

echo "host: $(hostname)  start: $(date)"
mkdir -p "$OUT"

srun --container-image="$CONTAINER" --container-mounts=/shared:/shared \
     env HOME="$HOME" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
         TORCHINDUCTOR_CACHE_DIR="$ROOT/.inductor_cache/ddp_check" \
     bash -c "cd '$ROOT' && \
      echo '=== nvidia-smi ===' && nvidia-smi --query-gpu=index,name,memory.total --format=csv && \
      echo && echo '=== 1/4  world=1 evaluation ===' && \
      '$ROOT/.venv/bin/python' -m train.checks.check_ddp --out '$OUT/w1.jsonl' && \
      echo && echo '=== 2/4  world=4 evaluation ===' && \
      '$ROOT/.venv/bin/torchrun' --standalone --nproc_per_node 4 \
          -m train.checks.check_ddp --out '$OUT/w4.jsonl' && \
      echo && echo '=== 3/4  diff ===' && \
      '$ROOT/.venv/bin/python' -m train.checks.check_ddp \
          --compare '$OUT/w1.jsonl' '$OUT/w4.jsonl' && \
      echo && echo '=== 4/4  world=4 training smoke, 20 steps ===' && \
      '$ROOT/.venv/bin/torchrun' --standalone --nproc_per_node 4 \
          -m train --model-name google/gemma-3-1b-it \
          --max-items 512 --max-steps 20 --eval-steps 10 --num-epochs 1 \
          --no-final-eval --num-workers 2 \
          --runs-jsonl '$OUT/smoke_runs.jsonl' --run-name ddp_smoke"

echo "srun exit: $?"
echo "end: $(date)"
