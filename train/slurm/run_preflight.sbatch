#!/bin/bash
#
# Everything that must pass on the REAL hardware before 18 runs are committed.
#
#     sbatch train/slurm/run_preflight.sbatch [STAGES]
#
# STAGES is a comma-separated subset of the stage numbers below (default: all),
# so a stage that had to be fixed can be re-run on its own without paying for
# the overfit run again.  The stages themselves are in `preflight_stages.sh`,
# which this runs inside the container.
#
# On B200, inside the pyxis container, because that is where the sweep runs and a
# check that passes somewhere else is not a check.  Blackwell hosts are Ubuntu
# 24.04 / python3.12 and this repo's venv is 3.10, so a BARE job there dies on
# imports; the container is a py3.10 base and the venv resolves inside it.  The
# venv interpreter is run DIRECTLY, never through PYTHONPATH -- PYTHONPATH would
# leave the image's dist-packages on the path and its torchvision then loads
# against our torch (`operator torchvision::nms does not exist`, job 130961).
#
# Stages, in the order a failure is cheapest to find:
#
#   1. label mask      -- does the supervised span still decode to exactly the
#                         answer, now that batches are left-padded?
#   2. left padding    -- loss(right) == loss(left) and the answer-tail logits
#                         slice collapses, on BOTH stacks and all three inputs.
#                         Also re-runs the memory probe (peak GiB per batch).
#   3. two-pass        -- the fast evaluator against generate-everything, on an
#                         untrained model (a spread over all three grading modes).
#   4. KV cache        -- is it live in pass 2?  Measured, not assumed.
#   5. smoke train     -- 24 items, 3 steps: loss moves, eval fires, best
#                         checkpoint reloads.
#   6. overfit + two-pass again -- 32 items, 40 epochs, purely to make stage 3's
#                         pass-1 branch non-vacuous (nothing clears pass 1 on an
#                         untrained model, so the sharper assertion never fires).
#
#SBATCH --partition=frida
#SBATCH --account=lkm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_BRD:B200
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=sl_qa_preflight
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/preflight_%j.out

set -uo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
CONTAINER=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh
BALLS="$ROOT/data/datasets/balls"
RUNS_JSONL="$ROOT/train/results/preflight_runs.jsonl"
STAGES="${1:-1,2,3,4,5,6}"

echo "host: $(hostname)   start: $(date)   stages: $STAGES"

srun --container-image="$CONTAINER" --container-mounts=/shared:/shared \
     env HOME="$HOME" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
         TORCHINDUCTOR_CACHE_DIR="$ROOT/.inductor_cache/preflight" \
     bash "$ROOT/train/slurm/preflight_stages.sh" "$STAGES"
echo "srun exit: $?"
echo "end: $(date)"
