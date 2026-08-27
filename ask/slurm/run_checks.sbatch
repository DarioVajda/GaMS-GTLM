#!/bin/bash
#
# The `ask` acceptance run: everything that needs a GPU, on the stack the arms
# were trained on.
#
#     sbatch ask/slurm/run_checks.sbatch            # every stage
#     sbatch ask/slurm/run_checks.sbatch 2,3        # just the parity checks
#
# The stages, and what each is for, are documented in check_stages.sh.  This
# file only allocates the node and enters the container, exactly as
# train/slurm/run_preflight.sbatch does: the venv is python3.10 and the
# Blackwell hosts are 3.12, so the pyxis image is what lets the interpreter
# resolve at all.
#
#SBATCH --partition=frida
#SBATCH --account=lkm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_BRD:B200
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=ask_checks
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/ask_checks_%j.out

set -uo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
CONTAINER=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh
STAGES="${1:-1,2,3,4,5}"

echo "host: $(hostname)   start: $(date)   stages: $STAGES"

srun --container-image="$CONTAINER" --container-mounts=/shared:/shared \
     env HOME="$HOME" PYTHONUNBUFFERED=1 \
     bash "$ROOT/ask/slurm/check_stages.sh" "$STAGES"

echo "srun exit: $?"
echo "end: $(date)"
