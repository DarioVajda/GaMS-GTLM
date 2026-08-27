#!/bin/bash
#
# Run `ask` on a compute node, for testing it without an interactive GPU.
#
#     sbatch ask/slurm/run_ask.sbatch --words brahialen "Kaj pomeni brahialen?"
#     sbatch ask/slurm/run_ask.sbatch --retrieve-only --show-ball --words plezalo x
#     sbatch ask/slurm/run_ask.sbatch --precompile --words brahialen "…"
#
# **This is the internal path, not the one the tool is built for.**  `ask` assumes
# a GPU in the terminal it runs in (D8): the normal way to use it is an
# interactive shell with a GPU allocated, where the progress line and the
# streamed answer are the point.  This wrapper exists so the same command can be
# checked on a compute node from the login node, and its output lands in a log
# where the spinner degrades to one line per stage on its own.
#
# Everything after the script name is passed through to `ask` verbatim.
#
# Same container and interpreter as train/slurm/run_train.sbatch: the venv is
# python3.10 while the Blackwell hosts are 3.12, so the pyxis image is what makes
# the interpreter resolve at all, and the venv python is invoked DIRECTLY rather
# than by putting its site-packages on PYTHONPATH -- that leaves the image's
# dist-packages on the path and its torchvision loads against our torch.  The
# PYTHONPATH set below carries only the two repo roots `bin/ask` would have set
# (D13): `train` resolves from the repo root, `qa`/`lib` only from `data/`.
#
#SBATCH --partition=frida
#SBATCH --account=lkm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_BRD:B200
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --job-name=ask
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/ask_%j.out

set -uo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
CONTAINER=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh

echo "host: $(hostname)  start: $(date)"
echo "args: $*"

# `ask` puts its own cache under ask/cache/<gpu-model> (D17) and sets
# TORCHINDUCTOR_CACHE_DIR itself when it serves flex.  Nothing is pinned here, so
# a job and an interactive session on the same GPU model share one cache.
# --container-workdir: pyxis starts in the image's own working directory, not
# the submit directory, so a relative path on the command line -- `--checkpoint
# checkpoints/...`, `--store data/stores/...` -- resolves against nothing.  The
# defaults no longer care (they resolve from the package), but an argument still
# does, and `bin/ask` cds to the repo for the same reason.
srun --container-image="$CONTAINER" --container-mounts=/shared:/shared \
     --container-workdir="$ROOT" \
     env HOME="$HOME" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
         PYTHONPATH="$ROOT:$ROOT/data" \
     "$ROOT/.venv/bin/python" -m ask "$@"

echo "srun exit: $?"
echo "end: $(date)"
