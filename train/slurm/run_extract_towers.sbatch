#!/bin/bash
#
# Write the multimodal Gemma-3 checkpoints' text towers out as text-only ones.
#
#     sbatch train/slurm/run_extract_towers.sbatch
#
# Steps 2 and 3 of the scaling study need `gemma-3-12b-it` and `gemma-3-4b-it`,
# which are `model_type: gemma3` with the text fields nested under `text_config`
# and the weights behind a vision tower.  `train/extract_text_tower.py` writes
# each one's text half out as a flat `gemma3_text` directory, which is what
# `GTLMGemma3Config` loads -- and is exactly the shape `GaMS3-12B-Instruct` is
# already published in, so after this every cell of the study loads through one
# code path.  See that module for why this is done offline rather than as a
# branch inside the adapter.
#
# CPU only: this is a download, a submodule save and a tokenizer copy.  No GPU
# is requested, so it never queues behind the training jobs it unblocks.
#
#SBATCH --partition=frida
#SBATCH --account=lkm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=extract_towers
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/extract_towers_%j.out

set -uo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
CONTAINER=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh

echo "host: $(hostname)  start: $(date)"

# The 4b first: it is a third of the download, so a broken assumption about the
# transformers layout surfaces in minutes rather than after the 12b transfer.
srun --container-image="$CONTAINER" --container-mounts=/shared:/shared \
     env HOME="$HOME" PYTHONUNBUFFERED=1 \
     bash -c "cd '$ROOT' && ./login.sh >/dev/null 2>&1; \
      '$ROOT/.venv/bin/python' -m train.extract_text_tower google/gemma-3-4b-it && \
      '$ROOT/.venv/bin/python' -m train.extract_text_tower google/gemma-3-12b-it"

echo "srun exit: $?"
echo "end: $(date)"
