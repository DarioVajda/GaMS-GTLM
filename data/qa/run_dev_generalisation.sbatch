#!/bin/bash
#
# Tier A and Tier C items for dev, published as datasets/dev_generalisation/.
#
#     sbatch data/qa/run_dev_generalisation.sbatch [args for qa.build_dev_generalisation]
#
# Generation takes seconds; the time is the extractor (GaMS3-12B, ~1,200 items on
# one GPU), relabel over the whole raw corpus, and the selftest over the union.
# Same container and environment as run_pipeline.sbatch, for the same reasons.
#
#SBATCH --partition=frida
#SBATCH --account=lkm
#SBATCH --constraint=GPU_BRD:B200|GPU_BRD:B300|GPU_BRD:H100
#SBATCH --exclude=ixb7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=dev_generalisation
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/dev_generalisation_%j.out

set -uo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
CONTAINER=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh
export HOME=${HOME:-/shared/home/dario.vajda}

echo "host: $(hostname)  start: $(date)"
srun --container-image="$CONTAINER" --container-mounts=/shared:/shared \
     env PATH=/shared/workspace/povejmo/graph_model/.venv/bin:$PATH \
         HF_HOME=/shared/workspace/povejmo/huggingface_cache \
         HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=true \
     bash -c "cd '$ROOT/data' && exec python -m qa.build_dev_generalisation $*"
rc=$?
echo "end: $(date)  rc=$rc"
exit $rc
