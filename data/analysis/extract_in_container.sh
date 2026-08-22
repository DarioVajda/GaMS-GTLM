#!/usr/bin/env bash
# Inner entry for run_extract_sharded.sbatch -- runs INSIDE the pyxis container.
#
# The Blackwell hosts (ixb*) are Ubuntu 24.04 / python3.12 and the project venv is
# built against 3.10, so a bare `srun` there dies on imports.  The container is a
# py3.10 base, which is what lets the same venv resolve; this is the pattern
# graph_model/sweep/slurm_launch.sh uses, reduced to what this job needs.
#
#   extract_in_container.sh <dataset> <prefix> <shard> <num_shards> [extra args]
set -uo pipefail

DATASET="$1"; PREFIX="$2"; SHARD="$3"; NUM_SHARDS="$4"; shift 4

ROOT=/shared/workspace/povejmo/gams_gtlm
export PATH=/shared/workspace/povejmo/graph_model/.venv/bin:$PATH
export HF_HOME=/shared/workspace/povejmo/huggingface_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "[inner] host=$(hostname) python=$(python -V 2>&1) shard=${SHARD}/${NUM_SHARDS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$ROOT/data"

exec python -m analysis.measure_extraction \
    --dataset "$DATASET" \
    --prompt "$ROOT/data/prompt_v3.txt" \
    --n 0 \
    --num-shards "$NUM_SHARDS" --shard "$SHARD" \
    --dump "${PREFIX}_items.part${SHARD}.jsonl" \
    "$@"
