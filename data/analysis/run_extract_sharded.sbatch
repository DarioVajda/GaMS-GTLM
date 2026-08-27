#!/bin/bash
#
# R4 -- the full extraction run, data-parallel across the Blackwell nodes.
#
#     sbatch analysis/run_extract_sharded.sbatch DATASET OUT_PREFIX [EXTRA ARGS]
#
# One stage on its own -- `sbatch data/run_pipeline.sbatch` runs all six, and
# shards across the GPUs of a single job rather than across an array.  This is
# the array version, for an extraction run on its own.
#
# A Slurm array: task k runs shard k of NUM_SHARDS and writes its own dump.
# Score the merged result afterwards without a GPU (qa/run_relabel.sbatch does
# this, or by hand):
#
#     python -m analysis.measure_extraction --score-only PREFIX.part*.jsonl \
#            --out score.json
#
# Sharding is a stride (`items[k::N]`), so every shard gets the same type mix --
# a contiguous split would hand one GPU all of T20's 21-word sentences.
#
# **Runs in the pyxis container.** ixb* are Ubuntu 24.04 / python3.12 and this
# venv is built against 3.10, so a bare srun there fails on imports; the
# container is a py3.10 base and the venv resolves inside it (verified on ixb1:
# torch 2.11+cu130, arch_list includes sm_100, bf16 matmul on a B200). Same
# pattern as graph_model/sweep/slurm_launch.sh.
#
# ixb7 is excluded as a CURRENT-STATE fault, not a permanent one: since its
# 2026-08-22T09:36 reboot its enroot mellanox hook fails on
# /dev/infiniband/rdma_cm, so no container starts there (reproduced by jobs
# 130295 and 130297; the same container starts fine on ixb1). Drop the --exclude
# once the node is fixed -- the constraint below already admits it.
#
#SBATCH --partition=frida
#SBATCH --account=lkm
#SBATCH --constraint=GPU_BRD:B200|GPU_BRD:B300
#SBATCH --exclude=ixb7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=extract_shard
#SBATCH --array=0-3
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/extract_%A_%a.out

set -uo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
CONTAINER=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh
export HOME=${HOME:-/shared/home/dario.vajda}

DATASET="${1:-$ROOT/data/datasets/work/generated_raw}"
PREFIX="${2:-$ROOT/data/datasets/work/extraction/items}"
shift 2 || true

NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-4}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

echo "host: $(hostname)  start: $(date)  shard ${SHARD}/${NUM_SHARDS}"

srun --container-image="$CONTAINER" --container-mounts=/shared:/shared \
     bash "$ROOT/data/analysis/extract_in_container.sh" \
          "$DATASET" "$PREFIX" "$SHARD" "$NUM_SHARDS" "$@"

echo "end: $(date)"
