#!/bin/bash
#
# The whole data build in one job: the graph store, then the five stages that
# make the QA dataset.  This is the preferred way to build the data; the
# per-stage `run_*.sbatch` scripts remain for rebuilding one stage at a time.
#
#     sbatch data/run_pipeline.sbatch                    # everything, defaults
#     sbatch data/run_pipeline.sbatch --types T1,T3      # ~3 min smoke build
#     sbatch data/run_pipeline.sbatch --rebuild-store    # force the store too
#     sbatch --gres=gpu:4 data/run_pipeline.sbatch       # more GPUs, faster
#                                                        # stage 3
#
# Every argument is passed through to `python -m pipeline` (`--help` lists
# them).  The job requests resources and enters the container; all of the build
# logic is Python, in data/pipeline/.
#
# ~21 min when the store already exists, ~32 min when it has to be built --
# measured on two B200s, job 133080; data/README.md carries the per-stage table.
# The GPUs sit idle through the CPU stages: at this scale that is around twenty
# GPU-minutes, which is a good trade for a build that is one job, one log and
# one thing to wait for.
#
# Resources.  --mem is set from a MEASURED peak.  The heaviest stage is the store
# build, which peaks at 48 GB as a job and 29 GB inside the builder; the gap is
# worker processes plus page cache for the 41 GB of N-Triples, which the cgroup
# charges here.  The whole QA chain peaked at 22 GB (job 132447).  64 G is the
# measured 48 plus a third.
#
# It was 160 G in the first draft of this file, from a 107.8 GB measurement that
# was real but not necessary: the parse pool prefetched without bound and the
# queue grew into whatever memory it was offered.  Bounding it (`_gated` in
# build_graph.py) cut the job to 48 GB and made it slightly faster.
#
# 16 CPUs is what build_graph's parallel parse can use -- past ~16 buys nothing
# -- and is ample for extraction dataloading.  3 h covers a from-scratch run
# including the store, which takes ~11 min here (667 s in job 133080).
#
# Two GPUs is a default, not a requirement: stage 3 shards across however many
# the job is given, so `sbatch --gres=gpu:1` and `--gres=gpu:8` both work and the
# program adapts at startup.  Two is roughly where the queue wait stops being
# repaid by the extra throughput -- one GPU costs ~13 minutes more, eight save
# ~3 over two.  When the cluster is busy, one GPU is often the fastest route to
# a finished build simply because it starts:
#
#     sbatch --constraint=GPU_BRD:H100 --gres=gpu:1 data/run_pipeline.sbatch
#
#SBATCH --partition=frida
#SBATCH --account=lkm
# Blackwell OR Hopper.  Nothing here needs Blackwell: the store build is CPU
# only, and the extractor is GaMS3-12B in bf16 (~24 GB), which is comfortable on
# an 80 GB H100 -- verified end to end on ixh (job 132546).  The Blackwells are
# heavily contended and the H100s often are not, so admitting them usually starts
# the job sooner than the extra throughput would have saved.
#
# Not included: axa's 40 GB A100s, where 12B plus batch-32 activations may not
# fit, and ana's 80 GB A100s, where the container has never been exercised.
# Either could be added once tried.
#SBATCH --constraint=GPU_BRD:B200|GPU_BRD:B300|GPU_BRD:H100
# ixb7 is excluded as a CURRENT-STATE fault, not a permanent one: since its
# 2026-08-22 reboot its enroot mellanox hook has failed on
# /dev/infiniband/rdma_cm, so no container starts there.  The node is back in
# `mixed` state, but that says nothing about the hook -- drop this line once a
# container has actually been seen to start on it.  Same exclusion as
# data/analysis/run_extract_sharded.sbatch.
#SBATCH --exclude=ixb7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=data_pipeline
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/pipeline_%j.out

set -uo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
CONTAINER=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh
export HOME=${HOME:-/shared/home/dario.vajda}

echo "host: $(hostname)  start: $(date)"
echo "args: $*"

srun --container-image="$CONTAINER" --container-mounts=/shared:/shared \
     bash "$ROOT/data/pipeline/in_container.sh" "$@"
rc=$?

echo "end: $(date)  rc=$rc"
exit $rc
