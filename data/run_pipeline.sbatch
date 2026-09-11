#!/bin/bash
#
# The whole data build in one job: the graph store, then the five stages that
# make the QA dataset.  This is the preferred way to build the data; the
# per-stage `run_*.sbatch` scripts remain for rebuilding one stage at a time.
#
#     sbatch data/run_pipeline.sbatch                    # everything, defaults
#     sbatch data/run_pipeline.sbatch --types T1,T3 --scale 0.02   # smoke build
#     sbatch data/run_pipeline.sbatch --rebuild-store    # force the store too
#     sbatch --gres=gpu:4 data/run_pipeline.sbatch       # more GPUs, faster
#                                                        # stage 3
#
# Every argument is passed through to `python -m pipeline` (`--help` lists
# them).  The job requests resources and enters the container; all of the build
# logic is Python, in data/pipeline/.
#
# ~2 1/4 h for the full 40k corpus when the store already exists -- 135 min on
# two H100s, job 141387; data/README.md carries the per-stage table.  The GPUs
# sit idle through the CPU stages, about 95 of those minutes, which is still a
# fair trade for a build that is one job, one log and one thing to wait for.
#
# Resources.  --mem is set from a MEASURED peak.  At the 40k corpus the QA chain
# is the heaviest part: 72 GB (job 141387), past the store build's 48 GB, and
# enough to have killed the old 64 G default.  96 G is the measured 72 plus a
# third.  The store build peaks at 48 GB as a job and 29 GB inside the builder;
# the gap is worker processes plus page cache for the 41 GB of N-Triples, which
# the cgroup charges here.
#
# It was 160 G in the first draft of this file, from a 107.8 GB measurement that
# was real but not necessary: the parse pool prefetched without bound and the
# queue grew into whatever memory it was offered.  Bounding it (`_gated` in
# build_graph.py) cut the job to 48 GB and made it slightly faster.
#
# 16 CPUs is what build_graph's parallel parse can use -- past ~16 buys nothing
# -- and is ample for extraction dataloading.  5 h covers a from-scratch run: the
# measured 135 min, the store's ~11 min (667 s in job 133080), and headroom for
# stage 2's availability pass, whose T35 scan alone has ranged 26-58 min
# depending on what else shares the node's CPUs.
#
# Two GPUs is a default, not a requirement: stage 3 shards across however many
# the job is given, so `sbatch --gres=gpu:1` and `--gres=gpu:8` both work and the
# program adapts at startup.  Two is roughly where the queue wait stops being
# repaid by the extra throughput.  Stage 3 is ~42 min on two at full size and
# shards evenly, so one GPU costs about that much again.  When the cluster is
# busy, one GPU can still be the fastest route to a finished build simply
# because it starts:
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
#SBATCH --mem=96G
#SBATCH --time=05:00:00
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
