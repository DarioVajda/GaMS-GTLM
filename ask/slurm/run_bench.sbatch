#!/bin/bash
#
# Both attention backends, on one GPU, back to back -- what a warm flex cache buys.
#
#     sbatch --partition=dev --constraint=GPU_SKU:A100_80GB_PCIE ask/slurm/run_bench.sbatch
#     sbatch ask/slurm/run_bench.sbatch                      # the default: B200
#
# One process per backend, deliberately: peak GPU memory is the number being
# compared, and two models resident in one process would report the second one's
# peak on top of the first one's weights.  The results land beside the log as
# JSON so the comparison can be re-printed without re-running the GPU work.
#
# Container and interpreter as in run_ask.sbatch: the venv is python3.10 while
# the Blackwell hosts are 3.12, so the pyxis image is what makes the interpreter
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
#SBATCH --time=01:00:00
#SBATCH --job-name=ask_bench
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/ask_bench_%j.out

set -uo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
CONTAINER=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh
OUT=$ROOT/data/logs/bench_${SLURM_JOB_ID}

echo "host: $(hostname)  start: $(date)"

run() {
    srun --container-image="$CONTAINER" --container-mounts=/shared:/shared \
         --container-workdir="$ROOT" \
         env HOME="$HOME" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
             HF_HOME=/shared/workspace/povejmo/huggingface_cache HF_HUB_OFFLINE=1 \
             PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
             PYTHONPATH="$ROOT:$ROOT/data" \
         "$ROOT/.venv/bin/python" -m ask.checks.bench_backends "$@"
}

# Anything after the script name goes to BOTH measurement runs and to neither
# comparison -- `--items 100 --repeats 1` turns this job from the timing
# benchmark into the drift measurement without a second script.
run --impl flex  --out "${OUT}_flex.json"  "$@" || echo "flex run exit $?"
run --impl eager --out "${OUT}_eager.json" "$@" || echo "eager run exit $?"
run --compare "${OUT}_flex.json" "${OUT}_eager.json"

echo "results: ${OUT}_{flex,eager}.json"
echo "end: $(date)"
