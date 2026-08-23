#!/bin/bash
#
# The step-cost check, on its own: what does a step actually cost on
# balls/v2, under the token-budget batching that replaced the fixed batch size?
#
#     sbatch train/run_timing.sbatch [DATA_ROOT] [EXTRA ARGS ...]
#
# 30 steps and one in-training evaluation over the real splits -- enough to price
# an epoch and an eval without paying for either.  --no-final-eval skips the slow
# graded dev/test pass, which on an untrained model generates for nearly every
# one of the 3,224 items.
#
#SBATCH --partition=frida
#SBATCH --nodelist=ana
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --job-name=sl_qa_timing
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/sl_qa_timing_%j.out

set -uo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
export PATH="$ROOT/.venv/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

DATA_ROOT="${1:-$ROOT/data/datasets/balls/v2}"
shift || true

echo "host: $(hostname)  start: $(date)   data_root: $DATA_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$ROOT"

python -m train --data-root "$DATA_ROOT" --max-steps 30 --eval-steps 25 \
    --accumulation-steps 4 --no-final-eval --num-workers 4 \
    --runs-jsonl "$ROOT/train/results/smoke_runs.jsonl" "$@"
echo "exit: $?"
echo "end: $(date)"
