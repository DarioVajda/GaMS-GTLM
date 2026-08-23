#!/bin/bash
#
# The cheap checks that must pass before a long run: plumbing, the two-pass
# evaluator's acceptance test, and a step-cost measurement, in one job.
#
#     sbatch train/run_smoke.sbatch [DATA_ROOT] [EXTRA ARGS ...]
#
#   1. smoke run    -- 24 items, 3 steps: does the loss move, does eval fire,
#                      does the best-checkpoint reload work?
#   2. overfit run  -- 32 items, 40 epochs, purely to produce a checkpoint that
#                      answers items exactly, so step 3's pass-1 branch is not
#                      vacuous.  Not a result.
#   3. two-pass test -- the fast evaluator against generate-everything, per item.
#                      Run twice: once on the untrained backbone (a spread over
#                      all three grading modes) and once on the overfit
#                      checkpoint (where pass 1 actually fires).
#   4. step timing  -- 20 steps over the REAL train split, so the p50-77-node /
#                      p99-293-node ball sizes are measured rather than assumed
#                      (train/config.py's "these graphs are ~32 nodes" premise
#                      died with balls/v2).
#
# ana is the A100-80GB node running Ubuntu 22.04 / python3.10, which is what the
# venv is built against -- so this needs no container, unlike the B200/B300 hosts.
#
#SBATCH --partition=frida
#SBATCH --nodelist=ana
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=sl_qa_smoke
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/sl_qa_smoke_%j.out

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

RUNS_JSONL="$ROOT/train/results/smoke_runs.jsonl"

echo; echo "=================== 1. smoke run (24 items, 3 steps) ==================="
python -m train --data-root "$DATA_ROOT" --max-items 24 --max-steps 3 \
    --eval-steps 2 --num-workers 0 --batch-size 1 --accumulation-steps 4 \
    --runs-jsonl "$RUNS_JSONL" "$@"
rc1=$?; echo "[smoke] exit $rc1"

echo; echo "=================== 2. overfit 32 items, so pass 1 fires ==============="
# The pass-1 branch of the two-pass evaluator is VACUOUS on an untrained model --
# nothing clears it, so the assertion "pass 1 passed => generation reproduces
# gold" is never actually evaluated.  Forty epochs over 32 items produces a
# checkpoint that answers most of them exactly, which is the state the assertion
# is about.  Nothing about this run is a result; it exists to make the test bite.
OVERFIT_DIR="$ROOT/checkpoints/sl_qa_overfit"
rm -rf "$OVERFIT_DIR"
python -m train --data-root "$DATA_ROOT" --max-items 32 --num-epochs 40 \
    --batch-size 1 --accumulation-steps 4 --eval-steps 100 --no-final-eval \
    --num-workers 0 --runs-jsonl "$RUNS_JSONL" "$@"
CKPT=$(ls -dt "$ROOT"/checkpoints/sl_qa/*/checkpoint-* 2>/dev/null | head -1)

echo; echo "=================== 3. two-pass == generate-everything ================="
python -m train.test_two_pass --data-root "$DATA_ROOT" --max-items 32 --split dev
rc2=$?
if [ -n "$CKPT" ]; then
    echo "--- again, against the overfit checkpoint $CKPT ---"
    python -m train.test_two_pass --data-root "$DATA_ROOT" --max-items 32 \
        --split train --no-spread --checkpoint "$CKPT"
    rc2b=$?
else
    echo "no checkpoint found; the pass-1 branch stays unexercised"; rc2b=1
fi

echo; echo "=================== 4. step timing on the real split ==================="
# eval-steps 10 so TWO in-training evals fire over 20 steps: the fast evaluator's
# cost on the full dev split is as much a part of the schedule budget as the
# training step is.  --no-final-eval skips the slow graded pass, which on an
# untrained model would generate for nearly every one of the 3,224 dev+test items.
python -m train --data-root "$DATA_ROOT" --max-steps 20 --eval-steps 10 \
    --no-final-eval --num-workers 4 --runs-jsonl "$RUNS_JSONL" "$@"
rc3=$?

echo; echo "smoke=$rc1  two_pass=$rc2/$rc2b  timing=$rc3"
echo "end: $(date)"
exit $(( rc1 | rc2 | rc2b | rc3 ))
