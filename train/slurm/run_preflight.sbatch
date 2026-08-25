#!/bin/bash
#
# Everything that must pass on the REAL hardware before 18 runs are committed.
#
#     sbatch train/slurm/run_preflight.sbatch [STAGES]
#
# STAGES is a comma-separated subset of the stage numbers below (default: all),
# so a stage that had to be fixed can be re-run on its own without paying for
# the overfit run again.
#
# On B200, inside the pyxis container, because that is where the sweep runs and a
# check that passes somewhere else is not a check.  Blackwell hosts are Ubuntu
# 24.04 / python3.12 and this repo's venv is 3.10, so a BARE job there dies on
# imports; the container is a py3.10 base and the venv resolves inside it.  The
# venv interpreter is run DIRECTLY, never through PYTHONPATH -- PYTHONPATH would
# leave the image's dist-packages on the path and its torchvision then loads
# against our torch (`operator torchvision::nms does not exist`, job 130961).
#
# Stages, in the order a failure is cheapest to find:
#
#   1. label mask      -- does the supervised span still decode to exactly the
#                         answer, now that batches are left-padded?
#   2. left padding    -- loss(right) == loss(left) and the answer-tail logits
#                         slice collapses, on BOTH stacks and all three inputs.
#                         Also re-runs the memory probe (peak GiB per batch).
#   3. two-pass        -- the fast evaluator against generate-everything, on an
#                         untrained model (a spread over all three grading modes).
#   4. KV cache        -- is it live in pass 2?  Measured, not assumed.
#   5. smoke train     -- 24 items, 3 steps: loss moves, eval fires, best
#                         checkpoint reloads.
#   6. overfit + two-pass again -- 32 items, 40 epochs, purely to make stage 3's
#                         pass-1 branch non-vacuous (nothing clears pass 1 on an
#                         untrained model, so the sharper assertion never fires).
#
#SBATCH --partition=frida
#SBATCH --account=povejmo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_BRD:B200
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=sl_qa_preflight
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/preflight_%j.out

set -uo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
CONTAINER=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh
BALLS="$ROOT/data/datasets/balls"
RUNS_JSONL="$ROOT/train/results/preflight_runs.jsonl"
STAGES="${1:-1,2,3,4,5,6}"

echo "host: $(hostname)   start: $(date)   stages: $STAGES"

srun --container-image="$CONTAINER" --container-mounts=/shared:/shared \
     env HOME="$HOME" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
         TORCHINDUCTOR_CACHE_DIR="$ROOT/.inductor_cache/preflight" \
     bash -c '
set -uo pipefail
ROOT='"$ROOT"'
BALLS='"$BALLS"'
RUNS_JSONL='"$RUNS_JSONL"'
STAGES=",'"$STAGES"',"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$PY" -c "import torch;print(\"torch\",torch.__version__,torch.cuda.get_device_name(0))"

rc=0
want () { case "$STAGES" in *,"$1",*) return 0;; *) return 1;; esac; }
step () { echo; echo "=========== $* ==========="; }

if want 1; then
step "1. label mask still supervises exactly the answer"
"$PY" -m train.checks.check_labels --n 10 || rc=1
fi

if want 2; then
step "2a. left padding -- GTLM stack, graph balls (spd + magnetic)"
"$PY" -m train.checks.test_left_pad --data-root "$BALLS/v2_clean" \
      --batch-size 4 --max-length 2048 || rc=1

step "2b. left padding -- GTLM stack, serialised (the 16k-token arm)"
"$PY" -m train.checks.test_left_pad --data-root "$BALLS/v2_clean_serialised" \
      --no-spd --no-magnetic --batch-size 4 --max-length 17408 || rc=1

step "2c. left padding -- PLAIN stack, serialised"
"$PY" -m train.checks.test_left_pad --data-root "$BALLS/v2_clean_serialised" \
      --plain-llm --batch-size 4 --max-length 17408 || rc=1

step "2d. left padding -- PLAIN stack, no retrieval"
"$PY" -m train.checks.test_left_pad --data-root "$BALLS/v2_clean_noretrieval" \
      --plain-llm --batch-size 16 --max-length 2048 || rc=1

# The same comparison in fp32.  bf16 carries ~8 mantissa bits, so a rounding
# difference there is a few parts in a thousand of the loss and the bf16 test can
# only say "no bigger than the control".  fp32 shrinks BOTH the control and the
# measurement by orders of magnitude, so if left padding changed what the model
# computes rather than the order it accumulates in, this is where it shows.
step "2e. left padding in FP32 -- the sharp version of 2a"
"$PY" -m train.checks.test_left_pad --data-root "$BALLS/v2_clean" --dtype fp32 \
      --batch-size 4 --max-length 2048 || rc=1

step "2f. left padding in FP32 -- PLAIN stack (where RoPE angles also move)"
"$PY" -m train.checks.test_left_pad --data-root "$BALLS/v2_clean_noretrieval" \
      --plain-llm --dtype fp32 --batch-size 16 --max-length 2048 || rc=1
fi

if want 3; then
step "3. two-pass == generate-everything (untrained, spread over the modes)"
"$PY" -m train.checks.test_two_pass --data-root "$BALLS/v2_clean" \
      --max-items 32 --split dev || rc=1
fi

if want 4; then
step "4a. is the KV cache live in pass 2?  (GTLM stack)"
"$PY" -m train.checks.probe_eval --data-root "$BALLS/v2_clean" --n 32 || rc=1

# On the flex backend the uncached arm cannot run at all, which is decisive but
# gives no timing.  The plain stack has no block-alignment constraint, so the
# A/B is a real measurement there.
step "4b. the same question with a timing answer (PLAIN stack)"
"$PY" -m train.checks.probe_eval --data-root "$BALLS/v2_clean_serialised" \
      --plain-llm --max-length 17408 --n 32 || rc=1
fi

if want 5; then
step "5. smoke train run (24 items, 3 steps)"
"$PY" -m train --data-root "$BALLS/v2_clean" --max-items 24 --max-steps 3 \
      --eval-steps 2 --num-workers 0 --batch-size 1 --accumulation-steps 4 \
      --runs-jsonl "$RUNS_JSONL" || rc=1
fi

if want 6; then
step "6. overfit 32 items, so pass 1 actually fires"
rm -rf "$ROOT/checkpoints/sl_qa_overfit"
"$PY" -m train --data-root "$BALLS/v2_clean" --max-items 32 --num-epochs 40 \
      --batch-size 1 --accumulation-steps 4 --eval-steps 100 --no-final-eval \
      --num-workers 0 --runs-jsonl "$RUNS_JSONL" || rc=1
CKPT=$(ls -dt "$ROOT"/checkpoints/sl_qa/*/checkpoint-* 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
    step "6b. two-pass again, against the overfit checkpoint $CKPT"
    "$PY" -m train.checks.test_two_pass --data-root "$BALLS/v2_clean" --max-items 32 \
          --split train --no-spread --checkpoint "$CKPT" || rc=1
else
    echo "no checkpoint found; the pass-1 branch stays unexercised"; rc=1
fi
fi

echo; echo "preflight rc=$rc"
exit $rc
'
echo "srun exit: $?"
echo "end: $(date)"
