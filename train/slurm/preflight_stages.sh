#!/bin/bash
#
# The preflight stages, run INSIDE the pyxis container by run_preflight.sbatch.
#
#     bash preflight_stages.sh STAGES
#
# A separate file rather than a `bash -c '…'` string in the sbatch: the stage
# commentary is prose, and an apostrophe in prose closes a single-quoted block.
# Same pattern as data/analysis/extract_in_container.sh.
set -uo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
BALLS="$ROOT/data/datasets/balls"
RUNS_JSONL="$ROOT/train/results/preflight_runs.jsonl"
STAGES=",${1:-1,2,3,4,5,6},"
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
"$PY" -m train.checks.check_left_pad --data-root "$BALLS" \
      --batch-size 4 --max-length 2048 || rc=1

step "2b. left padding -- GTLM stack, serialised (the 16k-token arm)"
"$PY" -m train.checks.check_left_pad --data-root "${BALLS}_serialised" \
      --no-spd --no-magnetic --batch-size 4 --max-length 17408 || rc=1

step "2c. left padding -- PLAIN stack, serialised"
"$PY" -m train.checks.check_left_pad --data-root "${BALLS}_serialised" \
      --plain-llm --batch-size 4 --max-length 17408 || rc=1

step "2d. left padding -- PLAIN stack, no retrieval"
"$PY" -m train.checks.check_left_pad --data-root "${BALLS}_noretrieval" \
      --plain-llm --batch-size 16 --max-length 2048 || rc=1

# The same comparison in fp32.  bf16 carries ~8 mantissa bits, so a rounding
# difference there is a few parts in a thousand of the loss and the bf16 test can
# only say "no bigger than the control".  fp32 shrinks BOTH the control and the
# measurement by orders of magnitude, so if left padding changed what the model
# computes rather than the order it accumulates in, this is where it shows.
step "2e. left padding in FP32 -- the sharp version of 2a"
"$PY" -m train.checks.check_left_pad --data-root "$BALLS" --dtype fp32 \
      --batch-size 4 --max-length 2048 || rc=1

step "2f. left padding in FP32 -- PLAIN stack (where RoPE angles also move)"
"$PY" -m train.checks.check_left_pad --data-root "${BALLS}_noretrieval" \
      --plain-llm --dtype fp32 --batch-size 16 --max-length 2048 || rc=1

# 2f alone does NOT cover the plain stack's real hazard.  Its sequences are
# L = 27..270 and Gemma-3's `sliding_window=512` never engages below that, so it
# checks left padding where the window CANNOT act.  The serialised arm is p50
# 2,166 / max ~14 k tokens, where 22 of 26 layers are windowed and the leading
# pads occupy part of that window -- a genuinely different computation from the
# right-padded one if anything is wrong.  2c (the bf16 version) is where the only
# preflight failure has ever landed, and bf16 cannot tell a real difference from
# its own rounding there; this stage can.
#
# The widest right-padded batches are the expensive part -- 113 GiB for the
# 13,973-token `mixed` one, measured -- which fits a B200's 183 GiB but would not
# fit a smaller card.  `check_left_pad` catches an OOM there, reports the row as
# `OOM` and carries on, so this stage still answers its question on a smaller
# GPU: the batches that decide it are the `long` ones at ~10 GiB.
step "2g. left padding in FP32 -- PLAIN stack, SERIALISED (the windowed case)"
"$PY" -m train.checks.check_left_pad --data-root "${BALLS}_serialised" \
      --plain-llm --dtype fp32 --batch-size 4 --max-length 17408 || rc=1
fi

if want 3; then
step "3. two-pass == generate-everything (untrained, spread over the modes)"
"$PY" -m train.checks.check_two_pass --data-root "$BALLS" \
      --max-items 32 --split dev || rc=1
fi

if want 4; then
step "4a. is the KV cache live in pass 2?  (GTLM stack)"
"$PY" -m train.checks.probe_eval --data-root "$BALLS" --n 32 || rc=1

# On the flex backend the uncached arm cannot run at all, which is decisive but
# gives no timing.  The plain stack has no block-alignment constraint, so the
# A/B is a real measurement there.
step "4b. the same question with a timing answer (PLAIN stack)"
"$PY" -m train.checks.probe_eval --data-root "${BALLS}_serialised" \
      --plain-llm --max-length 17408 --n 32 || rc=1
fi

if want 5; then
step "5. smoke train run (24 items, 3 steps)"
"$PY" -m train --data-root "$BALLS" --max-items 24 --max-steps 3 \
      --eval-steps 2 --num-workers 0 --batch-size 1 --accumulation-steps 4 \
      --runs-jsonl "$RUNS_JSONL" || rc=1
fi

if want 6; then
step "6. overfit 32 items, so pass 1 actually fires"
rm -rf "$ROOT/checkpoints/sl_qa_overfit"
"$PY" -m train --data-root "$BALLS" --max-items 32 --num-epochs 40 \
      --batch-size 1 --accumulation-steps 4 --eval-steps 100 --no-final-eval \
      --num-workers 0 --runs-jsonl "$RUNS_JSONL" || rc=1
CKPT=$(ls -dt "$ROOT"/checkpoints/sl_qa/*/checkpoint-* 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
    step "6b. two-pass again, against the overfit checkpoint $CKPT"
    "$PY" -m train.checks.check_two_pass --data-root "$BALLS" --max-items 32 \
          --split train --no-spread --checkpoint "$CKPT" || rc=1
else
    echo "no checkpoint found; the pass-1 branch stays unexercised"; rc=1
fi
fi

echo; echo "preflight rc=$rc"
exit $rc
