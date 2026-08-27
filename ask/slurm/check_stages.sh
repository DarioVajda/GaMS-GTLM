#!/usr/bin/env bash
# The `ask` acceptance stages, run INSIDE the container by run_checks.sbatch.
#
#   check_stages.sh 1,2,3,4,5
#
#   1  retrieval only     -- no model: the store opens, a ball is built, and the
#                            counts match what the corpus recorded for it.
#   2  parity, eager      -- ask's answer == evaluate's, item for item (D11).
#   3  parity, flex       -- the same on the backend the checkpoint was trained
#                            and evaluated under, plus the eager/flex drift the
#                            D17 fallback costs.
#   4  precompile         -- the (L, N) sweep, then an answer served on flex
#                            from the cache it just wrote.
#   5  the whole chain    -- the 12B extractor included: one question in, one
#                            grounded answer out, nothing supplied by hand.
#   6  the biggest ball   -- the test split's largest, 529 nodes / 11,421 tokens,
#                            on both backends.  Either it answers or D19 catches
#                            the OOM and says so; both are results, and a silent
#                            death is the only outcome that is not.
#
# Stage 5 loads 24 GB of extractor and stage 6 is the memory ceiling, so both are
# after the checks that say whether anything works at all.
set -uo pipefail

ROOT=/shared/workspace/povejmo/gams_gtlm
STAGES="${1:-1,2,3,4,5}"
PY="$ROOT/.venv/bin/python"
STORE="$ROOT/data/stores/kg_graph_gemma3"
Q="Navedi različne pomene besede brahialen."

export PYTHONPATH="$ROOT:$ROOT/data"
export HF_HOME=/shared/workspace/povejmo/huggingface_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# What bin/ask sets, and for the same reason: ball sizes vary by an order of
# magnitude within one process, so the allocator fragments and a large shape
# OOMs against memory that is free.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$ROOT"
has() { [[ ",$STAGES," == *",$1,"* ]]; }
rc=0
run() {
    echo; echo "======== $* ========"; echo
    "$@"; local s=$?
    echo "-------- exit $s --------"
    [ $s -ne 0 ] && rc=$s
    return 0
}

if has 1; then
    run "$PY" -m ask --store "$STORE" --retrieve-only --words brahialen "$Q"
    run "$PY" -m ask --store "$STORE" --retrieve-only --words plezalo \
        --json "Sklanjatev besede plezalo, prosim."
fi

if has 2; then
    run "$PY" -m ask.checks.check_parity --n 16 --impl eager
fi

if has 3; then
    run "$PY" -m ask.checks.check_parity --n 16 --impl flex --drift
fi

if has 4; then
    run "$PY" -m ask --store "$STORE" --words brahialen --precompile "$Q"
    # A second process, to prove the cache is what makes it warm rather than
    # anything left in memory.
    run "$PY" -m ask --store "$STORE" --words brahialen "$Q"
fi

if has 5; then
    run "$PY" -m ask --store "$STORE" "$Q"
    run "$PY" -m ask --store "$STORE" --json \
        "Katere sklanjatvene oblike ima folklora?"
fi

if has 6; then
    BIG="Oblika nebi pri besedi nebo — kateri sklon in katero število?"
    run "$PY" -m ask --store "$STORE" --words nebo --impl eager "$BIG"
    run "$PY" -m ask --store "$STORE" --words nebo --impl flex "$BIG"
fi

echo
echo "======== stages $STAGES done, exit $rc ========"
exit $rc
