#!/bin/bash
#
# Stage 1 of the QA dataset build: generate the items from a graph store.
#
#     sbatch run_generate.sbatch [STORE] [OUTDIR]
#
# One stage on its own -- `sbatch data/run_pipeline.sbatch` runs all six.  The
# default OUTDIR is where the pipeline keeps this stage's output, so running
# this by hand and then continuing with the pipeline lines up.
#
# Three phases, all single-threaded Python over an mmap'd store:
#   1. the D8 seed pool -- every core anchor traversed to its senses and their
#      children (~100 k anchors)
#   2. the availability matrix -- every generator RUN over every pool entry, so
#      "this type is available for this lemma" is a fact rather than a prediction
#   3. generation, then the selftest (QA_TASKS.md section 2) over the result
#
#SBATCH --partition=frida
#SBATCH --account=lkm
# Ubuntu 22.04 / python3.10 nodes only -- see data/README.md "Reproduce".
#SBATCH --nodelist=aga,ana,apl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=qa_generate
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/qa_gen_%j.out

set -euo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
export PYTHONPATH=/shared/workspace/povejmo/graph_model/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}
PY=$("$ROOT/data/lib/pick_python.sh")

STORE="${1:-$ROOT/data/stores/kg_graph_gemma3}"
OUT="${2:-$ROOT/data/datasets/work/generated_raw}"

echo "host: $(hostname)  start: $(date)"
echo "store: $STORE"
echo "out  : $OUT"
cd "$ROOT/data"

"$PY" -m qa.build_dataset --store "$STORE" --out "$OUT"

echo "--- selftest (QA_TASKS.md section 2) ---"
"$PY" -m qa.selftest --dataset "$OUT" --store "$STORE"

echo "--- C9: the grader over the gold itself, per split ---"
for s in train dev test; do
    echo "== $s =="
    "$PY" -m qa.grade "$OUT/$s.jsonl"
done
echo "end: $(date)"
