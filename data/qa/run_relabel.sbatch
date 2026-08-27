#!/bin/bash
#
# Score a finished extraction run, then relabel the dataset against it.
#
#     sbatch qa/run_relabel.sbatch DATASET PREFIX OUT
#     sbatch qa/run_relabel.sbatch \
#         data/datasets/work/generated_raw \
#         data/datasets/work/extraction/items \
#         data/datasets/work/relabelled
#
# One stage on its own -- `sbatch data/run_pipeline.sbatch` runs all six.  PREFIX
# names the extraction dump without its `_items.jsonl` suffix; the defaults are
# where the pipeline puts things, so the two paths interoperate.
#
# CPU only: the work is the surface index plus one union lookup per item.
#
#SBATCH --partition=frida
#SBATCH --account=lkm
#SBATCH --nodelist=aga,ana,apl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=qa_relabel
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/relabel_%j.out

set -euo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
PY=$("$ROOT/data/lib/pick_python.sh")

DATASET="${1:-$ROOT/data/datasets/work/generated_raw}"
PREFIX="${2:-$ROOT/data/datasets/work/extraction/items}"
OUT="${3:-$ROOT/data/datasets/work/relabelled}"

echo "host: $(hostname)  start: $(date)"
cd "$ROOT/data"

echo "--- merge the shards and score the whole run ---"
# Only merge shards when there are shards to merge: run through the pipeline the
# dump is already one file, and a glob that matches nothing would truncate it.
if compgen -G "${PREFIX}.part*.jsonl" > /dev/null; then
    cat "${PREFIX}".part*.jsonl > "${PREFIX}.jsonl"
fi
wc -l < "${PREFIX}.jsonl"
"$PY" -m analysis.measure_extraction \
    --score-only "${PREFIX}.jsonl" --out "$(dirname "$PREFIX")/score.json"

echo "--- relabel ---"
"$PY" -m qa.relabel "$DATASET" "${PREFIX}.jsonl" "$OUT"

echo "--- selftest on the relabelled set ---"
# Reported, not swallowed: this used to end in `|| true`, which made a failing
# selftest invisible.  It still does not stop the stage -- the artefact is
# written either way and what to do about a failure is a judgement call -- but
# the outcome is now on the record.
if "$PY" -m qa.selftest --dataset "$OUT" \
        --store "$ROOT/data/stores/kg_graph_gemma3"; then
    echo "selftest: PASS"
else
    echo "selftest: FAIL -- the relabelled set is written, but read the log above"
fi

echo "end: $(date)"
