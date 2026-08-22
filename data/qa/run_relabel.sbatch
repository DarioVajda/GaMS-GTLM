#!/bin/bash
#
# Score a finished extraction run, then relabel the dataset against it.
#
#     sbatch qa/run_relabel.sbatch DATASET PREFIX OUT
#     sbatch qa/run_relabel.sbatch \
#         data/datasets/generated/v2 \
#         data/analysis/results/extraction_v2 \
#         data/datasets/generated/v2_relabelled
#
# CPU only: the work is the surface index plus one union lookup per item.
#
#SBATCH --partition=frida
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

DATASET="${1:-$ROOT/data/datasets/generated/v2}"
PREFIX="${2:-$ROOT/data/analysis/results/extraction_v2}"
OUT="${3:-$ROOT/data/datasets/generated/v2_relabelled}"

echo "host: $(hostname)  start: $(date)"
cd "$ROOT/data"

echo "--- merge the shards and score the whole run ---"
cat "${PREFIX}"_items.part*.jsonl > "${PREFIX}_items.jsonl"
wc -l < "${PREFIX}_items.jsonl"
"$PY" -m analysis.measure_extraction \
    --score-only "${PREFIX}_items.jsonl" --out "${PREFIX}.json"

echo "--- relabel ---"
"$PY" -m qa.relabel "$DATASET" "${PREFIX}_items.jsonl" "$OUT"

echo "--- selftest on the relabelled set ---"
"$PY" -m qa.selftest --dataset "$OUT" \
    --store "$ROOT/data/stores/kg_graph_v7_gemma3" || true

echo "end: $(date)"
