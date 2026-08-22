#!/bin/bash
#
# Extract one text graph per QA item -- the input a GTLM reads.
#
#     sbatch run_build_balls.sbatch [TYPES] [DATASET] [OUT]
#     sbatch run_build_balls.sbatch "" datasets/generated/v2_relabelled \
#                                      datasets/balls/v2
#
# The DATASET must be relabelled (qa/relabel.py): the builder reads each item's
# `targets` -- the anchors a real extractor run resolved to -- and refuses an
# item that has none, because falling back to the generator's own `node_code` is
# the privileged knowledge D3 forbids.  An empty TYPES means all of them.
#
# CPU only; the work is store traversal, one anchor at a time.
#
#SBATCH --partition=frida
# Ubuntu 22.04 / python3.10 nodes only -- see data/README.md "Reproduce".
#SBATCH --nodelist=aga,ana,apl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=qa_balls
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/qa_balls_%j.out

set -euo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
export PYTHONPATH=/shared/workspace/povejmo/graph_model/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}
PY=$("$ROOT/data/lib/pick_python.sh")

TYPES="${1:-}"
DATASET="${2:-$ROOT/data/datasets/generated/v2_relabelled}"
OUT="${3:-$ROOT/data/datasets/balls/v2}"

echo "host: $(hostname)  start: $(date)"
echo "types: ${TYPES:-<all>}"
cd "$ROOT/data"
"$PY" -m qa.build_balls "$DATASET" "$OUT" --types "$TYPES"
echo "end: $(date)"
