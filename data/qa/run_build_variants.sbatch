#!/bin/bash
#
# The two baseline INPUT variants, derived from the GTLM arm's own ball files.
#
#     sbatch run_build_variants.sbatch [BALLS] [OUT_ROOT]
#
# Pure file transform -- no store, no GPU.  It reads datasets/balls/v2 and writes
# datasets/balls/v2_noretrieval and datasets/balls/v2_serialised, then verifies
# that every id, split and answer matches the reference.
#
#SBATCH --partition=frida
#SBATCH --account=lkm
# Ubuntu 22.04 / python3.10 nodes only -- see data/README.md "Reproduce".
#SBATCH --nodelist=aga,ana,apl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --job-name=qa_variants
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/qa_variants_%j.out

set -euo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
export PYTHONPATH=/shared/workspace/povejmo/graph_model/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
PY=$("$ROOT/data/lib/pick_python.sh")

BALLS="${1:-$ROOT/data/datasets/balls/v2}"
OUT_ROOT="${2:-$ROOT/data/datasets/balls}"

echo "host: $(hostname)  start: $(date)"
cd "$ROOT/data"
"$PY" -m qa.build_variants "$BALLS" "$OUT_ROOT"
"$PY" -m qa.check_variants "$BALLS" \
      "${BALLS}_noretrieval" "${BALLS}_serialised"
echo "end: $(date)"
