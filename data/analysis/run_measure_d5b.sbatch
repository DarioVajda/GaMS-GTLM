#!/bin/bash
#
# Re-measure the D5b constants (QA_DATASET_DESIGN.md 3.1b / 3.1c) against the v7
# store.  The published tables were taken on v5, whose collocation layer was
# missing 20 % of the curated phrases (README.md Finding 11), so every number
# that counts collocations moved.
#
#     sbatch run_measure_d5b.sbatch [STORE]
#
#SBATCH --partition=frida
# Ubuntu 22.04 / python3.10 nodes only -- see data/README.md "Reproduce".
#SBATCH --nodelist=aga,ana,apl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=d5b_measure
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/d5b_%j.out

set -euo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
export PYTHONPATH=/shared/workspace/povejmo/graph_model/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}
PY=$("$ROOT/data/lib/pick_python.sh")
STORE="${1:-$ROOT/data/stores/kg_graph_v7_gemma3}"

echo "host: $(hostname)  start: $(date)"
"$PY" "$ROOT/data/analysis/measure_d5b.py" "$STORE" \
      --out "$ROOT/data/analysis/results/d5b_v7.json"

# The same measurement on v6, so the v5-era published tables and the v7 ones are
# separated by ONE variable -- the dedup key -- rather than by "an older run".
echo "=============== control: the same measurement on v6 ==============="
"$PY" "$ROOT/data/analysis/measure_d5b.py" "$ROOT/data/stores/kg_graph_v6_gemma3" \
      --out "$ROOT/data/analysis/results/d5b_v6.json"
echo "end: $(date)"
