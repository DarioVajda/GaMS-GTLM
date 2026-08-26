#!/bin/bash
#
# Measure the collocation cap and sampler that build a ball: the fan-out it has
# to tame, what the cap costs in tokens, and how the log-damped weights compare
# with uniform and linear ones.  Backs QA_DATASET_DESIGN.md 3.1b / 3.1c.
#
#     sbatch run_measure_colloc_sampling.sbatch [STORE] [COMPARISON_STORE]
#
# With a second store it runs the identical measurement there too, so two stores
# can be separated by one variable rather than by "an older run".
#
#SBATCH --partition=frida
#SBATCH --account=lkm
# Ubuntu 22.04 / python3.10 nodes only -- see the root README.
#SBATCH --nodelist=aga,ana,apl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=colloc_sampling
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/colloc_sampling_%j.out

set -euo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
export PYTHONPATH=/shared/workspace/povejmo/graph_model/.venv/lib/python3.10/site-packages:${PYTHONPATH:-}
PY=$("$ROOT/data/lib/pick_python.sh")
STORE="${1:-$ROOT/data/stores/kg_graph_gemma3}"
COMPARE="${2:-}"

echo "host: $(hostname)  start: $(date)"
"$PY" "$ROOT/data/analysis/measure_colloc_sampling.py" "$STORE" \
      --out "$ROOT/data/analysis/results/colloc_sampling.json"

if [ -n "$COMPARE" ]; then
    echo "=============== comparison: $(basename "$COMPARE") ==============="
    "$PY" "$ROOT/data/analysis/measure_colloc_sampling.py" "$COMPARE" \
          --out "$ROOT/data/analysis/results/colloc_sampling_comparison.json"
fi
echo "end: $(date)"
