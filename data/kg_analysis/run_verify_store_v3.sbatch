#!/bin/bash
#SBATCH --partition=frida
# See run_save_v3.sbatch: only these nodes have the Python 3.10 the venv needs.
#SBATCH --nodelist=aga,ana,apl
#SBATCH --nodes=1
#SBATCH --ntasks=1
# The whole point of the store: this job re-runs the full 8-variant sizing study
# with no parse, no build and no tokenizer, so it needs 4 CPUs and a few GB
# instead of the 16 CPU / 65 GB the build peaked at.  The analysis loop is serial.
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --job-name=kg_verify_store
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/kg_analysis/verify_store_%j.out

set -euo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
export VENV=/shared/workspace/povejmo/graph_model/.venv
PY=$("$ROOT/data/kg_analysis/pick_python.sh")
if [ "$PY" != "$VENV/bin/python" ]; then
    export PYTHONPATH="$VENV/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "host: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}  start: $(date)"

# Same seed and settings as run 126763, but sourced entirely from the store.
"$PY" $ROOT/data/kg_analysis/build_gtlm_graph_v3.py \
  --load-graph "$ROOT/data/kg_graph_v3" \
  --out        "$ROOT/data/kg_analysis/results_v3_from_store.json" \
  --n-seeds 400 --max-hops 3 --seed 42

echo "--- diff against the published results_v3.json ---"
"$PY" - <<'PYEOF'
import json
R = "/shared/workspace/povejmo/gams_gtlm/data/kg_analysis/"
a = json.load(open(R + "results_v3.json"))
b = json.load(open(R + "results_v3_from_store.json"))
print("variants identical:", a["variants"] == b["variants"])
print("stats identical:   ", a["stats"] == b["stats"])
d = {k: (a["meta"].get(k), b["meta"].get(k)) for k in set(a["meta"]) | set(b["meta"])
     if a["meta"].get(k) != b["meta"].get(k)}
print("meta diffs:", d or "none")
PYEOF
echo "end: $(date)"
