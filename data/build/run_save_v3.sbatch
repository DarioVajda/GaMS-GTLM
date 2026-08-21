#!/bin/bash
#SBATCH --partition=frida
# Only aga/ana/apl still run Ubuntu 22.04 with /usr/bin/python3 = 3.10, which is
# what graph_model/.venv was built against; axa and every ix* node are 24.04 and
# ship 3.12 only, where the venv's site-packages are invisible.  Probed
# 2026-08-18 -- see data/README.md "Reproduce".
#SBATCH --nodelist=aga,ana,apl
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Sized from measurement.  CPUs: only the file parse is parallel -- 60s of a
# 1426s run at 32 workers -- so 16 costs ~60s of parse and nothing else; 64 was
# pure waste and left job 128324 stuck behind a PLANNED reservation on aga/apl.
# Memory is the one ask that cannot shrink much: run 126763 peaked at 68.7 GB
# RSS, in the *build* phase, where the parsed triples live as Python lists
# (12.9M writtenRep + 12.1M value strings, 14.7M usage pairs, ...) before they
# are folded into numpy.  90 GB is that peak plus ~30% headroom; the save path
# adds only the 280 MB offsets array and a bounded ~1M-string encode chunk.
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=00:45:00
#SBATCH --job-name=kg_save_v3
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/save_v3_%j.out

set -euo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
export VENV=/shared/workspace/povejmo/graph_model/.venv

# The graph_model venv lives on the shared filesystem but is NOT node-independent:
# .venv/bin/python is an absolute symlink to /usr/bin/python3, which is not 3.10
# on every node -- job 128295 died on `axa` with ModuleNotFoundError: numpy.
# pick_python.sh resolves an interpreter that can actually import the venv here.
PY=$("$ROOT/data/lib/pick_python.sh")
if [ "$PY" != "$VENV/bin/python" ]; then
    export PYTHONPATH="$VENV/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
    echo "note: venv launcher unusable on $(hostname); using $PY + PYTHONPATH"
fi

export HF_HOME=/shared/workspace/povejmo/huggingface_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true   # no forks after tokenizer load -> safe & faster

STORE=$ROOT/data/stores/kg_graph_v3

echo "host: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}  start: $(date)"
"$PY" -c "import sys, numpy, transformers; print('env OK ', sys.executable, sys.version.split()[0], 'numpy', numpy.__version__, 'transformers', transformers.__version__)"

# No GPU: this job is N-Triples parsing, dict building and a CPU (Rust) tokenizer.
# The only parallel stage is the file parse, which is already Pool-fanned over
# --workers; the CSR/mint/text stages are serial, so past ~64 CPUs there is
# nothing left to speed up.
#
# --no-analysis skips the 8 sizing variants (~11 min); results_v3.json already
# holds them and a store-backed rerun reproduces them byte-identically.
"$PY" $ROOT/data/build/build_gtlm_graph.py \
  --kg-dir "$ROOT/data/kg_raw/OntoLex DSB" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --save-graph "$STORE" \
  --no-analysis

echo "--- verifying the store ---"
"$PY" $ROOT/data/lib/graph_store.py "$STORE" --verify

echo "--- store contents ---"
ls -la "$STORE"
du -sh "$STORE"

echo "--- sample node texts ---"
"$PY" $ROOT/data/lib/graph_store.py "$STORE" --samples 3 >/dev/null 2>&1 || true
"$PY" - <<'PYEOF'
import sys; sys.path.insert(0, "/shared/workspace/povejmo/gams_gtlm/data/lib")
import numpy as np, graph_store
G = graph_store.load_graph("/shared/workspace/povejmo/gams_gtlm/data/stores/kg_graph_v3", verbose=False)
kind = np.asarray(G["kind"])
for k, nm in {0:"anchor",1:"form",2:"sense",3:"example",4:"translation",
              5:"collocation",6:"synonym",7:"antonym",8:"other"}.items():
    idx = np.flatnonzero(kind == k)[:3]
    print(f"--- {nm} ({int((kind==k).sum()):,}) ---")
    for i in idx:
        print("   ", G["text"][int(i)][:150])
PYEOF

echo "end: $(date)"
