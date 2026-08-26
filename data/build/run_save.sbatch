#!/bin/bash
#
# Build the graph store: stream-parse the raw N-Triples, reshape them into the
# untyped text graph GTLM reads, tokenize every node text, and save the result as
# memory-mappable arrays.  ~20-35 min, ~70 GB peak RSS.
#
# Usage:
#     sbatch run_save.sbatch [gemma3|gams2b] [REFERENCE_STORE]
#
#     sbatch run_save.sbatch gemma3     # -> data/stores/kg_graph_gemma3  (current)
#     sbatch run_save.sbatch gams2b     # -> data/stores/kg_graph_gams2b  (for diffing)
#
# The two stores differ in `token_len` and nothing else.  `gemma3` is the
# tokenizer every Gemma 3 size shares with GaMS3-12B, so one store serves the
# whole iteration ladder; `gams2b` exists to diff tokenizers.
#
# The build always ends with graph_store.py --verify, which is self-contained:
# shapes, offsets and the manifest are checked against the store itself.
#
# REFERENCE_STORE is optional and is for REGRESSION testing only -- pass an
# existing store and check_text.py additionally asserts that the new build did
# not lose anything relative to it: the IRI-backed layer and the synonym/antonym
# layer untouched, the collocation layer only ever growing, no member set
# carrying one phrase twice.  A first build has nothing to compare against and
# simply skips it.
#
#SBATCH --partition=frida
#SBATCH --account=lkm
# Only aga/ana/apl still run Ubuntu 22.04 with /usr/bin/python3 = 3.10, which is
# what graph_model/.venv was built against; axa and every ix* node are 24.04 and
# ship 3.12 only, where the venv's site-packages are invisible.
#SBATCH --nodelist=aga,ana,apl
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Only the file parse is parallel, and CPUs past ~16 buy nothing.  Memory peaks
# in the BUILD phase, where parsed triples still live as Python lists: ~71 GB,
# plus ~0.5 GB for the 3.5M-entry phrase intern table.
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=01:30:00
#SBATCH --job-name=kg_save
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/save_%j.out

set -euo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
export VENV=/shared/workspace/povejmo/graph_model/.venv

WHICH="${1:-gemma3}"
case "$WHICH" in
    gemma3) TOKENIZER=cjvt/GaMS3-12B-Instruct ;;
    gams2b) TOKENIZER=cjvt/GaMS-2B ;;
    *) echo "usage: sbatch $0 {gemma3|gams2b} [REFERENCE_STORE]" >&2; exit 2 ;;
esac
STORE=$ROOT/data/stores/kg_graph_$WHICH
REF="${2:-}"                                 # optional regression reference

if [ -n "$REF" ] && [ ! -d "$REF" ]; then
    echo "reference store $REF does not exist" >&2
    exit 2
fi

# The graph_model venv lives on the shared filesystem but is NOT node-independent:
# .venv/bin/python is an absolute symlink to /usr/bin/python3, which is not 3.10
# on every node -- job 128295 died on `axa` with ModuleNotFoundError: numpy.
PY=$("$ROOT/data/lib/pick_python.sh")
if [ "$PY" != "$VENV/bin/python" ]; then
    export PYTHONPATH="$VENV/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
    echo "note: venv launcher unusable on $(hostname); using $PY + PYTHONPATH"
fi

export HF_HOME=/shared/workspace/povejmo/huggingface_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true   # no forks after tokenizer load -> safe & faster

echo "host: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}  variant: $WHICH  start: $(date)"
echo "store: $STORE"
echo "ref  : ${REF:-<none -- regression check skipped>}"
"$PY" -c "import sys, numpy, transformers; print('env OK ', sys.executable, sys.version.split()[0], 'numpy', numpy.__version__, 'transformers', transformers.__version__)"

"$PY" $ROOT/data/build/build_gtlm_graph.py \
  --kg-dir "$ROOT/data/kg_raw/OntoLex DSB" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --tokenizer "$TOKENIZER" \
  --colloc-text phrase \
  --save-graph "$STORE" \
  --no-analysis

echo "--- verifying the store ---"
"$PY" $ROOT/data/lib/graph_store.py "$STORE" --verify

echo "--- store contents ---"
ls -la "$STORE"
du -sh "$STORE"

if [ -n "$REF" ]; then
    echo "--- regression check against $(basename "$REF") ---"
    "$PY" $ROOT/data/build/check_text.py "$STORE" "$REF"
else
    echo "--- regression check skipped (no reference store given) ---"
fi

echo "end: $(date)"
