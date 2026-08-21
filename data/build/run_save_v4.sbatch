#!/bin/bash
#
# v4 store: the TEXT-CONVENTION rebuild.  Structure is byte-identical to v3.1 --
# this build changes node text only, fixing flaw 7 in build_gtlm_graph.py:
#
#   (a) person was silently dropped from all 253,497 forms carrying it, because
#       VALUE_SL listed firstPerson/secondPerson/thirdPerson and the KG emits
#       first/second/third;
#   (b) FEATURE_PROPS listed "tense" and "mood", neither of which exists as a
#       predicate in this KG.  What carries that distinction is lexinfo:vform
#       (present / imperative / participle / infinitive / supine) -- 452,782
#       forms -- and it was not listed at all;
#   (c) aspect and clitic hang off the lexical-unit, but the feature branch only
#       accepted word-form subjects, so they were dropped even when listed.
#
#   definiteness (160,524 reachable forms) is mapped too: without it an
#   adjective's definite and indefinite forms carry byte-identical labels.
#
# Effect: the whole verb paradigm of a verb used to collapse to a handful of
# strings, with 2nd and 3rd person dual literally byte-identical.  Form nodes
# identical to a sibling fall 86,848 -> 418 (-99.5%, measured).  That matters
# specifically for GTLM: edges are untyped and the architecture is
# node-permutation equivariant, so two same-text sibling nodes are genuinely
# interchangeable once the extractor prunes their subtrees.
#
# Usage:
#     sbatch run_save_v4.sbatch gemma3     # -> data/stores/kg_graph_v4_gemma3  (current)
#     sbatch run_save_v4.sbatch gams2b     # -> data/stores/kg_graph_v4_gams2b  (for diffing)
#
#SBATCH --partition=frida
# Only aga/ana/apl still run Ubuntu 22.04 with /usr/bin/python3 = 3.10, which is
# what graph_model/.venv was built against; axa and every ix* node are 24.04 and
# ship 3.12 only, where the venv's site-packages are invisible.  Probed
# 2026-08-18 -- see data/README.md "Reproduce".
#SBATCH --nodelist=aga,ana,apl
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Only the file parse is parallel; memory peaks in the build phase, where parsed
# triples still live as Python lists.  The v3.1 gemma3 run peaked at 70.7 GB.
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --job-name=kg_save_v4
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/save_v4_%j.out

set -euo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
export VENV=/shared/workspace/povejmo/graph_model/.venv

WHICH="${1:-gemma3}"
case "$WHICH" in
    gemma3) TOKENIZER=cjvt/GaMS3-12B-Instruct ;;
    gams2b) TOKENIZER=cjvt/GaMS-2B ;;
    *) echo "usage: sbatch $0 {gemma3|gams2b}" >&2; exit 2 ;;
esac
STORE=$ROOT/data/stores/kg_graph_v4_$WHICH
REF=$ROOT/data/stores/kg_graph_v3_1_$WHICH   # same tokenizer, previous text convention

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
echo "ref  : $REF"
"$PY" -c "import sys, numpy, transformers; print('env OK ', sys.executable, sys.version.split()[0], 'numpy', numpy.__version__, 'transformers', transformers.__version__)"

"$PY" $ROOT/data/build/build_gtlm_graph.py \
  --kg-dir "$ROOT/data/kg_raw/OntoLex DSB" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --tokenizer "$TOKENIZER" \
  --save-graph "$STORE" \
  --no-analysis

echo "--- verifying the store ---"
"$PY" $ROOT/data/lib/graph_store.py "$STORE" --verify

echo "--- store contents ---"
ls -la "$STORE"
du -sh "$STORE"

# Acceptance: structure identical to the v3.1 store of the SAME tokenizer, text
# strictly additive, only anchors and forms touched, sibling duplicates down
# >=85%.  Standalone so it can be re-run against a built store without a rebuild.
echo "--- acceptance: v4 vs v3.1 (same tokenizer) ---"
"$PY" $ROOT/data/build/check_v4_text.py "$STORE" "$REF"

echo "end: $(date)"
