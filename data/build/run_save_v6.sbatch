#!/bin/bash
#
# v6 store: the NOUN-GENDER rebuild.  Structure is byte-identical to v5 -- this
# build changes the text of noun ANCHOR nodes only, fixing flaw 9 in
# build_gtlm_graph.py (recorded as M4 in data/QA_TASKS.md):
#
#   This KG carries lexinfo:gender at two levels, on disjoint parts of speech.
#   On a word-form it is the agreement feature of an adjective or an -l
#   participle -- 3,242,639 triples, rendered since v3.  On a lexical-unit it is
#   the inherent gender of a NOUN -- 310,362 triples, rendered by nothing at all
#   until now, so `iztočnica: miza (samostalnik, imenovalnik, ednina)` never said
#   that miza is feminine and T9 was not generatable from the store.
#
#   QA_TASKS.md proposed a one-line fix, UNIT_PROPS = ("aspect","gender",
#   "clitic").  That fix does not work on its own: `gender` was already in
#   FEATURE_PROPS, so the parse branch's `if local in FEATURE_PROPS / elif local
#   in UNIT_PROPS` chain routed every noun-entry gender into the FORM branch,
#   where the T_WORDFORM guard dropped it -- and the elif could never run.  v6
#   dispatches on the SUBJECT TYPE and consults both sets.
#
#   Placement: the gender goes on the ANCHOR only, and the entry's form leaves
#   inherit it instead of repeating it.  Verified on the v5 store: 200,000 of
#   200,000 sampled form leaves have an anchor at hop 1, so no extracted ball can
#   contain a noun form without the node carrying its gender.  Repeating it would
#   have written the label onto 5,176,265 form leaves instead of 310,362 anchors
#   (x16.7).
#
#   Verified over all 401 *-words.nt (data/analysis/scan_gender.py, 2026-08-21):
#   310,362 of 310,362 nouns carry exactly one entry gender, none carries two,
#   and no other part of speech carries one at all.
#
# Effect: unblocks T9 (spol_samostalnika) entirely and T8's gender slot.
#
# Usage:
#     sbatch run_save_v6.sbatch gemma3     # -> data/stores/kg_graph_v6_gemma3  (current)
#     sbatch run_save_v6.sbatch gams2b     # -> data/stores/kg_graph_v6_gams2b  (for diffing)
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
#SBATCH --time=01:30:00
#SBATCH --job-name=kg_save_v6
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/save_v6_%j.out

set -euo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
export VENV=/shared/workspace/povejmo/graph_model/.venv

WHICH="${1:-gemma3}"
case "$WHICH" in
    gemma3) TOKENIZER=cjvt/GaMS3-12B-Instruct ;;
    gams2b) TOKENIZER=cjvt/GaMS-2B ;;
    *) echo "usage: sbatch $0 {gemma3|gams2b}" >&2; exit 2 ;;
esac
STORE=$ROOT/data/stores/kg_graph_v6_$WHICH
REF=$ROOT/data/stores/kg_graph_v5_$WHICH     # same tokenizer, previous text convention

if [ ! -d "$REF" ]; then
    echo "missing reference store $REF -- the acceptance check needs it" >&2
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
echo "ref  : $REF"
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

# Acceptance: structure identical to the v5 store of the SAME tokenizer, only
# noun ANCHOR text touched, the only added label a gender, and the form leaves
# left alone so the inheritance claim actually holds.  Standalone so it can be
# re-run against a built store without a rebuild.
echo "--- acceptance: v6 vs v5 (same tokenizer) ---"
"$PY" $ROOT/data/build/check_v6_text.py "$STORE" "$REF"

echo "end: $(date)"
