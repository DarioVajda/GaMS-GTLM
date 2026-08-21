#!/bin/bash
#
# v5 store: the COLLOCATION-VERBALISATION rebuild.  Structure is byte-identical
# to v4 -- this build changes the text of collocation nodes only, fixing flaw 8
# in build_gtlm_graph.py:
#
#   v2/v3/v4 wrote a collocation as its two member lemmas joined by a plus
#       kolokacija: kisov + voda
#   on the recorded belief that "the inflected phrase is genuinely not stored".
#   It is stored.  A frac:Collocation node takes only rdfs:member / frac:head /
#   rdf:type in all 42 GB, and nothing in the dump ever points AT one (0 triples
#   with a collocation in the object position), so no traversal reaches the text
#   from either side.  The link is a NAMING CONVENTION:
#
#       <dependent-sense-D-lexical-unit-H> rdf:type frac:Collocation
#       <sense-D> ontolex:isSenseOf <lexical-unit-M>   (M a MultiWordExpression)
#       <lexical-unit-M> canonicalForm/writtenRep -> "kisova voda"
#
#   D names the sense of the multi-word entry that spells the pairing out, and
#   sense_lu / canon_of / wr already hold every hop.  code_of() was in fact
#   already parsing D out of that IRI to pack the node id -- using it as an
#   identifier ingredient, never dereferencing it.
#
#   Measured over the raw dump: 4,717,090 of 4,717,090 collocation nodes resolve,
#   3,744,473 distinct phrases, 75.3% differing from their constituent lemmas
#   concatenated.  91.5% of the 1,307 collocation phrases in the reference QA
#   file come back verbatim from this path alone.
#
# Effect: the 2.9M collocation nodes stop carrying a surface nobody would ever
# produce.  This is what unparks Group F of the QA dataset (data/QA_TASKS.md).
#
# Usage:
#     sbatch run_save_v5.sbatch gemma3     # -> data/stores/kg_graph_v5_gemma3  (current)
#     sbatch run_save_v5.sbatch gams2b     # -> data/stores/kg_graph_v5_gams2b  (for diffing)
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
#SBATCH --job-name=kg_save_v5
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/save_v5_%j.out

set -euo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
export VENV=/shared/workspace/povejmo/graph_model/.venv

WHICH="${1:-gemma3}"
case "$WHICH" in
    gemma3) TOKENIZER=cjvt/GaMS3-12B-Instruct ;;
    gams2b) TOKENIZER=cjvt/GaMS-2B ;;
    *) echo "usage: sbatch $0 {gemma3|gams2b}" >&2; exit 2 ;;
esac
STORE=$ROOT/data/stores/kg_graph_v5_$WHICH
REF=$ROOT/data/stores/kg_graph_v4_$WHICH     # same tokenizer, previous text convention

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

# Acceptance: structure identical to the v4 store of the SAME tokenizer, only
# collocation text touched, the pair form essentially gone, and each phrase
# anchored in the members it replaced.  Standalone so it can be re-run against a
# built store without a rebuild.
echo "--- acceptance: v5 vs v4 (same tokenizer) ---"
"$PY" $ROOT/data/build/check_v5_text.py "$STORE" "$REF"

echo "end: $(date)"
