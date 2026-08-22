#!/bin/bash
#
# v7 store: the COLLOCATION DEDUP rebuild.  This is the first version bump since
# v3 that changes STRUCTURE and not only node text, and it is a bug fix, not a
# feature.
#
#   Each collocation is reified once per participant, so several
#   frac:Collocation IRIs share one {sense_a, sense_b} member set.  v2 introduced
#   a dedup on that member set (flaw 2) and it was correct at the time: the nodes
#   were textless and every duplicate rendered as the same lemma pair.  v5 gave
#   the nodes their curated phrase (flaw 8) and left the key alone -- and the
#   duplicates do NOT carry the same phrase.  Measured on the v6 store:
#
#       439,370 member sets (15.3%) name more than one dependent sense
#       717,545 phrases (20.0% of all curated phrases) were being dropped
#       `iziti` + `zbirka` has 8 curated phrases; the store kept 1
#
#   and the loss was systematic, not random: the survivor was the lowest IRI
#   code, i.e. the earliest-entered entry.
#
#   v7 keys the dedup on (member set, PHRASE) -- one node per distinct node
#   TEXT, which is the only distinction the model can make anyway.  Unresolved
#   reifications render as the lemma pair, a function of the member set alone, so
#   they still collapse to one node.
#
# Expected effect (simulated against the v6 store before the rebuild):
#   collocation nodes 2,981,731 -> 3,569,711 (+19.7%)
#   store nodes      36,735,791 -> 37,323,771 (+1.6%)
#   phrases per anchor p50 4 -> 5, p90 125 -> 143
#   reference-file reproduction 82.8% -> 87.9% of 1,307 phrases
#
# Usage:
#     sbatch run_save_v7.sbatch gemma3     # -> data/stores/kg_graph_v7_gemma3  (current)
#     sbatch run_save_v7.sbatch gams2b     # -> data/stores/kg_graph_v7_gams2b  (for diffing)
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
# triples still live as Python lists.  The v3.1 gemma3 run peaked at 70.7 GB;
# v7 adds a 3.5M-entry phrase intern table on top, ~0.5 GB.
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=01:30:00
#SBATCH --job-name=kg_save_v7
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/save_v7_%j.out

set -euo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
export VENV=/shared/workspace/povejmo/graph_model/.venv

WHICH="${1:-gemma3}"
case "$WHICH" in
    gemma3) TOKENIZER=cjvt/GaMS3-12B-Instruct ;;
    gams2b) TOKENIZER=cjvt/GaMS-2B ;;
    *) echo "usage: sbatch $0 {gemma3|gams2b}" >&2; exit 2 ;;
esac
STORE=$ROOT/data/stores/kg_graph_v7_$WHICH
REF=$ROOT/data/stores/kg_graph_v6_$WHICH     # same tokenizer, previous convention

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

# Acceptance: the real (IRI-backed) layer and the synonym/antonym layer must be
# untouched, the collocation layer must only GROW -- every v6 phrase still
# present on the same sense pair -- and no pair may carry the same phrase twice.
echo "--- acceptance: v7 vs v6 (same tokenizer) ---"
"$PY" $ROOT/data/build/check_v7_text.py "$STORE" "$REF"

echo "end: $(date)"
