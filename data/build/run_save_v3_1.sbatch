#!/bin/bash
#
# v3.1: rebuild the store with two text fixes.  FLAW 5 -- sibling senses of a
# polysemous entry are no longer byte-identical (dictionary ordinal + a bounded
# first-example snippet where there is no definition).  FLAW 6 -- N-Triples
# literal escapes are decoded, so example text no longer carries a literal \".
#
# Writes a NEW store, data/stores/kg_graph_v3_1_gams2b -- named for the tokenizer that
# fills its token_len.  Superseded 2026-08-20 by run_save_v3_1_gemma3.sbatch,
# which rebuilds the identical graph with the Gemma 3 tokenizer that matches the
# model target; keep this script as the provenance of the _gams2b store.
#
# So the verified data/stores/kg_graph_v3 (whose store-backed rerun reproduces
# results_v3.json at MD5 11f4b10c7fbc1e4195795e896a285820) stays intact and the
# two can be compared node for node.
#
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
#SBATCH --job-name=kg_save_v3_1
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/save_v3_1_%j.out

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

STORE=$ROOT/data/stores/kg_graph_v3_1_gams2b

echo "host: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}  start: $(date)"
"$PY" -c "import sys, numpy, transformers; print('env OK ', sys.executable, sys.version.split()[0], 'numpy', numpy.__version__, 'transformers', transformers.__version__)"

# No GPU: this job is N-Triples parsing, dict building and a CPU (Rust) tokenizer.
# The only parallel stage is the file parse, which is already Pool-fanned over
# --workers; the CSR/mint/text stages are serial, so past ~64 CPUs there is
# nothing left to speed up.
#
# --no-analysis skips the 8 sizing variants (~11 min); results_v3.json already
# holds them and a store-backed rerun reproduces them byte-identically.
"$PY" $ROOT/data/build/build_gtlm_graph_v3.py \
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
G = graph_store.load_graph("/shared/workspace/povejmo/gams_gtlm/data/stores/kg_graph_v3_1_gams2b", verbose=False)
kind = np.asarray(G["kind"])
for k, nm in {0:"anchor",1:"form",2:"sense",3:"example",4:"translation",
              5:"collocation",6:"synonym",7:"antonym",8:"other"}.items():
    idx = np.flatnonzero(kind == k)[:3]
    print(f"--- {nm} ({int((kind==k).sum()):,}) ---")
    for i in idx:
        print("   ", G["text"][int(i)][:150])

# FLAW 6 acceptance check: no node text may carry an unresolved N-Triples escape.
esc = 0
for k_ in (2, 3):                      # sense snippets inherit example text
    idx = np.flatnonzero(kind == k_)
    idx = idx[:400000] if len(idx) > 400000 else idx
    esc += sum(1 for i in idx if '\\"' in G["text"][int(i)])
print(f"--- FLAW 6 check: {esc} sense/example nodes still contain a literal "
      f"backslash-quote (v3 had 3.48% of examples) ---")

# FLAW 5 acceptance check: sibling senses of a polysemous anchor must differ.
ip, ix, text = G["indptr"], G["indices"], G["text"]
anchors = np.flatnonzero(kind == 0)
rng = np.random.default_rng(0)
multi = dup = 0
for a in rng.choice(anchors, size=min(200000, len(anchors)), replace=False):
    ss = [v for v in ix[ip[a]:ip[a+1]] if kind[v] == 2]
    if len(ss) < 2:
        continue
    multi += 1
    ts = [text[v] for v in ss]
    if len(set(ts)) < len(ts):
        dup += 1
print(f"--- FLAW 5 check: {multi:,} polysemous anchors sampled, "
      f"{dup:,} still have duplicate sibling sense texts "
      f"({100.0*dup/max(multi,1):.2f}%; v3 was 97.7%) ---")
for a in rng.choice(anchors, size=len(anchors), replace=False):
    ss = [v for v in ix[ip[a]:ip[a+1]] if kind[v] == 2]
    if len(ss) >= 3:
        print("   ", text[a][:80])
        for v in ss[:4]:
            print("      ", text[v][:130])
        break
PYEOF

echo "end: $(date)"
