#!/bin/bash
#
# v3.1 rebuilt with the GEMMA 3 TOKENIZER.  Same builder, same text, same graph
# as data/stores/kg_graph_v3_1_gams2b -- the only intended difference is token_len.
#
# Why: the model target is Gemma 3 / GaMS3-12B, but every store so far filled
# token_len from cjvt/GaMS-2B (Gemma 2, vocab 256,000).  Measured 2026-08-20 on
# 20k node texts, that mismatch is worth -0.9% in aggregate -- small, but there
# is no reason to keep carrying it.
#
# Verified 2026-08-20: gemma-3-270m/1b/4b/12b/27b (pt and it) and
# cjvt/GaMS3-12B-Instruct all encode real node texts to BYTE-IDENTICAL ids, not
# merely equal counts; tokenizer.model is byte-identical across all of them.
# So this one store serves the whole iteration ladder, small Gemma 3 through
# GaMS3-12B, and the repo named below is only a label for which of the
# interchangeable repos was loaded.
#
#SBATCH --partition=frida
# Only aga/ana/apl still run Ubuntu 22.04 with /usr/bin/python3 = 3.10, which is
# what graph_model/.venv was built against; axa and every ix* node are 24.04 and
# ship 3.12 only, where the venv's site-packages are invisible.  Probed
# 2026-08-18 -- see data/README.md "Reproduce".
#SBATCH --nodelist=aga,ana,apl
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Sized from the v3.1 run this copies: only the file parse is parallel, and
# memory peaks in the build phase where parsed triples live as Python lists
# (run 126763 peaked at 68.7 GB RSS).  90 GB is that peak plus ~30% headroom.
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=00:45:00
#SBATCH --job-name=kg_save_v3_1_gemma3
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/save_v3_1_gemma3_%j.out

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

TOKENIZER=cjvt/GaMS3-12B-Instruct
STORE=$ROOT/data/stores/kg_graph_v3_1_gemma3
REF=$ROOT/data/stores/kg_graph_v3_1_gams2b

echo "host: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}  start: $(date)"
"$PY" -c "import sys, numpy, transformers; print('env OK ', sys.executable, sys.version.split()[0], 'numpy', numpy.__version__, 'transformers', transformers.__version__)"

"$PY" $ROOT/data/build/build_gtlm_graph_v3.py \
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

# Acceptance: this build must differ from the GaMS-2B store in token_len and in
# nothing else.  Any difference in text_blob/codes/indptr means the rebuild was
# not reproducible and the two stores cannot be compared node for node.
#
# indices.npy is the ONE array compared as a multiset rather than byte for byte.
# The builder merges parsed files with pool.imap_unordered (build_gtlm_graph_v3
# ~line 419), so files arrive in completion order and the neighbours of a node
# that several files touch -- shared examples, shared forms -- are appended in a
# run-dependent order.  Measured on this pair: 92,945 rows of 36,735,791 (0.25%),
# 89,957 of them example nodes, every one with an identical neighbour SET.  Node
# ids are unaffected because codes come from the IRI, not from insertion order,
# which is what keeps two stores node-for-node comparable.
echo "--- acceptance: gemma3 store vs gams2b store ---"
STORE="$STORE" REF="$REF" TOKENIZER="$TOKENIZER" "$PY" - <<'PYEOF'
import hashlib, json, os
import numpy as np

new, ref = os.environ["STORE"], os.environ["REF"]
ok = True

def md5(p, blocks=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while (b := f.read(blocks)):
            h.update(b)
    return h.hexdigest()

for name in ("node_codes.npy", "ntype.npy", "kind.npy", "mwe_set.npy",
             "indptr.npy", "text_off.npy", "text_blob.bin"):
    a, b = md5(os.path.join(new, name)), md5(os.path.join(ref, name))
    same = a == b
    ok &= same
    print(f"    {name:16s} identical={same}  {a}")

# indices: same neighbours per node, order within a row not guaranteed (above).
ip = np.load(os.path.join(new, "indptr.npy"), mmap_mode="r")
ia = np.load(os.path.join(new, "indices.npy"), mmap_mode="r")
ib = np.load(os.path.join(ref, "indices.npy"), mmap_mode="r")
n = len(ip) - 1
rows_reordered = 0
for a0 in range(0, n, 4_000_000):
    b0 = min(a0 + 4_000_000, n)
    lo, hi = int(ip[a0]), int(ip[b0])
    xa, xb = np.asarray(ia[lo:hi]), np.asarray(ib[lo:hi])
    if np.array_equal(xa, xb):
        continue
    deg = np.diff(np.asarray(ip[a0:b0 + 1]))
    rid = np.repeat(np.arange(b0 - a0, dtype=np.int64), deg)
    ka = np.sort(rid * np.int64(n) + xa)
    kb = np.sort(rid * np.int64(n) + xb)
    if not np.array_equal(ka, kb):          # a real edge difference
        print(f"    indices.npy      EDGE SET DIFFERS in rows [{a0},{b0})")
        ok = False
    rows_reordered += len(np.unique(rid[xa != xb]))
print(f"    indices.npy      neighbour sets identical, "
      f"{rows_reordered:,} of {n:,} rows reordered ({100*rows_reordered/n:.2f}%)")

tn = np.load(os.path.join(new, "token_len.npy"))
tr = np.load(os.path.join(ref, "token_len.npy"))
print(f"    token_len        n={len(tn):,}  equal_elementwise={100*(tn==tr).mean():.2f}%")
print(f"    token_len total  gemma3={tn.sum():,}  gams2b={tr.sum():,}  "
      f"ratio={tn.sum()/tr.sum():.4f}")
print(f"    token_len mean   gemma3={tn.mean():.3f}  gams2b={tr.mean():.3f}")
print(f"    token_len zero   gemma3={int((tn==0).sum()):,}  gams2b={int((tr==0).sum()):,}")

meta = json.load(open(os.path.join(new, "manifest.json")))["meta"]
print(f"    manifest tokenizer = {meta['tokenizer']!r}")
ok &= meta["tokenizer"] == os.environ["TOKENIZER"]
ok &= bool((tn == 0).sum() == (tr == 0).sum())      # same textless nodes
ok &= 0.97 < tn.sum() / tr.sum() < 1.0              # expected ~0.991, never larger

print("ACCEPTANCE:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
PYEOF

echo "end: $(date)"
