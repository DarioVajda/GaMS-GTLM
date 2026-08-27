#!/usr/bin/env python3
"""Are two stores byte-for-byte the same graph?

    python -m build.diff_stores NEW OLD

The companion to `check_store.py`, and deliberately a different question.
`check_store` asks whether a rebuild *lost* anything -- the IRI-backed layer
intact, the collocation layer only growing -- and tolerates a store that is
semantically equivalent but differently ordered.  This asks the stricter
question: are the bytes identical?

That matters because `data/README.md` claims the build is reproducible, and a
claim nothing checks is a claim that quietly stops being true.  Use this to
verify a rebuild after touching the builder; use `check_store` to decide whether
a rebuild that DOES differ is acceptable.

Exit 0 when every array, the text blob and the identity fields of the manifest
agree; 1 otherwise, naming what moved.  The manifest's `created` timestamp and
`builder_sha256` are excluded -- they are provenance, not content.
"""
import os
import sys
import json
import argparse

import numpy as np

ARRAYS = ("node_codes", "ntype", "kind", "mwe_set", "indptr", "indices",
          "token_len", "text_off")

# What the manifest says ABOUT the graph, as opposed to about the run that made
# it.  `created` moves every build and `builder_sha256` moves on a comment.
MANIFEST_CONTENT = ("n", "n_real", "n_edges", "n_half_edges", "n_textless",
                    "text_bytes", "graph_version", "format_version")
META_CONTENT = ("tokenizer", "kg_dir", "n_files", "files_limit",
                "sense_snippet", "sense_index", "colloc_text",
                "text_convention", "feature_props", "unit_props")


def _manifest(d):
    with open(os.path.join(d, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def run(new, old, verbose=True):
    """Compare two store directories.  0 if identical, 1 if not."""
    bad = []

    mn, mo = _manifest(new), _manifest(old)
    for k in MANIFEST_CONTENT:
        if mn.get(k) != mo.get(k):
            bad.append(f"manifest.{k}: {mn.get(k)!r} != {mo.get(k)!r}")
    for k in META_CONTENT:
        a, b = (mn.get("meta") or {}).get(k), (mo.get("meta") or {}).get(k)
        if a != b:
            bad.append(f"manifest.meta.{k}: {a!r} != {b!r}")

    for name in ARRAYS:
        pa = os.path.join(new, name + ".npy")
        pb = os.path.join(old, name + ".npy")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            if os.path.exists(pa) != os.path.exists(pb):
                bad.append(f"{name}.npy: present in only one store")
            continue
        a = np.load(pa, mmap_mode="r")
        b = np.load(pb, mmap_mode="r")
        if a.shape != b.shape:
            bad.append(f"{name}: shape {a.shape} != {b.shape}")
        elif not np.array_equal(np.asarray(a), np.asarray(b)):
            n_diff = int((np.asarray(a) != np.asarray(b)).sum())
            bad.append(f"{name}: {n_diff:,} of {a.size:,} elements differ")
        elif verbose:
            print(f"  ok  {name:12s} {a.shape}")

    # The blob is compared in chunks: it is ~2.8 GB and reading both whole would
    # cost 5.6 GB for a question answerable a megabyte at a time.
    pa = os.path.join(new, "text_blob.bin")
    pb = os.path.join(old, "text_blob.bin")
    sa, sb = os.path.getsize(pa), os.path.getsize(pb)
    if sa != sb:
        bad.append(f"text_blob: {sa:,} != {sb:,} bytes")
    else:
        off = 0
        with open(pa, "rb") as fa, open(pb, "rb") as fb:
            while True:
                ca, cb = fa.read(1 << 20), fb.read(1 << 20)
                if not ca:
                    break
                if ca != cb:
                    bad.append(f"text_blob: first difference near byte {off:,}")
                    break
                off += len(ca)
        if verbose and not any(x.startswith("text_blob") for x in bad):
            print(f"  ok  {'text_blob':12s} {sa:,} bytes")

    print()
    if bad:
        print(f"STORES DIFFER ({len(bad)}):")
        for b in bad:
            print(f"  {b}")
        return 1
    print("STORES ARE BYTE-IDENTICAL")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new")
    ap.add_argument("old")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    return run(args.new, args.old, verbose=not args.quiet)


if __name__ == "__main__":
    sys.exit(main())
