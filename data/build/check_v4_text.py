#!/usr/bin/env python3
"""Acceptance check for a v4 store against its v3.1 predecessor.

v4 changes node TEXT only (flaw 7 in build_gtlm_graph.py): vform / person /
definiteness are rendered on form nodes, aspect / clitic on anchors.  Structure
is untouched.  This script asserts exactly that, and measures the defect the
change exists to fix.

Four checks:

  1. STRUCTURE IDENTICAL.  node_codes / ntype / kind / mwe_set / indptr must be
     byte-identical; indices as a multiset per row (the builder merges parsed
     files with imap_unordered, so neighbour ORDER within a row is run-dependent
     -- see run_save_v3_1_gemma3.sbatch for the measurement).

  2. TEXT IS PURELY ADDITIVE.  No node's text may shrink, the surface (the part
     before " (") may never change, and the old parenthetical's items must be a
     SUBSEQUENCE of the new one's.  This is the strong form of "we only added
     labels": it fails if any existing label was reworded, reordered or dropped.

  3. ONLY ANCHORS AND FORMS CHANGED.  Senses, examples, collocations, synonyms
     and translations must be untouched.

  4. THE DEFECT IS FIXED.  Form nodes whose text is byte-identical to a sibling
     form of the same anchor must fall by >=85%.

Usage:  check_v4_text.py NEW_STORE OLD_STORE
"""
import os, sys, json, hashlib, collections
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import graph_store

K_ANCHOR, K_FORM = 0, 1
KIND_NAME = {0: "anchor", 1: "form", 2: "sense", 3: "example", 4: "translation",
             5: "collocation", 6: "synonym", 7: "antonym", 8: "other"}
DUP_DROP_MIN = 0.85


def md5(path, blocks=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while (b := f.read(blocks)):
            h.update(b)
    return h.hexdigest()


def check_structure(new, old):
    ok = True
    print("--- 1. structure: v4 must change text only ---")
    for name in ("node_codes.npy", "ntype.npy", "kind.npy", "mwe_set.npy",
                 "indptr.npy"):
        a, b = md5(os.path.join(new, name)), md5(os.path.join(old, name))
        same = a == b
        ok &= same
        print(f"    {name:16s} identical={same}  {a}")

    ip = np.load(os.path.join(new, "indptr.npy"), mmap_mode="r")
    ia = np.load(os.path.join(new, "indices.npy"), mmap_mode="r")
    ib = np.load(os.path.join(old, "indices.npy"), mmap_mode="r")
    n = len(ip) - 1
    reordered = 0
    for a0 in range(0, n, 4_000_000):
        b0 = min(a0 + 4_000_000, n)
        lo, hi = int(ip[a0]), int(ip[b0])
        xa, xb = np.asarray(ia[lo:hi]), np.asarray(ib[lo:hi])
        if np.array_equal(xa, xb):
            continue
        deg = np.diff(np.asarray(ip[a0:b0 + 1]))
        rid = np.repeat(np.arange(b0 - a0, dtype=np.int64), deg)
        if not np.array_equal(np.sort(rid * np.int64(n) + xa),
                              np.sort(rid * np.int64(n) + xb)):
            print(f"    indices.npy      EDGE SET DIFFERS in rows [{a0},{b0})")
            ok = False
        reordered += len(np.unique(rid[xa != xb]))
    print(f"    indices.npy      neighbour sets identical, {reordered:,} of "
          f"{n:,} rows reordered ({100 * reordered / n:.2f}%)")
    return ok


def paren_split(s):
    """'oblika: X (a, b)' -> ('oblika: X', ['a','b']).  No paren -> (s, [])."""
    i = s.rfind(" (")
    if i < 0 or not s.endswith(")"):
        return s, []
    return s[:i], s[i + 2:-1].split(", ")


def is_subsequence(small, big):
    it = iter(big)
    return all(x in it for x in small)


def check_text(tn, to, kind):
    print("\n--- 2/3. text diff: additive, anchors and forms only ---")
    n = len(kind)
    changed = collections.Counter()
    bad_shrink = bad_surface = bad_order = bad_kind = 0
    examples = []
    for i in range(n):
        a = to[i]
        b = tn[i]
        if a == b:
            continue
        k = int(kind[i])
        changed[k] += 1
        if k not in (K_ANCHOR, K_FORM):
            bad_kind += 1
            if bad_kind <= 3:
                print(f"    !! kind={KIND_NAME.get(k, k)} node {i} changed:\n"
                      f"       old {a!r}\n       new {b!r}")
            continue
        if len(b) < len(a):
            bad_shrink += 1
            if bad_shrink <= 3:
                print(f"    !! node {i} got SHORTER:\n       old {a!r}\n       new {b!r}")
            continue
        ha, pa = paren_split(a)
        hb, pb = paren_split(b)
        if ha != hb:
            bad_surface += 1
            if bad_surface <= 3:
                print(f"    !! node {i} surface changed:\n       old {ha!r}\n       new {hb!r}")
            continue
        if not is_subsequence(pa, pb):
            bad_order += 1
            if bad_order <= 3:
                print(f"    !! node {i} label set not additive:\n"
                      f"       old {pa}\n       new {pb}")
            continue
        if len(examples) < 8 and pa != pb:
            examples.append((a, b))

    total = sum(changed.values())
    print(f"    nodes with changed text : {total:,} of {n:,} "
          f"({100 * total / n:.2f}%)")
    for k, c in sorted(changed.items()):
        print(f"      {KIND_NAME.get(k, k):12s} {c:>10,}")
    print(f"    violations: shrank={bad_shrink:,}  surface changed={bad_surface:,}  "
          f"not additive={bad_order:,}  wrong kind={bad_kind:,}")
    print("    sample of what changed:")
    for a, b in examples:
        print(f"      - {a}\n      + {b}")
    return (bad_shrink == 0 and bad_surface == 0 and bad_order == 0
            and bad_kind == 0 and total > 0)


def sibling_dups(G, kind, label):
    """Form nodes whose text is byte-identical to another form of the SAME anchor."""
    indptr, indices, text = G["indptr"], G["indices"], G["text"]
    anchors = np.flatnonzero(kind == K_ANCHOR)
    dup_nodes = dup_anchors = 0
    for a in anchors:
        lo, hi = int(indptr[a]), int(indptr[a + 1])
        nb = indices[lo:hi]
        nb = nb[kind[nb] == K_FORM]
        if nb.shape[0] < 2:
            continue
        c = collections.Counter(text[int(j)] for j in nb)
        d = sum(v for v in c.values() if v > 1)
        if d:
            dup_nodes += d
            dup_anchors += 1
    print(f"    {label:6s} form nodes identical to a sibling: {dup_nodes:>9,}  "
          f"(anchors affected {dup_anchors:,})")
    return dup_nodes


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    new, old = sys.argv[1], sys.argv[2]
    print(f"new: {new}\nold: {old}\n")

    meta = json.load(open(os.path.join(new, "manifest.json")))["meta"]
    print(f"manifest text_convention = {meta.get('text_convention')!r}  "
          f"tokenizer = {meta.get('tokenizer')!r}")
    ok = meta.get("text_convention") == "v4"
    if not ok:
        print("    !! new store does not declare text_convention v4")

    kind = np.load(os.path.join(new, "kind.npy"))
    ok &= check_structure(new, old)

    Gn = graph_store.load_graph(new, verbose=False)
    Go = graph_store.load_graph(old, verbose=False)
    ok &= check_text(Gn["text"], Go["text"], kind)

    print("\n--- 4. the defect this change exists to fix ---")
    d_old = sibling_dups(Go, kind, "before")
    d_new = sibling_dups(Gn, kind, "after")
    if d_old:
        drop = 1 - d_new / d_old
        print(f"    reduction: {100 * drop:.1f}%  (required >= {100 * DUP_DROP_MIN:.0f}%)")
        ok &= drop >= DUP_DROP_MIN
    else:
        print("    !! old store reports no sibling duplicates -- check is broken")
        ok = False

    print("\nACCEPTANCE:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
