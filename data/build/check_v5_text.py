#!/usr/bin/env python3
"""Acceptance check for a v5 store against its v4 predecessor.

v5 changes the text of COLLOCATION nodes only (flaw 8 in build_gtlm_graph.py).
A frac:Collocation node carries no text of its own and nothing in the dump points
at one, so v2/v3/v4 wrote the two member lemmas joined by a plus:

    kolokacija: kisov + voda

The phrase is in the export after all -- on the multi-word entry whose sense id
is embedded in the collocation's own IRI, a link that exists as a naming
convention and never as a triple.  v5 dereferences it:

    kolokacija: kisova voda

The skip condition stays on the members, so node count and order are unchanged
and the graph structure must be byte-identical to the v4 store.

Five checks:

  1. STRUCTURE IDENTICAL.  node_codes / ntype / kind / mwe_set / indptr must be
     byte-identical; indices as a multiset per row (the builder merges parsed
     files with imap_unordered, so neighbour ORDER within a row is run-dependent).

  2. ONLY COLLOCATIONS CHANGED.  Anchors, forms, senses, examples, synonyms,
     antonyms and translations must be untouched.

  3. THE PAIR FORM IS GONE.  Collocation nodes still carrying " + " must be a
     residue, not the rule.

  4. THE PHRASE IS ANCHORED IN ITS MEMBERS.  A verbalised node's phrase must
     share a stem with at least one of the two lemmas it replaced -- this is what
     rules out the text having been resolved through the wrong entry.

  5. THE PHRASE IS NOT THE PAIR RESPELLED.  A large share must differ from the
     two lemmas concatenated, otherwise nothing was actually gained.

Usage:  check_v5_text.py NEW_STORE OLD_STORE
"""
import os, sys, json, hashlib, collections, re
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import graph_store

K_COLLOC = 5
KIND_NAME = {0: "anchor", 1: "form", 2: "sense", 3: "example", 4: "translation",
             5: "collocation", 6: "synonym", 7: "antonym", 8: "other"}
TAG = "kolokacija: "
PAIR_MIN = 0.99        # verbalised share required
ANCHOR_MIN = 0.95      # phrase shares a stem with a member lemma
INFLECT_MIN = 0.50     # phrase differs from the lemmas concatenated
STEM = 4


def md5(path, blocks=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while (b := f.read(blocks)):
            h.update(b)
    return h.hexdigest()


def check_structure(new, old):
    ok = True
    print("--- 1. structure: v5 must change text only ---")
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


def toks(s):
    return [t.lower() for t in re.findall(r"[^\W\d_]+", s, re.UNICODE)]


def check_text(tn, to, kind):
    print("\n--- 2/3/4/5. text diff ---")
    n = len(kind)
    changed = collections.Counter()
    bad_kind = 0
    n_col = n_pair_new = n_anchored = n_inflected = 0
    examples = []
    for i in range(n):
        a, b = to[i], tn[i]
        k = int(kind[i])
        if k == K_COLLOC:
            n_col += 1
            if " + " in b:
                n_pair_new += 1
            elif a.startswith(TAG) and b.startswith(TAG):
                lemmas = [x.strip() for x in a[len(TAG):].split(" + ")]
                phrase = b[len(TAG):]
                pt = toks(phrase)
                if any(any(t.startswith(l.lower()[:STEM]) or
                           l.lower().startswith(t[:STEM]) for t in pt)
                       for l in lemmas if l):
                    n_anchored += 1
                if phrase.lower() != " ".join(l.lower() for l in lemmas):
                    n_inflected += 1
                if len(examples) < 10 and a != b:
                    examples.append((a, b))
        if a == b:
            continue
        changed[k] += 1
        if k != K_COLLOC:
            bad_kind += 1
            if bad_kind <= 3:
                print(f"    !! kind={KIND_NAME.get(k, k)} node {i} changed:\n"
                      f"       old {a!r}\n       new {b!r}")

    total = sum(changed.values())
    print(f"    nodes with changed text : {total:,} of {n:,} ({100 * total / n:.2f}%)")
    for k, c in sorted(changed.items()):
        print(f"      {KIND_NAME.get(k, k):12s} {c:>10,}")
    print(f"    violations: wrong kind={bad_kind:,}")

    verb = n_col - n_pair_new
    ok = bad_kind == 0 and total > 0
    print(f"\n    collocation nodes            : {n_col:,}")
    for label, num, thr in (
            ("verbalised (no ' + ')", verb, PAIR_MIN),
            ("anchored in a member lemma", n_anchored, ANCHOR_MIN),
            ("differs from lemmas concat", n_inflected, INFLECT_MIN)):
        frac = num / max(n_col, 1)
        good = frac >= thr
        ok &= good
        print(f"    {label:29}: {num:>10,}  ({100 * frac:5.2f}%, "
              f"required >= {100 * thr:.0f}%)  {'ok' if good else 'FAIL'}")
    print(f"    still on the pair form       : {n_pair_new:,}")

    print("\n    sample of what changed:")
    for a, b in examples:
        print(f"      - {a}\n      + {b}")
    return ok


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    new, old = sys.argv[1], sys.argv[2]
    print(f"new: {new}\nold: {old}\n")

    meta = json.load(open(os.path.join(new, "manifest.json")))["meta"]
    print(f"manifest text_convention = {meta.get('text_convention')!r}  "
          f"colloc_text = {meta.get('colloc_text')!r}  "
          f"tokenizer = {meta.get('tokenizer')!r}")
    ok = meta.get("text_convention") == "v5"
    if not ok:
        print("    !! new store does not declare text_convention v5")

    kind = np.load(os.path.join(new, "kind.npy"))
    ok &= check_structure(new, old)

    Gn = graph_store.load_graph(new, verbose=False)
    Go = graph_store.load_graph(old, verbose=False)
    ok &= check_text(Gn["text"], Go["text"], kind)

    print("\nACCEPTANCE:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
