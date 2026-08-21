#!/usr/bin/env python3
"""Acceptance check for a v6 store against its v5 predecessor.

v6 changes the text of NOUN ANCHOR nodes only (flaw 9 / M4 in
build_gtlm_graph.py): entry-level `lexinfo:gender`, which this KG hangs off the
lexical-unit for nouns and off the word-form for adjectives and participles,
now reaches the anchor parenthetical:

    iztočnica: miza (samostalnik, imenovalnik, ednina)
 -> iztočnica: miza (samostalnik, ženski spol, imenovalnik, ednina)

The gender is rendered on the anchor ONLY; the entry's form leaves inherit it
from one hop away and must be untouched.

Six checks:

  1. STRUCTURE IDENTICAL.  node_codes / ntype / kind / mwe_set / indptr must be
     byte-identical; indices as a multiset per row (the builder merges parsed
     files with imap_unordered, so neighbour ORDER within a row is run-dependent
     -- see run_save_v3_1_gemma3.sbatch for the measurement).

  2. TEXT IS PURELY ADDITIVE.  No node's text may shrink, the surface (the part
     before " (") may never change, and the old parenthetical's items must be a
     SUBSEQUENCE of the new one's -- so nothing was reworded, reordered or lost.

  3. ONLY ANCHORS CHANGED.  Forms, senses, examples, collocations, synonyms and
     translations must be byte-identical.  Note this is STRICTER than v4's
     check, which allowed forms to change too: v6 must not touch a form node,
     and in particular must not start repeating the gender down the paradigm.
     This is the narrowing that data/QA_TASKS.md M4 called for -- v4's "nominal
     strings are byte-identical to v3.1" assertion moves onto form nodes, which
     is where it still holds.

  4. EVERY CHANGED ANCHOR IS A NOUN, AND THE ONLY ADDITION IS A GENDER.  The
     added labels must be exactly one of moški/ženski/srednji spol, and the
     anchor must carry the samostalnik POS tag.  This is what rules out the
     dispatch rewrite having leaked an entry-level property onto some other POS.

  5. COVERAGE.  Essentially every noun anchor must now carry a gender: the scan
     of the raw dump (data/analysis/scan_gender.py) found 310,362 of 310,362
     noun units with exactly one, so anything materially short of the noun
     anchor count means the parse is still dropping some.

  6. FORM LEAVES DID NOT INHERIT THE LABEL BY COPY.  Noun form leaves must NOT
     contain a gender label -- if they do, the change was implemented as
     repetition rather than as inheritance, which is the cost this design exists
     to avoid.  (Adjective and participle forms legitimately carry one; they are
     excluded by only looking at forms hanging off a noun anchor.)

Usage:  check_v6_text.py NEW_STORE OLD_STORE
"""
import os, sys, json, hashlib, collections
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import graph_store

K_ANCHOR, K_FORM = 0, 1
KIND_NAME = {0: "anchor", 1: "form", 2: "sense", 3: "example", 4: "translation",
             5: "collocation", 6: "synonym", 7: "antonym", 8: "other"}

GENDERS = ("moški spol", "ženski spol", "srednji spol")
NOUN_TAG = "samostalnik"
# Measured 1.000000 on kg_graph_v6_gemma3: every noun entry in the export has a
# gender and every noun anchor renders it, including the anchors recovered from
# an inflected form when the canonical form has no @sl writtenRep.  Demanding
# literally all of them is therefore the honest threshold -- anything less would
# let a future regression hide in the slack.
COVERAGE_MIN = 1.0


def md5(path, blocks=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while (b := f.read(blocks)):
            h.update(b)
    return h.hexdigest()


def check_structure(new, old):
    ok = True
    print("--- 1. structure: v6 must change text only ---")
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
    print("\n--- 2/3/4. text diff: additive, noun anchors only, gender only ---")
    n = len(kind)
    changed = collections.Counter()
    bad_shrink = bad_surface = bad_order = bad_kind = 0
    bad_added = bad_notnoun = 0
    added_hist = collections.Counter()
    examples = []
    for i in range(n):
        a = to[i]
        b = tn[i]
        if a == b:
            continue
        k = int(kind[i])
        changed[k] += 1
        if k != K_ANCHOR:
            bad_kind += 1
            if bad_kind <= 5:
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
        # exactly one label added, and it is a gender
        added = [x for x in pb if x not in pa]
        added_hist["+".join(added)] += 1
        if len(added) != 1 or added[0] not in GENDERS:
            bad_added += 1
            if bad_added <= 5:
                print(f"    !! node {i} added something other than a gender: "
                      f"{added}\n       old {a!r}\n       new {b!r}")
            continue
        if NOUN_TAG not in pb:
            bad_notnoun += 1
            if bad_notnoun <= 5:
                print(f"    !! node {i} gained a gender but is not a noun:\n"
                      f"       new {b!r}")
            continue
        if len(examples) < 8:
            examples.append((a, b))

    total = sum(changed.values())
    print(f"    nodes with changed text : {total:,} of {n:,} "
          f"({100 * total / n:.2f}%)")
    for k, c in sorted(changed.items()):
        print(f"      {KIND_NAME.get(k, k):12s} {c:>10,}")
    print(f"    labels added: " +
          ", ".join(f"{v or '(none)'}={c:,}" for v, c in added_hist.most_common(6)))
    print(f"    violations: shrank={bad_shrink:,}  surface changed={bad_surface:,}  "
          f"not additive={bad_order:,}  wrong kind={bad_kind:,}  "
          f"non-gender addition={bad_added:,}  not a noun={bad_notnoun:,}")
    print("    sample of what changed:")
    for a, b in examples:
        print(f"      - {a}\n      + {b}")
    ok = (bad_shrink == 0 and bad_surface == 0 and bad_order == 0
          and bad_kind == 0 and bad_added == 0 and bad_notnoun == 0
          and total > 0)
    return ok, total


def check_coverage(G, kind, n_changed):
    print("\n--- 5/6. coverage, and forms must NOT repeat the label ---")
    text = G["text"]
    indptr = np.asarray(G["indptr"])
    indices = np.asarray(G["indices"])
    anchors = np.flatnonzero(kind == K_ANCHOR)

    # A noun anchor is one whose PARENTHETICAL carries the samostalnik POS tag.
    # Testing `"samostalnik" in text` instead matches the LEMMA and sweeps in 68
    # multi-word grammatical-terminology entries -- "abstraktni samostalnik",
    # "spol samostalnika", "moški samostalniki" -- which are MWEs, carry no
    # partOfSpeech at all (no multi file has one), and so have no parenthetical
    # to hold a gender.  That substring test reported 310,430 noun anchors and a
    # spurious 99.978% coverage on a store that is in fact complete.
    n_noun = n_noun_gender = 0
    noun_anchor_ids = []
    for i in anchors:
        t = text[int(i)]
        parts = paren_split(t)[1]
        if NOUN_TAG not in parts:
            continue
        n_noun += 1
        noun_anchor_ids.append(int(i))
        if any(g in parts for g in GENDERS):
            n_noun_gender += 1
    cov = n_noun_gender / max(n_noun, 1)
    print(f"    noun anchors: {n_noun:,}")
    print(f"      carrying a gender: {n_noun_gender:,} ({100 * cov:.3f}%)  "
          f"(required >= {100 * COVERAGE_MIN:.0f}%)")
    print(f"      anchors changed vs v5: {n_changed:,}")
    ok = cov >= COVERAGE_MIN

    # form leaves of noun anchors must carry no gender label
    rng = np.random.default_rng(0)
    sample = rng.choice(np.asarray(noun_anchor_ids),
                        size=min(50_000, len(noun_anchor_ids)), replace=False)
    n_forms = n_forms_gender = 0
    shown = 0
    for i in sample:
        i = int(i)
        for j in indices[indptr[i]:indptr[i + 1]]:
            j = int(j)
            if kind[j] != K_FORM:
                continue
            n_forms += 1
            tj = text[j]
            if any(g in paren_split(tj)[1] for g in GENDERS):
                n_forms_gender += 1
                if shown < 5:
                    print(f"    !! noun form leaf carries a gender: {tj!r}")
                    shown += 1
    print(f"    noun form leaves sampled: {n_forms:,}  "
          f"carrying a gender: {n_forms_gender:,} (required 0)")
    ok &= n_forms_gender == 0
    return ok


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    new, old = sys.argv[1], sys.argv[2]
    print(f"new: {new}\nold: {old}\n")

    meta = json.load(open(os.path.join(new, "manifest.json")))["meta"]
    print(f"manifest text_convention = {meta.get('text_convention')!r}  "
          f"tokenizer = {meta.get('tokenizer')!r}")
    print(f"manifest unit_props      = {meta.get('unit_props')!r}")
    ok = meta.get("text_convention") == "v6"
    if not ok:
        print("    !! new store does not declare text_convention v6")
    if "gender" not in (meta.get("unit_props") or []):
        print("    !! new store does not declare gender in unit_props")
        ok = False

    kind = np.load(os.path.join(new, "kind.npy"))
    ok &= check_structure(new, old)

    Gn = graph_store.load_graph(new, verbose=False)
    Go = graph_store.load_graph(old, verbose=False)
    ok_text, n_changed = check_text(Gn["text"], Go["text"], kind)
    ok &= ok_text
    ok &= check_coverage(Gn, kind, n_changed)

    print("\nACCEPTANCE:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
