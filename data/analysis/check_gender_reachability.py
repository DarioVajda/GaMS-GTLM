#!/usr/bin/env python3
"""Q3's premise, checked on a built store.

The proposal is to render noun gender on the ANCHOR only and let every other
node of the entry inherit it.  That is safe exactly when a noun's form leaves
are adjacent to their anchor, so any ball that contains a form also contains
the anchor that carries the gender.

Reports:
  * how many noun form leaves sit at hop 1 from an anchor
  * how many noun form nodes there are, i.e. what repeating the gender on every
    form would have cost
Usage:  check_gender_reachability.py STORE
"""
import os, sys, collections
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import graph_store

K_ANCHOR, K_FORM = 0, 1
NOUN = "(samostalnik"


def main():
    store = sys.argv[1]
    G = graph_store.load_graph(store, verbose=False)
    kind = np.asarray(G["kind"])
    indptr = np.asarray(G["indptr"])
    indices = np.asarray(G["indices"])
    texts = G["text"]
    n = len(kind)
    print(f"store {store}: {n:,} nodes")

    anchors = np.flatnonzero(kind == K_ANCHOR)
    print(f"anchors: {len(anchors):,}")

    # noun anchors, by their POS tag in the text
    noun_anchor = np.zeros(n, dtype=bool)
    n_noun = 0
    for i in anchors:
        if NOUN in texts[int(i)]:
            noun_anchor[i] = True
            n_noun += 1
    print(f"noun anchors: {n_noun:,}")

    # every neighbour of a noun anchor that is a form leaf
    noun_forms = 0
    forms_seen = set()
    for i in np.flatnonzero(noun_anchor):
        for j in indices[indptr[i]:indptr[i + 1]]:
            j = int(j)
            if kind[j] == K_FORM:
                noun_forms += 1
                forms_seen.add(j)
    print(f"form leaves adjacent to a noun anchor: {noun_forms:,} "
          f"({len(forms_seen):,} distinct)")

    # the converse: does every form leaf have an anchor at hop 1?
    forms = np.flatnonzero(kind == K_FORM)
    rng = np.random.default_rng(0)
    sample = rng.choice(forms, size=min(200_000, len(forms)), replace=False)
    with_anchor = 0
    for i in sample:
        i = int(i)
        for j in indices[indptr[i]:indptr[i + 1]]:
            if kind[int(j)] == K_ANCHOR:
                with_anchor += 1
                break
    print(f"form leaves total: {len(forms):,}")
    print(f"sampled {len(sample):,} form leaves; "
          f"{with_anchor:,} ({100.0*with_anchor/len(sample):.3f}%) "
          f"have an anchor at hop 1")

    print(f"\ncost avoided by anchor-only rendering: the gender would otherwise "
          f"be repeated on {len(forms_seen):,} noun form leaves "
          f"instead of {n_noun:,} anchors "
          f"(x{len(forms_seen)/max(n_noun,1):.1f}).")

    print("\nsample noun anchors and their forms (current text):")
    shown = 0
    for i in np.flatnonzero(noun_anchor):
        t = texts[int(i)]
        if "spol" in t:
            continue
        print(f"  ANCHOR {t}")
        k = 0
        for j in indices[indptr[i]:indptr[i + 1]]:
            j = int(j)
            if kind[j] == K_FORM:
                print(f"      {texts[j]}")
                k += 1
                if k >= 3:
                    break
        shown += 1
        if shown >= 5:
            break


if __name__ == "__main__":
    main()
