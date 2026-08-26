#!/usr/bin/env python3
"""Identify the noun anchors that carry no gender in a store.

Written for a 68-anchor discrepancy: the store held 310,430 noun anchors but only
310,362 with a gender, i.e. 68 more nouns than data/analysis/scan_gender.py found
in the raw dump.  Either the scan misses nouns or the builder drops genders, and
the two have very different consequences, so the difference has to be resolved
rather than rounded away.

Prints the offending anchors with their lexical-unit ids so they can be grepped
in the raw N-Triples.

Usage:  python -m analysis.find_genderless_nouns STORE
"""
import sys

import numpy as np

from lib import graph_store

K_ANCHOR = 0
TYPE_SHIFT = 56
GENDERS = ("moški spol", "ženski spol", "srednji spol")
NOUN = "samostalnik"


def main():
    G = graph_store.load_graph(sys.argv[1], verbose=False)
    kind = np.asarray(G["kind"])
    codes = np.asarray(G["node_codes"])
    texts = G["text"]

    bad = []
    n_noun = 0
    for i in np.flatnonzero(kind == K_ANCHOR):
        i = int(i)
        t = texts[i]
        if NOUN not in t:
            continue
        n_noun += 1
        if not any(g in t for g in GENDERS):
            code = int(codes[i]) if i < len(codes) else -1
            bad.append((i, code, code >> TYPE_SHIFT, code & ((1 << TYPE_SHIFT) - 1), t))

    print(f"noun anchors: {n_noun:,}   without a gender: {len(bad):,}")
    print(f"\n{'node':>10} {'type':>5} {'payload':>12}  text")
    for i, code, ty, payload, t in bad:
        print(f"{i:>10} {ty:>5} {payload:>12}  {t}")


if __name__ == "__main__":
    main()
