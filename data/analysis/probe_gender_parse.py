#!/usr/bin/env python3
"""Parse ONE noun-batch words file and report where gender lands.

Entry-level gender must reach `unit` and be rendered into the anchor
parenthetical, leaving the form strings alone.

Usage:  python -m analysis.probe_gender_parse [FILE]
"""
import os
import sys
from collections import defaultdict

from build import build_graph as B
from lib.paths import kg_raw_file

PATH = sys.argv[1] if len(sys.argv) > 1 else kg_raw_file("220-words.nt")


def main():
    print(f"UNIT_PROPS   = {B.UNIT_PROPS}")
    print(f"FEATURE_PROPS= {B.FEATURE_PROPS}")
    res = B.parse_file(PATH)

    unit_map = defaultdict(dict)
    for c, p, v in res["unit"]:
        unit_map[c][p] = v
    feat_map = defaultdict(dict)
    for c, p, v in res["feat"]:
        feat_map[c][p] = v
    pos_map = dict(res["pos"])
    wr = {}
    for c, s in res["wr"]:
        prev = wr.get(c)
        if prev is None or (s.lower(), s) < (prev.lower(), prev):
            wr[c] = s
    canon = {int(a): int(b) for a, b in res["canon"]}

    nouns = [lu for lu, p in pos_map.items() if p == "noun"]
    with_gender = [lu for lu in nouns if "gender" in unit_map.get(lu, {})]
    print(f"\nnoun lexical units in {os.path.basename(PATH)}: {len(nouns):,}")
    print(f"  with entry-level gender parsed: {len(with_gender):,} "
          f"({100.0*len(with_gender)/max(len(nouns),1):.2f}%)")

    others = [lu for lu, p in pos_map.items() if p != "noun"
              and "gender" in unit_map.get(lu, {})]
    print(f"  NON-noun units that picked up an entry gender: {len(others):,} "
          f"(expected 0)")

    # aspect/clitic must be unharmed by the dispatch rewrite
    n_aspect = sum(1 for d in unit_map.values() if "aspect" in d)
    n_clitic = sum(1 for d in unit_map.values() if "clitic" in d)
    print(f"  aspect units: {n_aspect:,}   clitic units: {n_clitic:,}")

    print("\nrendered anchors (new text):")
    shown = 0
    for lu in nouns:
        f = canon.get(lu)
        if f is None or f not in wr:
            continue
        s = B.TAG_ANCHOR + wr[f] + B.feat_string(
            feat_map.get(f, {}), pos_map.get(lu), unit_map.get(lu))
        print("  " + s)
        shown += 1
        if shown >= 12:
            break

    print("\nrendered form leaves (must be UNCHANGED -- no gender):")
    shown = 0
    for a, b in res["other"][:400]:
        b = int(b)
        if b not in wr:
            continue
        t = B.TAG_FORM + wr[b] + B.feat_string(feat_map.get(b, {}))
        if "spol" in t:
            continue
        print("  " + t)
        shown += 1
        if shown >= 6:
            break


if __name__ == "__main__":
    main()
