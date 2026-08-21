#!/usr/bin/env python3
"""Regression check for the M4 fix on ADJECTIVES, where gender is a genuine
form feature: the dispatch rewrite must leave those form strings alone.

Usage:  test_gender_adj.py [FILE]
"""
import os, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build"))
import build_gtlm_graph as B

PATH = sys.argv[1] if len(sys.argv) > 1 else (
    "/shared/workspace/povejmo/gams_gtlm/data/kg_raw/OntoLex DSB/1-words.nt")


def main():
    res = B.parse_file(PATH)
    feat_map = defaultdict(dict)
    for c, p, v in res["feat"]:
        feat_map[c][p] = v
    unit_map = defaultdict(dict)
    for c, p, v in res["unit"]:
        unit_map[c][p] = v
    pos_map = dict(res["pos"])
    wr = {}
    for c, s in res["wr"]:
        prev = wr.get(c)
        if prev is None or (s.lower(), s) < (prev.lower(), prev):
            wr[c] = s

    n_formgender = sum(1 for d in feat_map.values() if "gender" in d)
    print(f"{os.path.basename(PATH)}: word-forms with a gender feature: "
          f"{n_formgender:,}  (must be > 0 -- adjectives)")
    print(f"lexical units with an entry gender: {len(unit_map):,}")

    adj = {lu for lu, p in pos_map.items() if p == "adjective"}
    adj_lu_gender = sum(1 for lu in adj if "gender" in unit_map.get(lu, {}))
    print(f"adjective units: {len(adj):,}, of which with ENTRY gender: "
          f"{adj_lu_gender:,} (expected 0)")

    print("\nadjective form leaves with gender (must still render it):")
    shown = 0
    for a, b in res["other"]:
        a = int(a); b = int(b)
        if a not in adj or b not in wr or "gender" not in feat_map.get(b, {}):
            continue
        print("  " + B.TAG_FORM + wr[b] + B.feat_string(feat_map.get(b, {})))
        shown += 1
        if shown >= 8:
            break
    if not shown:
        print("  (none found in this batch)")


if __name__ == "__main__":
    main()
