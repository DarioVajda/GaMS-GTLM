#!/usr/bin/env python3
"""Derive each item's extraction targets and final label from a REAL extractor run.

D3 says dataset extraction runs the identical pipeline to production: an external
model names the word, the surface index resolves it, and the union of matches is
the model's input.  Generation cannot know what the extractor will return, so the
label cannot be final at generation time.  This stage closes that gap.

    python -m qa.relabel datasets/work/generated_raw \\
        datasets/work/extraction/items.jsonl datasets/work/relabelled

For every item, in order:

  keep         the extractor's strings resolve to the item's OWN anchor
               -> unchanged; targets = the whole union it returned
  wrong_ball   they do, but this item is drawn as a synthetic negative
               -> targets = a DECOY word from the question; gold -> sentinel
  extract_miss they do not resolve to the anchor (empty or wrong)
               -> gold -> sentinel; targets = whatever it did return

Two things this deliberately does NOT do.  It never opens a neighbourhood: the
output names `targets` as node codes and leaves hop count, K_mwe and K_colloc to
the ball builder, so the extraction contract can be frozen before the sampling
policy is.  And it never rescues an item by consulting the generator's own anchor
-- that privileged knowledge is exactly what D3 forbids.

**`resolved`, not string equality, decides the branch.**  Measured over the
extraction run: the extractor hits the gold target string on 96.8 % of positives
but lands on the right anchor 98.4 % of the time, because it often returns the
lemma where the question named an inflected form (T21: 60 % vs 90 %).  Branching
on string equality would flip a tenth of T21's positives to `ni podatka` with the
answer sitting in the ball.
"""
import os
import re
import json
import hashlib
import argparse
import collections

import numpy as np

from qa import spec
from qa.build_dataset import NEGATIVE_GROUP
from qa.store import open_store
from lib.errors import StageError

WORD = re.compile(r"\w+", re.UNICODE)
# A word in this share of all questions is template boilerplate, not content.
# Derived from the corpus rather than hand-listed: the metalanguage the templates
# use (`beseda`, `oblika`, `sklon`, `glagol`) falls out on its own, and so does
# whatever a future template introduces.
BOILERPLATE_DF = 0.01


def key_rng(key, seed=0):
    """RNG seeded from what the draw is FOR -- never from a loop counter (D5b b)."""
    h = hashlib.blake2b(str(key).encode(), digest_size=8).digest()
    return np.random.default_rng((int.from_bytes(h, "big") ^ int(seed))
                                 & ((1 << 63) - 1))


def group_key(r):
    """The identity a synthetic-negative draw is made for.

    T12 and T14 must be negative in both or in neither (C17) -- T14 counts what
    T12 lists, so a lemma cannot be answerable under one and not the other.
    Drawing per ITEM breaks that: the two have different ids, so they flip
    independently, and the first run split `oškropljen`.  Keying the draw on the
    group leader makes the pair decide once, together.  Extraction itself never
    splits a pair (0 of 76 measured), so this is the only coupling needed.
    """
    return (NEGATIVE_GROUP.get(r["type"], r["type"]), r["lemma"], r["split"])


def target_string(r):
    sl = r.get("slots") or {}
    return sl.get("F") or sl.get("O") or sl.get("L") or r["lemma"]


def boilerplate(items):
    df = collections.Counter()
    for r in items:
        for w in {x.casefold() for x in WORD.findall(r["question"])}:
            df[w] += 1
    cut = BOILERPLATE_DF * len(items)
    return {w for w, c in df.items() if c >= cut}


def decoys(r, idx, boiler):
    """Candidate wrong words FROM THE QUESTION, best tier first.

    A decoy has to be a word a real extractor might plausibly have returned
    instead.  The eval says that is either a content word of an embedded sentence
    (T19/T20) or a grammatical term the question names -- GaMS returned
    `["apneti", "namenilnik", "velelnik"]` and `["lemo", "momgolsko"]`, never a
    function word.  A randomly chosen unrelated word would make the negative
    trivially separable, which is the shortcut D7 warns about.
    """
    tgt = target_string(r).casefold()
    sent = (r.get("slots") or {}).get("S") or ""

    def usable(w):
        wl = w.casefold()
        return (wl != tgt and wl not in tgt and tgt not in wl
                and wl in idx)

    in_sent = {w.casefold() for w in WORD.findall(sent)}
    tiers = collections.defaultdict(list)
    for w in WORD.findall(r["question"]):
        if not usable(w):
            continue
        wl = w.casefold()
        if wl in in_sent and wl not in boiler:
            tiers["sentence"].append(wl)
        elif wl not in boiler:
            tiers["content"].append(wl)
        else:
            tiers["term"].append(wl)
    for name in ("sentence", "content", "term"):
        if tiers[name]:
            return name, sorted(set(tiers[name]))
    return None, []


def to_negative(r):
    """Rewrite a record's answer side to the sentinel, in build_dataset's shape."""
    r["answer"] = spec.sentinel_line()
    r["gold_items"] = [spec.SENTINEL]
    r["negative"] = True
    sp = spec.SPEC[r["type"]]
    g = {"mode": sp["mode"], "sep": sp["sep"], "arity": sp["arity"],
         "regex": sp["regex"]}
    if r["type"] == "T17":
        g.update(quantity_band=(r.get("slots") or {}).get("band"),
                 n_asked=(r.get("slots") or {}).get("N"),
                 n_all=0, all_items=[])
    r["grading"] = g
    return r


def run(dataset, extraction, out, wrong_rate=0.015, seed=0, store=None):
    """Relabel `dataset` against a real extraction run, writing to `out`.

    The body `main()` used to hold, so the pipeline can call this stage as a
    function while the command line keeps behaving exactly as it did.
    """
    args = argparse.Namespace(dataset=dataset, extraction=extraction, out=out,
                              wrong_rate=wrong_rate, seed=seed, store=store)

    ext = {}
    with open(args.extraction, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ext[d["id"]] = d

    splits = {}
    for split in ("train", "dev", "test"):
        p = os.path.join(args.dataset, f"{split}.jsonl")
        if os.path.exists(p):
            splits[split] = [json.loads(l) for l in open(p, encoding="utf-8")]
    allitems = [r for v in splits.values() for r in v]
    boiler = boilerplate(allitems)
    print(f"{len(allitems):,} items, {len(ext):,} extraction rows, "
          f"{len(boiler)} boilerplate words", flush=True)

    store = open_store(args.store)
    codes = np.asarray(store.codes)
    idx = store.surface_index()

    used = collections.Counter()      # decoy -> times used, for diversity
    stats = collections.Counter()
    tiers = collections.Counter()
    missing = 0
    os.makedirs(args.out, exist_ok=True)

    # ---- pass 1: resolution, and which groups are eligible for a wrong ball --
    info = {}
    eligible = collections.defaultdict(list)
    for split, rows in splits.items():
        for r in rows:
            e = ext.get(r["id"])
            if e is None:
                continue
            got = [x.casefold() for x in e.get("parsed") or []]
            anchors = sorted({int(a) for x in got for a in idx.get(x, ())})
            i = int(np.searchsorted(codes, r["node_code"]))
            own = i if (i < len(codes)
                        and int(codes[i]) == r["node_code"]) else None
            resolved = own is not None and own in anchors
            tier, cands = decoys(r, idx, boiler)
            info[r["id"]] = {"anchors": anchors, "resolved": resolved,
                             "tier": tier, "cands": cands}
            if resolved and not r["negative"]:
                eligible[group_key(r)].append(r["id"])

    # A group flips only if EVERY member drew it and every member has a decoy --
    # otherwise C17's "negative in both or in neither" breaks again.
    flip = set()
    for gk, ids in eligible.items():
        if not all(info[i]["cands"] for i in ids):
            continue
        if key_rng(gk, args.seed).random() < args.wrong_rate:
            flip.update(ids)

    # Which extraction run produced these labels, recorded on every item.  The
    # last two path components rather than the basename: the dumps used to be
    # named `extraction_v2_items.jsonl` and identified themselves, and are now
    # `<somewhere>/extraction/items.jsonl`, where the basename says nothing.
    run_name = "/".join(args.extraction.rstrip("/").split(os.sep)[-2:])

    # ---- pass 2: apply -------------------------------------------------------
    for split, rows in splits.items():
        with open(os.path.join(args.out, f"{split}.jsonl"), "w",
                  encoding="utf-8") as g:
            for r in rows:
                d = info.get(r["id"])
                if d is None:
                    missing += 1
                    stats["no_extraction_row"] += 1
                    g.write(json.dumps(r, ensure_ascii=False) + "\n")
                    continue

                anchors, resolved = d["anchors"], d["resolved"]
                policy, decoy, tier = "keep", None, None
                if not resolved:
                    policy = "extract_miss"
                elif r["id"] in flip:
                    decoy = min(d["cands"], key=lambda w: (used[w], w))
                    used[decoy] += 1
                    tier = d["tier"]
                    anchors = sorted({int(a) for a in idx.get(decoy, ())})
                    policy = "wrong_ball"

                if policy != "keep":
                    to_negative(r)
                    r["negative_flavour"] = ("extract_miss"
                                             if policy == "extract_miss"
                                             else "wrong_ball")
                r["targets"] = [int(codes[a]) for a in anchors]
                r["extraction"] = {"strings": ext[r["id"]].get("parsed") or [],
                                   "resolved": bool(resolved),
                                   "policy": policy,
                                   "run": run_name}
                if decoy:
                    r["extraction"]["decoy"] = decoy
                    r["extraction"]["decoy_tier"] = tier
                    tiers[tier] += 1
                stats[policy] += 1
                stats["n"] += 1
                stats["empty_targets"] += (not anchors)
                stats["multi_target"] += (len(anchors) > 1)
                g.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = stats["n"]
    print(f"\n{'policy':>14} {'items':>7} {'share':>8}")
    for k in ("keep", "wrong_ball", "extract_miss"):
        print(f"{k:>14} {stats[k]:>7,} {100*stats[k]/n:7.2f} %")
    print(f"\nempty target set (single-node ball): {stats['empty_targets']:,} "
          f"({100*stats['empty_targets']/n:.2f} %)")
    print(f"union of >1 anchor:                  {stats['multi_target']:,} "
          f"({100*stats['multi_target']/n:.2f} %)")
    if tiers:
        print(f"decoy tiers: {dict(tiers)}")
        print(f"distinct decoys: {len(used)}, most used: {used.most_common(5)}")
    if missing:
        print(f"\nWARNING: {missing:,} items had no extraction row and were "
              f"passed through unchanged -- run the extractor over the whole "
              f"dataset before trusting this output")
    print(f"\n[wrote] {args.out}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("extraction", help="JSONL from analysis/measure_extraction.py")
    ap.add_argument("out")
    ap.add_argument("--wrong-rate", type=float, default=0.015,
                    help="share of resolvable items turned into wrong-ball "
                         "negatives; the measured wrong-anchor rate is 1.1 %%")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store", default=None)
    args = ap.parse_args()

    try:
        run(dataset=args.dataset, extraction=args.extraction, out=args.out,
            wrong_rate=args.wrong_rate, seed=args.seed, store=args.store)
    except StageError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
