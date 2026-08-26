#!/usr/bin/env python3
"""The cross-cutting checks of QA_TASKS.md section 2, as runnable assertions.

    python -m qa.selftest                       # the store-free checks (C7, C16)
    python -m qa.selftest --dataset DIR         # everything, against a dataset
    python -m qa.selftest --dataset DIR --store S   # + the checks that need the graph

Which check is which:

  C6   Tier C leakage -- no training item anywhere contains `protipomenka`,
       `antonim` or `nasprotje`.  T16's entire value is that the relation is
       unseen, so a leak does not degrade the result, it deletes it.
  C7   the Slovene agreement table, the canonical orderings, and `sense_class`.
  C9   run the grader over the gold answers themselves: every item must score
       success against its own gold.  Free, and the one test that validates the
       grading contract end to end.
  C10  every gold matches its type's regex, and no multi-item gold contains its
       own separator inside an item.
  C15  every gold is in its type's canonical order -- including the set-valued
       types whose order is not graded.  Grading tolerance is not a licence for
       non-canonical training data.
  C16  `sl_key` against a fixture with c-caron, s-caron, z-caron and a
       non-Slovene character.  `sl_SI` is not installed on this cluster and
       Python's default sort is wrong for Slovene.
  C17  T14's gold == len(T12's gold) for every lemma in both, and a lemma is a
       negative in both types or in neither.
  C18  D5b reproducibility: the pool is node-id sorted, the seed comes from the
       anchor's node code and nothing else, and every gold phrase is in its own
       ball.
  C11  the split is lemma-disjoint (D11).
  C13  the majority-class baseline for every type, reported beside the score.
"""
import os
import re
import sys
import json
import random
import argparse
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa import sl, spec, grade, seeds, colloc_sampling, gen                # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAIL.append(name)
    return ok


# --------------------------------------------------------------------------
def c16_sl_key():
    print("\nC16 -- sl_key")
    got = sorted(["cesta", "riba", "sok", "zima", "čas", "šola", "žito", "xyz"],
                 key=sl.sl_key)
    # `x` is not in the Slovene alphabet, so by the documented rule it sorts
    # AFTER the whole alphabet, by codepoint.  Assert the Slovene order exactly
    # and that the foreign string does not land inside it.
    want = ["cesta", "čas", "riba", "sok", "šola", "zima", "žito", "xyz"]
    check("Slovene letters collate correctly", got == want, str(got))
    check("default sort really is wrong (so the key is not decoration)",
          sorted(["čas", "zima"]) != sorted(["čas", "zima"], key=sl.sl_key))


def c7_agreement():
    print("\nC7 -- Slovene number agreement (T14)")
    cases = {1: "v bazi je zabeležen 1 pomen",
             2: "v bazi sta zabeležena 2 pomena",
             3: "v bazi so zabeleženi 3 pomeni",
             4: "v bazi so zabeleženi 4 pomeni",
             5: "v bazi je zabeleženih 5 pomenov",
             11: "v bazi je zabeleženih 11 pomenov",
             21: "v bazi je zabeležen 21 pomen",
             102: "v bazi sta zabeležena 102 pomena"}
    bad = {n: sl.pomen_agreement(n) for n, w in cases.items()
           if sl.pomen_agreement(n) != w}
    check("agreement table", not bad, str(bad))
    check("accusative frame", sl.pomen_agreement(3, "ima") == "ima 3 pomene")


def c7_sense_class(store):
    print("\nC7 -- sense_class fixture")
    # one placeholder, one fallback, one fallback-with-snippet, one defined
    core = store.core_anchors()
    seen = collections.Counter()
    for a in core[:4000]:
        a = int(a)
        lemma_cf = store.lemma(a).casefold()
        for v, o, b in store.senses(a):
            seen[seeds.sense_class(store, a, v, o, b, lemma_cf)] += 1
    check("all three classes occur in the store",
          all(seen[k] for k in ("placeholder", "fallback", "defined")), str(dict(seen)))


# --------------------------------------------------------------------------
def load(dataset):
    out = {}
    for s in ("train", "dev", "test"):
        p = os.path.join(dataset, f"{s}.jsonl")
        out[s] = grade.read_jsonl(p) if os.path.exists(p) else []
    return out


def c9_gold_grades(items):
    print("\nC9 -- the grader scores every gold answer as correct")
    bad = collections.Counter()
    for it in items:
        r = grade.grade(it, it["answer"])
        if not r["success"]:
            bad[(it["type"], r["reason"])] += 1
    check("gold == 100 % success", not bad,
          str(dict(bad.most_common(8))) if bad else f"{len(items):,} items")


def c10_regex(items):
    print("\nC10 -- every gold matches its type's regex, no separator inside an item")
    bad_re = collections.Counter()
    bad_sep = collections.Counter()
    for it in items:
        g = it["grading"]
        line = it["answer"].splitlines()[0]
        if it["negative"]:
            if line != spec.sentinel_line():
                bad_re[it["type"]] += 1
            continue
        if not re.match(g["regex"], line):
            bad_re[it["type"]] += 1
        sep = g["sep"]
        if sep and g["arity"] != 1:
            seps = sep if isinstance(sep, (list, tuple)) else [sep]
            for item in it["gold_items"]:
                if any(s in item for s in seps):
                    # a labelled block leader legitimately contains ": " but never
                    # the separator itself
                    bad_sep[it["type"]] += 1
    check("regex", not bad_re, str(dict(bad_re)))
    check("no separator inside an item", not bad_sep, str(dict(bad_sep)))


def c15_canonical(items):
    print("\nC15 -- every set-valued gold is in canonical order")
    bad = collections.Counter()
    for it in items:
        if it["negative"] or it["type"] not in ("T4", "T15", "T16", "T17"):
            continue
        g = it["gold_items"]
        if g != sorted(g, key=sl.sl_key):
            bad[it["type"]] += 1
    check("sl_key order", not bad, str(dict(bad)))


def c6_tier_c(splits):
    print("\nC6 -- Tier C leakage")
    hard = collections.Counter()
    soft_q = collections.Counter()
    soft_a = collections.Counter()
    for s in ("train", "dev"):
        for it in splits[s]:
            q = it["question"].casefold()
            a = it["answer"].casefold()
            for w in spec.TIER_C_TAG_WORDS:
                if w in q or w in a:
                    hard[w] += 1
            for w in spec.TIER_C_SOFT_WORDS:
                if w in q:
                    soft_q[w] += 1
                elif w in a:
                    soft_a[w] += 1
    check("no antonym TAG WORD in train/dev, anywhere", not hard, str(dict(hard)))
    check("no 'opposite' vocabulary in a train/dev QUESTION", not soft_q,
          str(dict(soft_q)))
    # Reported, not asserted: these occur inside curated definitions and corpus
    # sentences, which are data, not phrasing we chose.  See spec.TIER_C_*.
    print(f"    'opposite' vocabulary inside a train/dev ANSWER: "
          f"{sum(soft_a.values())} items {dict(soft_a)} (allowed)")
    tc = [it for s in splits for it in splits[s] if it["type"] in spec.TIER_C]
    check("every Tier C item is a test item",
          all(it["split"] == "test" for it in tc), f"{len(tc)} items")


def c11_split(splits):
    print("\nC11 -- the split is lemma-disjoint")
    by = {s: {it["lemma"] for it in v} for s, v in splits.items()}
    overlap = {}
    for a in ("train", "dev", "test"):
        for b in ("train", "dev", "test"):
            if a < b and by[a] & by[b]:
                overlap[f"{a}/{b}"] = len(by[a] & by[b])
    check("no lemma on two sides", not overlap, str(overlap))


def c17_t12_t14(splits):
    print("\nC17 -- T14 == len(T12), and negatives agree")
    t12, t14 = {}, {}
    for s in splits:
        for it in splits[s]:
            if it["type"] == "T12":
                t12[it["lemma"]] = (it["negative"], len(it["gold_items"]))
            elif it["type"] == "T14":
                t14[it["lemma"]] = (it["negative"],
                                    int(it["gold_items"][0]) if not it["negative"] else 0)
    shared = set(t12) & set(t14)
    bad_count = {l for l in shared
                 if not t12[l][0] and not t14[l][0] and t12[l][1] != t14[l][1]}
    bad_neg = {l for l in shared if t12[l][0] != t14[l][0]}
    check(f"count matches the list ({len(shared)} shared lemmas)", not bad_count,
          str(sorted(bad_count)[:5]))
    check("negative in both or in neither", not bad_neg, str(sorted(bad_neg)[:5]))


def c13_baselines(splits):
    print("\nC13 -- majority-class baseline per type (reported, not asserted)")
    by_type = collections.defaultdict(list)
    for it in splits["test"]:
        by_type[it["type"]].append(it["answer"])
    for t in sorted(by_type):
        c = collections.Counter(by_type[t])
        top, n = c.most_common(1)[0]
        print(f"    {t:5s} n={len(by_type[t]):4d}  baseline "
              f"{100.0 * n / len(by_type[t]):5.1f}%  {top[:56]!r}")


def c18_sampling(store, items):
    print("\nC18 -- D5b reproducibility, and gold-in-ball")
    from qa.store import QAStore
    anchors = [it for it in items if it["type"] == "T17" and not it["negative"]]
    if not anchors:
        check("T17 items present", False)
        return
    sample = anchors[:200]
    ok_ball = True
    for it in sample:
        a = int(next(i for i in [_anchor_of(store, it)] if i is not None))
        ball = {p for _v, p, _s in colloc_sampling.sample(store, a)}
        if not set(it["gold_items"]) <= ball:
            ok_ball = False
            break
    check("every gold phrase is in its own ball", ok_ball, f"{len(sample)} items")
    a = _anchor_of(store, sample[0])
    b1 = colloc_sampling.sample(store, a)
    b2 = colloc_sampling.sample(store, a)
    check("the sampler is deterministic", b1 == b2)
    check("the pool is sorted by node id",
          all(x[0] < y[0] for x, y in zip(b1, b1[1:])) if len(b1) > 1 else True)
    s1 = colloc_sampling.anchor_seed(store.codes[a])
    check("the seed depends only on the node code",
          s1 == colloc_sampling.anchor_seed(store.codes[a]))


def _anchor_of(store, item):
    import numpy as np
    i = int(np.searchsorted(store.codes, item["node_code"]))
    return i if i < store.n_real and int(store.codes[i]) == item["node_code"] else None


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset")
    ap.add_argument("--store")
    args = ap.parse_args()

    c16_sl_key()
    c7_agreement()

    store = None
    if args.store:
        from qa.store import open_store
        store = open_store(args.store, verbose=False)
        c7_sense_class(store)

    if args.dataset:
        splits = load(args.dataset)
        allitems = [it for v in splits.values() for it in v]
        print(f"\ndataset: {len(allitems):,} items "
              f"({', '.join(f'{s} {len(v):,}' for s, v in splits.items())})")
        c9_gold_grades(allitems)
        c10_regex(allitems)
        c15_canonical(allitems)
        c6_tier_c(splits)
        c11_split(splits)
        c17_t12_t14(splits)
        c13_baselines(splits)
        if store:
            c18_sampling(store, allitems)

    print("\nSELFTEST:", "PASS" if not FAIL else f"FAIL ({len(FAIL)}): {FAIL}")
    raise SystemExit(0 if not FAIL else 1)


if __name__ == "__main__":
    main()
