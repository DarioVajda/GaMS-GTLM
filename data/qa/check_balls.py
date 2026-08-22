#!/usr/bin/env python3
"""C18 (d) on the real thing: is every gold item actually inside the ball?

    python -m qa.check_balls datasets/generated/v2_relabelled datasets/balls/v2

`qa/selftest.py` checks containment against `d5b.sample(store, a)` -- the
collocation slice, at the sampler's default K, for one anchor.  That validated a
ball nobody ever built: the item's real input is the union of every anchor D3's
lookup returned, under the builder's own K, and it holds forms and senses and
MWEs too.  This reads the written balls and asks the question directly, so a cap
that is too tight shows up as missing gold rather than as a number nobody
compared.

Positives only.  A negative's gold is the sentinel by construction, so
containment is meaningless for it -- and 3.6 % of items are the single-node
`ni v bazi` ball on purpose.

**The predicate is per-node, and it is on ATOMS, not on the answer string.**  A
literal substring test is wrong here and scores four types at 0 %: gold is
*rendered* for a human reader (`tožilnik množine`, `nedoločnik: eksploatirati`)
while the node states the same fact in the store's own convention
(`oblika: zalednike (tožilnik, množina)`).  Nothing is missing -- the two just
spell it differently.  So a gold item is contained when **some single node**
carries all of its atoms: the surface, and every label that qualifies it.  One
node, not the whole ball, or `tožilnik množine` would pass on a ball holding
`tožilnik ednine` and `imenovalnik množine` separately.

Two normalisations, both closed and both listed below: number labels appear in
the genitive in an answer and the nominative in the graph (`množine` /
`množina`), and M2's auxiliary is dropped.

**One recorded exemption: T5 and T6.**  M2 composes the periphrastic tenses from
the *-l* participle (graph content) plus a `biti` auxiliary (not injected into
the ball -- QA_TASKS.md, M2, reversed 2026-08-22).  So their gold legitimately
contains tokens the input does not, and the invariant for them reads *every gold
**lexical** item is in the ball*.  This script strips a leading auxiliary before
matching those two types and reports them separately, rather than either failing
them or quietly passing everything.
"""
import os
import re
import sys
import json
import glob
import argparse
import collections

# The 16 `biti` forms M2 composes with -- present and future, all persons and
# numbers.  Listed rather than read from the store because this script must run
# without opening 37 M nodes.
AUX = frozenset("sem si je sva sta smo ste so "
                "bom boš bo bova bosta bomo boste bodo".split())

# An answer says "tožilnik MNOŽINE", the graph says "(tožilnik, MNOŽINA)".  The
# grammatical-number vocabulary is three words, so the genitive is mapped back
# rather than guessed at with a suffix rule.
GEN2NOM = {"ednine": "ednina", "dvojine": "dvojina", "množine": "množina"}

# Everything M2 COMPOSES rather than reads: the auxiliary, and the two tense
# names themselves.  `preteklik` and `prihodnjik` are not `vform` values in this
# KG -- that is the whole of M2 -- so no node can carry them, while `sedanjik`
# is stored and stays in the predicate.  A T5 cell reads
# `preteklik: sem cirkuliral`; its one graph atom is `cirkuliral`.
COMPOSED = AUX | {"preteklik", "prihodnjik"}

WORD = re.compile(r"[^\W_]+", re.UNICODE)


def atoms(gold, composed=False):
    """The words a node must ALL carry for this gold item to be present.

    `nedoločnik: eksploatirati` -> {nedoločnik, eksploatirati}; the label is kept
    rather than stripped, because "the ball holds this surface" is a weaker claim
    than "the ball holds this surface under this label" and the second one is
    what the question asked.

    `composed` drops the M2 atoms, and is passed only for T5/T6 -- so a missing
    *participle* still fails those types, which is the part of their gold the
    graph really does own.
    """
    parts = {GEN2NOM.get(p, p) for p in WORD.findall(gold.casefold())}
    return (parts - COMPOSED) or parts if composed else parts


def membership_need(r):
    """How many allowed phrases the ball must hold for the item to be gradeable.

    A membership item (T17) is NOT graded against the sampled `gold_items` -- it
    is graded against `all_items`, the anchor's complete collocation set, plus a
    count band (`qa/grade.py` count_ok).  So containment here does not mean "the
    ball holds those particular phrases"; it means "the ball holds enough
    phrases the grader will accept".  The floor per band, `min(., n_all)`
    throughout because under-supply is never an error:

        none / vague_small   2      vague_large   5      exact   n_asked (2..10)
    """
    g = r["grading"]
    n_all = int(g["n_all"])
    band = g["quantity_band"]
    if band == "exact":
        return min(int(g["n_asked"]), n_all)
    return min(5 if band == "vague_large" else 2, n_all)


def membership_have(r, b):
    """Distinct allowed phrases the ball actually holds.

    Matched on `sl.norm`, the same normalisation the grader's `all_items` uses,
    and deduplicated -- an anchor can carry one phrase on two sense pairs
    (`Gadova Peč` / `Gadova peč`), which is two nodes and one gradeable answer.
    """
    from qa import sl
    allow = set(r["grading"]["all_items"])
    have = set()
    for text in b["nodes"]:
        if text.startswith("kolokacija: "):
            p = sl.norm(text[len("kolokacija: "):])
            if p in allow:
                have.add(p)
    return len(have)


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("balls")
    ap.add_argument("--show", type=int, default=8, help="example misses to print")
    args = ap.parse_args()

    gen = load(args.dataset)
    balls = load(args.balls)
    print(f"{len(gen):,} items, {len(balls):,} balls")

    hit = collections.Counter()
    tot = collections.Counter()
    items_bad = collections.Counter()
    items_n = collections.Counter()
    empty = collections.Counter()
    misses = []

    for iid, b in balls.items():
        r = gen.get(iid)
        if r is None or r["negative"]:
            continue
        gold = [g for g in (r.get("gold_items") or []) if isinstance(g, str) and g]
        if not gold:
            continue
        t = r["type"]
        if (r.get("grading") or {}).get("mode") == "membership":
            need, have = membership_need(r), membership_have(r, b)
            tot[t] += 1
            items_n[t] += 1
            if have >= need:
                hit[t] += 1
            else:
                items_bad[t] += 1
                misses.append((iid, t, need, [f"{have} allowed phrases in ball, "
                                              f"band needs {need}"]))
            continue

        nodes = [set(WORD.findall(x.casefold())) for x in b["nodes"]]
        nodes = [{GEN2NOM.get(w, w) for w in s} for s in nodes]
        # `velelnik: /` asserts that the cell is EMPTY -- the verb has no
        # imperative.  The ball is right to hold no such node, so the claim is
        # satisfied by absence and there is nothing to contain.  Counted, not
        # silently passed.
        vacuous = [g for g in gold if g.rstrip().endswith("/")]
        gold = [g for g in gold if not g.rstrip().endswith("/")]
        empty[t] += len(vacuous)
        miss = [g for g in gold
                if not any(atoms(g, t in ("T5", "T6")) <= s for s in nodes)]
        tot[t] += len(gold)
        hit[t] += len(gold) - len(miss)
        items_n[t] += 1
        if miss:
            items_bad[t] += 1
            misses.append((iid, t, len(gold), miss[:3]))

    print(f"\n{'type':>5} {'items':>6} {'bad':>5} {'gold':>7} {'in ball':>8} "
          f"{'containment':>12} {'empty cells':>12}")
    for t in sorted(tot, key=lambda x: (len(x), x)):
        print(f"{t:>5} {items_n[t]:>6} {items_bad[t]:>5} {tot[t]:>7} {hit[t]:>8} "
              f"{100 * hit[t] / tot[t]:11.1f} % {empty[t]:>12}")
    n, h = sum(tot.values()), sum(hit.values())
    print(f"{'ALL':>5} {sum(items_n.values()):>6} {sum(items_bad.values()):>5} "
          f"{n:>7} {h:>8} {100 * h / n:11.1f} % {sum(empty.values()):>12}")

    if misses:
        print(f"\n{len(misses):,} items miss at least one gold item; first "
              f"{min(args.show, len(misses))}:")
        for iid, t, ng, m in misses[:args.show]:
            print(f"  {iid} ({t}, {ng} gold)  missing {m}")
    else:
        print("\nevery gold item of every positive is in its own ball")


if __name__ == "__main__":
    main()
