#!/usr/bin/env python3
"""C18 (d) on the real thing: is every gold item actually inside the ball?

    python -m qa.check_balls datasets/generated datasets/balls

`qa/selftest.py` checks containment against `colloc_sampling.sample(store, a)` -- the
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
the ball -- QA_TASKS.md 0.7).  So their gold legitimately
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

from qa import grade, sl, spec

# The 16 `biti` forms M2 composes with -- present and future, all persons and
# numbers.  Listed rather than read from the store because this script must run
# without opening 37 M nodes.
AUX = frozenset("sem si je sva sta smo ste so "
                "bom boš bo bova bosta bomo boste bodo".split())

# An answer says "tožilnik MNOŽINE", the graph says "(tožilnik, MNOŽINA)".  The
# grammatical-number vocabulary is three words, so the genitive is mapped back
# rather than guessed at with a suffix rule.
#: An answer declines its labels; the graph states them in the nominative.  The
#: person joined this table with 0.1, which put the person/number of a verb cell
#: into the LABEL (`sedanjik 1. osebe ednine`) where the old line left it
#: implicit in the position.
GEN2NOM = {"ednine": "ednina", "dvojine": "dvojina", "množine": "množina",
           "osebe": "oseba"}

WORD = re.compile(r"[^\W_]+", re.UNICODE)

# Everything M2 COMPOSES rather than reads: the auxiliary, and the two tense
# names themselves.  `preteklik` and `prihodnjik` are not `vform` values in this
# KG -- that is the whole of M2 -- so no node can carry them, while `sedanjik`
# is stored and stays in the predicate.
#
# `oseba` joins them for a composed cell and only there.  `preteklik 1. osebe
# ednine: sem cirkuliral` is a *-l* participle plus an auxiliary, and a participle
# has no person in Slovene -- the person is on the `sem`, which is not in the
# ball.  The present-tense cells of the same two types DO carry it
# (`oblika: naškrobim (sedanjik, 1. oseba, ednina)`), so this is decided per gold
# item from its label, never per type: stripping it for all 27 cells would stop
# the check from noticing a present cell labelled with the wrong person.
COMPOSED = AUX | {"preteklik", "prihodnjik", "oseba"}

#: The eight declared labels of 0.1, as atoms.  A declared label is one the ball
#: cannot carry -- that is what declaring it means -- so requiring it here would
#: contradict C25, which excuses exactly these.  `qa/spec.py` holds the list and
#: the reasoning; this is the same budget seen from the other side.
DECLARED_ATOMS = frozenset(
    w for label in spec.DECLARED_LABELS for w in WORD.findall(label.casefold()))

#: Types whose VALUE is composed too, not just the label.  T14 answers with a
#: count of the ball's defined senses; a count is not a node, so there is nothing
#: to contain.  Recorded and reported rather than passed on a coincidence -- the
#: old check scored T14 at 100 % because a one-token gold of `3` matched any node
#: with a 3 in it.
#: T23 joins it for the same reason one step further out: its value is `da` or
#: `ne`, a VERDICT ABOUT the ball rather than a node in it, and a `ne` item is
#: precisely the case where the thing being asked about is absent.  Requiring
#: containment would make every correct `ne` look like a defect.
COMPOSED_VALUE = ("T14", "T23")


def is_composed(gold):
    """True for a cell M2 composes -- decided by the label, not by the type."""
    return gold.split(":", 1)[0].startswith(("preteklik", "prihodnjik"))


def atoms(gold):
    """The words a node must ALL carry for this gold item to be present.

    `nedoločnik: eksploatirati` -> {nedoločnik, eksploatirati}; the label is kept
    rather than stripped, because "the ball holds this surface" is a weaker claim
    than "the ball holds this surface under this label" and the second one is
    what the question asked.  0.1 made that distinction matter far more: the
    label now carries what the position used to, so this is the check that a
    paradigm cell is filled with the right cell's surface -- and it is the only
    check that verifies T12's sense ordinal, which C25 sees only as `pomen`.

    Two atom sets come off: the eight declared labels, which no ball carries by
    definition, and M2's composed material for the two periphrastic tenses.  What
    remains for those is the participle, which is the part of their gold the
    graph really does own.
    """
    full = {GEN2NOM.get(p, p) for p in WORD.findall(gold.casefold())}
    parts = full - DECLARED_ATOMS
    if is_composed(gold):
        # The person is two atoms, `1` and `oseba`, and BOTH sit on the auxiliary
        # rather than on the participle -- so the ordinal goes with the noun.
        parts -= COMPOSED
        parts = {p for p in parts if not p.isdigit()}
    # Never an empty set: `set() <= anything` is True, so a gold item that
    # stripped down to nothing would pass against every ball ever built.
    return parts or full


def gold_strings(r):
    """A positive's gold as `oznaka: vrednost` strings, for `atoms`.

    0.1 made `gold_items` a list of PAIRS.  The old shape was a list of strings
    and the old filter here was `isinstance(g, str)` -- which, against pairs,
    quietly selects nothing and turns the whole containment check into a pass
    over an empty list.  Rejecting the old shape outright is the only version of
    this that cannot fail silently.
    """
    out = []
    for g in r.get("gold_items") or []:
        if not isinstance(g, (list, tuple)) or len(g) != 2:
            raise TypeError(
                f"{r['id']} has a pre-reformat `gold_items` ({g!r}). Convert the "
                f"corpus with `python -m qa.migrate_pairs` -- containment cannot "
                f"be checked against a positional line, and passing here would "
                f"mean the check ran over nothing.")
        if g[1]:
            out.append(f"{g[0]}: {g[1]}")
    return out


def allowed_values(r):
    """The VALUES of a membership item's allow-list, normalised.

    `all_items` is a list of pairs under 0.1 (`["kolokacija", "mineralna voda"]`)
    and the ball's member nodes are bare phrases, so the comparison is over the
    value half.  The label half is checked by the grader, which is where it
    belongs -- here the question is only whether the ball shows enough of them.
    """
    return {sl.norm(v) for _oznaka, v in (grade.contract(r).get("all_items") or [])}


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
    g = grade.contract(r)
    n_all = int(g["n_all"])
    band = g["quantity_band"]
    if band == "exact":
        return min(int(g["n_asked"]), n_all)
    return min(5 if band == "vague_large" else 2, n_all)


def member_nodes(r, b):
    """Normalised member-set candidates the ball holds, for r's type.

    The prefix comes from `spec.MEMBER_KIND` rather than a literal, so adding a
    membership type does not silently skip this check.
    """
    kind = spec.MEMBER_KIND.get(r["type"])
    if not kind:
        return set()
    pre = kind + ": "
    return {sl.norm(t[len(pre):]) for t in b["nodes"] if t.startswith(pre)}


def membership_have(r, b):
    """Distinct allowed phrases the ball actually holds.

    Matched on `sl.norm`, the same normalisation the grader's `all_items` uses,
    and deduplicated -- an anchor can carry one phrase on two sense pairs
    (`Gadova Peč` / `Gadova peč`), which is two nodes and one gradeable answer.
    """
    return len(member_nodes(r, b) & allowed_values(r))


def contract_gap(r, b):
    """Member candidates the ball SHOWS but the contract does not ALLOW.

    Must be empty.  If it is not, the model is being shown evidence it will be
    marked wrong for using: `all_items` came from one anchor while D3 unioned
    several into the ball.  Stage 4's `build_balls.member_contract` derives the
    allow-list FROM the ball, so this is empty by construction on a current
    dataset; it stays as a guard against one built by an older pipeline.
    """
    if r.get("negative") or grade.contract(r)["mode"] != "membership":
        return set()
    return member_nodes(r, b) - allowed_values(r)


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["id"]] = r
    return out


def run(dataset, balls, show=8):
    """Is every gold item of every positive present in that item's own ball?

    0 if the corpus is sound, 1 if any membership item's contract fails to cover
    a member node its ball contains -- the model would be marked wrong for using
    evidence it was shown.
    """
    args = argparse.Namespace(dataset=dataset, balls=balls, show=show)

    gen = load(args.dataset)
    balls = load(args.balls)
    print(f"{len(gen):,} items, {len(balls):,} balls")

    hit = collections.Counter()
    tot = collections.Counter()
    items_bad = collections.Counter()
    items_n = collections.Counter()
    empty = collections.Counter()
    misses = []
    gaps = []

    # Cheap, and it fails the whole check rather than one type: a type whose gold
    # is a DRAW from a seeded sample but is graded against one fixed list makes
    # every number below meaningless.
    spec.check_set_rule()

    for iid, b in balls.items():
        r = gen.get(iid)
        if r is None or r["negative"]:
            continue
        gap = contract_gap(r, b)
        if gap:
            gaps.append((iid, r["type"], sorted(gap)[:3], len(gap)))
        gold = gold_strings(r)
        if not gold:
            continue
        t = r["type"]
        if grade.contract(r)["mode"] == "membership":
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
        # A cell the KG does not fill is an ABSENT KEY under 0.9, so there is no
        # gold item to contain and nothing reaches here.  The old positional line
        # asserted the emptiness with `velelnik: /`; those are counted separately
        # so the column does not silently go to zero and read as a regression.
        vacuous = [g for g in gold if g.rstrip().endswith("/")]
        gold = [g for g in gold if not g.rstrip().endswith("/")]
        if t in COMPOSED_VALUE:
            # The value is composed, not read (see COMPOSED_VALUE).  Counted as
            # vacuous rather than scored on whichever node happens to hold the
            # digit.
            empty[t] += len(gold)
            items_n[t] += 1
            continue
        empty[t] += len(vacuous)
        miss = [g for g in gold if not any(atoms(g) <= s for s in nodes)]
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

    if gaps:
        n_g = sum(x[3] for x in gaps)
        print(f"\n*** CONTRACT GAP: {len(gaps):,} membership items show "
              f"{n_g:,} member node(s) their `all_items` does not allow.")
        print("    The model would be marked wrong for using evidence it was "
              "shown. Run stage 4 so `finalise_contract` widens the allow-list "
              "to the ball (qa/build_balls.py); see the set rule in qa/spec.py.")
        for iid, t, ex, k in gaps[:args.show]:
            print(f"  {iid} ({t}) {k} not allowed, e.g. {ex}")
        return 1
    print("every membership item's contract covers every member node in its ball")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("balls")
    ap.add_argument("--show", type=int, default=8, help="example misses to print")
    args = ap.parse_args()
    return run(dataset=args.dataset, balls=args.balls, show=args.show)


if __name__ == "__main__":
    sys.exit(main())
