#!/usr/bin/env python3
"""The cross-cutting checks of QA_TASKS.md section 2, as runnable assertions.

    python -m qa.selftest                       # the store-free checks (C7, C16)
    python -m qa.selftest --dataset DIR         # everything, against a dataset
    python -m qa.selftest --dataset DIR --store S   # + the checks that need the graph
    python -m qa.selftest --dataset DIR --balls DIR # + the checks that need the balls

Which check is which:

  C6   Tier C leakage -- no training item anywhere contains `protipomenka`,
       `antonim` or `nasprotje`.  T16's entire value is that the relation is
       unseen, so a leak does not degrade the result, it deletes it.  Checked
       over the WHOLE item, slot values included: Group H takes the relation as
       a slot, so a held-out relation can enter training as a question's subject
       rather than as its answer.
  C7   the Slovene agreement table, the canonical orderings, the answer-label
       renderers, `norm`'s idempotence, and `sense_class`.
  C9   run the grader over the gold answers themselves: every item must score
       success against its own gold.  Free, and the one test that validates the
       grading contract end to end.
  C10  every gold matches THE one shape of 0.9 -- there is no per-type regex
       left -- no value contains ` | `, and no label contains a colon.
  C15  every gold is in its type's canonical order -- including the set-valued
       types whose order is not graded.  Grading tolerance is not a licence for
       non-canonical training data.
  C16  `sl_key` against a fixture with c-caron, s-caron, z-caron and a
       non-Slovene character.  `sl_SI` is not installed on this cluster and
       Python's default sort is wrong for Slovene.
  C17  T14's gold == len(T12's gold) for every lemma in both, and a lemma is a
       negative in both types or in neither.
  C18  D5b reproducibility: the pool is node-id sorted and the seed comes from
       the anchor's node code and nothing else.  Containment -- C18 (d) -- needs
       the written balls, so it lives in `qa/check_balls.py`.
  C11  the split is lemma-disjoint (D11), over EVERY lemma an item names rather
       than over its seed.
  C13  the majority-class baseline for every type, reported beside the score.
  C28  no POSITIVE item ships the single-node `ni v bazi` ball.  Needs --balls.

C26 (constituent validity over T30/T34) and C27 (D5c is a no-op on word balls)
are specified in QA_TASKS.md section 2 and are NOT here: neither has anything to
assert yet.  C26 needs the MWE constituent accessor that does not exist, and C27
needs D5c itself.  A check that runs over nothing passes over nothing, and would
read in this list as coverage we do not have; each lands with the change that
gives it something to check.
"""
import os
import re
import sys
import argparse
import collections

from qa import sl, spec, grade, pairs, seeds, colloc_sampling

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


def c7_labels():
    """The answer labels of 0.1, and the normalizer they are compared under.

    These renderers used to produce a question's metalanguage, where a bug reads
    as clumsy Slovene.  They now produce the ANSWER's labels, where the same bug
    silently relabels gold -- so they get a fixture rather than a reviewer.
    """
    print("\nC7 -- the answer labels (0.1) and the normalizer")
    check("nominal cell", sl.cell_label("tožilnik", "ednina") == "tožilnik ednine")
    check("verb cell",
          sl.person_label("preteklik", "1. oseba", "ednina") == "preteklik 1. osebe ednine")
    check("verb cell, gendered",
          sl.person_label("prihodnjik", "3. oseba", "množina", "ženski spol")
          == "prihodnjik 3. osebe množine (ženski spol)")
    check("18 nominal labels, all distinct", len(set(sl.GRID_LABELS)) == 18)
    check("9 verb cells, all distinct", len(set(sl.PERSON_CELLS)) == 9)
    # `rstrip(".")` can uncover whitespace the periods were hiding, and an
    # allow-list is normalised once when written and again when compared -- so a
    # normalizer that moves on the second application marks a correct answer
    # wrong.  Observed on T19-000364.
    fixture = ["  Ena   dva ...", "Trije.", "štiri", "Pet .. ", "ne-ni"]
    check("norm is idempotent",
          all(sl.norm(sl.norm(s)) == sl.norm(s) for s in fixture),
          str([sl.norm(s) for s in fixture]))
    # The round trip 0.1 rests on: render, parse, get the pairs back.
    p = [["pomen 1", "razlaga: z dvopičjem"], ["pomen 2", "druga"]]
    check("render/parse round-trips a value containing a colon",
          grade.parse(pairs.render(p))
          == sorted((sl.norm(k), sl.norm(v)) for k, v in p))


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


def corpus_is_pairs(items):
    """Precondition, not a numbered check: is this corpus in the 0.1 shape?

    Every check below reads `gold_items` as `[oznaka, vrednost]` pairs.  Against
    a pre-reformat corpus they do not fail cleanly -- they raise from inside a
    comprehension, or worse, filter to nothing and pass over an empty list.  So
    the shape is settled once, in one line, before anything is scored.
    """
    bad = collections.Counter()
    for it in items:
        if it.get("negative"):
            continue
        for g in it.get("gold_items") or []:
            if not isinstance(g, (list, tuple)) or len(g) != 2:
                bad[it["type"]] += 1
                break
    ok = check("gold_items are [oznaka, vrednost] pairs", not bad,
               str(dict(bad.most_common(6))) if bad else "")
    if not bad:
        return True
    print("    this corpus predates the labelled-pair rule (QA_TASKS.md 0.1). "
          "Convert it with `python -m qa.migrate_pairs`, or rebuild it; the "
          "checks below are skipped rather than run against a shape they cannot "
          "read.")
    return ok


def c9_gold_grades(items, pre_ball=False):
    print("\nC9 -- the grader scores every gold answer as correct")
    bad = collections.Counter()
    graded = skipped = 0
    for it in items:
        if pre_ball and grade.needs_ball(it):
            skipped += 1
            continue
        graded += 1
        r = grade.grade(it, it["answer"])
        if not r["success"]:
            bad[(it["type"], r["reason"])] += 1
    detail = f"{graded:,} items"
    if skipped:
        detail += (f" ({skipped:,} membership positives not gradeable yet -- "
                   f"stage 5 writes their allow-list)")
    check("gold == 100 % success", not bad,
          str(dict(bad.most_common(8))) if bad else detail)


def c10_shape(items):
    """One regex for all 34 types (0.9), and the two rules that keep it parseable.

    There is no per-type regex left and no single-item exemption.  The value rule
    is what the T19 and T33 seed filters must satisfy: a corpus sentence
    containing ` | ` would split into two fields and take its label with it.
    """
    print("\nC10 -- THE one shape; no ` | ` inside a field, no colon inside a label")
    shape = re.compile(spec.SHAPE)
    bad_re = collections.Counter()
    bad_sep = collections.Counter()
    bad_label = collections.Counter()
    for it in items:
        line = it["answer"].splitlines()[0]
        if it["negative"]:
            if line != spec.sentinel_line():
                bad_re[it["type"]] += 1
            continue
        if not shape.match(line):
            bad_re[it["type"]] += 1
        for oznaka, vrednost in it["gold_items"]:
            if " | " in vrednost or " | " in oznaka:
                bad_sep[it["type"]] += 1
            if ":" in oznaka:
                bad_label[it["type"]] += 1
    check("the one shape", not bad_re, str(dict(bad_re)))
    check("no ` | ` inside a field", not bad_sep, str(dict(bad_sep)))
    check("no colon inside a label", not bad_label, str(dict(bad_label)))


def c15_canonical(items):
    """Set-valued golds must be in a TOTAL canonical order, not merely a sorted one.

    Checked against `sl_sort_key`, not `sl_key`.  `sl_key` casefolds, so
    `zastava` and `Zastava` tie under it -- and `sorted` is stable, so a list in
    either order passes a comparison against `sorted(g, key=sl_key)`.  That is
    precisely how a set-iteration-order dependency survived here: the check
    could not see the thing it existed to catch.

    An unstable order never changed a score -- and under 0.1 it cannot, because
    nothing is positional any more.  It changed whether the corpus could be
    rebuilt byte-for-byte, which is what this asserts.  Grading tolerance is not
    a licence for non-canonical training data: the model must see exactly one
    ordering for a given set, or it is being taught noise on a surface it is
    forced to emit.
    """
    print("\nC15 -- every set-valued gold is in canonical order")
    bad = collections.Counter()
    for it in items:
        if it["negative"] or it["type"] not in spec.VALUE_SORTED:
            continue
        # By VALUE: these types carry one repeated label (`sopomenka`), so
        # sorting the pairs would sort on a constant and assert nothing.
        g = [v for _oznaka, v in it["gold_items"]]
        if g != sorted(g, key=sl.sl_sort_key):
            bad[it["type"]] += 1
    check("sl_sort_key order", not bad, str(dict(bad)))


def _slot_text(item):
    """Every string the item's slots carry, as one casefolded blob.

    A slot value is neither question nor answer and was checked as neither.  It
    has to be: T23/T24/T25/T27 take the RELATION as a slot, so a held-out
    relation reaches training as the question's subject -- and `Ali ima beseda X
    protipomenko?` leaks T16 whether its gold says `da` or `ne`.  The slot is
    also where a rendering bug hides one: a frame that drops a slot still leaves
    it on the row.
    """
    return " ".join(str(v) for v in (item.get("slots") or {}).values()).casefold()


def c6_tier_c(splits):
    print("\nC6 -- Tier C leakage")
    hard = collections.Counter()
    hard_slot = collections.Counter()
    soft_q = collections.Counter()
    soft_a = collections.Counter()
    for s in ("train", "dev"):
        for it in splits[s]:
            q = it["question"].casefold()
            a = it["answer"].casefold()
            sl_text = _slot_text(it)
            for w in spec.TIER_C_TAG_WORDS:
                if w in q or w in a:
                    hard[w] += 1
                elif w in sl_text:
                    hard_slot[w] += 1
            for w in spec.TIER_C_SOFT_WORDS:
                if w in q:
                    soft_q[w] += 1
                elif w in a:
                    soft_a[w] += 1
    check("no antonym TAG WORD in train/dev, anywhere", not hard, str(dict(hard)))
    check("no held-out relation as a train/dev SLOT VALUE", not hard_slot,
          str(dict(hard_slot)))
    check("no 'opposite' vocabulary in a train/dev QUESTION", not soft_q,
          str(dict(soft_q)))
    # Reported, not asserted: these occur inside curated definitions and corpus
    # sentences, which are data, not phrasing we chose.  See spec.TIER_C_*.
    print(f"    'opposite' vocabulary inside a train/dev ANSWER: "
          f"{sum(soft_a.values())} items {dict(soft_a)} (allowed)")
    tc = [it for s in splits for it in splits[s] if it["type"] in spec.TIER_C]
    check("every Tier C item is a test item",
          all(it["split"] == "test" for it in tc), f"{len(tc)} items")


#: Slot keys that name a LEMMA the item is about, beyond its seed.  `L2` is
#: Group H's second anchor (T25-T27); `L` and `F` are the seed's own lemma and
#: one of its forms, and a form belongs to its lemma's entry, so it is the lemma
#: behind it that has to be disjoint -- `lemma` already carries that.
LEMMA_SLOTS = ("L2",)


def item_lemmas(item):
    """Every lemma this item names, not just the one it was seeded from.

    T25, T26 and T27 name two.  A test item whose SECOND lemma is a training seed
    breaks lemma-disjointness exactly as a duplicated seed would, and checking
    only `lemma` would never see it -- the pair filter (H.3) is the generation
    rule, this is the assertion that it held.
    """
    out = {item["lemma"]}
    slots = item.get("slots") or {}
    out |= {str(slots[k]) for k in LEMMA_SLOTS if slots.get(k)}
    return out


def c11_split(splits):
    print("\nC11 -- the split is lemma-disjoint, over every lemma an item names")
    by = {s: {l for it in v for l in item_lemmas(it)} for s, v in splits.items()}
    overlap = {}
    for a in ("train", "dev", "test"):
        for b in ("train", "dev", "test"):
            if a < b and by[a] & by[b]:
                overlap[f"{a}/{b}"] = sorted(by[a] & by[b])[:5]
    second = sum(1 for v in splits.values() for it in v if len(item_lemmas(it)) > 1)
    check("no lemma on two sides", not overlap,
          str(overlap) if overlap else f"{sum(map(len, by.values())):,} lemmas, "
          f"{second:,} items naming a second one")


def c17_t12_t14(splits):
    print("\nC17 -- T14 == len(T12), and negatives agree")
    t12, t14 = {}, {}
    for s in splits:
        for it in splits[s]:
            if it["type"] == "T12":
                t12[it["lemma"]] = (it["negative"], len(it["gold_items"]))
            elif it["type"] == "T14":
                # `[["število pomenov", "3"]]` -- the VALUE is the count.
                t14[it["lemma"]] = (it["negative"],
                                    int(it["gold_items"][0][1]) if not it["negative"] else 0)
    shared = set(t12) & set(t14)
    bad_count = {l for l in shared
                 if not t12[l][0] and not t14[l][0] and t12[l][1] != t14[l][1]}
    bad_neg = {l for l in shared if t12[l][0] != t14[l][0]}
    check(f"count matches the list ({len(shared)} shared lemmas)", not bad_count,
          str(sorted(bad_count)[:5]))
    check("negative in both or in neither", not bad_neg, str(sorted(bad_neg)[:5]))


def c28_balls_resolve(balls, items):
    """C28 -- no POSITIVE item ships the single-node `ni v bazi` ball.

    This is the check the phrase types would most have benefited from having
    earlier.  A subject the surface index cannot resolve still produces a
    well-formed ball, a well-formed question and a gold answer nothing in it
    supports; `qa/build_balls.py` counts these as `empty_ball`, and this is the
    assertion that the count is zero where it matters rather than merely printed.

    For a NEGATIVE the single-node ball is correct and expected -- the entity is
    not in the graph, which is what the item says.  So the check is over
    positives, which is also what makes it non-vacuous before the phrase types
    exist: 11,179 items today, every one of them a single word.

    Multi-word subjects are reported separately.  They are the cohort that will
    start failing this the moment a phrase type has positives, and today they are
    42 items, all of them T8 negatives, 37 of which resolve to nothing at all.
    """
    print("\nC28 -- every positive item's ball resolves")
    # A ball row carries no slots, so the subject comes from the generated record
    # it was built from, joined on id.
    subject = {it["id"]: str((it.get("slots") or {}).get("L") or "") for it in items}
    bad = collections.Counter()
    positives = 0
    mw_total = mw_empty = 0
    for r in balls:
        nodes = r.get("nodes") or []
        empty = len(nodes) == 1 and "(ni v bazi)" in nodes[0]
        multiword = " " in subject.get(r["id"], "")
        mw_total += multiword
        mw_empty += multiword and empty
        if r.get("negative"):
            continue
        positives += 1
        if empty:
            bad[r["type"]] += 1
    check("no positive ships the `ni v bazi` ball", not bad,
          str(dict(bad)) if bad else f"{positives:,} positives")
    print(f"    multi-word subjects: {mw_total} items, {mw_empty} of them "
          f"unresolved (reported -- the phrase cohort)")


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
    """C18 (a)-(c): the sampler is reproducible.

    Containment -- C18 (d), every gold item is in the ball -- is NOT checked
    here.  It cannot be: this module only has the store, and an item's real
    input is the union of every anchor the extractor returned, under the
    builder's own K, holding forms and senses and MWEs as well as the
    collocation slice.  `qa/check_balls.py` reads the written balls and asks
    the question directly.
    """
    print("\nC18 -- D5b reproducibility")
    anchors = [it for it in items if it["type"] == "T17" and not it["negative"]]
    if not anchors:
        check("T17 items present", False)
        return
    sample = anchors[:200]
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
def run(dataset=None, store=None, pre_ball=False, balls=None):
    """Every check in QA_TASKS.md section 2.  0 if all passed, 1 if any failed.

    `FAIL` is module state, so it is cleared on entry: the pipeline calls this
    twice in one process -- once on the generated set, once on the relabelled
    one -- and without the reset the second call would inherit the first's
    failures and report them against the wrong dataset.

    `pre_ball=True` for a dataset that has not reached stage 5: its membership
    positives have no allow-list yet, so C9 skips them and says how many.  It is
    the caller that knows which stage its dataset came from, not this module.
    """
    del FAIL[:]

    args = argparse.Namespace(dataset=dataset, store=store, balls=balls)

    c16_sl_key()
    c7_agreement()
    c7_labels()

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
        if not corpus_is_pairs(allitems):
            print("\nSELFTEST:", f"FAIL ({len(FAIL)}): {FAIL}")
            return 1
        c9_gold_grades(allitems, pre_ball=pre_ball)
        c10_shape(allitems)
        c15_canonical(allitems)
        c6_tier_c(splits)
        c11_split(splits)
        c17_t12_t14(splits)
        if args.balls:
            ballrows = [r for v in load(args.balls).values() for r in v]
            c28_balls_resolve(ballrows, allitems)
        c13_baselines(splits)
        if store:
            c18_sampling(store, allitems)

    print("\nSELFTEST:", "PASS" if not FAIL else f"FAIL ({len(FAIL)}): {FAIL}")
    return 0 if not FAIL else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset")
    ap.add_argument("--store")
    ap.add_argument("--balls", help="the written balls, for C28")
    args = ap.parse_args()
    return run(dataset=args.dataset, store=args.store, balls=args.balls)


if __name__ == "__main__":
    sys.exit(main())
