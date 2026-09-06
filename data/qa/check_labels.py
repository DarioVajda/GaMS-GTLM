#!/usr/bin/env python3
"""C25: is every answer label READ OFF the ball, or invented by the generator?

    python -m qa.check_labels datasets/balls

QA_TASKS.md 0.1 makes the answer label its own content with the vocabulary the
graph uses to label it -- `sopomenka: pasiven`, `mestnik ednine: gori`.  The
whole point is the unseen relation: `protipomenka: sodelovanje` is constructible
by a model supervised only on synonyms, because the label is sitting in the ball
and can be copied.

Nothing in the OUTPUT enforces that.  A generator reading the tag off the node
and a generator holding `SPEC["T15"]["label"] = "sopomenka"` emit byte-identical
answers, score identically, and differ by one line that looks equally reasonable
in review.  They ask the model to do two different things: copy a label it can
see, or recall one it cannot.  The second is the 19-way classifier this format
was designed to delete, wearing the new format's clothes.

So the check is not about generator style.  It is a property of each ITEM: is
the label string this item's answer must emit actually visible in the ball this
item ships?

**The five declared constants are a budget, not an exemption.**  Their count is
how many labels the model must still memorise rather than read.  A new type is
free to need a sixth -- but the diff that adds it also grows DECLARED below,
where somebody reviews it, instead of disappearing into a per-type table.

Matching is WHOLE-WORD over node text only, never over the answer.  Substrings
lie here: `vid` occurs inside `dvovidski` and scores the aspect label 38 %
derived when the true figure is 0.2 %, all of it coincidence (`kolokacija:
okvariti vid`).

Positives only -- a negative's answer is the sentinel and carries no labels.

Two modes.  The default takes each item's labels from its own `gold_items`
pairs -- exact, per item, and pass/fail: this is C25 proper, and it is the
default now that the corpus carries pairs.

`--legacy` takes them from the per-type candidate table below instead, for a
corpus still in the pre-reformat shape.  That table is a SUPERSET of what any one
item emits, so it can over-report a missing label and never under-report one; it
reports and does not fail.  Its one run against the pre-reformat corpus produced
the table in QA_TASKS.md 0.1, and every apparent miss there was an item whose
answer has no such pair (a `/` gap in T7 and T11, a non-noun in T8) -- the check
confirming itself.
"""
import argparse
import collections
import json
import pathlib
import re
import sys
import unicodedata

# ── The eight labels the graph does not carry (QA_TASKS.md 0.1) ─────────────
# The list itself lives in `qa/spec.py`, with the reasoning for each entry, so
# that this check and the containment check (C18 d, `qa/check_balls.py`) excuse
# exactly the same eight strings.  Two copies would drift, and the direction they
# would drift in is a label quietly becoming unreadable in one check and declared
# in the other.
#
# `več` / `manj` / `enako` are the only three labels the fifteen Group H types
# add -- every other one of them answers with a label some existing type already
# uses, which is the property that makes those types indistinguishable from each
# other in output space.
from qa.spec import DECLARED_LABELS as DECLARED

CASES = ("imenovalnik", "rodilnik", "dajalnik", "tožilnik", "mestnik", "orodnik")
NUMS = ("ednine", "dvojine", "množine")
GRID = tuple(f"{c} {n}" for n in NUMS for c in CASES)
TENSES = ("sedanjik", "preteklik", "prihodnjik")
PERSONS = tuple(f"{t} {p}. osebe {n}" for t in TENSES for p in (1, 2, 3) for n in NUMS)

LEGACY = {
    "T1": GRID, "T2": GRID, "T3": GRID, "T20": GRID, "T21": GRID,
    "T4": ("iztočnica",),
    "T5": PERSONS, "T6": PERSONS,
    "T7": ("nedoločnik", "namenilnik", "velelnik"),
    "T8": ("besedna vrsta", "spol", "vid"),
    "T9": ("spol",), "T10": ("vid",),
    "T11": ("osnovnik", "primernik", "presežnik"),
    "T12": ("pomen",), "T14": ("število pomenov",),
    "T15": ("sopomenka",), "T16": ("protipomenka",),
    "T17": ("kolokacija",), "T19": ("zgled",),
}

# An answer declines its labels (`tožilnik ednine`, `3. osebe`) where the node
# text states them in the nominative.  Closed table, same discipline as
# check_balls.py -- never a stemmer.
NORM = {
    "ednine": "ednina", "dvojine": "dvojina", "množine": "množina",
    "osebe": "oseba", "pomenov": "pomen", "pomena": "pomen", "pomeni": "pomen",
    "moškega": "moški", "ženskega": "ženski", "srednjega": "srednji",
    "spola": "spol",
}


def nfc(s):
    return unicodedata.normalize("NFC", s).casefold()


def atoms(oznaka):
    """`sedanjik 1. osebe ednine` -> the words a ball must carry for it.

    Parentheses come off with the punctuation.  A composed tense that agrees for
    gender is labelled `preteklik 3. osebe ednine (ženski spol)`, mirroring how
    0.5 renders a feature bundle in the node text -- and the node text writes the
    bundle's own parentheses around a different set of words, so `(ženski` would
    never be found as a literal.
    """
    out = []
    for w in nfc(oznaka).split():
        w = w.strip(".,()")
        if not w or w.isdigit():          # the person ordinal is positional noise
            continue
        out.append(NORM.get(w, w))
    return out


def word(text, s):
    return re.search(r"(?<![^\W\d_])" + re.escape(s) + r"(?![^\W\d_])", text)


def carries(text, oznaka):
    return all(word(text, a) for a in atoms(oznaka))


def is_declared(oznaka):
    """A constant may be one COMPONENT of a compound label.

    T5's cells read `preteklik 1. osebe ednine`: the person and number halves are
    derived, the tense half is not, and the label as a whole is only as readable
    as its worst component.
    """
    return any(word(nfc(oznaka), c) for c in DECLARED)


def labels_of(item, legacy):
    if not legacy:
        return [o for o, _ in item["gold_items"]]
    return LEGACY.get(item["type"], ())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("balls", type=pathlib.Path,
                    help="directory holding {train,dev,test}.jsonl WITH nodes")
    ap.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False,
                    help="take labels from the per-type table rather than from "
                         "gold_items pairs, for a pre-reformat corpus; reports "
                         "rather than fails")
    args = ap.parse_args(argv)

    hit = collections.Counter()
    tot = collections.Counter()
    misses = collections.defaultdict(list)
    items = 0
    for split in ("train", "dev", "test"):
        path = args.balls / f"{split}.jsonl"
        if not path.exists():
            continue
        with path.open() as fh:
            for line in fh:
                d = json.loads(line)
                if d.get("negative"):
                    continue
                text = nfc(" \n".join(d["nodes"]))
                items += 1
                for oznaka in labels_of(d, args.legacy):
                    key = (d["type"], oznaka)
                    tot[key] += 1
                    if carries(text, oznaka):
                        hit[key] += 1
                    else:
                        misses[key].append(d["id"])

    derived = constant = partial = 0
    print(f"{'type':5} {'oznaka':34} {'in ball':>17}  status")
    print("-" * 74)
    for key in sorted(tot, key=lambda k: (int(k[0][1:]), k[1])):
        t, oznaka = key
        n, h = tot[key], hit[key]
        if is_declared(oznaka):
            status, constant = "CONSTANT (declared)", constant + 1
        elif h == n:
            status, derived = "derived", derived + 1
        else:
            status, partial = f"partial -- {n - h} item(s) below", partial + 1
        print(f"{t:5} {oznaka:34} {h:6}/{n:<6} {100*h/n:5.1f}%  {status}")

    if partial:
        print("\nItems whose ball lacks a label the type CAN use. In --legacy the "
              "label set is a per-type superset, so each of these is expected to "
              "be an item that emits no such pair -- confirm, do not assume:")
        for key in sorted(misses, key=lambda k: (int(k[0][1:]), k[1])):
            if is_declared(key[1]):
                continue
            print(f"  {key[0]:5} {key[1]:26} {', '.join(misses[key])}")

    print(f"\n{items:,} positive items · {derived} labels derived from the ball · "
          f"{constant} label(s) hitting a declared constant · {partial} partial")

    if not args.legacy:
        bad = [k for k in tot if hit[k] != tot[k] and not is_declared(k[1])]
        if bad:
            print("\nC25 FAILS. A label outside DECLARED is not read from the "
                  "graph, so the model must memorise it per type -- the defect "
                  "QA_TASKS.md 0.1 removes. Derive it, or add it to DECLARED and "
                  "accept the cost.", file=sys.stderr)
            return 1
        return 0

    print("--legacy is a report, not a verdict: the per-type label table is a "
          "superset of what any item emits, so a miss is not yet evidence of a "
          "defect. C25 becomes a pass/fail check under --no-legacy, once "
          "gold_items carries real pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
