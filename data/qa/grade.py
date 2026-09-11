#!/usr/bin/env python3
"""The grader.  A script, not a model -- QA_TASKS.md 0.8.

No embedding model, no LLM judge, no fuzzy similarity.  Every gradeable string is
present verbatim in the model's input (these types are retrieval probes over an
extracted subgraph), so exact match after a fixed shallow normalization is both
fair and meaningful; a model that paraphrases has not done the task.

One headline number: `success`, the percentage of items whose answer satisfies
their type's condition.  `f1` is computed per item as a DIAGNOSTIC -- over PAIRS,
so it now separates "right value under the wrong label" from "wrong value", which
the positional version could not -- and is never reported as a score.

Two modes, one constant per type, never inferred at run time:

    pairs        the answer is a multiset of `oznaka: vrednost` (0.1)
    membership   a DRAW from a set: subset of ALL, plus a count rule (0.8.3)

`sequence` and `multiset` are gone.  They differed only in whether position was
compared, and under 0.1 nothing is positional: what a position used to carry, the
label carries.  The one comparison covers a paradigm cell and a synonym list
alike.

The grader never opens the store.  Everything it needs -- including the full
`ALL(anchor)` set for membership items -- travels inside the item.
"""
import json
import argparse
import collections

from . import spec
from .sl import norm

PREFIX = "ODGOVOR:"

# The contract has two layers, and confusing them is what produces a whole run
# graded against a stale constant.
#
#   TYPE level   `mode` -- one constant per type, the same for all 12,490 items.
#                It lives in `qa/spec.py` and is read from there, NEVER from the
#                row.  Copying it into every row is how T19 went on being graded
#                `sequence` for a whole run after the spec said otherwise: a
#                dataset on disk is a snapshot of what the spec said the day it
#                was written, and there is no mechanism by which it learns
#                otherwise.  0.1 shrank this layer from four fields to one, which
#                is most of the point.
#
#   ITEM level   `all_items`, `n_all`, `n_pool`, `quantity_band`, `n_asked` --
#                facts about THIS item that the type cannot know.  These live in
#                the row, because that is the only place they can live.
#
# `spec` may supply a DEFAULT for an item-level field when the type fixes it for
# every item (T19 asks for exactly one example), and the row always wins.
TYPE_LEVEL = ("mode",)
ITEM_DEFAULTS = {"quantity_band": "band", "n_asked": "n_asked"}

#: The fields an older pipeline wrote into the row and 0.1 deleted.  Named here
#: so `qa/build_balls.py` can strip them and say how many it stripped, rather
#: than silently carrying a shape contract that no longer means anything.
RETIRED_ROW_FIELDS = ("sep", "arity", "regex")


def contract(item):
    """The grading contract for `item`: type constants + this item's facts.

    The single accessor.  Nothing should read `item["grading"]` directly -- that
    is what lets a stale row override the spec.
    """
    sp = spec.SPEC[item["type"]]
    g = dict(item.get("grading") or {})
    for k in RETIRED_ROW_FIELDS:
        g.pop(k, None)
    g.update({k: sp[k] for k in TYPE_LEVEL})
    for field, key in ITEM_DEFAULTS.items():
        if g.get(field) is None and sp.get(key) is not None:
            g[field] = sp[key]
    return g


def parse(answer):
    """The first ODGOVOR: line as a sorted multiset of (oznaka, vrednost).

    None if there is no such line or a field carries no label -- an unlabelled
    field is unparseable rather than a value with an empty label, because the
    whole content of 0.1 is that the answer says what each value IS.

    Split on ` | `, then on the FIRST `: ` in each field: a label may not contain
    a colon and a value may, which is what makes a definition or a corpus
    sentence safe as a value with no escaping and no per-type separator.

    The 0.2 sentinel has no `: ` at all and so parses to None by construction.
    It is compared as a line, before this function is reached.
    """
    if answer is None:
        return None
    for line in answer.splitlines():
        line = line.strip()
        if not line.startswith(PREFIX):
            continue
        body = line[len(PREFIX):].strip()
        out = []
        for field in body.split(" | "):
            oznaka, sep, vrednost = field.partition(": ")
            if not sep:
                return None
            out.append((norm(oznaka), norm(vrednost)))
        return sorted(out)
    return None


def count_ok(k, band, n_all, n_asked=None, n_pool=None):
    """QA_TASKS.md Group F.  Under-supply is never an error -- 3 phrases where 5
    were asked for is the refusal-to-invent case.

    Two supplies, and they bound opposite ends.  The CEILING is `n_all`, the
    store's set, which is what may truthfully be named.  The FLOOR is `n_pool`,
    what the ball let the target be drawn from (stage 5 writes it): the model
    can only be required to produce what it was shown.  Using `n_all` for both
    marked an honest 7 wrong on "exactly 8" when the ball held 7 of 30.  A row
    without `n_pool` grades exactly as before.
    """
    lo = n_all if n_pool is None else min(int(n_pool), n_all)
    if band == "none":
        return min(2, lo) <= k <= min(15, n_all) or k == n_all
    if band == "vague_small":
        return min(2, lo) <= k <= min(6, n_all) or k == n_all
    if band == "vague_large":
        return k >= min(5, lo)                   # deliberately uncapped
    if band == "exact":
        return min(int(n_asked), lo) <= k <= min(int(n_asked), n_all)
    raise ValueError(f"unknown quantity band {band!r}")


def _f1(pred, gold):
    if not pred and not gold:
        return 1.0
    pc, gc = collections.Counter(pred), collections.Counter(gold)
    overlap = sum((pc & gc).values())
    if not overlap:
        return 0.0
    p = overlap / len(pred)
    r = overlap / len(gold)
    return 2 * p * r / (p + r)


def _allow_pairs(item, allow):
    """A membership item's allow-list as normalised (oznaka, vrednost) pairs.

    0.1 made `all_items` a list of PAIRS, because the answer names its members as
    `kolokacija: mineralna voda` and a bare phrase would let a true phrase in
    under a relation the item never asked about.  A pre-reformat row holds bare
    strings, and unpacking those characterwise is a `ValueError` five frames deep
    that says nothing about the corpus -- so it is named here instead.
    """
    out = set()
    for x in allow:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(
                f"{item.get('id', '?')} carries a pre-reformat `all_items` "
                f"({x!r}). This corpus predates the labelled-pair rule "
                f"(QA_TASKS.md 0.1); convert it with `python -m qa.migrate_pairs` "
                f"or rebuild it. Nothing here can grade a positional line.")
        out.add((norm(x[0]), norm(x[1])))
    return out


def _sentinel(answer):
    """True if the first ODGOVOR: line is exactly the 0.2 sentinel."""
    if answer is None:
        return False
    for line in answer.splitlines():
        line = line.strip()
        if line.startswith(PREFIX):
            return norm(line[len(PREFIX):]) == norm(spec.SENTINEL)
    return False


def grade(item, prediction):
    """-> {'success': bool, 'f1': float, 'reason': str}

    `reason` is a short machine-readable tag for the failure, so a failure
    breakdown is a Counter and not a regex over prose.
    """
    g = contract(item)
    mode = g["mode"]

    # The sentinel is settled first, in both directions.  It carries no pairs, so
    # it cannot be compared as one; and a model that answers the sentinel to a
    # positive item has abstained, which is a wrong answer and not an
    # unparseable one -- the distinction the Tier C false-sentinel rate is about.
    gold_is_sentinel = _sentinel(item["answer"])
    pred_is_sentinel = _sentinel(prediction)
    if gold_is_sentinel or pred_is_sentinel:
        ok = gold_is_sentinel and pred_is_sentinel
        return {"success": ok, "f1": 1.0 if ok else 0.0,
                "reason": "ok" if ok else
                          ("false_sentinel" if pred_is_sentinel else "missed_sentinel")}

    pred = parse(prediction)
    gold = parse(item["answer"])
    if pred is None:
        return {"success": False, "f1": 0.0, "reason": "unparseable"}

    if mode == "pairs":
        ok = pred == gold
        return {"success": ok, "f1": _f1(pred, gold),
                "reason": "ok" if ok else "mismatch"}

    if mode == "membership":
        f1 = _f1(pred, gold)
        deduped = list(dict.fromkeys(pred))
        if len(deduped) != len(pred):
            return {"success": False, "f1": f1, "reason": "repeated_item"}
        if g.get("all_items") is None:
            raise ValueError(
                f"{item.get('id', '?')} ({item['type']}) is graded "
                f"`membership` but its row carries no `all_items`. The "
                f"allow-list is per-ITEM data derived from the ball, so it "
                f"cannot come from `qa/spec.py` -- rebuild the dataset with "
                f"`qa/build_balls.py` (stage 5), which writes it.  Grading a "
                f"dataset that has not reached stage 5 yet is not an error: "
                f"pass `pre_ball=True` to skip these items rather than "
                f"raising on them.")
        allow = _allow_pairs(item, g["all_items"])
        outside = [p for p in pred if p not in allow]
        if outside:
            return {"success": False, "f1": f1, "reason": "not_in_all"}
        if not count_ok(len(pred), g["quantity_band"], g["n_all"],
                        g.get("n_asked"), g.get("n_pool")):
            return {"success": False, "f1": f1, "reason": "bad_count"}
        return {"success": True, "f1": f1, "reason": "ok"}

    raise ValueError(f"unknown grading mode {mode!r}")


def needs_ball(item):
    """True if this item cannot be graded until stage 5 has written its ball.

    A membership POSITIVE is scored against `grading.all_items`, an allow-list
    derived from the ball over all of the item's anchors -- so it does not exist
    before `qa/build_balls.py` runs.  A negative is scored on the sentinel alone
    and is gradeable from the moment it is generated.
    """
    if spec.SPEC[item["type"]]["mode"] != "membership":
        return False
    if item.get("negative"):
        return False
    return (item.get("grading") or {}).get("all_items") is None


def grade_all(items, predictions, pre_ball=False):
    """items: [dict], predictions: {item id -> str}.  -> (rows, summary).

    `pre_ball=True` says the dataset has not reached stage 5, so the membership
    positives have no allow-list yet: they are skipped and counted rather than
    raising.  Everything else is graded exactly as it would be at the end.
    """
    skipped = 0
    rows = []
    for it in items:
        if pre_ball and needs_ball(it):
            skipped += 1
            continue
        r = grade(it, predictions.get(it["id"]))
        r["id"] = it["id"]
        r["type"] = it["type"]
        r["band"] = it["band"]
        r["negative"] = bool(it.get("negative"))
        rows.append(r)
    s = summarise(rows)
    s["skipped_pre_ball"] = skipped
    return rows, s


def summarise(rows):
    def pct(sel):
        sel = list(sel)
        return (100.0 * sum(r["success"] for r in sel) / len(sel)) if sel else float("nan")

    out = {"n": len(rows), "success": pct(rows),
           "by_type": {}, "by_band": {}, "reasons": {}}
    for key in ("type", "band"):
        seen = sorted({r[key] for r in rows})
        out[f"by_{key}"] = {v: pct(r for r in rows if r[key] == v) for v in seen}
    out["reasons"] = dict(collections.Counter(r["reason"] for r in rows))
    out["success_positive"] = pct(r for r in rows if not r["negative"])
    out["success_negative"] = pct(r for r in rows if r["negative"])
    return out


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def run(items, predictions=None, verbose=True, pre_ball=False):
    """Grade `items` (a .jsonl path) and return the summary dict.

    With no `predictions`, grades the gold against itself -- C9, which must come
    out at 100 % on every split.  The pipeline calls this per split and reads
    `success`; `verbose=False` keeps the per-type tables out of its log.
    `pre_ball=True` skips the membership positives, which have no allow-list
    before stage 5 -- see `needs_ball`.
    """
    items = read_jsonl(items)
    if predictions:
        preds = {r["id"]: r["prediction"] for r in read_jsonl(predictions)}
    else:
        preds = {it["id"]: it["answer"] for it in items}

    rows, s = grade_all(items, preds, pre_ball=pre_ball)
    if not verbose:
        return s
    print(f"items: {s['n']:,}   success: {s['success']:.2f}%   "
          f"(positives {s['success_positive']:.2f}%, "
          f"negatives {s['success_negative']:.2f}%)")
    print("\nby type:")
    for t, v in sorted(s["by_type"].items()):
        print(f"  {t:5s} {v:6.2f}%")
    print("\nby band:")
    for b, v in sorted(s["by_band"].items()):
        print(f"  {b:5s} {v:6.2f}%")
    print("\nfailure reasons:", s["reasons"])
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("items", help="the dataset .jsonl")
    ap.add_argument("--predictions", help="jsonl of {'id':..., 'prediction':...}; "
                                          "omit to grade the gold against itself (C9)")
    ap.add_argument("--json", action="store_true", help="dump the summary as JSON")
    args = ap.parse_args()

    s = run(items=args.items, predictions=args.predictions,
            verbose=not args.json)
    if args.json:
        print(json.dumps(s, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
