#!/usr/bin/env python3
"""The grader.  A script, not a model -- QA_TASKS.md 0.8.

No embedding model, no LLM judge, no fuzzy similarity.  Every gradeable string is
present verbatim in the model's input (these types are retrieval probes over an
extracted subgraph), so exact match after a fixed shallow normalization is both
fair and meaningful; a model that paraphrases has not done the task.

One headline number: `success`, the percentage of items whose answer satisfies
their type's condition.  `f1` is computed per item as a DIAGNOSTIC -- it
separates "missed a sense" from "invented one" from "right content, wrong order"
-- and is never reported as a score.

Three modes, one constant per type, never inferred at run time:

    sequence     positions carry meaning (a paradigm cell, a tense)
    multiset     a set whose order is our convention, not the data's
    membership   collocations: subset of ALL, plus a count rule (0.8.3)

The grader never opens the store.  Everything it needs -- including the full
`ALL(anchor)` set for membership items -- travels inside the item.
"""
import re
import json
import argparse
import collections

from . import spec
from .sl import norm

PREFIX = "ODGOVOR:"

# The contract has two layers, and confusing them is what produced the T19 and
# T17 defects of 2026-08-23.
#
#   TYPE level   `mode`, `sep`, `arity`, `regex` -- one constant per type, the
#                same for all 12,490 items.  These live in `qa/spec.py` and are
#                read from there, NEVER from the row.  Copying them into every
#                row is how T19 went on being graded `sequence` for a whole run
#                after the spec said otherwise: a dataset on disk is a snapshot
#                of what the spec said the day it was written, and there is no
#                mechanism by which it learns otherwise.
#
#   ITEM level   `all_items`, `n_all`, `quantity_band`, `n_asked` -- facts about
#                THIS item that the type cannot know.  These live in the row,
#                because that is the only place they can live.
#
# `spec` may supply a DEFAULT for an item-level field when the type fixes it for
# every item (T19 asks for exactly one example), and the row always wins.
TYPE_LEVEL = ("mode", "sep", "arity", "regex")
ITEM_DEFAULTS = {"quantity_band": "band", "n_asked": "n_asked"}


def contract(item):
    """The grading contract for `item`: type constants + this item's facts.

    The single accessor.  Nothing should read `item["grading"]` directly -- that
    is what lets a stale row override the spec.
    """
    sp = spec.SPEC[item["type"]]
    g = dict(item.get("grading") or {})
    g.update({k: sp[k] for k in TYPE_LEVEL})
    for field, key in ITEM_DEFAULTS.items():
        if g.get(field) is None and sp.get(key) is not None:
            g[field] = sp[key]
    return g


def parse(answer, sep=" | ", arity=None):
    """The first ODGOVOR: line, split and normalized.  None if unparseable.

    An `arity: 1` type is NEVER split: 65 % of usage examples contain a comma and
    4 of 51,172 contain a `|`, so splitting a single-item answer would shred it
    (QA_TASKS.md 0.9).

    `sep` may be a LIST, for the types whose line is grouped ("ednina; dvojina;
    množina", "sedanjik; preteklik; prihodnjik").  Splitting on any of them
    recovers the flat positional list the grader compares.
    """
    if answer is None:
        return None
    for line in answer.splitlines():
        line = line.strip()
        if not line.startswith(PREFIX):
            continue
        body = line[len(PREFIX):].strip()
        if arity == 1 or sep in (None, "", "—"):
            return [norm(body)]
        if isinstance(sep, (list, tuple)):
            pat = "|".join(re.escape(s) for s in sep)
            return [norm(x) for x in re.split(pat, body)]
        return [norm(x) for x in body.split(sep)]
    return None


def count_ok(k, band, n_all, n_asked=None):
    """QA_TASKS.md Group F.  `min(., n_all)` everywhere: under-supply is never an
    error -- 3 phrases where 5 were asked for is the refusal-to-invent case."""
    if band == "none":
        return 2 <= k <= min(15, n_all) or k == n_all
    if band == "vague_small":
        return 2 <= k <= min(6, n_all) or k == n_all
    if band == "vague_large":
        return k >= min(5, n_all)                # deliberately uncapped
    if band == "exact":
        return k == min(int(n_asked), n_all)
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


def grade(item, prediction):
    """-> {'success': bool, 'f1': float, 'reason': str}

    `reason` is a short machine-readable tag for the failure, so a failure
    breakdown is a Counter and not a regex over prose.
    """
    g = contract(item)
    sep, mode, arity = g.get("sep"), g["mode"], g.get("arity")
    pred = parse(prediction, sep=sep, arity=arity)
    gold = parse(item["answer"], sep=sep, arity=arity)
    if pred is None:
        return {"success": False, "f1": 0.0, "reason": "unparseable"}

    if mode == "sequence":
        ok = pred == gold
        return {"success": ok, "f1": _f1(pred, gold),
                "reason": "ok" if ok else "mismatch"}

    if mode == "multiset":
        ok = sorted(pred) == sorted(gold)
        return {"success": ok, "f1": _f1(pred, gold),
                "reason": "ok" if ok else "mismatch"}

    if mode == "membership":
        f1 = _f1(pred, gold)
        deduped = list(dict.fromkeys(pred))
        if len(deduped) != len(pred):
            return {"success": False, "f1": f1, "reason": "repeated_item"}
        # A negative is graded on the sentinel alone and never touches the
        # allow-list, so it is settled BEFORE the allow-list is required -- a
        # negative legitimately has no member set to draw from.
        if item.get("negative"):
            ok = pred == gold                    # the sentinel, exactly
            return {"success": ok, "f1": f1,
                    "reason": "ok" if ok else "mismatch"}
        if g.get("all_items") is None:
            raise ValueError(
                f"{item.get('id', '?')} ({item['type']}) is graded "
                f"`membership` but its row carries no `all_items`. The "
                f"allow-list is per-ITEM data derived from the ball, so it "
                f"cannot come from `qa/spec.py` -- rebuild the dataset with "
                f"`qa/build_balls.py` (stage 4), which writes it.")
        allow = set(g["all_items"])
        outside = [p for p in pred if p not in allow]
        if outside:
            return {"success": False, "f1": f1, "reason": "not_in_all"}
        if not count_ok(len(pred), g["quantity_band"], g["n_all"],
                        g.get("n_asked")):
            return {"success": False, "f1": f1, "reason": "bad_count"}
        return {"success": True, "f1": f1, "reason": "ok"}

    raise ValueError(f"unknown grading mode {mode!r}")


def grade_all(items, predictions):
    """items: [dict], predictions: {item id -> str}.  -> (rows, summary)."""
    rows = []
    for it in items:
        r = grade(it, predictions.get(it["id"]))
        r["id"] = it["id"]
        r["type"] = it["type"]
        r["band"] = it["band"]
        r["negative"] = bool(it.get("negative"))
        rows.append(r)
    return rows, summarise(rows)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("items", help="the dataset .jsonl")
    ap.add_argument("--predictions", help="jsonl of {'id':..., 'prediction':...}; "
                                          "omit to grade the gold against itself (C9)")
    ap.add_argument("--json", action="store_true", help="dump the summary as JSON")
    args = ap.parse_args()

    items = read_jsonl(args.items)
    if args.predictions:
        preds = {r["id"]: r["prediction"] for r in read_jsonl(args.predictions)}
    else:
        preds = {it["id"]: it["answer"] for it in items}

    rows, s = grade_all(items, preds)
    if args.json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
        return
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


if __name__ == "__main__":
    main()
