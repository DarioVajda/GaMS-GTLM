"""Re-check the T17 degenerate-repetition finding against the current dataset.

    .venv/bin/python -m train.analysis.analyse_t17 train/results/arms_v3

The finding, made on `arms_v2`: 63 of 115 GTLM T17 items emitted verbatim repeats
(`A | A | A | B`) and were graded `repeated_item`.  The diagnosis was greedy
decoding at `repetition_penalty=1.0`, which the two-pass evaluator's correctness
argument *requires* — pass 1 declares an item correct on the grounds that greedy
decoding would have emitted gold, and any decoding fix invalidates that shortcut
and makes every evaluation roughly 4x more expensive.

**That diagnosis was made against single-item golds**, and the rebuild that
followed it (the corpus arms_v3 trained on, and the one in `datasets/` today)
changed the T17 targets: 14 of 115 test answers differ and many now carry four
items.  A model that emits four items where gold has four is doing something
quite different from one that emits four where gold has one, even when both are
graded `repeated_item`.  So the rate is re-measured here, broken down by how many
items the gold actually asks for — which is the thing the old reading could not
see.

Reads the per-item dumps `train/evaluate.py` writes
(`train/results/predictions/<run>/test.jsonl`) and joins them to the run records
in `<sweep_dir>/runs.jsonl` so each dump is attributed to its arm.
"""
import os
import json
import argparse
import collections

from .._io import load_items, read_jsonl
from ..qa_contract import contract, parse
from .report_arms import arm_of

# train/analysis/ -> train/, then train/results/predictions.
PRED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "predictions")


def analyse(rows, items, type_="T17"):
    """Per-arm counts, split by the gold's own item count."""
    stat = collections.Counter()
    by_gold_n = collections.defaultdict(collections.Counter)
    for r in rows:
        if r.get("type") != type_ or r.get("prediction") is None:
            continue
        item = items.get(r["id"])
        if item is None:
            continue
        g = contract(item)
        gold = parse(r["gold"], g["sep"], g["arity"]) or []
        pred = parse(r["prediction"], g["sep"], g["arity"]) or []
        deduped = list(dict.fromkeys(pred))
        repeated = len(deduped) != len(pred)
        stat["items"] += 1
        stat["repeated"] += repeated
        stat["success"] += bool(r.get("success"))
        key = len(gold)
        by_gold_n[key]["items"] += 1
        by_gold_n[key]["repeated"] += repeated
        by_gold_n[key]["success"] += bool(r.get("success"))
    return stat, by_gold_n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_dir")
    ap.add_argument("--split", default="test")
    ap.add_argument("--type", default="T17")
    a = ap.parse_args(argv)

    runs = read_jsonl(os.path.join(a.sweep_dir, "runs.jsonl"))
    if not runs:
        print(f"{a.sweep_dir}: no runs recorded")
        return 1
    items = load_items(runs[0]["items_root"], a.split)

    per_arm = collections.defaultdict(list)
    missing = []
    for rec in runs:
        dump = os.path.join(PRED_DIR, rec["run_name"], f"{a.split}.jsonl")
        if not os.path.exists(dump):
            missing.append(rec["run_name"])
            continue
        with open(dump, encoding="utf-8") as f:
            per_arm[arm_of(rec)] += [json.loads(line) for line in f if line.strip()]
    if missing:
        print(f"[warn] {len(missing)} run(s) have no {a.split} dump: "
              f"{', '.join(missing[:3])}{' …' if len(missing) > 3 else ''}\n")
    if not per_arm:
        print("no prediction dumps found")
        return 1

    print(f"{a.type} degenerate repetition on `{a.split}`, "
          f"summed over seeds ({runs[0]['items_root']}):\n")
    print("| arm | items | repeated | share | success |")
    print("|---|--:|--:|--:|--:|")
    detail = {}
    for arm in sorted(per_arm):
        stat, by_n = analyse(per_arm[arm], items, a.type)
        detail[arm] = by_n
        n = stat["items"] or 1
        print(f"| {arm} | {stat['items']} | {stat['repeated']} | "
              f"{stat['repeated'] / n:.3f} | {stat['success'] / n:.3f} |")

    print("\nBroken down by how many items the GOLD asks for — the distinction "
          "the original\nreading could not make, because it was taken against "
          "single-item golds:\n")
    golds = sorted({k for by_n in detail.values() for k in by_n})
    print("| arm | " + " | ".join(f"gold n={k}" for k in golds) + " |")
    print("|---" * (len(golds) + 1) + "|")
    for arm in sorted(detail):
        cells = []
        for k in golds:
            c = detail[arm].get(k)
            cells.append(f"{c['repeated']}/{c['items']}" if c else "—")
        print(f"| {arm} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
