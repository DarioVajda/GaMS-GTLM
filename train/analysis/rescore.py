"""Re-grade the four arms' prediction dumps under a repaired grading contract.

> **Historical — `arms_v2` only.** Both repairs are now applied at SOURCE
> (`qa/spec.py`, `qa/grade.py`, `qa/build_balls.py`) and `generated/v2_clean` is
> what the defaults read, so an `arms_v3` run is already graded under the
> repaired contract and there is nothing here to re-apply. This module stays
> because it is the record of what the repair moved in the `arms_v2` numbers;
> do not point it at `arms_v3` output.

`data/qa/repair_grading.py` fixes two contracts that rejected answers which
satisfy the question as asked (T19's arbitrary tie-break over a set of recorded
examples; T17's incomplete `all_items` allow-list).  Both repairs only ever widen
what counts as correct, so no run has to be retrained: the per-item dumps written
by `train/evaluate.py` carry the generated string, and re-grading is a CPU join.

The repaired contract is derived from `balls/v2` and applied **identically to all
four arms**, including the two baselines -- a grader that varied by arm would be a
worse defect than the one being repaired.

Monotonicity is asserted, not assumed: if any item that passed under the original
contract fails under the repaired one, the run aborts.  That is what licenses
re-using the pass-1 rows, whose dumped prediction is the teacher-forced argmax.

    .venv/bin/python -m train.analysis.rescore
"""
import argparse
import collections
import json
import os
import statistics as st

from ..qa_contract import grade

ARM_NAMES = {
    ("v2", True): "GTLM (spd + magnetic)",
    ("v2", False): "GTLM, no bias",
    ("v2_serialised", False): "serialised graph",
    ("v2_noretrieval", False): "no retrieval",
}
ORDER = ["GTLM (spd + magnetic)", "serialised graph",
         "no retrieval", "GTLM, no bias"]


def load_items(root, split):
    path = os.path.join(root, f"{split}.jsonl")
    return {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def arm_of(run):
    r = json.loads(run)
    inp = r["input"]
    return ARM_NAMES[(inp, bool(r["spd"]))] if inp == "v2" else ARM_NAMES[(inp, False)]


def rescore_run(pred_dir, items, split, allow_regressions=False):
    """-> (n, strict_ok, repaired_ok, per_type, flips) or None if no dump."""
    path = os.path.join(pred_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        return None
    n = strict = repaired = 0
    per_type = collections.defaultdict(lambda: [0, 0, 0])   # n, strict, repaired
    flips = collections.Counter()
    regressions = []
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        it = items.get(row["id"])
        if it is None:
            continue
        was = bool(row["success"])
        now = bool(grade(it, row["prediction"])["success"])
        if was and not now:
            regressions.append((row["id"], row["type"]))
        n += 1
        strict += was
        repaired += now
        t = per_type[row["type"]]
        t[0] += 1
        t[1] += was
        t[2] += now
        if was != now:
            flips[row["type"]] += 1
    if regressions and not allow_regressions:
        raise AssertionError(
            f"the repair is NOT monotone -- {len(regressions)} item(s) that "
            f"passed now fail, e.g. {regressions[:5]}. A grading repair that can "
            f"turn a correct answer incorrect must not be applied silently.\n"
            f"If this is a deliberate TIGHTENING rather than a repair, pass "
            f"--allow-regressions and say so in the write-up. Scoring against "
            f"`generated/v2_clean` is one such case: its allow-list is the ball "
            f"(a retrieval probe -- name what you were shown) where "
            f"`v2_graded`'s was the whole store (a knowledge probe -- name "
            f"anything true). That is a semantic choice, not a bug fix.")
    return n, strict, repaired, per_type, flips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-jsonl", default="train/results/arms_v2/runs.jsonl")
    ap.add_argument("--pred-root", default="train/results/predictions")
    ap.add_argument("--items-root", default="data/datasets/generated/v2_graded")
    ap.add_argument("--split", default="test")
    ap.add_argument("--markdown", default="")
    ap.add_argument("--allow-regressions", action="store_true",
                    help="permit a contract that turns a correct answer "
                         "incorrect (a deliberate tightening, not a repair)")
    a = ap.parse_args()

    items = load_items(a.items_root, a.split)
    by_arm = collections.defaultdict(list)
    per_type_acc = collections.defaultdict(lambda: collections.defaultdict(list))
    flips_total = collections.Counter()

    for line in open(a.runs_jsonl, encoding="utf-8"):
        r = json.loads(line)
        out = rescore_run(os.path.join(a.pred_root, r["run_name"]), items,
                          a.split, a.allow_regressions)
        if out is None:
            print(f"[warn] no {a.split} dump for {r['run_name']}")
            continue
        n, strict, repaired, per_type, flips = out
        arm = arm_of(line)
        by_arm[arm].append((strict / n, repaired / n))
        for t, (tn, ts, tr) in per_type.items():
            per_type_acc[arm][t].append(tr / tn)
        flips_total.update(flips)

    lines = []
    lines.append(f"| arm | seeds | {a.split} accuracy (as graded) | "
                 f"(repaired) | change |")
    lines.append("|---|--:|--:|--:|--:|")
    for arm in ORDER:
        v = by_arm.get(arm)
        if not v:
            continue
        s = [x[0] for x in v]
        r = [x[1] for x in v]
        lines.append(
            f"| **{arm}** | {len(v)} | {st.mean(s):.4f} ± {st.pstdev(s):.4f} | "
            f"**{st.mean(r):.4f} ± {st.pstdev(r):.4f}** | "
            f"{st.mean(r) - st.mean(s):+.4f} |")
    table = "\n".join(lines)
    print(table)

    print("\nItems whose verdict changed, by type:")
    for t, c in flips_total.most_common():
        print(f"  {t:>5} {c}")

    types = sorted({t for d in per_type_acc.values() for t in d},
                   key=lambda x: int(x[1:]))
    pt = ["", f"Per type ({a.split} accuracy, repaired contract, mean over seeds):",
          "", "| type | " + " | ".join(ORDER) + " |",
          "|---|" + "---|" * len(ORDER)]
    for t in types:
        cells = []
        for arm in ORDER:
            v = per_type_acc[arm].get(t)
            cells.append(f"{st.mean(v):.3f}" if v else "—")
        pt.append(f"| {t} | " + " | ".join(cells) + " |")
    print("\n".join(pt))

    if a.markdown:
        os.makedirs(os.path.dirname(a.markdown), exist_ok=True)
        with open(a.markdown, "w", encoding="utf-8") as f:
            f.write(table + "\n" + "\n".join(pt) + "\n")
        print(f"\n[wrote] {a.markdown}")


if __name__ == "__main__":
    main()
