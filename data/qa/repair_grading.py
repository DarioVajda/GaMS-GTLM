#!/usr/bin/env python3
"""Repair two grading contracts that penalise a correct answer.

    ┌─────────────────────────────────────────────────────────────────────┐
    │ SUPERSEDED as of 2026-08-23.  Both defects are now fixed at source:  │
    │ `qa/spec.py` (T19's mode + the set rule), `qa/grade.py` (`contract`, │
    │ so per-type constants are read from the spec and a stale row cannot  │
    │ override it) and `qa/build_balls.py` (`member_contract`, which builds │
    │ the allow-list over ALL of an item's anchors).  Re-running stage 4    │
    │ produces the corrected contract by itself, and `qa/check_balls.py`   │
    │ fails the build if it does not.                                      │
    │                                                                      │
    │ This script is kept ONLY to reproduce `generated/v2_graded`, the     │
    │ frozen contract the twelve completed four-arm runs were scored       │
    │ against.  Do not use it on new data -- run the pipeline.             │
    └─────────────────────────────────────────────────────────────────────┘


Both defects were found on 2026-08-23 by reading the per-item prediction dumps
from the four-arm run (see train/README.md), and both are cases of the grader rejecting an
answer that satisfies the question as asked.

**T19 -- an arbitrary tie-break over a set.**  The question is *"V kakšnem stavku
se pojavi ⟨L⟩?"* -- "in what sentence does ⟨L⟩ appear?"  82 of 115 test items
carry more than one `zgled:` node and every one of them answers it, but the
grader was `sequence`/arity-1: exact match against whichever example happened to
come first in file order (the gold is the first candidate in 101 of 103 cases).

That is not a property of the data.  The KG stores the examples of a sense as a
SET; their order is an artefact of serialisation.  Grading on it measures whether
an architecture happens to preserve sibling order -- and penalises permutation
equivariance, which is a *desirable* property of a graph model, not a defect.
T19 becomes `membership` over the recorded examples, with an exact count of one.

**T17 -- an incomplete allow-list.**  `membership` was already the right mode, but
`all_items` is built from a deduplicated member set that drops collocations the
KG does record and the ball still shows the model.  A model that reads its own
input and returns a real collocation from it was scored `not_in_all`.  The
allow-list is widened to every `kolokacija:` node actually present in the ball.

Both repairs are **monotone**: the allow-list only grows and the gold is always
inside it, so every answer that passed before still passes.  Nothing that was
counted correct becomes incorrect, which is what makes re-scoring the existing
prediction dumps sound (`train/analysis/rescore.py`) rather than requiring a retrain.

Candidates come from `balls/v2`, which is the same ball every arm was built from
(`v2_serialised` re-encodes it, `v2_noretrieval` blanks the nodes but keeps the
id), so the repaired contract is identical for all four arms.  Deriving it from
each arm's own input would make the grader arm-dependent, which would be a far
worse bug than the one being fixed.

    .venv/bin/python -m qa.repair_grading            # from data/
"""
import argparse
import collections
import json
import os

from .sl import norm

SPLITS = ("train", "dev", "test")
# type -> the node prefix whose texts are the recorded set for that question
CANDIDATE_KIND = {"T19": "zgled", "T17": "kolokacija"}


def candidates(row, kind):
    """Normalised texts of every `kind:` node in the ball, in file order."""
    out = []
    for n in row["nodes"]:
        t = n["text"] if isinstance(n, dict) else n
        if t.startswith(kind + ":"):
            out.append(norm(t.split(":", 1)[1].strip()))
    return list(dict.fromkeys(out))


def repair(item, ball_row, stats):
    """Patch `item['grading']` in place.  Returns True if anything changed."""
    t = item["type"]
    kind = CANDIDATE_KIND.get(t)
    if kind is None:
        return False
    g = item["grading"]
    cands = candidates(ball_row, kind)

    if t == "T19":
        if g["mode"] == "membership":
            return False                      # already repaired
        # The gold must be inside the allow-list or the repair would make a
        # previously-correct answer wrong -- assert rather than hope.
        gold = norm(item["answer"].split("ODGOVOR:", 1)[1].strip())
        allow = list(dict.fromkeys(cands + ([gold] if gold else [])))
        if not item.get("negative") and gold not in cands:
            stats["T19_gold_not_a_node"] += 1
        g.update(mode="membership", quantity_band="exact", n_asked=1,
                 n_all=len(allow), all_items=allow)
        stats["T19_repaired"] += 1
        return True

    # T17: widen, never narrow.
    before = list(g.get("all_items") or [])
    allow = list(dict.fromkeys(before + cands))
    if len(allow) == len(before):
        return False
    g["all_items"] = allow
    g["n_all"] = len(allow)
    stats["T17_widened"] += 1
    stats["T17_items_added"] += len(allow) - len(before)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-root", default="datasets/generated/v2_final")
    ap.add_argument("--ball-root", default="datasets/balls/v2")
    ap.add_argument("--out", default="datasets/generated/v2_graded",
                    help="written split by split; pass the items-root to patch "
                         "in place (not recommended -- keep the raw contract)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    stats = collections.Counter()
    for split in SPLITS:
        src = os.path.join(a.items_root, f"{split}.jsonl")
        if not os.path.exists(src):
            continue
        balls = {}
        with open(os.path.join(a.ball_root, f"{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["type"] in CANDIDATE_KIND:
                    balls[r["id"]] = r
        n = 0
        with open(src, encoding="utf-8") as f, \
                open(os.path.join(a.out, f"{split}.jsonl"), "w",
                     encoding="utf-8") as g:
            for line in f:
                item = json.loads(line)
                b = balls.get(item["id"])
                if b is not None:
                    repair(item, b, stats)
                    n += 1
                g.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[repair] {split}: {n} T17/T19 items inspected", flush=True)

    print("\n" + "\n".join(f"  {k:<24} {v}" for k, v in sorted(stats.items())))
    print(f"\n[wrote] {a.out}")


if __name__ == "__main__":
    main()
