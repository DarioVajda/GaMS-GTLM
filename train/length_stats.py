"""Packed input-sequence length, GTLM's graph ball vs the serialised prompt.

The two arms carry the SAME ball -- `qa/check_variants.py` verifies that -- but
present it differently, so "how long is the input" has two different answers and
they are what the cost difference between the arms is made of.

Length here is the **packed** token count: the sum over a graph's nodes of their
`input_ids`, which is the sequence the model actually attends over (nodes are
concatenated, prefix nodes first and the prompt node last).  For the serialised
arm that sum is one node; for GTLM it is `n_nodes` of them.

Structural features are switched off: SPD and the magnetic Laplacian do not
change tokenization, and computing them here would cost minutes for nothing.

    .venv/bin/python -m train.length_stats
"""
import argparse
import json
import os

from transformers import AutoTokenizer

from train.batching import packed_lengths
from train.config import RunConfig
from train.data import load_split

ARMS = [
    ("GTLM (graph ball)", "data/datasets/balls/v2", 2048),
    ("serialised graph", "data/datasets/balls/v2_serialised", 17408),
]
PCTS = [50, 75, 90, 95, 99, 100]


def pct(xs, p):
    """Nearest-rank percentile, so every reported value is a real observation."""
    if not xs:
        return 0
    s = sorted(xs)
    if p >= 100:
        return s[-1]
    k = max(0, min(len(s) - 1, int(round(p / 100 * len(s) + 0.5)) - 1))
    return s[k]


def summarise(xs):
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "min": min(xs),
        **{f"p{p}": pct(xs, p) for p in PCTS},
        "total": sum(xs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="google/gemma-3-1b-it")
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--out", default="train/results/arms_v2/length_stats.json")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_name)
    splits = [s for s in a.splits.split(",") if s]
    report = {}

    for name, root, max_length in ARMS:
        cfg = RunConfig(model_name=a.model_name, data_root=root,
                        max_length=max_length, max_items=a.max_items,
                        spd=False, magnetic=False, rrwp=False)
        per_split, all_lengths, all_nodes = {}, [], []
        for split in splits:
            sp = load_split(cfg, tok, split, with_generation=False)
            lens = packed_lengths(sp.ds)
            nodes = [len(r["nodes"]) + 1 for r in sp.rows]   # + the prompt node
            per_split[split] = {"tokens": summarise(lens),
                                "nodes": summarise(nodes)}
            all_lengths += lens
            all_nodes += nodes
            print(f"[len] {name:<20} {split:<5} n={len(lens):>6} "
                  f"mean={sum(lens)/len(lens):>8.1f} p99={pct(lens, 99):>6} "
                  f"max={max(lens):>6}", flush=True)
        report[name] = {"data_root": root, "max_length": max_length,
                        "splits": per_split,
                        "all": {"tokens": summarise(all_lengths),
                                "nodes": summarise(all_nodes)}}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[wrote] {a.out}")

    hdr = f"{'':<22}" + "".join(f"{k:>10}" for k in
                                ["n", "mean", "min", "p50", "p90", "p99", "max"])
    print("\nPacked tokens per item (all splits):\n" + hdr)
    for name in report:
        t = report[name]["all"]["tokens"]
        print(f"{name:<22}" + "".join(
            f"{v:>10}" for v in [t["n"], round(t["mean"], 1), t["min"],
                                 t["p50"], t["p90"], t["p99"], t["p100"]]))
    print("\nNodes per item (all splits):\n" + hdr)
    for name in report:
        t = report[name]["all"]["nodes"]
        print(f"{name:<22}" + "".join(
            f"{v:>10}" for v in [t["n"], round(t["mean"], 1), t["min"],
                                 t["p50"], t["p90"], t["p99"], t["p100"]]))


if __name__ == "__main__":
    main()
