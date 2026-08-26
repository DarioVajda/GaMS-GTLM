#!/usr/bin/env python3
"""C19 -- the collocation mix the dataset ACTUALLY got, not the one `voda` got.

`QA_DATASET_DESIGN.md` section 3.1c justifies D5b's `w = log1p(partner degree)`
weighting on a single anchor, `voda`, and on a power law fitted over all senses.
Neither says what the sampler puts in the average ball once the dataset exists.
C19 is the open item that asks for the realised figure; this closes it.

For every T17 item in a generated dataset, redraw the anchor's ball with the same
`colloc_sampling.sample` call the generator used (it is a pure function of the anchor's node
code, so this is the same ball, not a resample) and report:

  * the pool size before the K = 15 cap, so the share of items where the cap
    actually bit is visible rather than assumed;
  * partner-proxy percentiles over every sampled collocation -- the distribution
    section 3.1c reports for one anchor, over the whole dataset;
  * the HUB-PARTNER share.  This is the number the weighting was chosen to hold
    down: a partner sense that collocates with tens of thousands of other senses
    (`biti`, `imeti`, `se`) contributes a slot that teaches nothing about the
    anchor, and linear weighting filled 10.2 of 15 slots with them on `voda`
    where log1p filled 1.7.  "Hub" is defined by DEGREE against the store's own
    distribution, not by a word list -- that is the quantity the weight reads,
    and it needs no assumption about which lemmas are function words.
  * the same ball redrawn with LINEAR weights, so section 3.1c's central claim is
    checked on the realised dataset rather than on one anchor.

A first version of this script counted a slot as "function word" whenever its
PHRASE contained a preposition or auxiliary, and reported 39.4 %.  That measure
is wrong and the number is meaningless: `boj proti kriminaliteti` and `kosilo iz
nahrbtnika` are exactly the collocations the type exists to teach.  What matters
is what sits at the *other end of the pairing*, not what the phrase is spelled
with.

Run on a compute node (it opens the store):

    srun --partition=dev --nodelist=ana --time=30 --mem=32G \
        python3 analysis/measure_ball_mix.py datasets/generated/v1
"""
import os
import sys
import json
import argparse
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa.store import open_store                                   # noqa: E402
from qa import colloc_sampling, sl                                            # noqa: E402


# A partner counts as a HUB when its collocation degree is at or above this many
# senses.  Set from the store's own distribution rather than picked: section 3.1b
# measures the p99 of `sense -> kolokacija` fanout at 1,055 in v7, so a partner
# above it is in the top percent of the graph by this exact measure.  Reported
# alongside the raw percentiles so the threshold can be second-guessed.
HUB_DEGREE = 1055
# One threshold is one arbitrary cut, and the answer turned out to depend on it,
# so the share is reported across a sweep: p99 of the global fanout, a round
# middle, v6's largest hub (`imeti`) and v7's (`biti`).
HUB_SWEEP = (1055, 5000, 14233, 64842)


def pct(a, qs=(50, 90, 99)):
    if not len(a):
        return {}
    return {f"p{q}": float(np.percentile(a, q)) for q in qs}


def _linear_sample(store, a, cand, k, seed):
    """The same Efraimidis-Spirakis draw with `w = proxy` instead of log1p.

    The control section 3.1c compares against.  Same seed, same sorted pool, so
    the only difference between the two balls is the weighting.
    """
    deg = store.colloc_degree()
    w = np.array([float(deg[p]) if p >= 0 else 0.0 for _v, _p, p in cand])
    rng = np.random.default_rng(colloc_sampling.anchor_seed(store.codes[a], seed))
    keys = rng.random(len(cand)) ** (1.0 / np.maximum(w, 1e-12))
    take = np.argpartition(-keys, k - 1)[:k]
    return [cand[int(i)] for i in sorted(take)]


def measure(store, items, k=colloc_sampling.K_DEFAULT, seed=0):
    deg = store.colloc_degree()
    pool_sizes, proxies, capped = [], [], 0
    hubs_log, hubs_lin, uniq_partners = [], [], []
    seen_anchor = set()

    for it in items:
        a = it.get("anchor")
        if a is None or a in seen_anchor:
            continue
        seen_anchor.add(a)
        cand = colloc_sampling.pool(store, a)
        if not cand:
            continue
        pool_sizes.append(len(cand))
        ball = colloc_sampling.sample(store, a, k=k, dataset_seed=seed)
        parts = [p for _v, _p, p in ball if p >= 0]
        proxies += [int(deg[p]) for p in parts]
        uniq_partners.append(len(set(parts)))
        hubs_log.append([sum(1 for p in parts if deg[p] >= t) for t in HUB_SWEEP])
        # the linear control only means anything where the cap actually bit
        if len(cand) > k:
            capped += 1
            lin = [p for _v, _p, p in _linear_sample(store, a, cand, k, seed)
                   if p >= 0]
            hubs_lin.append([sum(1 for p in lin if deg[p] >= t)
                             for t in HUB_SWEEP])

    n = len(pool_sizes)
    slots = sum(min(p, k) for p in pool_sizes)
    return {
        "anchors": n,
        "hub_degree_threshold": HUB_DEGREE,
        "pool_size": pct(np.array(pool_sizes)) | {
            "mean": float(np.mean(pool_sizes)) if n else 0.0,
            "max": int(max(pool_sizes)) if n else 0},
        "capped_share": round(100.0 * capped / max(n, 1), 1),
        "partner_proxy": pct(np.array(proxies)) | {
            "mean": float(np.mean(proxies)) if proxies else 0.0,
            "max": int(max(proxies)) if proxies else 0,
            "n": len(proxies)},
        "distinct_partners_per_ball": {
            "mean": float(np.mean(uniq_partners)) if n else 0.0},
        "hub_partners_per_ball": _hub_table(hubs_log, hubs_lin, pool_sizes, k),
    }


def _hub_table(hubs_log, hubs_lin, pool_sizes, k):
    """Mean hub slots out of K, per threshold, log1p against the linear control.

    Restricted to the anchors where the cap actually bit: below K every candidate
    is in the ball whatever the weights are, so including those anchors would
    dilute the comparison with rows on which the two schemes cannot differ.
    """
    log = np.array([h for h, p in zip(hubs_log, pool_sizes) if p > k])
    lin = np.array(hubs_lin)
    out = {}
    for i, t in enumerate(HUB_SWEEP):
        out[f"deg>={t}"] = {
            "log1p": round(float(log[:, i].mean()), 2) if len(log) else 0.0,
            "linear": round(float(lin[:, i].mean()), 2) if len(lin) else 0.0,
            "of_k": k,
        }
    out["capped_anchors"] = int(len(log))
    out["all_anchors_log1p_share_pct"] = round(
        100.0 * float(np.sum([h[0] for h in hubs_log])) / max(
            sum(min(p, k) for p in pool_sizes), 1), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--store", default=None)
    ap.add_argument("--k", type=int, default=colloc_sampling.K_DEFAULT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    store = open_store(args.store)
    items = []
    for s in ("train", "dev", "test"):
        p = os.path.join(args.dataset, f"{s}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["type"] == "T17" and not r["negative"]:
                    items.append(r)
    print(f"T17 positives: {len(items):,}")

    # the record stores the node CODE; map it back to the node index
    codes = np.asarray(store.codes)
    for r in items:
        i = int(np.searchsorted(codes, r["node_code"]))
        r["anchor"] = i if i < len(codes) and int(codes[i]) == r["node_code"] else None

    out = measure(store, items, k=args.k)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    bands = collections.Counter(r["grading"].get("quantity_band") for r in items)
    print("quantity bands:", dict(bands))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
