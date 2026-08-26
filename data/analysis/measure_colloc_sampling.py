#!/usr/bin/env python3
"""Measure the collocation cap and its sampler (QA_DATASET_DESIGN.md 3.1b, 3.1c).

Everything a store-dependent number in those two sections rests on, recomputed
from the store itself:

  3.1b  the `sense -> kolokacija` fan-out distribution, the hub list, the share of
        senses above K, and the ball-cost table at several caps
  3.1c  the power-law fit that justifies log-damped sampling weights, and the
        four-scheme comparison on one high-fan-out anchor

What is NOT re-derived here: the *content-based* rank for the upward `sestavina`
cap.  The ball-cost table varies the COLLOCATION cap with that one already
applied, exactly as 3.1b does, and which MWEs are kept changes a ball's contents
but not its size -- so the cost column is unaffected and the cheap
lowest-node-id rank is used.  Stated rather than assumed.

    python measure_colloc_sampling.py [STORE] [--anchors 400] [--voda voda]
"""
import os
import sys
import json
import math
import random
import argparse
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
sys.path.insert(0, DATA)
sys.path.insert(0, os.path.join(DATA, "lib"))

from qa.store import open_store, K_ANCHOR, K_SENSE, K_COLLOC   # noqa: E402
from qa import colloc_sampling                                             # noqa: E402

K_DEFAULT = 15


def pct(a, p):
    return int(np.percentile(a, p)) if len(a) else 0


# --------------------------------------------------------------------------
def fanout(store):
    """3.1b: the sense -> kolokacija fan-out, over every sense in the store."""
    deg = store.colloc_degree()
    kind = store.kind
    senses = np.flatnonzero(kind[:store.n_real] == K_SENSE)
    d = deg[senses]
    bearing = senses[d > 0]
    db = deg[bearing]
    out = {
        "collocation_bearing_senses": int(len(bearing)),
        "p50": pct(db, 50), "p90": pct(db, 90), "p99": pct(db, 99),
        "max": int(db.max()) if len(db) else 0,
        "above_K": int((db > K_DEFAULT).sum()),
        "above_K_share": float((db > K_DEFAULT).mean()) if len(db) else 0.0,
    }
    top = bearing[np.argsort(-db)[:8]]
    out["hubs"] = [(store.text(int(i))[:40], int(deg[i])) for i in top]
    return out, bearing, db


# --------------------------------------------------------------------------
def anchor_proxy(store, limit=None, progress=None):
    """D9's proxy for every core anchor: collocation + MWE memberships."""
    core = store.core_anchors()
    if limit:
        core = core[:limit]
    deg = store.colloc_degree()
    kind, mwe, nr = store.kind, store.mwe, store.n_real
    out = np.zeros(len(core), dtype=np.int64)
    for i, a in enumerate(core):
        a = int(a)
        n_col = n_mwe = 0
        for v in store.nbrs(a):
            v = int(v)
            if kind[v] == K_SENSE:
                n_col += int(deg[v])
            elif v < nr and kind[v] == K_ANCHOR and mwe[v]:
                n_mwe += 1
        out[i] = n_col + n_mwe
        if progress and (i + 1) % 20000 == 0:
            progress(i + 1, len(core))
    return core, out


# --------------------------------------------------------------------------
def ball_tokens(store, a, k_col, k_mwe=K_DEFAULT):
    """Token cost of the hop-2 ball around `a` under the two caps (D4/D5/D5b).

    `k_col = None` means uncapped; 0 means collocations excluded entirely.
    """
    kind, mwe, nr = store.kind, store.mwe, store.n_real
    tok = store.G["token_len"]
    seen = {a}

    h1 = [int(v) for v in store.nbrs(a)]
    up = sorted(v for v in h1 if v < nr and kind[v] == K_ANCHOR and mwe[v])
    if len(up) > k_mwe:
        up = up[:k_mwe]                 # count, not identity -- see the docstring
    rest = [v for v in h1 if not (v < nr and kind[v] == K_ANCHOR and mwe[v])]
    h1 = rest + up
    seen.update(h1)

    # the anchor's collocation budget is pooled across its senses (D5b)
    if k_col == 0:
        keep_col = set()
    elif k_col is None:
        keep_col = None                 # keep everything
    else:
        keep_col = {v for v, _p, _s in colloc_sampling.sample(store, a, k=k_col)}

    for v in h1:
        for w in store.nbrs(v):
            w = int(w)
            if kind[w] == K_COLLOC:
                if keep_col is not None and w not in keep_col:
                    continue
            seen.add(w)
    idx = np.fromiter(seen, dtype=np.int64, count=len(seen))
    return int(tok[idx].sum())


def ball_cost_table(store, anchors, caps=(0, 15, 50, 100, None), progress=None):
    rows = {}
    for cap in caps:
        costs = []
        for j, a in enumerate(anchors):
            costs.append(ball_tokens(store, int(a), cap))
            if progress and (j + 1) % 50 == 0:
                progress(cap, j + 1, len(anchors))
        c = np.array(costs)
        rows["uncapped" if cap is None else str(cap)] = {
            "p50": pct(c, 50), "p90": pct(c, 90), "p99": pct(c, 99),
            "max": int(c.max())}
    return rows


# --------------------------------------------------------------------------
def power_law(db):
    """3.1c: the log-log rank/value fit over the collocation fan-out."""
    v = np.sort(db)[::-1].astype(float)
    v = v[v > 0]
    r = np.arange(1, len(v) + 1, dtype=float)
    x, y = np.log(r), np.log(v)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"slope": float(slope), "r2": 1.0 - ss_res / ss_tot, "n": int(len(v))}


def scheme_comparison(store, a, k=K_DEFAULT, seeds=30, hub_threshold=2000):
    """3.1c: four selection schemes on one anchor, mean +- sd over `seeds` draws."""
    cand = colloc_sampling.pool(store, a)
    deg = store.colloc_degree()
    proxy = np.array([float(deg[p]) if p >= 0 else 0.0 for _v, _p, p in cand])
    out = {"pool_size": len(cand),
           "pool": {"p10": pct(proxy, 10), "p50": pct(proxy, 50),
                    "p90": pct(proxy, 90), "max": int(proxy.max()) if len(proxy) else 0,
                    "above_2000": int((proxy > hub_threshold).sum())}}

    def draw(w, seed):
        rng = np.random.default_rng(seed)
        keys = rng.random(len(cand)) ** (1.0 / np.maximum(w, 1e-12))
        return np.argpartition(-keys, k - 1)[:k]

    schemes = {
        "uniform": np.ones_like(proxy),
        "log1p": np.log1p(proxy),
        "linear": np.maximum(proxy, 1e-9),
    }
    for name, w in schemes.items():
        med, fw, sample = [], [], None
        for s in range(seeds):
            take = draw(w, s)
            med.append(float(np.median(proxy[take])))
            fw.append(int((proxy[take] > hub_threshold).sum()))
            if sample is None:
                sample = [cand[int(i)][1] for i in take[:3]]
        out[name] = {"partner_proxy_p50": float(np.mean(med)),
                     "hub_partners_mean": float(np.mean(fw)),
                     "hub_partners_sd": float(np.std(fw)),
                     "sample": sample}
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store", nargs="?", default=None)
    ap.add_argument("--anchors", type=int, default=400)
    ap.add_argument("--voda", default="voda")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "colloc_sampling.json"))
    args = ap.parse_args()

    store = open_store(args.store)
    print(f"[store] {store.path}  convention={store.convention}  "
          f"{store.n:,} nodes", flush=True)
    report = {"store": os.path.basename(store.path),
              "convention": store.convention,
              "nodes": store.n,
              "half_edges": int(len(store.indices))}

    print("[3.1b] fan-out ...", flush=True)
    fo, bearing, db = fanout(store)
    report["fanout"] = fo
    print(json.dumps(fo, ensure_ascii=False, indent=2), flush=True)

    print("[3.1b] anchor proxy ...", flush=True)
    core, proxy = anchor_proxy(
        store, progress=lambda i, n: print(f"    {i:,}/{n:,}", flush=True))
    top = core[np.argsort(-proxy)[:args.anchors]]
    report["top_anchors"] = [(store.lemma(int(a)), int(p)) for a, p in
                             zip(top[:10], np.sort(proxy)[::-1][:10])]
    print("    top:", report["top_anchors"], flush=True)

    print(f"[3.1b] ball cost over the top {args.anchors} anchors ...", flush=True)
    report["ball_cost"] = ball_cost_table(
        store, top,
        progress=lambda c, j, n: print(f"    cap={c} {j}/{n}", flush=True))
    print(json.dumps(report["ball_cost"], indent=2), flush=True)

    print("[3.1c] power law ...", flush=True)
    report["power_law"] = power_law(db)
    print(report["power_law"], flush=True)

    idx = store.surface_index()
    cands = idx.get(args.voda.casefold(), ())
    # The reverse index covers otherForm as well as canonicalForm, so `voda` also
    # resolves to `vod` (its genitive singular).  Prefer the entry whose LEMMA is
    # the query -- taking the first candidate silently measured the wrong word.
    cands = ([c for c in cands if store.lemma(int(c)).casefold() == args.voda.casefold()]
             or list(cands))
    if cands:
        a = int(cands[0])
        print(f"[3.1c] scheme comparison on {store.lemma(a)!r} ...", flush=True)
        report["schemes"] = {args.voda: scheme_comparison(store, a)}
        print(json.dumps(report["schemes"], ensure_ascii=False, indent=2), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
