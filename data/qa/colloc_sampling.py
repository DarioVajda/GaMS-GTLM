#!/usr/bin/env python3
"""How many collocations enter a ball, and which ones (QA_DATASET_DESIGN.md D5b).

`sense -> kolokacija` is a hub: 99.9 % of `imeti`'s 14,249 collocations sit on
`pomen 1`, so an uncapped hop-2 ball around a common word costs a p50 of 22,704
tokens.  The cap is K, pooled across ALL of an anchor's senses -- per-sense would
still admit 17 x K for `voda` -- and the K are SAMPLED rather than ranked:
without replacement, with `w = log(1 + proxy(partner))`.  Every deterministic key
tried was biased toward one slice of a power-law distribution.

`qa/build_balls.py` passes K = 10.  K_DEFAULT below is what the generator draws
its own gold with, before any ball exists, which is why membership targets are
re-verbalised against the ball afterwards.

The generator and the ball builder must draw the same set or the gold is
unanswerable from the input, so the selection lives here as a pure function of
`(anchor node code, K, dataset seed)` and both call it rather than reimplement
it.  That is what decouples them without letting them drift.

Four reproducibility requirements are the contract (selftest C18):

  (a) the candidate pool is sorted by node id before drawing -- CSR adjacency
      order is NOT stable across builds (the parse is parallel and unordered), so
      an unsorted pool means the same seed picks different phrases after a
      rebuild;
  (b) the RNG seed derives from the anchor's node code and nothing else -- never
      from a loop counter or extraction order;
  (c) rebuilding the store leaves every ball byte-identical;
  (d) every gold phrase is in its own ball.

(a) is enforced by `QAStore.sense_children`, (b) here, (c) follows from (a)+(b),
and (d) is asserted by selftest.py.
"""
import hashlib

import numpy as np

K_DEFAULT = 15


def anchor_seed(node_code, dataset_seed=0):
    """Deterministic per-anchor RNG seed: blake2b(node code) ^ DATASET_SEED."""
    h = hashlib.blake2b(str(int(node_code)).encode(), digest_size=8).digest()
    return (int.from_bytes(h, "big") ^ int(dataset_seed)) & ((1 << 63) - 1)


def pool(store, a):
    """[(colloc_node, phrase, partner_sense)] for one anchor, sorted by node id.

    The partner is the member sense that is NOT one of this anchor's own -- the
    other end of the pairing, whose frequency the weight reads.
    """
    own = {s for s, _o, _b in store.senses(a)}
    out = []
    for v, phrase in store.collocations(a):
        nb = [int(x) for x in store.nbrs(v)]
        partner = next((x for x in nb if x not in own), nb[0] if nb else -1)
        out.append((v, phrase, partner))
    return out


def weights(store, cand):
    """log1p of each candidate's partner proxy.

    Every partner sense is a member of the collocation being weighted, so its
    collocation degree is at least 1 and the weight is at least log(2) > 0 --
    which is the point of log1p over log (D5b: `log(1)` is 0 and would give every
    hapax partner zero probability).
    """
    deg = store.colloc_degree()
    return np.log1p([float(deg[p]) if p >= 0 else 0.0 for _v, _p, p in cand])


def sample(store, a, k=K_DEFAULT, dataset_seed=0):
    """The K collocations D5b puts in this anchor's ball, in node-id order.

    Weighted sampling without replacement by the Efraimidis-Spirakis method:
    draw `u_i ~ U(0,1)`, key `u_i ** (1 / w_i)`, keep the k largest.  One RNG
    draw per candidate, no rejection loop, and the result depends only on the
    seed and the (sorted) pool.
    """
    cand = pool(store, a)
    if len(cand) <= k:
        return cand
    w = weights(store, cand)
    rng = np.random.default_rng(anchor_seed(store.codes[a], dataset_seed))
    keys = rng.random(len(cand)) ** (1.0 / np.maximum(w, 1e-12))
    take = np.argpartition(-keys, k - 1)[:k]
    return [cand[int(i)] for i in sorted(take)]
