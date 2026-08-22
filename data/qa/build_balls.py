#!/usr/bin/env python3
"""Turn relabelled QA items into the small text graphs a GTLM actually reads.

    sbatch qa/run_build_balls.sbatch                 # -> datasets/balls/v2

**This builder consumes `targets`, never `node_code`.**  D3 fixes the production
pipeline as extractor -> verbatim surface lookup -> union of every match, and
says dataset extraction runs the identical procedure.  `qa/relabel.py` already
ran it: each item carries the node codes a real extractor run resolved to.  The
v1 builder instead looked the item's own anchor up by `node_code`, which is the
privileged knowledge D3 forbids -- it handed every item the right ball even when
the extractor would never have found it, and it made the 9.3 % of items whose
lookup returns several units invisible.  Reading `targets` is what puts
*selection among what resolution returned* -- the model's job, per D3 -- into the
input.

Output is one JSON object per line:

    {"id", "type", "split", "question", "answer",
     "nodes": ["iztočnica: gora (…)", "oblika: gore (…)", …],
     "edges": [[i, j], …],          # the induced subgraph, undirected pairs
     "anchors": [0, 31], "anchor": 0, "n_nodes": N, "n_tokens": T}

`anchors` is a list because a union ball has one per matched lexical unit; the
prompt node attaches to all of them (train/data.py).  `anchor` is kept as
`anchors[0]` so nothing downstream breaks on the rename.

The policy, uniform across all 19 question types (D4):

  hop 2 from every matched unit, with two directional caps and nothing else.

Hop 2 is the floor -- synonyms, antonyms and collocations are reified nodes one
hop past the sense -- and the ceiling, because those nodes carry both lemmas in
their own text.  It does **not** branch on question type: giving T3 the paradigm
and T9 the anchor alone would be cheaper and would also make every score
meaningless, because the model would be told which relation to read before it
read anything.

  D5   `sestavina` is directional.  MWE -> constituent word always (bounded at
       22 across the whole graph).  Word -> MWE capped at K_MWE, ranked by a
       content-independent rule: senses carrying a definition or example first,
       then fewest constituents, then node id.  Only 114,033 of 400,180 words
       are in any MWE at all, but the hubs are function words -- `biti` is a
       constituent of 423,510 of them.
  D5b  `sense -> kolokacija` capped at K_COLLOC, pooled across ALL of the
       anchor's senses (99.9 % of `imeti`'s collocations sit on `pomen 1`, so a
       per-sense cap would still admit 17 x K for `voda`), and *sampled* rather
       than truncated -- see qa/d5b.py.

Both K default to 10.

**M2: the periphrastic auxiliary is NOT injected.**  QA_TASKS.md 1094 settled the
opposite in 2026-08-21 -- put `biti`'s 17 form nodes in every verb's ball so T5
and T6 containment is literally true.  Reversed 2026-08-22, for three reasons.
It breaks train/inference parity: no extractor returns `biti` from *"Kako se
glagol pisati spreže v prihodnjiku?"*, so the injection exists only in this
builder and every conjugation ball at inference would be missing 17 nodes the
model trained on.  It is type-shaping in a POS costume: nothing in this graph
needs the auxiliary paradigm except conjugation questions, and D4's real test is
whether the ball policy encodes advance knowledge of what will be asked.  And
the content is not worth retrieving -- 16 tokens of the most frequent verb in
Slovene, identical for every anchor, which any Slovene-pretrained backbone has
cold.  So C18's containment claim reads *every gold lexical item is in the ball;
the auxiliary is composed*, and T5/T6 (1,049 items, 8.4 %) are its one recorded
exemption.
"""
import os
import sys
import json
import argparse
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa.store import open_store, K_ANCHOR, K_SENSE, K_COLLOC       # noqa: E402
from qa import d5b, gen, sl, spec                                  # noqa: E402

K_MWE = 10          # D5, upward `sestavina`
K_COLLOC_CAP = 10   # D5b, `sense -> kolokacija`

# Ranking 423,510 MWE candidates costs a text read each; above this many, fall
# back to the lowest-node-id prefix before ranking.  A cost guard, not a policy:
# it is content-independent either way, and `--report` prints how often it fires
# so a silent bias cannot creep in unnoticed.
MWE_SCAN_CAP = 20_000


def _mwe_sort_key(store, m):
    """D5's rank for one MWE: contentful senses first, then fewest constituents.

    Content-independent of the *question*, which is what keeps D4 intact -- it
    reads the MWE entry's own completeness, never what is being asked about it.
    """
    contentful = 0
    for s, _o, body in store.senses(m):
        if body:
            contentful = 1
            break
    n_const = sum(1 for v in store.nbrs(m)
                  if store.kind[int(v)] == K_ANCHOR and not store.mwe[int(v)])
    return (-contentful, n_const, int(m))


def _split_hop1(store, a):
    """The anchor's neighbours, split into (plain, mwe-anchors)."""
    plain, mwes = [], []
    for v in store.nbrs(a):
        v = int(v)
        if store.kind[v] == K_ANCHOR and store.mwe[v]:
            mwes.append(v)
        else:
            plain.append(v)
    return plain, mwes


def ball_nodes(store, a, k_mwe=K_MWE, k_colloc=K_COLLOC_CAP, stats=None):
    """The node ids of one anchor's hop-2 ball, D5 and D5b applied.

    A plain BFS to depth 2 with exactly two expansion filters: the root's
    upward `sestavina` edges, and the collocation edges of the root's own
    senses.  Everything else -- forms, senses, examples, synonyms, antonyms,
    translations, and the constituents/forms/senses of a kept MWE -- is
    traversed unrestricted, because nothing else in the graph explodes.
    """
    plain, mwes = _split_hop1(store, a)
    if len(mwes) > k_mwe:
        cand = mwes
        if len(cand) > MWE_SCAN_CAP:
            if stats is not None:
                stats["mwe_scan_capped"] += 1
            cand = sorted(cand)[:MWE_SCAN_CAP]
        mwes = sorted(sorted(cand, key=lambda m: _mwe_sort_key(store, m))[:k_mwe])

    # D5b picks from the pool over ALL of the anchor's senses, so it has to be
    # resolved once for the anchor rather than per sense.
    keep_colloc = {int(v) for v, _p, _s in
                   d5b.sample(store, a, k=k_colloc)}

    own_senses = {int(v) for v in plain if store.kind[int(v)] == K_SENSE}
    hop1 = plain + mwes
    nodes = {int(a)} | set(hop1)
    for u in hop1:
        restrict = u in own_senses
        for v in store.nbrs(u):
            v = int(v)
            if restrict and store.kind[v] == K_COLLOC and v not in keep_colloc:
                continue
            nodes.add(v)
    return nodes


def induced_edges(store, order):
    """Every store edge with both ends in the ball, as index pairs into `order`.

    The induced subgraph rather than the BFS tree: a collocation node names both
    its members, and dropping the edge back to the second member would leave the
    graph claiming a relation the text already states.
    """
    pos = {n: i for i, n in enumerate(order)}
    edges = []
    for n, i in pos.items():
        for v in store.nbrs(n):
            v = int(v)
            j = pos.get(v)
            if j is not None and j > i:
                edges.append([i, j])
    return edges


def reverbalise(r, texts):
    """Re-draw a membership item's TARGET from the ball it will actually see.

    Only T17 is membership mode today.  Its answer string was verbalised at
    generation time from `d5b.sample(K=15)` -- before any ball existed -- while
    the ball holds K=10, so 26.5 % of its target phrases named collocations the
    model cannot see, across 62.6 % of its positives.  Training on that teaches
    exactly the failure the graph exists to prevent: emit a plausible-sounding
    collocation whether or not the evidence is in front of you.

    Grading is untouched and needs no change -- `membership` accepts any subset
    of `all_items` of the right size (`qa/grade.py` count_ok), and the ball was
    verified to hold enough of them for every item (`qa/check_balls.py`).  What
    moves is only which of the acceptable answers we supervise on: the ones that
    are visible.

    The band's count logic is the generator's, unchanged, applied to the ball's
    pool instead of the sampler's -- so an `exact` item still answers with
    exactly `n_asked` phrases and a `vague_large` item still answers long.
    """
    g = r.get("grading") or {}
    if g.get("mode") != "membership" or r["negative"]:
        return None
    allow = set(g["all_items"])
    pool = gen.dedup_by_norm(
        t[len("kolokacija: "):] for t in texts
        if t.startswith("kolokacija: ") and sl.norm(t[len("kolokacija: "):]) in allow)
    if not pool:
        return None
    band, n_all = g["quantity_band"], int(g["n_all"])
    if band == "exact":
        want = min(int(g["n_asked"]), n_all)
    else:
        want = min(gen.BAND_LEN[band], n_all)
    want = min(want, len(pool))
    if want < 1:
        return None
    items = pool[:want]
    return spec.PREFIX + g["sep"].join(items), items


def build(store, targets, k_mwe, k_colloc, cache, stats):
    """(node texts, edges, anchor positions) for the union of `targets`."""
    nodes = set()
    for a in targets:
        if a not in cache:
            cache[a] = ball_nodes(store, a, k_mwe, k_colloc, stats)
        nodes |= cache[a]
    # Sorted by node id, so the ball is a pure function of (store, policy) and
    # not of the order the extractor happened to return its strings in (C18 a).
    order = sorted(nodes)
    pos = {n: i for i, n in enumerate(order)}
    return ([store.text(n) for n in order],
            induced_edges(store, order),
            [pos[a] for a in targets])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="a RELABELLED dataset -- items need `targets`")
    ap.add_argument("out")
    ap.add_argument("--store", default=None)
    ap.add_argument("--k-mwe", type=int, default=K_MWE)
    ap.add_argument("--k-colloc", type=int, default=K_COLLOC_CAP)
    ap.add_argument("--types", default="", help="comma-separated subset")
    args = ap.parse_args()

    store = open_store(args.store)
    codes = np.asarray(store.codes)
    tok_len = np.asarray(store.G["token_len"])
    want = {t.strip() for t in args.types.split(",") if t.strip()}
    os.makedirs(args.out, exist_ok=True)

    cache, stats = {}, collections.Counter()
    print(f"policy: hop 2, K_mwe={args.k_mwe}, K_colloc={args.k_colloc}\n",
          flush=True)

    for split in ("train", "dev", "test"):
        src = os.path.join(args.dataset, f"{split}.jsonl")
        if not os.path.exists(src):
            continue
        n, sizes, toks, nanch = 0, [], [], []
        with open(src, encoding="utf-8") as f, \
                open(os.path.join(args.out, f"{split}.jsonl"), "w",
                     encoding="utf-8") as g:
            for line in f:
                r = json.loads(line)
                if want and r["type"] not in want:
                    continue
                if "targets" not in r:
                    raise SystemExit(
                        f"{src}: item {r['id']} has no `targets` -- run "
                        f"qa/relabel.py against an extraction run first")

                targets = []
                for c in r["targets"]:
                    i = int(np.searchsorted(codes, c))
                    if i < len(codes) and int(codes[i]) == c:
                        targets.append(i)
                if targets:
                    texts, edges, anchors = build(store, sorted(set(targets)),
                                                  args.k_mwe, args.k_colloc,
                                                  cache, stats)
                    t = sum(int(tok_len[n_]) for n_ in sorted(
                        {x for a in targets for x in cache[a]}))
                else:
                    # The extractor returned nothing the lookup could reach.
                    # The truthful input for "this word is not in the base" is a
                    # single node saying so -- not a reason to drop the item.
                    texts = [f"iztočnica: {r['lemma']} (ni v bazi)"]
                    edges, anchors, t = [], [0], len(texts[0]) // 4
                    stats["empty_ball"] += 1

                answer, gold = r["answer"], r.get("gold_items")
                rv = reverbalise(r, texts)
                if rv is not None:
                    if rv[0] != answer:
                        stats["reverbalised"] += 1
                    answer, gold = rv

                sizes.append(len(texts))
                toks.append(t)
                nanch.append(len(anchors))
                stats["union"] += len(anchors) > 1
                g.write(json.dumps({
                    "id": r["id"], "type": r["type"], "split": split,
                    "question": r["question"], "answer": answer,
                    "gold_items": gold,
                    "negative": r["negative"], "band": r["band"],
                    "nodes": texts, "edges": edges,
                    "anchors": anchors, "anchor": anchors[0],
                    "n_nodes": len(texts), "n_tokens": t,
                    "targets": r["targets"],
                    "policy": (r.get("extraction") or {}).get("policy"),
                }, ensure_ascii=False) + "\n")
                n += 1
        if sizes:
            print(f"[{split}] {n:,} items  "
                  f"nodes mean {np.mean(sizes):6.1f} "
                  f"p50 {np.percentile(sizes, 50):5.0f} "
                  f"p90 {np.percentile(sizes, 90):6.0f} "
                  f"p99 {np.percentile(sizes, 99):6.0f} max {max(sizes):6,}  |  "
                  f"tokens p50 {np.percentile(toks, 50):5.0f} "
                  f"p90 {np.percentile(toks, 90):6.0f} "
                  f"p99 {np.percentile(toks, 99):7.0f} max {max(toks):7,}",
                  flush=True)

    print(f"\ndistinct anchors ballooned: {len(cache):,}")
    print(f"union balls (>1 anchor):    {stats['union']:,}")
    print(f"single-node `ni v bazi`:    {stats['empty_ball']:,}")
    print(f"targets re-verbalised:      {stats['reverbalised']:,} "
          f"(membership items whose answer named a phrase outside their ball)")
    if stats["mwe_scan_capped"]:
        print(f"MWE scan cap hit on {stats['mwe_scan_capped']} anchors "
              f"(ranked a {MWE_SCAN_CAP:,}-candidate prefix by node id)")
    print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
