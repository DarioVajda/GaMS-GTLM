#!/usr/bin/env python3
"""Turn relabelled QA items into the small text graphs a GTLM actually reads.

    sbatch qa/run_build_balls.sbatch                 # -> datasets/balls
                                                     #  + datasets/generated

**Two outputs, one target.**  `reverbalise()` re-draws a membership item's answer
from the ball it will actually see, so the answer the ball ships and the answer
the dataset ships must be the same string.  `--dataset-out` writes the second
one.  `datasets/generated` is then the authority for `answer` and `gold_items`;
the relabelled set this reads (stage 3, in `datasets/work/`) stays immutable, and
the rewrite is auditable as a diff between the two directories.

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
       than truncated -- see qa/colloc_sampling.py.

Both K default to 10.

**M2: the periphrastic auxiliary is NOT injected.**  The opposite was specified
first -- put `biti`'s 17 form nodes in every verb's ball so T5 and T6
containment is literally true.  Reversed, for three reasons (QA_TASKS.md 0.7).
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
import json
import argparse
import contextlib
import collections

import numpy as np

from qa.store import open_store, K_ANCHOR, K_SENSE, K_COLLOC
from qa import colloc_sampling, gen, grade, pairs, sl, spec
from lib.errors import StageError

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
                   colloc_sampling.sample(store, a, k=k_colloc)}

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

    Returned in ascending `(i, j)` order.  Within one node the store lists its
    neighbours in CSR order, which is not part of the store's contract, so an
    unsorted edge list would make the written ball depend on it (C18 c).
    """
    pos = {n: i for i, n in enumerate(order)}
    edges = []
    for n, i in pos.items():
        for v in store.nbrs(n):
            v = int(v)
            j = pos.get(v)
            if j is not None and j > i:
                edges.append([i, j])
    edges.sort()
    return edges


#: membership type -> the store accessor for everything it may legitimately name.
MEMBER_SOURCE = {"T17": "collocations", "T19": "examples"}


def member_pool(r, texts):
    """The member nodes of `r`'s BALL -- what the target is drawn from.

    Distinct from the allow-list: this is what the model can *see*, so it is what
    we supervise on (`reverbalise`), and it is a subset of what we *accept*.
    """
    kind = spec.MEMBER_KIND.get(r["type"])
    if not kind:
        return []
    pre = kind + ": "
    return gen.dedup_by_norm(t[len(pre):] for t in texts if t.startswith(pre))


def member_allow(store, r, targets):
    """Everything `r` may legitimately name: the store's set over ALL its anchors.

    Deliberately WIDER than the ball.  An answer is correct when it is true, and
    a phrase the store records for this lemma is true whether the model read it
    off the graph or already knew it -- rewarding recall of real Slovene is not a
    failure mode we want to grade against.  The ball's members are a subset of
    this (they are drawn from the same store, under the K cap), so a model that
    only ever reads its input is fully covered.

    Over ALL anchors, which is the bug this replaces: `gen.all_phrases` used one
    anchor while D3 unions several into a ball, so the model would be shown
    members the allow-list has never heard of and scored `not_in_all` for naming
    them.  Only union balls are affected, so the defect is invisible unless the
    lemma is a homograph.
    """
    attr = MEMBER_SOURCE.get(r["type"])
    if not attr:
        return []
    get = getattr(store, attr)
    return sorted({sl.norm(p) for a in targets for _v, p in get(a) if p})


def member_contract(store, r, texts, targets, stats):
    """Set a membership item's `all_items`/`n_all`, and return its target pool.

    Two different sets, and conflating them is what produced the T17 defect:

        ALLOW (`all_items`)   everything true of this item's anchors -- what the
                              grader accepts.  Wider than the ball on purpose.
        POOL  (return value)  the members of the ball -- what the model can see,
                              so what we supervise on.

    Stage 4 is the first stage holding both the item and its resolved anchors,
    so it is the only place either can be computed correctly.
    """
    # Drop the per-TYPE constants an older pipeline copied into the row.  They
    # are read from `qa/spec.py` now (`grade.contract`), and leaving stale
    # copies on disk is the thing that made T19 gradeable as `sequence` long
    # after the spec said otherwise.
    row = r.setdefault("grading", {})
    for key in grade.TYPE_LEVEL + grade.RETIRED_ROW_FIELDS:
        if row.pop(key, None) is not None:
            stats["stale_type_fields_dropped"] += 1

    g = grade.contract(r)
    if g["mode"] != "membership" or r["negative"]:
        return None
    pool = member_pool(r, texts)
    # The ball is drawn from the store under a K cap, so it is a subset -- but
    # union rather than assume, so a policy change upstream can never make a
    # visible member unnameable.
    # PAIRS, not bare phrases (0.8.3): under 0.1 the answer names its members as
    # `kolokacija: mineralna voda`, so the allow-list has to check the label
    # alongside the phrase or membership would accept a true phrase under a
    # relation the item never asked about.
    kind = spec.MEMBER_KIND[r["type"]]
    allow = [[kind, v] for v in sorted({*member_allow(store, r, targets),
                                        *(sl.norm(p) for p in pool)})]
    was = {tuple(x) if isinstance(x, (list, tuple)) else (kind, x)
           for x in (row.get("all_items") or [])}
    if was != {tuple(x) for x in allow}:
        stats["contract_rebuilt"] += 1
        stats["contract_items_delta"] += len(allow) - len(was)
    # Written even when empty: an empty allow-list is a real, gradeable state
    # (nothing this item may name), and leaving the key absent would instead
    # make the item silently ungradeable.
    row["all_items"], row["n_all"] = allow, len(allow)
    return pool or None


def reverbalise(r, pool):
    """Re-draw a membership item's TARGET from the ball it will actually see.

    T17's answer was verbalised at generation time from `colloc_sampling.sample(K=15)` --
    before any ball existed -- while the ball holds K=10, so 26.5 % of its target
    phrases named collocations the model cannot see, across 62.6 % of its
    positives.  Training on that teaches exactly the failure the graph exists to
    prevent: emit a plausible-sounding collocation whether or not the evidence is
    in front of you.  T19 is the same story with one example instead of many.

    The band's count logic is the generator's, unchanged, applied to the ball's
    pool instead of the sampler's -- so an `exact` item still answers with
    exactly `n_asked` phrases and a `vague_large` item still answers long.
    """
    if not pool:
        return None
    g = grade.contract(r)
    n_all = int(g["n_all"])
    band = g["quantity_band"]
    if band == "exact":
        want = min(int(g["n_asked"]), n_all)
    else:
        want = min(gen.BAND_LEN[band], n_all)
    want = min(want, len(pool))
    if want < 1:
        return None
    line, gold = pairs.answer_of(r["type"], r.get("slots"), pool[:want])
    return line, gold


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


def run(dataset, out, store=None, k_mwe=K_MWE, k_colloc=K_COLLOC_CAP, types="",
        dataset_out=None):
    """Build one ball per item, and the dataset those balls re-verbalise.

    The body `main()` used to hold, so the pipeline can call this stage as a
    function while the command line keeps behaving exactly as it did.
    """
    args = argparse.Namespace(dataset=dataset, out=out, store=store,
                              k_mwe=k_mwe, k_colloc=k_colloc, types=types,
                              dataset_out=dataset_out)

    store = open_store(args.store)
    codes = np.asarray(store.codes)
    tok_len = np.asarray(store.G["token_len"])
    want = {t.strip() for t in args.types.split(",") if t.strip()}
    os.makedirs(args.out, exist_ok=True)
    if args.dataset_out:
        os.makedirs(args.dataset_out, exist_ok=True)

    cache, stats = {}, collections.Counter()
    print(f"policy: hop 2, K_mwe={args.k_mwe}, K_colloc={args.k_colloc}\n",
          flush=True)

    for split in ("train", "dev", "test"):
        src = os.path.join(args.dataset, f"{split}.jsonl")
        if not os.path.exists(src):
            continue
        n, sizes, toks, nanch = 0, [], [], []
        ds_sink = (open(os.path.join(args.dataset_out, f"{split}.jsonl"), "w",
                        encoding="utf-8")
                   if args.dataset_out else contextlib.nullcontext())
        with open(src, encoding="utf-8") as f, \
                open(os.path.join(args.out, f"{split}.jsonl"), "w",
                     encoding="utf-8") as g, ds_sink as d:
            for line in f:
                r = json.loads(line)
                if want and r["type"] not in want:
                    continue
                if "targets" not in r:
                    raise StageError(
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
                # The allow-list comes from the ball, and the target is drawn
                # from that same pool -- so the contract, the input and the
                # supervision cannot disagree about what this item is.
                rv = reverbalise(
                    r, member_contract(store, r, texts, targets, stats))
                if rv is not None:
                    if rv[0] != answer:
                        stats["reverbalised"] += 1
                    answer, gold = rv
                    # The dataset row carries the SAME target, so the ball and
                    # the dataset cannot disagree about what is being supervised.
                    r["answer"], r["gold_items"] = answer, gold
                if d is not None:
                    d.write(json.dumps(r, ensure_ascii=False) + "\n")

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
    print(f"allow-lists rebuilt:        {stats['contract_rebuilt']:,} changed "
          f"({stats['contract_items_delta']:+,} member items vs what the row "
          f"carried -- the store over ALL anchors, not just the first)")
    print(f"stale type fields dropped:  {stats['stale_type_fields_dropped']:,} "
          f"(mode copies, and the sep/arity/regex 0.1 retired; qa/spec.py owns "
          f"what is left)")
    if stats["mwe_scan_capped"]:
        print(f"MWE scan cap hit on {stats['mwe_scan_capped']} anchors "
              f"(ranked a {MWE_SCAN_CAP:,}-candidate prefix by node id)")
    print(f"\n[wrote] {args.out}")
    if args.dataset_out:
        print(f"[wrote] {args.dataset_out}  (dataset rows, membership targets "
              f"re-verbalised -- THIS is the authority for `answer`)")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="a RELABELLED dataset -- items need `targets`")
    ap.add_argument("out")
    ap.add_argument("--store", default=None)
    ap.add_argument("--k-mwe", type=int, default=K_MWE)
    ap.add_argument("--k-colloc", type=int, default=K_COLLOC_CAP)
    ap.add_argument("--types", default="", help="comma-separated subset")
    ap.add_argument("--dataset-out", default=None,
                    help="also rewrite the DATASET rows here, carrying the "
                         "re-verbalised membership target (see `reverbalise`). "
                         "Without it the corrected answer would live only in the "
                         "ball and every consumer reading the dataset's `answer` "
                         "for T17 would get a target naming phrases the model "
                         "cannot see.  Pass a NEW directory: the stage-3 artefact "
                         "stays immutable and the rewrite is auditable as a diff.")
    args = ap.parse_args()

    try:
        run(dataset=args.dataset, out=args.out, store=args.store,
            k_mwe=args.k_mwe, k_colloc=args.k_colloc, types=args.types,
            dataset_out=args.dataset_out)
    except StageError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
