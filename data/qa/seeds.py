#!/usr/bin/env python3
"""The seed pool (D8), the frequency proxy (D9), the bands (D10) and the split (D11).

One pass over the store's core anchors materialises everything every generator
needs to decide which lemmas it may use and which band each falls in.  The pass
is the expensive part of generation (~72 k anchors, each traversed to its senses
and their children), so it is done once and handed around as a `Pool` object.

`sense_class` lives here rather than in the type generators because T12 and T14
must call the SAME function -- QA_TASKS.md Group D is explicit that a second
implementation is how the two silently drift apart, and C17 asserts they never do.
"""
import os
import json
import hashlib
import collections

import numpy as np

from . import sl

# D10 -- fixed log-spaced thresholds on the proxy, never quantiles.
BAND_EDGES = (0, 1, 5, 17, 65, 257, 1025)          # lower bound of B0..B6
BAND_NAMES = ("B0", "B1", "B2", "B3", "B4", "B5", "B6")
# D10 sampling quota, in per cent.  Tilted upward: B5+B6 rise from a natural
# 17.9 % of the pool to 30 % of the dataset.
BAND_QUOTA = (5, 10, 15, 20, 20, 20, 10)

# D11 / D16 -- lemma-disjoint split, ~10 k train (<=1 k dev) + ~2 k test.
SPLIT_SHARE = {"train": 0.75, "dev": 0.07, "test": 0.18}


def band_of(proxy):
    for i in range(len(BAND_EDGES) - 1, -1, -1):
        if proxy >= BAND_EDGES[i]:
            return BAND_NAMES[i]
    return BAND_NAMES[0]


def sense_class(store, a, sense_node, ordinal, body, lemma_cf):
    """placeholder | fallback | defined -- QA_TASKS.md Group D, shared by T12/T14.

    placeholder  the entry reified: the sense's own id equals the entry's id.
                 Never carries a definition; often the busiest node around.
    fallback     the text is the headword repeated (the builder writes
                 `body = definition or lemma`), so there is no gloss here.
    defined      anything else -- a real skos:definition.

    The rule is stated on the TEXT, not on the ids, because that is what the
    model sees: `pomen 1: otrok` (placeholder) and `pomen 2: otrok` (ordinary
    fallback) are identical in shape, and any policy that keeps one and drops the
    other is not derivable from the model's input.
    """
    if sense_node < store.n_real and store.lu_id(sense_node) == store.lu_id(a):
        return "placeholder"
    if not body or body.casefold() == lemma_cf:
        return "fallback"
    return "defined"


class Entry:
    """Everything the generators need about one seed lemma, computed once."""
    __slots__ = ("a", "lu", "lemma", "pos", "feat", "proxy", "band", "split",
                 "n_def", "n_syn", "n_ant", "n_ex", "n_col")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


class Pool:
    def __init__(self, store, entries):
        self.store = store
        self.entries = entries
        self.by_lemma = {e.lemma: e for e in entries}

    def __len__(self):
        return len(self.entries)

    def filter(self, pred):
        return [e for e in self.entries if pred(e)]

    def band_counts(self):
        return collections.Counter(e.band for e in self.entries)


def defined_senses(store, a):
    """[(node, ordinal, body)] of the senses T12 lists -- `defined`, deduped.

    Deduplication is by definition text: 0.40 % of pool anchors carry the same
    definition on two senses (`naslikati` -> *ustvariti sliko* twice).  T14 counts
    the length of THIS list, so the dedup has to happen here and only here.
    """
    lemma_cf = store.lemma(a).casefold()
    out = []
    seen = set()
    for v, ordinal, body in store.senses(a):
        if sense_class(store, a, v, ordinal, body, lemma_cf) != "defined":
            continue
        key = sl.norm(body)
        if key in seen or not key:
            continue
        seen.add(key)
        out.append((v, ordinal, body))
    return out


def ambiguous_lemma_units(path=None):
    """Lexical-units whose LEMMA is not determined by the source.

    `analysis/scan_multi_writtenrep.py` finds the entries whose canonical form
    carries two different words as `writtenRep` -- 958 in the KG, 196 of them
    core.  Some are legitimate variants (*molsti* / *molzti*, the *s* / *z*
    allomorphs); others are cross-entry contamination, and the builder cannot
    tell them apart because the RDF offers no signal:

        word-form-2773103  writtenRep  \"\"\"se\"\"\"@sl
        word-form-2773103  writtenRep  \"\"\"prikazati\"\"\"@sl

    That is the reflexive clitic `se`, which the builder's tie-break (least
    capitalised, then alphabetical) renders as `iztočnica: prikazati (zaimek,
    naslonska oblika)` -- a wrong lemma on the second-largest hub in the graph,
    and therefore wrong gold in every question about it.

    Excluded from the seed pool wholesale, variants included: even a legitimate
    variant leaves "which one is the lemma" undecided, and 196 of ~72,000 pool
    lemmas is 0.27 % -- far cheaper than one wrong answer on a high-frequency
    function word.
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "analysis", "results",
            "ambiguous_lemma_units.json")
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {int(k) for k in json.load(f)["units"]}


def entry_for(store, a, split=None, require_content=True):
    """One `Entry` for anchor `a`, or None when it is not usable as a seed.

    Factored out of `build_pool` so the same construction serves the union
    absence test in build_dataset: that test has to run the REAL generators over
    an arbitrary co-extracted anchor, and a generator needs a fully populated
    Entry.  Building it any other way would let the two drift.
    """
    a = int(a)
    lemma = store.lemma(a)
    if not lemma:
        return None
    n_def = len(defined_senses(store, a))
    n_col = len(store.collocations(a))
    n_syn = len(store.partners(a, 6))
    n_ant = len(store.partners(a, 7))
    n_ex = len(store.examples(a))
    if require_content and not (n_def or n_col or n_syn or n_ex):
        return None
    proxy = n_col + _mwe_memberships(store, a)
    return Entry(a=a, lu=store.lu_id(a), lemma=lemma, pos=store.pos(a),
                 feat=store.anchor_features(a), proxy=proxy,
                 band=band_of(proxy), split=split, n_def=n_def, n_syn=n_syn,
                 n_ant=n_ant, n_ex=n_ex, n_col=n_col)


def build_pool(store, progress=None):
    """D8: core entries with at least one of {definition, synonym, collocation,
    usage example}, with the proxy, band and split decided."""
    entries = []
    core = store.core_anchors()
    excluded = ambiguous_lemma_units()
    n_excluded = 0
    for i, a in enumerate(core):
        a = int(a)
        if store.lu_id(a) in excluded:
            if store.lemma(a):
                n_excluded += 1
            continue
        e = entry_for(store, a)
        if e is not None:
            entries.append(e)
        if progress and (i + 1) % 10000 == 0:
            progress(i + 1, len(core), len(entries))
    if progress:
        progress(len(core), len(core), len(entries))
    print(f"[pool] excluded {n_excluded:,} entries with a source-ambiguous lemma",
          flush=True)
    assign_split(entries)
    return Pool(store, entries)


def _mwe_memberships(store, a):
    n = 0
    for v in store.nbrs(a):
        v = int(v)
        if v < store.n_real and store.kind[v] == 0 and store.mwe[v]:
            n += 1
    return n


def assign_split(entries, seed="gtlm-qa-v1"):
    """D11: partition LEMMAS, stratified so every split shares the band profile.

    The split key is a hash of the lemma, so it is stable under any change to the
    pool's size or order -- adding a type, or regenerating against a new store,
    must not move a lemma from test to train.
    """
    by_band = collections.defaultdict(list)
    for e in entries:
        h = hashlib.blake2b(f"{seed}:{e.lemma}".encode(), digest_size=8).digest()
        by_band[e.band].append((int.from_bytes(h, "big"), e))
    for band, rows in by_band.items():
        # Sort on the hash ALONE.  A plain `rows.sort()` falls through to
        # comparing Entries whenever two hashes tie, which raises -- and two
        # entries can share a hash because the hash is over the LEMMA and one
        # lemma can have several entries (`voda` / `Voda`).
        rows.sort(key=lambda r: r[0])
        n = len(rows)
        n_train = int(round(n * SPLIT_SHARE["train"]))
        n_dev = int(round(n * SPLIT_SHARE["dev"]))
        for j, (_h, e) in enumerate(rows):
            e.split = ("train" if j < n_train
                       else "dev" if j < n_train + n_dev else "test")


def save_pool(pool, path):
    with open(path, "w", encoding="utf-8") as f:
        for e in pool.entries:
            f.write(json.dumps(e.as_dict(), ensure_ascii=False) + "\n")


def band_report(pool):
    c = pool.band_counts()
    tot = sum(c.values())
    lines = [f"{'band':4s} {'lemmas':>8s} {'share':>7s} {'quota':>6s}"]
    for b, q in zip(BAND_NAMES, BAND_QUOTA):
        lines.append(f"{b:4s} {c[b]:8,d} {100.0 * c[b] / max(tot, 1):6.1f}% {q:5d}%")
    lines.append(f"{'all':4s} {tot:8,d}")
    return "\n".join(lines)
