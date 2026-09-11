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


def constituents(store, a):
    """The word anchors an MWE is composed of, via its `sestavina` edges.

    Empty for a WORD anchor, and the guard is load-bearing rather than defensive.
    The filter below reads "an anchor neighbour that is not itself an MWE", which
    for a phrase means its parts -- but run on a word it would mean *the other
    words that word is linked to*, and T30 would answer "what is X made of?" with
    a list of X's relatives.  Only ever called on phrases until now, so nothing
    had exercised the other direction.
    """
    from .store import K_ANCHOR
    if not store.mwe[a]:
        return []
    return [int(v) for v in store.nbrs(a)
            if store.kind[int(v)] == K_ANCHOR and not store.mwe[int(v)]]


def constituent_occurs(store, c, phrase, cache=None):
    """C26: does constituent anchor `c` actually SHOW UP in `phrase`?

    Its own lemma, or one of its `oblika:` forms, as a whole word -- `razprava`
    is not in *razvneti razpravo* and `razpravo` is.

    The check is needed because the KG's own constituent edges are not always
    right, and the wrong ones are concentrated: over 2,365 sampled phrases, 49
    anchors account for every failure and the top ten for 93.6 % of them. They
    are function words whose anchor carries a misleading lemma -- node 100762
    reads `iztočnica: prikazati (zaimek, naslonska oblika)` and is the clitic
    *se*, in 270 phrases; nodes 260/261/277 read `celoti`, `na` and `silo`,
    are all tagged `predlog`, and stand in for *v* and *z*.

    So the answer would name a word that is not in the phrase, in a shape that
    looks perfectly well formed -- which is why this is a filter as well as a
    check.  It costs 26.4 % of T30's items.

    `cache` maps an anchor to its surfaces, and it is not an optimisation to be
    weighed -- it is the difference between a 6-minute generation stage and a
    40-minute one.  Without it, `store.forms` is walked again for every phrase a
    constituent appears in, and the constituents that appear in the most phrases
    are the function words with the most forms.  Pass one dict for the whole run.
    """
    if sl.occurs(store.lemma(c), phrase):
        return True
    if cache is None:
        return any(sl.occurs(s, phrase) for _v, s, _f in store.forms(c))
    surfaces = cache.get(c)
    if surfaces is None:
        surfaces = cache[c] = tuple({s for _v, s, _f in store.forms(c) if s})
    return any(sl.occurs(s, phrase) for s in surfaces)


def phrase_proxy(store, a, cache=None):
    """D9's frequency proxy for a PHRASE: the MINIMUM over its constituents'.

    A phrase has no collocations and belongs to no larger MWE, so the word
    proxy (`n_col + mwe_memberships`) is ~0 for all of them: measured, 316 of
    319 phrase seeds land in B0 and the whole family collapses into one band.

    `min` rather than max/sum/median, decided by measurement over 6,000
    phrases -- share of the family in each band, against a core pool that runs
    29.9/12.3/13.9/14.8/15.4/9.4/4.3 across B0..B6:

        min     0.0  1.5  2.7  7.4 23.1 39.9 25.4      <- spreads
        median  0.0  0.1  0.1  0.4  3.4 20.6 75.4
        max     0.0  0.1  0.1  0.2  1.4 10.6 87.7
        sum     0.0  0.0  0.1  0.1  0.9  7.9 91.0

    max, sum and median are all dominated by the phrase's function word --
    `biti` is a constituent of 423,510 phrases -- so they band the family by
    its most common part and pile 75-91 % into B6, which is exactly as flat as
    the B0 collapse they were meant to fix.  `min` is also the reading that
    means anything: a phrase is at most as common as its rarest constituent.

    B0 comes out empty by construction, and that is correct rather than a
    miss -- every constituent belongs to at least the phrase being scored, so
    the minimum proxy is at least 1.

    `cache` maps constituent node -> proxy.  Pass one: `store.proxy` walks the
    node's whole adjacency, phrases share their function words, and without it
    a pool build re-walks `biti`'s 423,510 edges once per phrase that uses it.
    """
    cs = constituents(store, a)
    if not cs:
        return 0
    if cache is None:
        return min(store.proxy(c) for c in cs)
    out = None
    for c in cs:
        p = cache.get(c)
        if p is None:
            p = cache[c] = store.proxy(c)
        out = p if out is None or p < out else out
    return out


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
                 "n_def", "n_syn", "n_ant", "n_ex", "n_col",
                 # Group H counters.  `mwe` is what the phrase types select on;
                 # `n_mwe`/`n_const` are the two counts that are facts about an
                 # entry's position in the MWE graph rather than about a
                 # relation, and T24/T27 compare counts across entries, so they
                 # have to be exact here rather than recomputed per generator.
                 "mwe", "n_mwe", "n_const",
                 # T36's relation.  Counted here rather than probed per entry
                 # because the negative pool is chosen by a predicate that runs
                 # over the WHOLE pool once per split, and a sense walk there
                 # costs more than the one extra walk this adds to the build.
                 "n_trans")

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


def entry_for(store, a, split=None, require_content=True, proxy_cache=None):
    """One `Entry` for anchor `a`, or None when it is not usable as a seed.

    Factored out of `build_pool` so the same construction serves the union
    absence test in build_dataset: that test has to run the REAL generators over
    an arbitrary co-extracted anchor, and a generator needs a fully populated
    Entry.  Building it any other way would let the two drift.

    Works for a phrase anchor as well as a word one; the two differ only in how
    the proxy is computed (see `phrase_proxy`).  The content rule is deliberately
    NOT tightened for phrases: 95.4 % of MWEs pass it on a usage example alone,
    and that example-bearing half is the capability the phrase family exists to
    reach.  Bounding the phrase pool is `build_phrase_pool`'s job, by sampling.
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
    n_trans = len(store.translations(a))
    mwe = bool(store.mwe[a])
    n_mwe = _mwe_memberships(store, a)
    n_const = len(constituents(store, a)) if mwe else 0
    proxy = (phrase_proxy(store, a, proxy_cache) if mwe else n_col + n_mwe)
    return Entry(a=a, lu=store.lu_id(a), lemma=lemma, pos=store.pos(a),
                 feat=store.anchor_features(a), proxy=proxy,
                 band=band_of(proxy), split=split, n_def=n_def, n_syn=n_syn,
                 n_ant=n_ant, n_ex=n_ex, n_col=n_col,
                 mwe=mwe, n_mwe=n_mwe, n_const=n_const, n_trans=n_trans)


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


# How many phrase seeds to draw.  The word pool is ~72 k and the corpus it
# supports is 12,490 items; the phrase family is a minority of the type
# inventory, so a pool of the same order is generous and nothing about it wants
# to be 3.76 M.
# Raised from 80,000 when the rest of H.5 landed, and the reason is one type:
# T32 asks a phrase for its DEFINITION and only 1.4 % of MWEs carry one, so an
# 80 k sample yielded 973 train seeds against the 1,250 a 40,000-item train
# split needs -- measured, not projected.  200 k puts it at ~2,400 with margin,
# and fixes T34's dev side too (32 seeds, where the split agreement of a median
# 3 constituents costs 0.07**3).  Every other phrase type had headroom at 80 k.
#
# Raised again to 500,000 once the real generators could be measured instead of
# proxied.  The TRAIN side was never the binding constraint: at 200 k T32 had
# 2,416 train seeds and ~196 test ones, against a flat 200-per-type test split,
# and T34's dev side sampled at zero.  Both are the small splits -- 18 % and
# 7 % -- so a pool sized by the train share leaves them short, and the fix is
# the pool rather than the caps.  The cost is the pool build: ~130 s -> ~320 s,
# paid once per run.
PHRASE_POOL = 500_000


def build_phrase_pool(store, limit=PHRASE_POOL, seed=20260821, progress=None):
    """D8 for phrases: a BOUNDED sample of MWE anchors, entries built for the draw.

    `build_pool` can walk its whole population because there are 100,801 core
    anchors.  There are 3,940,417 MWE anchors, and `entry_for`'s content rule
    keeps 95.4 % of them -- so the same approach projects a pool of ~3.76 M
    entries and a 14.3-minute pass before a single item is generated.

    The fix is to bound the pool by SAMPLING rather than by content.  Tightening
    the content rule instead is the tempting mistake: requiring a definition cuts
    the pool to ~62,849, a pleasant number bought by discarding the 97.6 % of
    phrases whose content is a usage example -- which is precisely the half
    QA_TASKS.md H.5 wants these types to reach.

    Deterministic: the permutation is seeded, so the draw is a pure function of
    (store, limit, seed) exactly as `build_pool` is a pure function of the store.
    """
    import numpy as np

    mwe = store.mwe_anchors()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(mwe))
    entries, cache, seen = [], {}, 0
    for j in order:
        if len(entries) >= limit:
            break
        seen += 1
        e = entry_for(store, int(mwe[int(j)]), proxy_cache=cache)
        if e is not None:
            entries.append(e)
        if progress and seen % 10000 == 0:
            progress(seen, len(mwe), len(entries))
    print(f"[phrase pool] {len(entries):,} entries from {seen:,} sampled of "
          f"{len(mwe):,} MWE anchors ({len(cache):,} constituent proxies cached)",
          flush=True)
    assign_split(entries)
    return entries


def same_split_partners(entries, store, kind):
    """H.3: {entry -> [partner entry, ...]} over one relation, same split only.

    T25/T26/T27 name two lemmas, and C11 is a statement about lemmas rather than
    about items: a test item naming a training lemma breaks the split exactly as
    a duplicated seed would.  `assign_split` hashes ONE lemma, so a pair's two
    halves agree only by chance -- measured over the whole pool, 60.0 % for
    `sopomenka` and 59.5 % for `protipomenka`, both sitting on the Σ share²
    ≈ 0.5998 baseline, which is what "by chance" looks like when it is true.

    So the pair is drawn from within one split.  Capacity after the filter, and
    the reason it must be recomputed per relation rather than scaled from the
    coverage table: `sopomenka` keeps 96,456 of 206,476 partner mentions and
    `protipomenka` 3,044 of 5,654 -- the thinner relation keeps the LARGER
    share, because its partners are likelier to be seeds themselves.
    """
    by_lemma = {}
    for e in entries:
        by_lemma.setdefault(e.lemma.casefold(), e)
    out = {}
    for e in entries:
        keep = []
        for p in store.partners(e.a, kind):
            o = by_lemma.get(p.casefold())
            if o is not None and o.split == e.split and o.a != e.a:
                keep.append(o)
        if keep:
            out[e.a] = keep
    return out


def constituents_agree(store, entry, word_split):
    """H.3, one level down: does every seeded constituent share the phrase's split?

    A phrase type whose ANSWER names its constituents (T30, T34) names word
    lemmas, and C11 is a check over every lemma an item names -- so a train
    phrase built on a test seed breaks lemma-disjointness exactly as a
    duplicated seed does.  This is the same argument H.3 makes for synonym
    pairs, and it needs the same filter.

    Measured on 5,000 phrase seeds against the 71,152-lemma word pool: only
    **39.9 %** agree, a steeper loss than H.3's ~40 % because a phrase has a
    median of 3 constituents and each agrees only by chance.  It is a filter and
    not a blocker: 39.9 % of an 80 k phrase pool is ~32,000 usable seeds, against
    the few hundred a single type draws.

    Applied ONLY by the constituent-naming types.  A phrase type that asks for a
    definition or an example names no constituent lemma and must not pay this
    cost -- which is why it is a helper here rather than a rule inside
    `build_phrase_pool`.

    `word_split` maps a casefolded word lemma to its split; build it once from
    the word pool.  A constituent that is not a seed at all is not a leak: it
    never appears as a training seed, so it cannot collide with one.
    """
    for c in constituents(store, entry.a):
        s = word_split.get(store.lemma(c).casefold())
        if s is not None and s != entry.split:
            return False
    return True


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
