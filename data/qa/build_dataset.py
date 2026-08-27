#!/usr/bin/env python3
"""Generate the QA dataset: questions, answers, and everything the grader needs.

    python -m qa.build_dataset --store data/stores/kg_graph_gemma3 \
                               --out   data/datasets/work/generated_raw

Writes `train.jsonl`, `dev.jsonl`, `test.jsonl`, plus `pool.jsonl` (the seed pool
with its bands) and `report.json` (the realised type x band x split matrix, the
availability matrix behind it, and the negative mix).

The item record -- the first artefact everything downstream reads:

    id                "T12-000123"
    type / type_name  "T12" / "pomen/razlaga_pomena"
    lemma / lu_id / node_code    the anchor, in the same id space as the store
    band / proxy      D10 band and the D9 proxy it came from
    split / tier      train|dev|test, and core|A|C (D12)
    template_id       "T12/03" -- which frame, so a per-frame breakdown is free
    question          plain prose, no delimiters (D3)
    answer            the gradeable line, and nothing else (0.1)
    gold_items        the answer's items, pre-split, for analysis
    slots             the frame's slot values
    negative          bool, and negative_flavour when true
    grading           the contract the grader reads: mode, sep, arity, regex,
                      plus all_items/n_all/quantity_band/n_asked for T17

Quotas follow D16 (~10 k train incl. <=1 k dev, ~2 k test) and D10's band shares,
each capped by what the graph can actually supply and the shortfall redistributed
across bands with headroom -- Section 5 of QA_DATASET_DESIGN.md calls for exactly
this rather than silently producing a skewed set.
"""
import os
import json
import random
import hashlib
import argparse
import collections

from qa import sl, seeds, gen, spec, templates
from qa.store import open_store
from lib.errors import StageError

TYPES = [t for t in spec.SPEC]
TARGET = {"train": 9000, "dev": 1000, "test": 2000}
NEGATIVE_RATE = 0.10
UNLISTED_SHARE = 0.30             # of the negatives; the rest are ordinary lemmas


def _rng(*parts):
    """A `random.Random` seeded from what it is FOR, not from call order.

    Generation used one sequential stream, so touching any code path shifted
    every later draw and a one-line change produced a wholly different corpus.
    Keying the stream on (seed, type, split, lemma, purpose) makes a change local:
    edit the negative path and only negatives move.  Same reasoning as D5b's
    "derive the RNG seed from the anchor's node code -- never from a loop counter".
    """
    h = hashlib.blake2b(":".join(str(p) for p in parts).encode(), digest_size=8)
    return random.Random(int.from_bytes(h.digest(), "big"))


# --------------------------------------------------------------------------
# who can be a negative for each type
# --------------------------------------------------------------------------
def _neg_absent(field):
    return lambda ctx, e: getattr(e, field) == 0


NEGATIVE_SEED = {
    # category mismatch -- an ordinary lemma the question does not apply to
    "T1":  lambda ctx, e: e.pos == "glagol",
    "T2":  lambda ctx, e: e.pos == "glagol",
    "T3":  lambda ctx, e: e.pos == "glagol",
    "T5":  lambda ctx, e: e.pos not in ("glagol", None),
    "T6":  lambda ctx, e: e.pos not in ("glagol", None),
    "T7":  lambda ctx, e: e.pos not in ("glagol", None),
    "T9":  lambda ctx, e: e.pos not in ("samostalnik", None),
    "T10": lambda ctx, e: e.pos not in ("glagol", None),
    # the relation is genuinely absent
    "T11": lambda ctx, e: (e.pos in ("pridevnik", "prislov")
                           and gen.gradation(ctx.store, e.a, e.lemma) is None),
    "T12": _neg_absent("n_def"),
    "T14": _neg_absent("n_def"),
    "T15": _neg_absent("n_syn"),
    "T16": _neg_absent("n_ant"),
    "T17": _neg_absent("n_col"),
    "T19": _neg_absent("n_ex"),
    # T4, T8, T20, T21 are handled specially -- see negative_item()
}


# Types whose negative asserts "this ENTRY does not record X".  T20/T21 are
# excluded because their negative is pairing-scoped, not entry-scoped -- the gold
# says the form does not belong to the named lemma (T21) or does not occur in the
# given sentence (T20), and a co-extracted anchor that analyses the form on its
# own terms does not contradict either.  T4's negatives are 100 % `unlisted`, so
# there is nothing here to test.
UNION_TESTED = set(TYPES) - {"T4", "T20", "T21"}


def _union_is_silent(ctx, type_key, surface, own):
    """True when NO anchor the surface resolves to can answer this type.

    The absence test must run over the UNION D3 will hand the model, not over
    the seed anchor alone: otherwise a negative ships with gold its own ball
    contradicts, because a second lexical unit with the same lemma records what
    the item calls unrecorded (`konec`), or a co-extracted homonym does (`rahlo`
    has no gradation, `rahel` does).

    Runs the real generator rather than a per-type predicate written here, the
    way `availability()` does.  A predicate would be a second copy of each
    generator's eligibility rule and would drift from it silently.
    """
    fn = gen.GENERATORS.get(type_key)
    if fn is None:
        return True
    for a in ctx.surface.get(surface.casefold(), ()):
        a = int(a)
        if a == own:
            continue            # the seed's own silence is already established
        other = seeds.entry_for(ctx.store, a)
        if other is None:
            continue
        try:
            if fn(ctx, other, _rng("union", type_key, a)) is not None:
                return False
        except Exception:                                       # noqa: BLE001
            continue
    return True


def negative_item(ctx, type_key, e, rng, flavour):
    """(slots, flavour) for one negative, or None.

    Flavours (0.2): `unlisted` names a REAL word the D3 lookup cannot reach
    (qa/unlisted.py); `absent` and `mismatch` are both ordinary lemmas genuinely
    missing the relation, kept apart in the record so the mix is measurable
    rather than assumed.

    `unlisted` words are real and unreachable, never misspellings -- see
    qa/unlisted.py.
    """
    lemma = e.lemma
    if flavour == "unlisted":
        words = ctx.unlisted
        if not words:
            return None
        lemma = words[rng.randrange(len(words))]
    slots = {"L": lemma}
    if type_key in ("T2",):
        n = rng.choice(list(sl.NUMBERS))
        slots.update(number=n, **gen.number_slots(n))
    elif type_key in ("T3",):
        c = rng.choice(list(sl.CASES)); n = rng.choice(list(sl.NUMBERS))
        slots.update(case=c, number=n, **gen.case_slots(c), **gen.number_slots(n))
    elif type_key == "T6":
        t = rng.choice(list(sl.TENSES))
        slots.update(tense=t, **gen.tense_slots(t))
    elif type_key == "T17":
        slots.update(band="none", n_all=0)
    elif type_key in ("T4",):
        # T4 has no ordinary-lemma negative: every real form belongs to some
        # lemma, so "no lemma is recorded" is only true of a string the lookup
        # cannot reach.  Its negatives are therefore 100 % flavour (a), which is
        # the one documented deviation from the 70/30 mix (0.2).
        if flavour != "unlisted":
            return None
        slots = {"F": lemma, "L": lemma}
    elif type_key in ("T20", "T21"):
        # An ordinary-lemma negative here is a real form of a DIFFERENT entry
        # offered under this lemma -- both words exist, the pairing does not.
        other = ctx.pool.entries[rng.randrange(len(ctx.pool.entries))]
        grid = gen.nominal_grid(ctx.store, other.a)
        surfaces = sorted({s for v in grid.values() for s in v})
        mine = {s for v in gen.nominal_grid(ctx.store, e.a).values() for s in v}
        surfaces = [s for s in surfaces if s not in mine]
        if not surfaces:
            return None
        if type_key == "T20":
            ex = [t for _v, t in ctx.store.examples(e.a) if t]
            if not ex:
                return None
            sent = ex[0]
            # The borrowed form must not actually OCCUR in the sentence.  Without
            # this the item can be a false negative: `aneksije` is a form of a
            # different entry, but nothing stopped the draw from landing on a
            # string that happens to appear in this sentence, in which case the
            # question has a true answer and the gold says there is none.
            surfaces = [s for s in surfaces if not gen.occurrences(s, sent)]
            if not surfaces:
                return None
            slots = {"F": surfaces[rng.randrange(len(surfaces))], "L": lemma,
                     "S": sent}
        else:
            slots = {"F": surfaces[rng.randrange(len(surfaces))], "L": lemma}
        flavour = "mismatch"
    return slots, flavour


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def render_question(type_key, slots, rng, allow_tier_a, negative, pos=None):
    key = spec.template_key(type_key, slots)
    if negative:
        pool_frames = templates.neutral_frames(key)
    else:
        pool_frames = templates.frames_for_pos(key, pos)
    pool_frames = [f for f in pool_frames
                   if allow_tier_a or templates.tier_of(key, f) != "A"]
    if not pool_frames:
        return None, None, None
    frame = pool_frames[rng.randrange(len(pool_frames))]
    try:
        q = frame.format(**slots)
    except KeyError:
        return None, None, None
    return q, templates.frame_id(key, frame), templates.tier_of(key, frame)


def make_item(ctx, type_key, e, slots, items, *, split, rng, negative=False,
              flavour=None, allow_tier_a=False, idx=0):
    sp = spec.SPEC[type_key]
    q, tid, tier = render_question(type_key, slots, rng, allow_tier_a, negative,
                                   pos=e.pos)
    if q is None:
        return None
    if negative:
        answer = spec.sentinel_line()
        gold = [spec.SENTINEL]
    else:
        sep = sp["sep"]
        if sp["arity"] == 1 or sep is None:
            answer = spec.PREFIX + items[0]
        elif isinstance(sep, (list, tuple)):
            answer = spec.PREFIX + _join_grouped(type_key, items)
        else:
            answer = spec.PREFIX + sep.join(items)
        gold = items
    # ITEM-level facts only.  `mode`, `sep`, `arity` and `regex` are per-TYPE
    # constants and are read from `qa/spec.py` at grading time (`grade.contract`)
    # -- writing them here would put 12,490 stale copies on disk, which is how
    # T19 kept being graded `sequence` after the spec said `membership`.
    grading = {}
    # `all_items` is deliberately NOT set here.  It is the set of members the
    # model will be SHOWN, and no ball exists yet at generation time; deriving it
    # from one anchor is what made D3's union balls ungradeable.  Stage 4 owns
    # it (`build_balls.member_contract`).
    if sp["mode"] == "membership":
        grading.update(quantity_band=slots.get("band"),
                       n_asked=slots.get("N"))
    return {
        "id": f"{type_key}-{idx:06d}",
        "type": type_key,
        "type_name": sp["name"],
        "lemma": e.lemma,
        "lu_id": e.lu,
        "node_code": int(ctx.store.codes[e.a]),
        "band": e.band,
        "proxy": e.proxy,
        "split": split,
        "tier": "C" if type_key in spec.TIER_C else ("A" if tier == "A" else "core"),
        "template_id": tid,
        "question": q,
        "answer": answer,
        "gold_items": gold,
        "slots": {k: v for k, v in slots.items() if k not in ("n_all",)},
        "negative": bool(negative),
        "negative_flavour": flavour,
        "grading": grading,
    }


def _tier_c_safe(item):
    """Keep a training item only if it cannot leak the held-out antonym relation.

    Tag words (`protipomenka`, `antonim`) are banned outright; the ordinary
    Slovene words for "opposite" are banned from the QUESTION, which we author,
    and allowed in the ANSWER, which is the KG's own data (the entry `obratno`
    is defined as *na nasproten način*).  See spec.TIER_C_TAG_WORDS.
    """
    if item["split"] == "test":
        return True
    q = item["question"].casefold()
    a = item["answer"].casefold()
    if any(w in q or w in a for w in spec.TIER_C_TAG_WORDS):
        return False
    return not any(w in q for w in spec.TIER_C_SOFT_WORDS)


def _join_grouped(type_key, items):
    """Re-insert the group separator for the two grouped types (spec.SPEC)."""
    if type_key == "T1":
        return "; ".join(", ".join(items[i:i + 6]) for i in (0, 6, 12))
    if type_key == "T5":
        return "; ".join(", ".join(items[i:i + 9]) for i in (0, 9, 18))
    raise ValueError(type_key)


# --------------------------------------------------------------------------
def availability(ctx, rng, types, sample_cap=None):
    """{type: {band: [entries]}} -- who can actually answer each type.

    Computed by RUNNING the generator, not by predicting it: a type is available
    for an entry iff the generator returns a gold answer for it.  Slower than a
    heuristic and the only version that cannot be wrong.
    """
    avail = {t: collections.defaultdict(list) for t in types}
    errors = collections.Counter()
    first_error = {}
    entries = ctx.pool.entries
    for i, e in enumerate(entries):
        if sample_cap and i >= sample_cap:
            break
        for t in types:
            try:
                out = gen.GENERATORS[t](ctx, e, random.Random(0))
            except Exception as exc:                # noqa: BLE001
                errors[t] += 1
                first_error.setdefault(t, f"{e.lemma}: {exc!r}")
                out = None
            if out is not None:
                avail[t][e.band].append(e)
    # A generator raising is a bug, not a "this entry does not qualify".  Swallow
    # it so one bad entry cannot kill an hour-long job, but never silently: an
    # exception rate above a few per mille means the type is broken, not thin.
    if errors:
        print(f"[avail] GENERATOR EXCEPTIONS: {dict(errors)}", flush=True)
        for t, msg in first_error.items():
            print(f"    {t}: {msg}", flush=True)
        worst = max(errors.values()) / max(len(entries), 1)
        if worst > 0.001:
            raise RuntimeError(f"generator exception rate {worst:.3%} -- fix it")
    return avail


def allocate(want_total, per_band_quota, capacity):
    """Split `want_total` over bands by quota, capped by capacity, redistributing.

    Section 5 of QA_DATASET_DESIGN.md: thin cells (a "B0 x usage example" cell is
    nearly empty) must be shrunk and the shortfall borrowed, not silently
    produced as a skewed set.
    """
    bands = list(per_band_quota)
    want = {b: int(round(want_total * q / 100.0)) for b, q in per_band_quota.items()}
    take = {b: min(want[b], capacity.get(b, 0)) for b in bands}
    short = want_total - sum(take.values())
    while short > 0:
        head = [b for b in bands if capacity.get(b, 0) > take[b]]
        if not head:
            break
        per = max(1, short // len(head))
        for b in head:
            add = min(per, capacity[b] - take[b], short)
            take[b] += add
            short -= add
            if short <= 0:
                break
    return take


def generate(ctx, out_dir, seed=20260821, types=None, verbose=True, scale=1.0):
    """Generate the dataset.  `scale` shrinks every split's target proportionally.

    `scale` exists so the pipeline can be exercised end to end without building a
    full corpus: `--types` narrows the type MIX but not the size, because each
    split's budget is divided among whatever types were asked for.  At the
    default 1.0 the arithmetic below is exactly what it always was.
    """
    types = types or TYPES
    # A follower must run after its leader or the replay is not there yet, and
    # C17 would fail exactly as silently as before.  Ordering it here rather than
    # relying on TYPES happening to be sorted the right way.
    types = sorted(types, key=lambda t: (t in NEGATIVE_GROUP, types.index(t)))
    _NEG_REPLAY.clear()
    rng = random.Random(seed)
    if verbose:
        print(f"[avail] running every generator over {len(ctx.pool):,} pool entries "
              f"x {len(types)} types ...", flush=True)
    avail = availability(ctx, rng, types)
    if verbose:
        for t in types:
            n = sum(len(v) for v in avail[t].values())
            print(f"    {t:5s} eligible {n:7,d}", flush=True)

    quota = dict(zip(seeds.BAND_NAMES, seeds.BAND_QUOTA))
    items = {s: [] for s in TARGET}
    counter = collections.Counter()
    report = {"availability": {}, "realised": {}, "negatives": {}}

    for t in types:
        by_band = avail[t]
        report["availability"][t] = {b: len(v) for b, v in sorted(by_band.items())}
        tier_c = t in spec.TIER_C
        for split, total in TARGET.items():
            if tier_c and split != "test":
                continue
            per_type = total * scale / len(types)
            # entries available for this type AND on this side of the split
            cap_entries = {b: [e for e in by_band.get(b, []) if e.split == split]
                           for b in seeds.BAND_NAMES}
            cap = {b: len(v) for b, v in cap_entries.items()}
            take = allocate(int(round(per_type)), quota, cap)
            n_neg = int(round(sum(take.values()) * NEGATIVE_RATE))
            for b, k in take.items():
                pool_b = list(cap_entries[b])
                _rng(seed, t, split, b, "order").shuffle(pool_b)
                for e in pool_b[:k]:
                    idx = counter[t]; counter[t] += 1
                    try:
                        out = gen.GENERATORS[t](ctx, e, _rng(seed, t, e.lemma,
                                                             "gen"))
                    except Exception as exc:            # noqa: BLE001
                        # availability() already ran every generator over every
                        # entry, so a raise here means a code path only the real
                        # rng reaches.  Record it and carry on rather than losing
                        # an hour of work; the count is in the report.
                        report.setdefault("gen_errors", collections.Counter())[
                            f"{t}:{type(exc).__name__}"] += 1
                        continue
                    if out is None:
                        continue
                    slots, gold = out
                    it = make_item(ctx, t, e, slots, gold, split=split,
                                   rng=_rng(seed, t, e.lemma, "tmpl"),
                                   allow_tier_a=(split == "test"), idx=idx)
                    if it and _tier_c_safe(it):
                        items[split].append(it)
            # negatives, drawn from the same band distribution as the positives
            made_neg = emit_negatives(ctx, t, split, take, n_neg, rng, counter,
                                      items, seed)
            report["negatives"].setdefault(t, {})[split] = made_neg

    for split, rows in items.items():
        rows.sort(key=lambda r: (r["type"], r["id"]))
        d = report["realised"].setdefault(split, {})
        for r in rows:
            e = d.setdefault(r["type"], {"n": 0, "negative": 0, "tier_A": 0,
                                         "bands": collections.Counter()})
            e["n"] += 1
            e["negative"] += bool(r["negative"])
            e["tier_A"] += (r["tier"] == "A")
            e["bands"][r["band"]] += 1
    report["totals"] = {s: len(v) for s, v in items.items()}
    report["negative_flavours"] = dict(collections.Counter(
        r["negative_flavour"] for v in items.values() for r in v if r["negative"]))
    report["templates_used"] = len({r["template_id"] for v in items.values()
                                    for r in v})
    if "gen_errors" in report:
        report["gen_errors"] = dict(report["gen_errors"])

    os.makedirs(out_dir, exist_ok=True)
    for split, rows in items.items():
        with open(os.path.join(out_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=dict)
    return items, report


# Which flavour a type's ORDINARY-LEMMA negatives take (0.2).  `mismatch` means
# the lemma exists and the question's category does not apply to it; `absent`
# means the lemma exists and simply lacks the relation.
MISMATCH_TYPES = {"T1", "T2", "T3", "T5", "T6", "T7", "T9", "T10", "T20", "T21"}


def _ordinary_pool(ctx, t, split):
    """Seeds for a type's ordinary-lemma negatives, or [] if it has none."""
    if t == "T8":
        # An MWE entry carries no partOfSpeech at all -- a truthful negative that
        # teaches a real boundary of the resource.
        return [e for e in _mwe_seeds(ctx) if e.split == split]
    if t in ("T20", "T21"):
        return [e for e in ctx.pool.entries if e.split == split]
    if t == "T4":
        return []                       # see negative_item(): no such negative exists
    pred = NEGATIVE_SEED.get(t)
    if pred is None:
        return []
    return [e for e in ctx.pool.entries if e.split == split and pred(ctx, e)]


# Types whose negatives must be THE SAME LEMMAS (check C17).  T14 answers "how
# many senses" for exactly the senses T12 lists, so a lemma that is a negative in
# one and a positive in the other teaches the training data to say *ni podatka v
# bazi* and *ima 3 pomene* about the same word.  Drawing each type's negatives
# from its own random stream made agreement a coincidence -- it held in the first
# generation and broke in the second (`imitacijski`).  The leader's choices are
# recorded and replayed, which is exact rather than probable: the two share
# `_neg_absent("n_def")`, so the pools are identical, and `negative_item` has no
# T12- or T14-specific branch, so a choice that works for one works for the other.
NEGATIVE_GROUP = {"T14": "T12"}
_NEG_REPLAY = {}


def emit_negatives(ctx, t, split, take, n_neg, rng, counter, items, seed):
    """~10 % of a type's items, 30 % `unlisted` and 70 % ordinary lemmas.

    Negatives are drawn from the same band distribution as the positives, so the
    band never predicts negativity, and at most one per (lemma, type).
    """
    leader = NEGATIVE_GROUP.get(t)
    if leader is not None and (leader, split) in _NEG_REPLAY:
        return _replay_negatives(ctx, t, split, rng, counter, items,
                                 _NEG_REPLAY[(leader, split)], seed)
    bands = [b for b, k in take.items() for _ in range(k)]
    if not bands or not n_neg:
        return {}
    rng.shuffle(bands)
    ordinary = _ordinary_pool(ctx, t, split)
    real = [e for e in ctx.pool.entries if e.split == split]
    if not ordinary and not real:
        return {}

    made = collections.Counter()
    used = set()
    chosen = []
    tries = 0
    rejected_union = 0
    while sum(made.values()) < n_neg and tries < n_neg * 40:
        tries += 1
        want_fake = (rng.random() < UNLISTED_SHARE) or not ordinary
        flavour = ("unlisted" if want_fake
                   else ("mismatch" if t in MISMATCH_TYPES else "absent"))
        pool_b = real if want_fake else ordinary
        band = bands[sum(made.values()) % len(bands)]
        cands = [e for e in pool_b if e.band == band and e.lemma not in used] \
            or [e for e in pool_b if e.lemma not in used]
        if not cands:
            break
        e = cands[rng.randrange(len(cands))]
        out = negative_item(ctx, t, e, _rng(seed, t, split, e.lemma, "neg"),
                            flavour)
        if out is None:
            continue
        slots, flavour = out
        # The union, not the seed anchor, decides whether this is really silent.
        if flavour != "unlisted" and t in UNION_TESTED \
                and not _union_is_silent(ctx, t, slots.get("L") or e.lemma, e.a):
            rejected_union += 1
            used.add(e.lemma)
            continue
        idx = counter[t]; counter[t] += 1
        it = make_item(ctx, t, e, slots, None, split=split,
                       rng=_rng(seed, t, split, slots.get("L") or e.lemma,
                                "tmpl"),
                       negative=True, flavour=flavour,
                       allow_tier_a=(split == "test"), idx=idx)
        if it and _tier_c_safe(it):
            items[split].append(it)
            used.add(e.lemma)
            made[flavour] += 1
            chosen.append((e, flavour, slots.get("L")))
    if t in NEGATIVE_GROUP.values():
        _NEG_REPLAY[(t, split)] = chosen
    if rejected_union:
        made["_rejected_union"] = rejected_union
    return dict(made)


def _replay_negatives(ctx, t, split, rng, counter, items, chosen, seed):
    """Emit exactly the negatives another type already chose (C17).

    The `unlisted` surface is replayed too, not re-drawn: the draw is random, and
    a *different* unlisted word for the same seed would still satisfy C17 but
    would make the two types' negative questions gratuitously unalike.
    """
    made = collections.Counter()
    for e, flavour, lemma in chosen:
        # "absent" so the unlisted word is not re-drawn; the recorded surface is
        # written back over `L` immediately below.
        out = negative_item(ctx, t, e, _rng(seed, t, split, e.lemma, "neg"),
                            "absent")
        if out is None:
            continue
        slots, _ = out
        slots["L"] = lemma
        idx = counter[t]; counter[t] += 1
        it = make_item(ctx, t, e, slots, None, split=split,
                       rng=_rng(seed, t, split, lemma, "tmpl"),
                       negative=True, flavour=flavour,
                       allow_tier_a=(split == "test"), idx=idx)
        if it and _tier_c_safe(it):
            items[split].append(it)
            made[flavour] += 1
    return dict(made)


_MWE_CACHE = {}


def _mwe_seeds(ctx, limit=6000):
    """MWE anchors -- T8's ordinary-lemma negative: they carry no POS at all.

    Given a real band and a real split, both for reasons that would otherwise be
    silent defects: a hard-coded band would put every T8 negative in B0, and a
    hard-coded split would let one MWE lemma appear on both sides and break the
    lemma-disjointness C11 asserts.
    """
    if "mwe" in _MWE_CACHE:
        return _MWE_CACHE["mwe"]
    import numpy as np
    st = ctx.store
    a = np.flatnonzero(st.kind[:st.n_real] == 0)
    a = a[st.mwe[a]][:limit * 4]
    out = []
    for i in a:
        i = int(i)
        lemma = st.lemma(i)
        if not lemma or st.pos(i) is not None:
            continue
        proxy = len(st.collocations(i)) + seeds._mwe_memberships(st, i)
        out.append(seeds.Entry(a=i, lu=st.lu_id(i), lemma=lemma, pos=None,
                               feat={}, proxy=proxy, band=seeds.band_of(proxy),
                               split=None, n_def=0, n_syn=0, n_ant=0, n_ex=0,
                               n_col=0))
        if len(out) >= limit:
            break
    seeds.assign_split(out)
    _MWE_CACHE["mwe"] = out
    return out


# --------------------------------------------------------------------------
def run(store=None, out=None, seed=20260821, types="", scale=1.0):
    """Generate the dataset into `out`.  Returns (items by split, the report).

    The body `main()` used to hold, so the pipeline can call this stage as a
    function while the command line keeps behaving exactly as it did.
    """
    args = argparse.Namespace(store=store, out=out, seed=seed, types=types,
                              scale=scale)

    store = open_store(args.store)
    print(f"[store] {store.path}  convention={store.convention}  "
          f"{store.n:,} nodes", flush=True)
    # The generator reads collocation nodes expecting a curated phrase, and noun
    # anchors expecting their gender, so it warns when the store was built
    # without either.  It used to compare against the literal "v7", which stopped
    # matching when stores began recording their text convention instead of a
    # version -- and so warned on every run, about the correct store.
    missing = [c for c in ("collocation-phrases", "entry-gender")
               if c not in (store.convention or "")]
    if missing:
        print(f"    !! this store's text convention is {store.convention!r}; "
              f"the generator expects {missing} -- T15/T18 and noun gender will "
              f"be wrong", flush=True)

    def prog(i, n, kept):
        print(f"[pool] {i:,}/{n:,} anchors, {kept:,} kept", flush=True)

    pool = seeds.build_pool(store, progress=prog)
    print(f"[pool] {len(pool):,} seed lemmas\n{seeds.band_report(pool)}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    seeds.save_pool(pool, os.path.join(args.out, "pool.jsonl"))

    ctx = gen.Ctx(store, pool)
    print("[aux] biti: " + ", ".join(f"{k} {len(v)} cells"
                                     for k, v in ctx.aux.items()), flush=True)
    for k, v in ctx.aux.items():
        if len(v) != 9:
            raise StageError(f"the {k} auxiliary table has {len(v)} of 9 cells -- "
                             f"T5/T6 would emit gaps in every composed tense")

    types = [t.strip() for t in args.types.split(",") if t.strip()] or None
    items, report = generate(ctx, args.out, seed=args.seed, types=types,
                             scale=args.scale)
    for s, rows in items.items():
        print(f"[out] {s}: {len(rows):,} items", flush=True)
    return items, report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--types", default="", help="comma-separated subset")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="shrink every split's target by this factor, e.g. 0.02 "
                         "for a fast end-to-end test (default: 1.0, the real "
                         "corpus). --types narrows the type MIX, not the size.")
    args = ap.parse_args()

    try:
        run(store=args.store, out=args.out, seed=args.seed, types=args.types,
            scale=args.scale)
    except StageError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
