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
    gold_items        the answer as [oznaka, vrednost] pairs -- also the item's
                      key set (0.9), so a per-label breakdown is free
    slots             the frame's slot values
    negative          bool, and negative_flavour when true
    grading           the ITEM-level half of the contract: all_items / n_all /
                      quantity_band / n_asked for the membership types, and
                      nothing at all for the rest.  `mode` is per-TYPE and is
                      read from `qa/spec.py`, never from the row

Quotas follow D16 (~10 k train incl. <=1 k dev, ~2 k test) and D10's band shares,
each capped by what the graph can actually supply and the shortfall redistributed
across bands with headroom -- Section 5 of QA_DATASET_DESIGN.md calls for exactly
this rather than silently producing a skewed set.
"""
import os
import time
import json
import random
import hashlib
import argparse
import collections

from qa import sl, seeds, gen, spec, pairs, templates
from qa.store import open_store
from lib.errors import StageError

#: Every type that is BUILT.  `spec.SPEC` describes more than this -- a parked
#: type keeps its row there, and its generator and frames, so that the reason it
#: is out stays next to the code rather than in a commit message.  See
#: `spec.PARKED`.
TYPES = [t for t in spec.SPEC if t not in spec.PARKED]

#: Per-split budgets, in two different currencies on purpose.
#:
#: `train` and `dev` are TOTALS, divided among whatever types are enabled -- the
#: corpus is sized as a whole and the per-type count follows from it.
#:
#: `test` is PER TYPE, which is a different thing and not a rounding of the same
#: thing.  A test split divided from a total gives each type whatever the
#: division leaves it, so a per-type score is computed over a denominator that
#: moves whenever a type is added; fixing it at 200 makes every type's score
#: read against the same denominator, and makes a difference between two types
#: comparable without a footnote.  The cost is that the test total now depends on
#: how many types are enabled (34 x 200 = 6,800 here), which is the right way
#: round: the reporting unit is the type.
TARGET = {"train": 40000, "dev": 2000}
TEST_PER_TYPE = 200
SPLITS = ("train", "dev", "test")
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
    # ── Group H ────────────────────────────────────────────────────────────
    # T22's negative is a number or case the entry does not fill, so its pool is
    # every entry that could have one: a verb (no nominal paradigm at all) or a
    # nominal with a gap.  Which of the two an entry turns out to be is decided
    # in `negative_item`, against the UNION -- this predicate only has to be
    # cheap, because `_ordinary_pool` walks all 72,334 entries with it.
    "T22": lambda ctx, e: e.pos == "glagol" or e.pos in gen.NOMINAL_POS,
    # Cheap prefilter only: which RELATION is absent is decided in
    # `negative_item`, over the union.  Almost every entry lacks one of the four,
    # so this narrows nothing much and is not meant to.
    "T24": lambda ctx, e: not (e.n_syn and e.n_ex and e.n_def),
    "T28": _neg_absent("n_syn"),
    "T29": _neg_absent("n_def"),
    "T31": _neg_absent("n_mwe"),
    "T35": _neg_absent("n_col"),
    "T36": _neg_absent("n_trans"),
    # T4, T8, T20, T21 are handled specially -- see negative_item().
    # T25 and T27 have no ordinary-lemma negative for T23's reason: a pair that
    # is not in the relation is one of T25's two ANSWERS, and two words whose
    # counts are both zero is not an item T27 can build.  Their negatives are
    # `unlisted` only.  T30's is in MISMATCH_TYPES; T32/T33/T34 are phrase types
    # whose negative is a word asked a phrase's question, likewise.
}


# Types whose negative asserts "this ENTRY does not record X".  T20/T21 are
# excluded because their negative is pairing-scoped, not entry-scoped -- the gold
# says the form does not belong to the named lemma (T21) or does not occur in the
# given sentence (T20), and a co-extracted anchor that analyses the form on its
# own terms does not contradict either.  T4's negatives are 100 % `unlisted`, so
# there is nothing here to test.
UNION_TESTED = set(TYPES) - {
    "T4", "T20", "T21",
    # Group H adds seven more, all for T20/T21's reason rather than a new one:
    # their negative is scoped to something NARROWER than the entry, so running
    # the whole generator over a co-extracted anchor answers a question the item
    # never asked.  Each does its own union check, over the right scope:
    #
    #   T22  a selection of cells, not the paradigm  (t22_absent_selection)
    #   T24  one relation, not every countable one   (relation_present)
    #   T27  one relation over a PAIR                (relation_present)
    #   T25  the pair, and `ne` is an answer anyway  (unlisted negatives only)
    #   T26  the pair                                (unlisted negatives only)
    #   T29  the borrowed sentence, not the entry
    #   T35  the phrase, not the anchor              (collocation_owners)
    "T22", "T24", "T25", "T26", "T27", "T29", "T35",
}


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
    elif type_key == "T23":
        # Every T23 frame names a relation, so a negative needs one too -- and
        # without this branch `render_question` raises KeyError on the missing
        # slots and the negative is dropped, silently, all of them.
        #
        # NEVER the held-out relation.  A negative's answer is the sentinel and
        # teaches nothing about the relation it names, so asking about
        # `protipomenka` here would spend the Tier C budget for nothing.
        #
        # The item is the distinction the type exists to draw: `ne` means the
        # word is in the base and the relation is not, the sentinel means the
        # word is not in the base at all.
        rel = spec.T23_RELATIONS[rng.randrange(len(spec.T23_RELATIONS))]
        slots.update(sl.relation_slots(rel))
    elif type_key == "T22":
        sel = gen.t22_absent_selection(ctx, lemma, e.a, rng)
        if sel is None:
            return None                 # this entry fills every axis
        slots["IZBOR"] = sl.selection_phrase(*sel)
        flavour = "mismatch" if e.pos == "glagol" else flavour
    elif type_key in ("T24", "T27"):
        # The relation has to be one the entry genuinely LACKS, over the whole
        # union.  Drawing it uniformly would put `oblika` in the question of an
        # entry that has forms, and ship a sentinel contradicted by the ball --
        # these two types count several relations, so "this entry is a negative"
        # is never true of the entry, only of a (entry, relation) pair.
        #
        # Never a held-out relation, for T23's reason: a negative's answer is the
        # sentinel and teaches nothing about the relation it names, so asking
        # about `protipomenka` or `prevod` would spend the Tier C budget for
        # nothing.
        others = [e] if type_key == "T24" else None
        if type_key == "T27":
            partner = gen._pair_partner(ctx, e, rng)
            if partner is None:
                return None
            others = [e, partner]
            slots["L2"] = partner.lemma
            # The direction is asked whatever the answer is, and a negative's
            # frames are drawn from the same pool: without it every {PRIM} frame
            # would fail to render and the negative would vanish silently.
            slots["PRIM"] = "več" if rng.random() < 0.5 else "manj"
        empty = [r for r in spec.COUNTABLE
                 if not any(gen.relation_present(ctx, o.lemma, o.a, r)
                            for o in others)]
        if not empty:
            return None
        slots.update(sl.relation_slots(empty[rng.randrange(len(empty))]))
    elif type_key in ("T25", "T26"):
        other = gen._pair_partner(ctx, e, rng)
        if other is None:
            return None
        slots["L2"] = other.lemma
        if type_key == "T25":
            rel = spec.PAIR_RELATIONS[rng.randrange(len(spec.PAIR_RELATIONS))]
            slots.update(sl.relation_slots(rel))
    elif type_key == "T28":
        # The frame names a sense, and a negative has none to name -- so it
        # names the word's own lemma as the sense, which is what a user asking
        # about a word with no senses would write.  Without a POMEN slot
        # `render_question` raises KeyError and every T28 negative is dropped,
        # silently, exactly as T23's comment records for its own branch.
        slots["POMEN"] = lemma
        slots["ORD"] = 1
    elif type_key == "T29":
        # A sentence or phrase that belongs to no sense of this word.  Borrowed
        # from another entry, and required NOT to occur on this one, or the
        # item is a false negative.
        other = ctx.pool.entries[rng.randrange(len(ctx.pool.entries))]
        cand = [t for _v, t in ctx.store.examples(other.a)
                if t and len(t.split()) <= gen.MAX_EXAMPLE_WORDS]
        mine = {t for _v, t in ctx.store.examples(e.a)}
        cand = [t for t in cand if t not in mine]
        if not cand:
            return None
        slots["Z"] = cand[rng.randrange(len(cand))]
    elif type_key == "T31":
        slots.update(band="none", n_all=0)
    elif type_key == "T34":
        slots["L"] = f"{lemma} {gen.GAP_MARK}"
        slots["ZVEZA"] = slots["L"]
        flavour = "absent"
    elif type_key == "T35":
        # The phrase must be nobody's collocation, checked along the same route
        # the pipeline resolves it by.  Without this the sentinel is simply
        # false whenever the MWE happens to coincide with a collocation phrase,
        # which H.5 measures at 29.6 %.
        if gen.collocation_owners(ctx, lemma):
            return None
        slots["ZVEZA"] = lemma
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
    # Which axes this item fixes is read off the slots the generator produced,
    # so the question can never name an axis the answer does not vary over, nor
    # leave one it does vary over unnamed (0.1 clause 3).
    axes = frozenset(a for a in ("gender", "definiteness") if slots.get(a))
    if negative:
        pool_frames = templates.neutral_frames(key, axes)
    else:
        pool_frames = templates.frames_for_pos(key, pos, axes)
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
        # One shape for all of them (0.1).  `qa/pairs.py` owns which label each
        # cell carries, so the answer and `gold_items` cannot disagree and no
        # type keeps a private output template here.
        answer, gold = pairs.answer_of(type_key, slots, items)
    # ITEM-level facts only.  `mode` is a per-TYPE constant and is read from
    # `qa/spec.py` at grading time (`grade.contract`) -- writing it here would
    # put 12,490 stale copies on disk, which is how T19 kept being graded
    # `sequence` after the spec said `membership`.
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


# --------------------------------------------------------------------------
def seed_entries(ctx, type_key):
    """The population a type is seeded from.

    Every type but the phrase family draws from the word pool.  A phrase type
    drawing from it would not FAIL -- its generator would simply return None for
    all 72,334 word entries and the type would be reported unavailable, which is
    the silent kind of wrong.  `spec.PHRASE_SEEDED` is what makes the loop ask
    the right population, and it is derived from the SPEC table so a new phrase
    type declares its pool where it declares everything else.
    """
    if type_key in spec.PHRASE_SEEDED:
        return ctx.phrase_pool
    return ctx.pool.entries


#: How many times a band's own quota an early stop must have collected.  1.0
#: would already give an identical allocation -- `allocate` takes
#: min(want_b, capacity_b) and only redistributes when a band is SHORT -- so
#: this is headroom against that reasoning being subtly wrong, not a tuning knob.
AVAIL_MARGIN = 2


def availability(ctx, rng, types, sample_cap=None, need=None, seed=20260821,
                 verbose=False):
    """{type: {band: [entries]}} -- who can actually answer each type.

    Computed by RUNNING the generator, not by predicting it: a type is available
    for an entry iff the generator returns a gold answer for it.  Slower than a
    heuristic and the only version that cannot be wrong.

    Type-outer rather than entry-outer, because the population is now per type
    (`seed_entries`).

    **`need` stops a type's scan once no further entry could be used.**  A split
    takes `cap x quota_b / 100` entries from band b and nothing more, so proving
    the 60,000th eligible entry of a 1,250-item type eligible buys exactly
    nothing -- and it is not free: measured over the 34 types, the full pass is
    106 minutes, four fifths of it spent on six types that are available for
    nearly every entry they are offered.  With `need` it is about 15.

    Two properties make the stop safe rather than merely fast.  The scan order is
    SHUFFLED, so what it collects is a random sample of the population and not
    its lowest node ids -- which are the oldest entries, and, at the head of the
    word pool, all proper nouns.  And the stop fires only when every (split,
    band) cell holds `AVAIL_MARGIN x` what allocation can spend there, so a thin
    band still drags the scan over the whole pool, exactly as before.
    """
    avail = {t: collections.defaultdict(list) for t in types}
    errors = collections.Counter()
    first_error = {}
    seen = collections.Counter()
    stopped = {}
    quota = dict(zip(seeds.BAND_NAMES, seeds.BAND_QUOTA))
    for t in types:
        entries = seed_entries(ctx, t)
        want = _avail_need(t, need, quota)
        if want:
            entries = list(entries)
            _rng(seed, t, "avail").shuffle(entries)
        have = collections.Counter()
        t0 = time.time()
        for i, e in enumerate(entries):
            if sample_cap and i >= sample_cap:
                break
            if want and _avail_enough(have, want):
                stopped[t] = i
                break
            seen[t] += 1
            try:
                # The SAME rng generation will use for this entry.  It was
                # `random.Random(0)`, so "eligible" meant eligible under a draw
                # generation never makes: T35 shuffles its phrases and tries 12,
                # and an entry that passed here failed there -- 827 of its 1,290
                # train items were made in job 141387.
                out = gen.GENERATORS[t](ctx, e, _rng(seed, t, e.lemma, "gen"))
            except Exception as exc:                # noqa: BLE001
                errors[t] += 1
                first_error.setdefault(t, f"{e.lemma}: {exc!r}")
                out = None
            if out is not None:
                avail[t][e.band].append(e)
                have[(e.split, e.band)] += 1
        # Per type as it finishes, not as a block at the end: this pass is the
        # longest thing in the build, and a stage that prints nothing for an
        # hour cannot be told apart from a stage that has hung.
        if verbose:
            n = sum(len(v) for v in avail[t].values())
            print(f"    {t:5s} eligible {n:7,d}  of {seen[t]:7,d} scanned  "
                  f"{time.time()-t0:6.1f}s"
                  f"{'  (stopped early)' if t in stopped else ''}", flush=True)
    # A generator raising is a bug, not a "this entry does not qualify".  Swallow
    # it so one bad entry cannot kill an hour-long job, but never silently: an
    # exception rate above a few per mille means the type is broken, not thin.
    if errors:
        print(f"[avail] GENERATOR EXCEPTIONS: {dict(errors)}", flush=True)
        for t, msg in first_error.items():
            print(f"    {t}: {msg}", flush=True)
        # Per type, against the population THAT type was run over -- a phrase
        # type's rate must not be divided by the word pool's size.
        worst = max(errors[t] / max(seen[t], 1) for t in errors)
        if worst > 0.001:
            raise RuntimeError(f"generator exception rate {worst:.3%} -- fix it")
    if stopped:
        print("[avail] scan stopped early (every band already holds more than "
              "allocation can spend): "
              + ", ".join(f"{t} after {n:,}" for t, n in sorted(stopped.items())),
              flush=True)
    return avail


def _avail_need(type_key, need, quota):
    """{(split, band): entries worth collecting} for one type, or None.

    None means "scan everything", which is what a caller with no caps to respect
    gets -- `availability` is also called on its own by the checks.
    """
    if not need:
        return None
    want = {}
    for split, cap in need.items():
        if type_key in spec.TIER_C and split != "test":
            continue
        for b, q in quota.items():
            n = int(round(cap * q / 100.0)) * AVAIL_MARGIN
            if n:
                want[(split, b)] = n
    return want or None


def _avail_enough(have, want):
    return all(have[k] >= n for k, n in want.items())


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


def _emit_positive(ctx, t, e, split, seed, counter, items, report):
    """Make one positive from entry `e`; True if an item was appended.

    False is not an error.  `make_item` can still refuse (no frame renders for
    these slots) and `_tier_c_safe` can drop a train item, so the caller walks on
    to the next entry -- that walk is what keeps a type at its cap.
    """
    idx = counter[t]; counter[t] += 1
    try:
        out = gen.GENERATORS[t](ctx, e, _rng(seed, t, e.lemma, "gen"))
    except Exception as exc:                        # noqa: BLE001
        # availability() ran this generator on this entry with this same rng, so
        # a raise here means the generator is not a pure function of (store,
        # entry, rng).  Recorded and carried past rather than losing an hour of
        # work; the count is in the report.
        report.setdefault("gen_errors", collections.Counter())[
            f"{t}:{type(exc).__name__}"] += 1
        return False
    if out is None:
        return False
    slots, gold = out
    it = make_item(ctx, t, e, slots, gold, split=split,
                   rng=_rng(seed, t, e.lemma, "tmpl"),
                   allow_tier_a=(split == "test"), idx=idx)
    if it and _tier_c_safe(it):
        items[split].append(it)
        return True
    return False


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

    # The per-type CAP, per split.  A total is divided only among the types that
    # actually take that split, so the two Tier C types -- which are test-only --
    # do not silently shrink every training type's share by holding a slice of
    # the budget they never spend.
    #
    # Computed BEFORE the availability pass, because it is what tells that pass
    # when to stop: a type that can spend 1,250 entries has no use for the
    # 60,000th proof that it could have used one more.
    trainable = [t for t in types if t not in spec.TIER_C] or list(types)
    per_type_cap = {s: TARGET[s] * scale / len(trainable) for s in TARGET}
    per_type_cap["test"] = TEST_PER_TYPE * scale

    if verbose:
        n_calls = sum(len(seed_entries(ctx, t)) for t in types)
        print(f"[avail] running each of {len(types)} generator(s) over its own "
              f"seed pool -- at most {n_calls:,} calls ...", flush=True)
    avail = availability(ctx, rng, types, need=per_type_cap, seed=seed,
                         verbose=verbose)

    quota = dict(zip(seeds.BAND_NAMES, seeds.BAND_QUOTA))
    items = {s: [] for s in SPLITS}
    counter = collections.Counter()
    report = {"availability": {}, "realised": {}, "negatives": {}}
    report["per_type_cap"] = {s: int(round(v)) for s, v in per_type_cap.items()}

    for t in types:
        by_band = avail[t]
        report["availability"][t] = {b: len(v) for b, v in sorted(by_band.items())}
        tier_c = t in spec.TIER_C
        for split in SPLITS:
            if tier_c and split != "test":
                continue
            # A CAP, not a quota to be filled: a type that cannot supply it
            # produces fewer items and no other type produces more.  The corpus
            # total is therefore an outcome rather than a target, which is what
            # keeps a per-type score comparable across types.
            per_type = int(round(per_type_cap[split]))
            # entries available for this type AND on this side of the split
            cap_entries = {b: [e for e in by_band.get(b, []) if e.split == split]
                           for b in seeds.BAND_NAMES}
            cap = {b: len(v) for b, v in cap_entries.items()}
            # Negatives count INSIDE the cap.  They were 10 % on top of it, so
            # "200 per type" shipped 220 and the 40,000 train target shipped
            # 42,013 (job 141387).  10 % of what the type can SUPPLY, not of the
            # cap, or a thin type (T11, ~460 items) would be a quarter negatives.
            supply = sum(allocate(per_type, quota, cap).values())
            n_neg = int(round(min(per_type, supply) * NEGATIVE_RATE))
            take = allocate(per_type - n_neg, quota, cap)
            # Walk each band's shuffled pool until k items are MADE, rather than
            # over its first k entries: an entry can still yield no item (no
            # frame renders, `_tier_c_safe` drops it), and `pool_b[:k]` left
            # those holes unfilled -- T1 shipped 179 of 200 test items.
            made_pos, spare = 0, []
            for b, k in take.items():
                pool_b = list(cap_entries[b])
                _rng(seed, t, split, b, "order").shuffle(pool_b)
                got = used = 0
                for e in pool_b:
                    if got >= k:
                        break
                    used += 1
                    got += _emit_positive(ctx, t, e, split, seed, counter,
                                          items, report)
                made_pos += got
                spare += pool_b[used:]
            # negatives, drawn from the same band distribution as the positives
            made_neg = emit_negatives(ctx, t, split, take, n_neg, rng, counter,
                                      items, seed)
            report["negatives"].setdefault(t, {})[split] = made_neg
            # Whatever is still missing -- a band that ran dry, negatives that
            # came up short -- is topped up with positives from the entries no
            # band spent, so a type with the capacity lands on its cap exactly.
            # From any band: the quota shapes the bulk, and a handful of top-ups
            # cannot move it.
            short = per_type - made_pos - sum(
                v for k, v in made_neg.items() if not k.startswith("_"))
            _rng(seed, t, split, "topup").shuffle(spare)
            for e in spare:
                if short <= 0:
                    break
                short -= _emit_positive(ctx, t, e, split, seed, counter, items,
                                        report)
            if short > 0:
                report.setdefault("short", {}).setdefault(t, {})[split] = short

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
MISMATCH_TYPES = {"T1", "T2", "T3", "T5", "T6", "T7", "T9", "T10", "T20", "T21",
                  # T30 asks what a PHRASE is made of; a single-word entry is one
                  # the question does not apply to, and saying so is a boundary of
                  # the resource rather than a gap in it.
                  "T30"}


def _ordinary_pool(ctx, t, split):
    """Seeds for a type's ordinary-lemma negatives, or [] if it has none.

    T23 is deliberately absent: its POSITIVES already answer `ne` when a relation
    is missing, so an "ordinary lemma that lacks the relation" is not a negative
    of this type at all -- it is one of its two answers.  What remains is the
    lemma that is not in the base, which is the `unlisted` flavour.
    """
    if t == "T30":
        # A word, asked a phrase's question.  Drawn from the WORD pool while the
        # positives come from the phrase pool -- the one place the two meet.
        return [e for e in ctx.pool.entries if e.split == split]
    if t == "T8":
        # An MWE entry carries no partOfSpeech at all -- a truthful negative that
        # teaches a real boundary of the resource.
        return [e for e in _mwe_seeds(ctx) if e.split == split]
    if t in ("T20", "T21"):
        return [e for e in ctx.pool.entries if e.split == split]
    if t == "T4":
        return []                       # see negative_item(): no such negative exists
    if t in ("T32", "T33"):
        # A phrase the store holds and this relation does not reach: 98.6 % of
        # MWEs have no defined sense and 6.6 % no example.  Drawn from the
        # PHRASE pool, not the word pool -- a word asked a phrase's question is
        # T30's mismatch, and one boundary per flavour (see emit_negatives).
        field = "n_def" if t == "T32" else "n_ex"
        return [e for e in ctx.phrase_pool
                if e.split == split and not getattr(e, field)]
    if t == "T34":
        # A word that is in no phrase at all, offered as a fragment.  Nothing
        # completes it, and saying so is a true statement about the base rather
        # than an invented gap.
        return [e for e in ctx.pool.entries
                if e.split == split and not e.n_mwe]
    if t == "T35":
        # A phrase nobody records as a collocation.  An MWE seed rather than a
        # word, because the question hands the model a PHRASE and asks whose
        # collocation it is -- handing it a word would test the frame, not the
        # relation.
        return [e for e in _mwe_seeds(ctx) if e.split == split]
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
        # A phrase type's `unlisted` negative would be a single real word framed
        # as a phrase, which is the mismatch negative it already has, wearing a
        # second label.  One boundary per flavour.
        want_fake = ((rng.random() < UNLISTED_SHARE
                      and t not in spec.PHRASE_SEEDED) or not ordinary)
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
    # A type that asked for negatives and produced none is a defect, and it is a
    # QUIET one: the type still ships thousands of positives and the report shows
    # an empty dict next to a type nobody was reading closely.  T23 hit it -- a
    # frame slot its negatives did not fill made `render_question` return None for
    # every attempt -- and nothing else in the build would have said so.
    if n_neg and not sum(v for k, v in made.items() if not k.startswith("_")):
        print(f"[neg] *** {t}/{split}: wanted {n_neg} negatives, made NONE after "
              f"{tries} tries. The type's negatives are being rejected or dropped "
              f"-- a slot its frames need and `negative_item` does not fill is the "
              f"usual cause.", flush=True)
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
    st = ctx.store
    cache, out = {}, []
    for i in st.mwe_anchors()[:limit * 4]:
        i = int(i)
        if st.pos(i) is not None:
            continue
        # `entry_for`, not a hand-built Entry.  This used to construct one
        # inline, which meant a second place that had to know every field: the
        # moment Group H added `mwe`/`n_mwe`/`n_const` to the record, these
        # entries were missing them and `as_dict()` raised on the first one
        # written.  It also hard-coded the word proxy, which is ~0 for every
        # phrase and put the whole T8 negative pool in B0 -- the exact defect
        # this function's docstring warns about.
        e = seeds.entry_for(st, i, require_content=False, proxy_cache=cache)
        if e is None:
            continue
        out.append(e)
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
