#!/usr/bin/env python3
"""The per-type answer generators.

Each `gen_TX(ctx, entry, rng)` returns `(slots, items)` or None when the entry
cannot support the type -- `slots` are the template's slot values, `items` the
gradeable items in canonical order.  The driver (build_dataset.py) turns that
into an item record; nothing here formats a question or decides a quota.

Two rules hold for every generator in this file:

  * **the gold is read off node TEXT**, never off the RDF or off an id, because
    node text is what the model is shown and the whole grading contract rests on
    the answer string being present verbatim in the input;
  * **canonical order is generated even where it is not graded** (C15).  Grading
    tolerance is not a licence for sloppy training data: the model must see one
    ordering for a given set or it is being taught noise on a surface it is
    forced to emit.
"""
import re
import collections

from . import sl
from . import colloc_sampling
from . import seeds
from . import spec
from .spec import GAP
from .store import K_SYN, K_ANT, K_COLLOC, K_EXAMPLE, K_SENSE, K_ANCHOR, TAG

# --------------------------------------------------------------------------
# shared context
# --------------------------------------------------------------------------


class Ctx:
    """Store plus the few global tables the generators share."""

    def __init__(self, store, pool):
        self.store = store
        self.pool = pool
        self.aux = _auxiliary_tables(store)
        self._surface = None
        self._unlisted = None
        self._word_split = None
        self._phrase_pool = None
        self._partners = {}
        self._by_split = None
        #: anchor -> its surfaces, for C26.  One dict for the whole run; see
        #: seeds.constituent_occurs for why it is not optional.
        self.form_cache = {}

    @property
    def surface(self):
        if self._surface is None:
            self._surface = self.store.surface_index()
        return self._surface

    @property
    def phrase_pool(self):
        """The MWE seed pool (H.5), built once and shared by every phrase type.

        Separate from `pool` rather than merged into it because the two are sized
        by different constraints: the word pool is everything that passes the
        content rule (72,334), while the phrase population is 3.76 M and is bound
        by SAMPLING instead -- see seeds.build_phrase_pool.
        """
        if self._phrase_pool is None:
            self._phrase_pool = seeds.build_phrase_pool(self.store)
        return self._phrase_pool

    @property
    def word_split(self):
        """casefolded word lemma -> its split, for H.3's constituent rule.

        Built from the WORD pool, which is the population whose lemmas C11 keeps
        disjoint; a phrase type that names constituents has to answer to it.
        """
        if self._word_split is None:
            self._word_split = {}
            for e in self.pool.entries:
                self._word_split.setdefault(e.lemma.casefold(), e.split)
        return self._word_split

    def partners(self, kind):
        """H.3's same-split partner index for one relation, built once.

        `seeds.same_split_partners` is a full pass over the pool, and T25, T26
        and T27 all want the same answer -- so it is cached here rather than
        recomputed three times inside `availability`, which already runs every
        generator over every seed.
        """
        if kind not in self._partners:
            self._partners[kind] = seeds.same_split_partners(
                self.pool.entries, self.store, kind)
        return self._partners[kind]

    @property
    def by_split(self):
        """{split: [entry]} -- the distractor population for H.3.

        A pair must sit inside one split (C11 is about lemmas, not items), and
        the NEGATIVE half of T25 and the partner of T26/T27 are drawn from here
        rather than from the whole pool for exactly that reason.
        """
        if self._by_split is None:
            self._by_split = collections.defaultdict(list)
            for e in self.pool.entries:
                self._by_split[e.split].append(e)
        return self._by_split

    @property
    def unlisted(self):
        """Real words the lookup cannot reach -- flavour (a)'s source."""
        if self._unlisted is None:
            from qa import unlisted
            self._unlisted = unlisted.load(self.store)
        return self._unlisted


def _auxiliary_tables(store):
    """`biti`'s present and future forms, keyed (person, number).

    M2 (QA_TASKS.md 0.7): the periphrastic tenses are composed from the -l
    participle and this auxiliary, and BOTH halves are graph content -- the
    auxiliary is an ordinary verb entry whose forms carry vform=sedanjik and
    vform=prihodnjik.  Reading it out of the store rather than hard-coding it is
    the point: if it were a constant in this file, the model would be supervised
    to produce a string it cannot read anywhere in its input.
    """
    present, future = {}, {}
    for a in store.core_anchors():
        a = int(a)
        if store.lemma(a) != "biti":
            continue
        for _v, s, f in store.forms(a):
            key = (f.get("person"), f.get("number"))
            if None in key:
                continue
            if f.get("vform") == "sedanjik":
                present.setdefault(key, s)
            elif f.get("vform") == "prihodnjik":
                future.setdefault(key, s)
    # Keyed by the TENSE that uses the table, not by the auxiliary's own vform:
    # the perfect is built with the PRESENT auxiliary (`sem delal`) and the future
    # with the future one (`bom delal`).
    return {"preteklik": present, "prihodnjik": future}


# --------------------------------------------------------------------------
# form views
# --------------------------------------------------------------------------
# Adjectives, and the gendered forms of numerals and pronouns, decline by GENDER
# as well.  They used to be excluded here, on the grounds that a (case, number)
# cell would then hold three surfaces and "a positional 18-cell line would not be
# well defined".  That was a true objection to the POSITIONAL contract, and 0.1
# deleted the positional contract: the distinguishing feature now moves into the
# label, exactly as T6 already does with `... (ženski spol)`.
#
# The exclusion was expensive.  26,612 of 100,801 core anchors are adjectives --
# second only to nouns -- and 1,476,688 form leaves (55.2 % of all of them) were
# unreachable by every type in the document, against 861,014 in the grid that six
# types are built on.  The unreachable half was larger than the reachable one and
# was one part of speech.
NOMINAL_POS = {"samostalnik", "števnik", "zaimek", "pridevnik"}


def paradigm_leaves(store, a):
    """(surface, case, number, gender, definiteness) over the case/number
    paradigm, for any part of speech.

    Replaces the old `set(f) != {"case", "number"}` test, which admitted a form
    only if case and number were the ONLY features it carried -- and so excluded
    every adjective, whose leaves always carry gender and degree as well.

    Gradation stays T11's (0.4): only the positive degree is a paradigm leaf
    here.  The inflected comparative and superlative are too thin to build a
    frame family on -- 106 of 4,000 adjectives carry a case-bearing comparative
    and 95 a superlative.
    """
    for _v, s, f in store.forms(a):
        if not (f.get("case") and f.get("number")):
            continue
        if f.get("vform") or f.get("person"):
            continue
        if f.get("degree") not in (None, "osnovnik"):
            continue
        yield s, f["case"], f["number"], f.get("gender"), f.get("definiteness")


def paradigm_genders(store, a):
    """The genders this entry's paradigm distinguishes, canonically ordered.

    Empty when the paradigm has no gender axis, which is the ordinary noun: a
    noun HAS a gender, but it is a property of the entry and not an axis its
    forms vary over, so it never enters a cell label.
    """
    if store.pos(a) not in NOMINAL_POS:
        return ()
    seen = {g for _s, _c, _n, g, _d in paradigm_leaves(store, a) if g}
    return tuple(g for g in sl.GENDERS if g in seen)


def nominal_grid(store, a, gender=None, definiteness=None, require_healthy=True):
    """{(case, number): {surface, ...}} for ONE gender of the paradigm.

    The key stays a (case, number) pair.  Widening it to carry gender and
    definiteness would have been the obvious move and it is the wrong one: every
    consumer destructures this key, `healthy_grid` looks up two cells of it by
    name, and the T20/T21 ambiguity test means something different from what it
    would then say.  Fixing the gender per item instead leaves all of that
    intact, and it is what 0.1 clause 3 requires anyway -- the question has to
    name the gender it is asking for, or the answer is not determined by it.

    `definiteness` is a cell-level tie-break, not a filter: the form carrying the
    requested definiteness wins, and the unmarked form fills every cell that has
    no such split.  Measured over 6,000 adjectives x 3 genders: 0 of 318,089
    cells hold more than one surface under this rule.

    Empty unless the paradigm is actually inflected -- see `healthy_grid`.  The
    check lives here rather than at the five call sites so that no type can
    forget it: a lemma-filled paradigm is wrong gold for T1/T2/T3 (a table of one
    repeated surface), wrong gold for T21 (an eighteen-way disjunction) and, worst
    of all, *plausible* gold for T20.
    """
    if store.pos(a) not in NOMINAL_POS:
        return {}
    cells = collections.defaultdict(set)
    for s, c, n, g, d in paradigm_leaves(store, a):
        if g != gender:
            continue
        cells[(c, n)].add((s, d))
    if not cells:
        return {}
    grid = {}
    for k, vs in cells.items():
        exact = {s for s, d in vs if d == definiteness}
        grid[k] = exact or {s for s, d in vs if d is None} or {s for s, _d in vs}
    # The lemma form is MERGED INTO THE ANCHOR by the builder, so it has no
    # `oblika:` leaf of its own -- without this the nominative singular of every
    # noun comes out as a gap.  The anchor's own parenthetical names the cell it
    # is rendered in (`iztočnica: gora (samostalnik, ženski spol, imenovalnik,
    # ednina)`), so the cell is read off the node, not assumed.
    #
    # The old guard refused to merge whenever the anchor carried `degree` or
    # `definiteness`.  Every adjective anchor carries both (`iztočnica: Molierjev
    # (pridevnik, imenovalnik, ednina, moški spol, osnovnik, nedoločna oblika)`),
    # so without this rewrite every adjective's nominative singular would have
    # come out as a gap -- in the one cell the citation form occupies.
    surface = store.lemma(a)
    feat = store.anchor_features(a)
    if surface and feat.get("case") and feat.get("number") \
            and not feat.get("vform") and not feat.get("person") \
            and feat.get("degree") in (None, "osnovnik") \
            and feat.get("gender") == gender \
            and feat.get("definiteness") in (None, definiteness):
        grid.setdefault((feat["case"], feat["number"]), set()).add(surface)
    if require_healthy and not healthy_grid(grid):
        return {}
    return grid


MIN_DISTINCT_FORMS = 6


def healthy_grid(grid):
    """Is this paradigm actually inflected, or is it lemma-filled?

    Some entries are lemma-filled rather than inflected.  The store's `odstotek`
    reads

        rodilnik    dvojina  odstotkov      <- right
        rodilnik    množina  odstotek       <- WRONG, should be `odstotkov`
        imenovalnik množina  odstotek       <- WRONG, should be `odstotki`

    which made `odstotkov` look UNAMBIGUOUS -- it occupies exactly one cell --
    only because the cell it should share is filled with the lemma.  T20 then
    labelled a genitive plural in a real sentence as a genitive dual.  D15's
    guarantee that "every label is correct by construction" rests on the
    ambiguity test, and the ambiguity test rests on the paradigm being complete.

    Two conditions, both cheap:

      * the nominative plural differs from the nominative singular -- never right
        for a Slovene noun that inflects at all, and the precise detector for
        this defect;
      * at least six distinct surfaces over the filled cells -- a real noun has
        eight to ten, an uninflected one has a single surface repeated.

    Measured over 49,078 noun entries with >= 12 filled cells: **8.98 % fail the
    first test**, and **3,335 (6.8 %) carry the lemma in all eighteen cells**.
    The population is mostly foreign proper nouns (*Baudelaire*, *Apollinaire*,
    *SMS*) plus ordinary words whose plural column was never filled (*pilot*).
    Note some lemmas have both a healthy and a defective entry -- `Moliere` and
    `Gilmore` appear on both sides -- so this must be decided per entry, never
    per lemma.

    **Unchanged for adjectives, and that is a result rather than an assumption.**
    Both tests were reasoned about nouns, and both keep their meaning when the
    grid is one gender of a larger paradigm: the nominative plural of an
    adjective still differs from its nominative singular within a gender, and
    six distinct surfaces over 18 cells is still the floor that separates an
    inflected entry from a lemma-filled one.  Re-measured over 6,000 adjectives
    x 3 genders: **87.1 % pass** (17,916 attempted), against ~91 % for nouns.
    Had the grid key been widened to carry gender instead, both `grid.get` calls
    below would have missed silently, the lemma-fill test would have stopped
    firing, and this would have degenerated to the six-surface floor -- which any
    54-cell adjective passes without being inflected at all.
    """
    if not grid:
        return False
    sg = grid.get(("imenovalnik", "ednina"))
    pl = grid.get(("imenovalnik", "množina"))
    if sg and pl and sg == pl:
        return False
    return len({s for v in grid.values() for s in v}) >= MIN_DISTINCT_FORMS


def definiteness_axis(store, a, gender, cells=None):
    """The definiteness values marked WITHIN `cells` (all of them if None).

    Empty for the feminine and neuter, and for every noun.  Scoping matters: of
    the 18 cells only the masculine nominative and accusative singular carry the
    split, so a question that names definiteness while asking about the dual is
    naming an axis its own answer does not vary over.  Each type therefore passes
    the cells it is actually asking about -- the whole paradigm for T1, one
    number for T2, one cell for T3.
    """
    seen = {d for _s, c, n, g, d in paradigm_leaves(store, a)
            if g == gender and d and (cells is None or (c, n) in cells)}
    return tuple(d for d in sl.DEFINITENESS if d in seen)


def pick_gender(store, a, rng):
    """The gender one item asks about, or None when the paradigm has no gender
    axis.

    Uniform over the genders the paradigm actually distinguishes, so a three-
    gender adjective contributes three times the item variety of a noun from the
    same seed -- which is the point of reaching them at all.
    """
    genders = paradigm_genders(store, a)
    return genders[rng.randrange(len(genders))] if genders else None


def pick_definiteness(store, a, gender, rng, cells=None):
    ds = definiteness_axis(store, a, gender, cells) if gender else ()
    return ds[rng.randrange(len(ds))] if ds else None


def axis_slots(gender, definiteness=None):
    """Question-metalanguage slots for the axes an item fixes.

    Empty for a noun, which is what keeps every existing frame usable unchanged:
    a frame may name exactly the axes its item provides (`templates.frame_axes`),
    so a noun still draws a noun frame and cannot draw an adjective one.
    """
    if not gender:
        return {}
    out = {"gender": gender, **gender_slots(gender)}
    if definiteness:
        out["definiteness"] = definiteness
        out["DOLOCNOST"] = definiteness
        out["DOLOCNOST_LOC"] = sl.DEFINITENESS_LOC[definiteness]
    return out


def paradigm_readings(store, a):
    """{surface: {(case, number, gender, definiteness), ...}} over the WHOLE
    paradigm, every gender at once.

    T1/T2/T3 answer WITH a surface, so they may fix a gender and let the question
    name it.  T20 and T21 answer WITH a reading, and their ambiguity test has to
    run over the entire paradigm: measured over 4,000 adjectives, **91.4 %** of
    surfaces that carry exactly one reading inside some gender carry more than
    one across the paradigm -- `Shakespearejevi` has seven.  Fixing a gender
    first and then asking "what form is this?" would label a masculine
    accusative plural as a feminine genitive singular, which is precisely the
    defect D15 and T20's unambiguity filter exist to prevent.
    """
    out = collections.defaultdict(set)
    if store.pos(a) not in NOMINAL_POS:
        return out
    for s, c, n, g, d in paradigm_leaves(store, a):
        out[s].add((c, n, g, d))
    return out


def _cell(grid, c, n):
    v = grid.get((c, n))
    if not v:
        return None
    if len(v) > 1:
        return False                 # a doublet: the entry is unusable
    return next(iter(v))


def nominal_line(grid, numbers=sl.NUMBERS):
    """The 6 x len(numbers) cells in canonical order, or None if a cell doubles."""
    out = []
    filled = 0
    for n in numbers:
        for c in sl.CASES:
            v = _cell(grid, c, n)
            if v is False:
                return None, 0
            out.append(v if v else GAP)
            filled += bool(v)
    return out, filled


def verb_grid(store, a):
    """(present, participle) tables keyed (person, number) / (gender, number)."""
    present, part = {}, {}
    for _v, s, f in store.forms(a):
        vf = f.get("vform")
        if vf == "sedanjik" and f.get("person") and f.get("number"):
            present.setdefault((f["person"], f["number"]), set()).add(s)
        elif vf == "deležnik na -l" and f.get("gender") and f.get("number"):
            part.setdefault((f["gender"], f["number"]), set()).add(s)
    return present, part


_NEG = re.compile(r"^(ne|ni)", re.I)


def _pick(surfaces):
    """One surface for a cell.  The non-negated spelling wins.

    3 of 10,242 core verbs have a doubled present cell, always a stored negated
    form -- sem/nisem, bom/nebom, bi/nebi (the last two are non-standard
    single-token spellings).  Without a rule the positional line is not
    deterministic; with this one it is, and it picks the form a dictionary cites.
    """
    if len(surfaces) == 1:
        return next(iter(surfaces))
    plain = sorted(s for s in surfaces if not _NEG.match(s))
    return (plain or sorted(surfaces))[0]


def person_cells(table, gender=None, aux=None):
    """9 cells in canonical order: 1ed 2ed 3ed 1dv 2dv 3dv 1mn 2mn 3mn."""
    out = []
    for n in sl.NUMBERS:
        for p in sl.PERSONS:
            if aux is None:
                v = table.get((p, n))
                out.append(_pick(v) if v else GAP)
            else:
                a = aux.get((p, n))
                pt = table.get((gender, n))
                out.append(f"{a} {_pick(pt)}" if a and pt else GAP)
    return out


# --------------------------------------------------------------------------
# Group A -- sklanjanje
# --------------------------------------------------------------------------
def gen_T1(ctx, e, rng):
    gender = pick_gender(ctx.store, e.a, rng)
    definiteness = pick_definiteness(ctx.store, e.a, gender, rng)
    grid = nominal_grid(ctx.store, e.a, gender, definiteness)
    cells, filled = nominal_line(grid)
    if cells is None or filled < 12:
        return None
    return {"L": e.lemma, **axis_slots(gender, definiteness)}, cells


def gen_T2(ctx, e, rng):
    number = rng.choice(list(sl.NUMBERS))
    gender = pick_gender(ctx.store, e.a, rng)
    # Only the cells of THIS number are in scope, so the dual and the plural
    # never name a definiteness their own six cells do not distinguish.
    definiteness = pick_definiteness(ctx.store, e.a, gender, rng,
                                     {(c, number) for c in sl.CASES})
    grid = nominal_grid(ctx.store, e.a, gender, definiteness)
    cells, filled = nominal_line(grid, numbers=(number,))
    if cells is None or filled < 5:
        return None
    cells = list(cells)
    cells[0] = f"{number}: {cells[0]}"
    return ({"L": e.lemma, "number": number, **number_slots(number),
             **axis_slots(gender, definiteness)}, cells)


def number_slots(n):
    f = sl.NUMBER_FORMS[n]
    return {"STEV_NOM": f["nom"], "STEV_GEN": f["gen"],
            "STEV_LOC": f["loc"], "STEV_ACC": f["acc"]}


def case_slots(c):
    f = sl.CASE_FORMS[c]
    return {"SKLON": f["nom"], "SKLON_LOC": f["loc"], "ORD": sl.CASE_ORD[c]}


def tense_slots(t):
    f = sl.TENSE_FORMS[t]
    return {"CAS_NOM": f["nom"], "CAS_LOC": f["loc"], "CAS_ACC": f["acc"]}


def gender_slots(g):
    return {"SPOL_NOM": sl.GENDER_ADJ[g],
            "SPOL_LOC": sl.GENDER_LOC[g].split()[0]}


def gen_T3(ctx, e, rng):
    gender = pick_gender(ctx.store, e.a, rng)
    # The cell is chosen first and the definiteness is scoped to THAT cell: T3
    # asks about one cell, and 16 of the 18 do not distinguish definiteness at
    # all.  Choosing it first would have put `v določni obliki` on a question
    # about the dual, which asserts an axis the answer does not vary over.
    probe = nominal_grid(ctx.store, e.a, gender)
    cells = [(c, n) for n in sl.NUMBERS for c in sl.CASES
             if _cell(probe, c, n) not in (None, False)]
    if not cells:
        return None
    c, n = cells[rng.randrange(len(cells))]
    definiteness = pick_definiteness(ctx.store, e.a, gender, rng, {(c, n)})
    grid = nominal_grid(ctx.store, e.a, gender, definiteness)
    value = _cell(grid, c, n)
    if value in (None, False):
        return None
    return ({"L": e.lemma, "case": c, "number": n,
             **case_slots(c), **number_slots(n),
             **axis_slots(gender, definiteness)},
            [value])


AMBIGUOUS_T4_SHARE = 0.18


def gen_T4(ctx, e, rng, ambiguous=None):
    """Lemmatisation.  The question names the FORM, so the reverse index is
    exercised on `otherForm` rather than `canonicalForm`.

    ~18 % of items take the **deliberate ambiguous slice**: a form that resolves
    to 2-3 lexical units, whose gold names ALL of them.  That slice is the
    cheapest source of D3's discrimination supervision and a direct probe of R5
    (the natural multi-entity rate), so it is a target share and not an accident.
    When an entry has no ambiguous form the item falls back to an unambiguous
    one rather than being dropped -- the alternative would bias the ambiguous
    slice toward whichever entries happen to have both.
    """
    grid = nominal_grid(ctx.store, e.a)
    surfaces = sorted({s for v in grid.values() for s in v if s.casefold() != e.lemma.casefold()})
    if not surfaces:
        return None
    idx = ctx.surface
    pairs = [(s, idx.get(s.casefold(), ())) for s in surfaces]
    if ambiguous is None:
        ambiguous = rng.random() < AMBIGUOUS_T4_SHARE
    cands = [(s, o) for s, o in pairs if 2 <= len(o) <= 3] if ambiguous else []
    if not cands:
        cands = [(s, o) for s, o in pairs if len(o) == 1]
    if not cands:
        return None
    form, owners = cands[rng.randrange(len(cands))]
    lemmas = sorted({ctx.store.lemma(o) for o in owners}, key=sl.sl_sort_key)
    return {"F": form, "L": e.lemma, "n_owners": len(owners)}, lemmas


def healthy_paradigm(store, a):
    """Every gender of the paradigm passes `healthy_grid`.

    T20 and T21 read the whole paradigm at once, so they cannot inherit the
    health check from a single `nominal_grid` call the way T1-T3 do.  Without
    this they would have lost it entirely for adjectives -- and a lemma-filled
    paradigm is *plausible* gold for T20, which is the worst failure of the
    three the check exists to stop.
    """
    for g in (paradigm_genders(store, a) or (None,)):
        ds = definiteness_axis(store, a, g)
        if not nominal_grid(store, a, g, ds[0] if ds else None):
            return False
    return True


def _reading_items(readings):
    """`tožilnik ednine ali orodnik ednine`, canonically ordered.

    A reading is (case, number, gender, definiteness); the last two are None for
    a noun and the label then renders exactly as it always did.
    """
    order = sorted(readings, key=lambda r: (sl.CASE_IDX[r[0]], sl.NUMBER_IDX[r[1]],
                                            sl.GENDER_IDX.get(r[2], -1),
                                            sl.DEFINITENESS_IDX.get(r[3], -1)))
    return [sl.cell_label(c, n, g, d) for c, n, g, d in order]


def gen_T21(ctx, e, rng, want_ambiguous=True):
    if not healthy_paradigm(ctx.store, e.a):
        return None
    by_surface = paradigm_readings(ctx.store, e.a)
    cands = [(s, r) for s, r in by_surface.items()
             if (len(r) >= 2) == want_ambiguous]
    if not cands:
        return None
    form, readings = cands[rng.randrange(len(cands))]
    return {"F": form, "L": e.lemma}, _reading_items(readings)


# --------------------------------------------------------------------------
# Group B -- spreganje
# --------------------------------------------------------------------------
def _verb_tables(ctx, e):
    if e.pos != "glagol" or e.lemma == "biti":
        return None
    present, part = verb_grid(ctx.store, e.a)
    if len(present) != 9:
        return None
    if any((g, n) not in part for g in sl.GENDERS for n in sl.NUMBERS):
        return None
    return present, part


def _tense_cells(ctx, present, part, tense, gender):
    if tense == "sedanjik":
        return person_cells(present)
    return person_cells(part, gender=gender, aux=ctx.aux[tense])


def gen_T5(ctx, e, rng):
    t = _verb_tables(ctx, e)
    if t is None:
        return None
    present, part = t
    items = []
    for tense in sl.TENSES:
        cells = _tense_cells(ctx, present, part, tense, "moški spol")
        if GAP in cells:
            return None
        cells[0] = f"{tense}: {cells[0]}"
        items.extend(cells)
    return {"L": e.lemma}, items


def gen_T6(ctx, e, rng):
    t = _verb_tables(ctx, e)
    if t is None:
        return None
    present, part = t
    tense = rng.choice(list(sl.TENSES))
    gender = None
    # The gender slot exists only for the composed tenses -- the present has no
    # gender in Slovene, so offering it there would be a false premise.
    if tense != "sedanjik" and rng.random() < 0.20:
        gender = rng.choice(["ženski spol", "srednji spol"])
    cells = _tense_cells(ctx, present, part, tense, gender or "moški spol")
    if GAP in cells:
        return None
    # The gender goes in PARENTHESES, mirroring how 0.5 renders every feature
    # bundle in the node text -- which is what lets `qa/check_labels.py` find the
    # label's words in the ball.  `qa/pairs.py` reads this leader back off the
    # first cell and distributes it over all nine.
    label = tense if gender is None else f"{tense} ({gender})"
    cells[0] = f"{label}: {cells[0]}"
    slots = {"L": e.lemma, "tense": tense, **tense_slots(tense)}
    if gender:
        slots["gender"] = gender
        slots.update(gender_slots(gender))
    return slots, cells


def gen_T7(ctx, e, rng):
    if e.pos != "glagol":
        return None
    store = ctx.store
    inf = sup = imp = None
    for _v, s, f in store.forms(e.a):
        vf = f.get("vform")
        if vf == "nedoločnik" and inf is None:
            inf = s
        elif vf == "namenilnik" and sup is None:
            sup = s
        elif vf == "velelnik" and f.get("person") == "2. oseba" \
                and f.get("number") == "ednina" and imp is None:
            imp = s
    if inf is None and store.anchor_features(e.a).get("vform") == "nedoločnik":
        inf = store.lemma(e.a)          # the anchor IS the infinitive
    if inf is None and sup is None and imp is None:
        return None
    return ({"L": e.lemma},
            [f"nedoločnik: {inf or GAP}", f"namenilnik: {sup or GAP}",
             f"velelnik: {imp or GAP}"])


# --------------------------------------------------------------------------
# Group C -- besedna vrsta
# --------------------------------------------------------------------------
# T8's noun answer IS T9's answer with the POS prepended, and its verb answer IS
# T10's.  One normalizer, called by all three, so the three cannot spell the same
# fact two ways (QA_TASKS.md T8).
def lexeme_property(store, a):
    f = store.anchor_features(a)
    pos = f.get("pos")
    if pos == "samostalnik":
        return pos, f.get("gender")
    if pos == "glagol":
        return pos, f.get("aspect")
    return pos, None


def gen_T8(ctx, e, rng):
    pos, prop = lexeme_property(ctx.store, e.a)
    if not pos:
        return None
    return {"L": e.lemma}, [pos] + ([prop] if prop else [])


def gen_T9(ctx, e, rng):
    pos, prop = lexeme_property(ctx.store, e.a)
    if pos != "samostalnik" or not prop:
        return None
    return {"L": e.lemma}, [prop]


def gen_T10(ctx, e, rng):
    pos, prop = lexeme_property(ctx.store, e.a)
    if pos != "glagol" or not prop:
        return None
    return {"L": e.lemma}, [prop]


# --------------------------------------------------------------------------
# T11 -- stopnjevanje
# --------------------------------------------------------------------------
def forms_with_anchor(store, a):
    """The entry's form leaves, plus the anchor itself.

    The citation form is merged into the anchor and has no leaf of its own, so a
    view that reads only `oblika:` nodes misses exactly the form a dictionary
    prints -- for an adjective that is the indefinite positive (`mikaven`), which
    is precisely what T11's first slot wants.
    """
    out = list(store.forms(a))
    feat = store.anchor_features(a)
    lemma = store.lemma(a)
    if lemma and feat:
        out.append((int(a), lemma, feat))
    return out


def _degree_cell(store, a, degree, prefer_definite):
    """Nominative singular masculine form of one degree, or None.

    Where `definiteness` is present, prefer the indefinite for the positive
    (`brunast`) and the definite for the superlative (`najbrunastejši`) -- how
    Slovene dictionaries cite them.
    """
    best = None
    for _v, s, f in forms_with_anchor(store, a):
        if f.get("degree") != degree:
            continue
        if f.get("case") != "imenovalnik" or f.get("number") != "ednina":
            continue
        g = f.get("gender")
        if g not in (None, "moški spol"):
            continue
        d = f.get("definiteness")
        rank = (0 if (d == "določna oblika") == prefer_definite else 1,
                0 if d is not None else 1, s)
        if best is None or rank < best[0]:
            best = (rank, s)
    return best[1] if best else None


def gradation(store, a, lemma):
    """(osnovnik, primernik, presežnik) or None if the paradigm is degenerate.

    **The degeneracy filter is the point of this type.**  The reference dataset
    is degenerate in 99/100 rows -- comparative == superlative == lemma -- and
    that is not only a generator bug there: this KG stores the same defect on
    real entries.  `mikaven` carries `word-form-908674 oblika: mikaven
    (imenovalnik, ednina, moški spol, primernik, določna oblika)`: the cell
    exists, the surface was never inflected.  An item built on it would teach
    that the comparative of *mikaven* is *mikaven*.

    So a graded cell counts only when it actually differs from the positive.
    Entries that fail become T11 negatives, which is what they truthfully are:
    no graded form is recorded for them.
    """
    pos = _degree_cell(store, a, "osnovnik", False) or lemma
    comp = _degree_cell(store, a, "primernik", False)
    sup = _degree_cell(store, a, "presežnik", True)
    if not comp or comp.casefold() in (pos.casefold(), lemma.casefold()):
        return None
    if sup and sup.casefold() in (pos.casefold(), comp.casefold(), lemma.casefold()):
        sup = None
    return pos, comp, sup


def gen_T11(ctx, e, rng):
    if e.pos not in ("pridevnik", "prislov"):
        return None
    g = gradation(ctx.store, e.a, e.lemma)
    if g is None:
        return None                          # the scarce resource defines the pool
    pos, comp, sup = g
    return {"L": e.lemma}, [pos, comp, sup or GAP]


# --------------------------------------------------------------------------
# Group D -- pomen
# --------------------------------------------------------------------------
def gen_T12(ctx, e, rng):
    """The defined senses, each carrying the ordinal the GRAPH gives it.

    Not a 1..n renumbering.  A ball renders the anchor's senses under the store's
    own ordinals, and the placeholder and fallback senses that `defined_senses`
    drops keep their numbers there -- `nakladnica`'s one real sense is `pomen 3`
    in the ball and was `pomen 1` in the answer.  Under 0.1 the ordinal is part
    of the LABEL, so renumbering makes the label contradict the node it names, in
    623 of 623 items: the model would be trained to answer `pomen 1` while
    looking at `pomen 3`.  The ordinal-less senses the store sorts last are
    labelled `pomen`, exactly as the node text spells them.
    """
    ds = seeds.defined_senses(ctx.store, e.a)
    if not ds:
        return None
    return {"L": e.lemma}, [f"{sense_label(o)}: {b}" for _v, o, b in ds]


def sense_label(ordinal):
    """`pomen 3`, or `pomen` for a sense the KG gives no number."""
    return f"pomen {ordinal}" if ordinal else "pomen"


def gen_T14(ctx, e, rng):
    ds = seeds.defined_senses(ctx.store, e.a)
    if not ds:
        return None
    return {"L": e.lemma}, [str(len(ds))]


# --------------------------------------------------------------------------
# Group E -- sense relations
# --------------------------------------------------------------------------
def _partners(ctx, e, kind, cap=10):
    p = ctx.store.partners(e.a, kind)
    return p[:cap] if p else None


def gen_T15(ctx, e, rng):
    p = _partners(ctx, e, 6)
    return ({"L": e.lemma}, p) if p else None


def gen_T16(ctx, e, rng):
    p = _partners(ctx, e, 7)
    return ({"L": e.lemma}, p) if p else None


# --------------------------------------------------------------------------
# Group F -- kolokacije
# --------------------------------------------------------------------------
BAND_SHARE = (("none", 0.40), ("vague_small", 0.20),
              ("vague_large", 0.20), ("exact", 0.20))
BAND_LEN = {"none": 5, "vague_small": 4, "vague_large": 15}


def gen_T17(ctx, e, rng):
    ball = colloc_sampling.sample(ctx.store, e.a)
    if not ball:
        return None
    r = rng.random()
    acc = 0.0
    band = "none"
    for b, share in BAND_SHARE:
        acc += share
        if r < acc:
            band = b
            break
    phrases = dedup_by_norm(p for _v, p, _s in ball)
    # |ALL| is the size of the set the GRADER holds, which is deduplicated the
    # same way -- not the number of collocation nodes.  `gadov` has two nodes,
    # `Gadova Peč` and `Gadova peč`, and exactly one phrase; counting nodes made
    # the `k == n_all` escape hatch unreachable and the item ungradeable.
    n_all = len(all_phrases(ctx, e))
    if band == "exact":
        n_asked = rng.randrange(2, 11)
        want = min(n_asked, len(phrases))
    else:
        n_asked = None
        want = min(BAND_LEN[band], len(phrases))
    if want < 1:
        return None
    items = phrases[:want]
    slots = {"L": e.lemma, "band": band, "n_all": n_all}
    if n_asked is not None:
        slots["N"] = n_asked
        slots.update(count_slots(n_asked))
    return slots, items


def dedup_by_norm(phrases):
    """Canonically ordered, one surface per phrase AS THE GRADER SEES IT.

    v7 folds case-only duplicates within one member set, but an anchor can carry
    the same phrase on two different sense pairs -- `Gadova Peč` and `Gadova peč`
    are two nodes on two pairs -- and the grader matches case-insensitively, so a
    gold list holding both would fail its own `dedup(answer) == answer` rule.
    The surviving surface is the least capitalised one, matching how the builder
    chooses between several writtenRep values for one form.
    """
    best = {}
    for p in phrases:
        k = sl.norm(p)
        rank = (sum(1 for ch in p if ch.isupper()), p)
        if k not in best or rank < best[k][0]:
            best[k] = (rank, p)
    return sorted((v[1] for v in best.values()), key=sl.sl_key)


def count_slots(n):
    """The counted noun phrases the `exact` band's frames need, agreeing with n."""
    return {"N_KOL": sl.counted(n, "kolokacija"),
            "N_ZVEZ": sl.counted(n, "besedna zveza"),
            "N_KOLZVEZ": sl.counted(n, "kolokativna zveza"),
            "N_KOLPRIM": sl.counted(n, "kolokativni primer"),
            "N_TIPZVEZ": sl.counted(n, "tipična besedna zveza"),
            "N_POGKOL": sl.counted(n, "pogosta kolokacija"),
            "N_SOPOJ": sl.counted(n, "zabeležena sopojavitev")}


def all_phrases(ctx, e):
    """Every collocation phrase the store holds for ONE anchor, normalized.

    Used for `n_all` at sampling time -- how many the store has, which is what
    the quantity band is drawn against.  It is NOT the grader's allow-list and
    must not be used as one: D3 unions several anchors into a ball, so this is
    neither a superset nor a subset of what the model is shown.  The allow-list
    is `build_balls.member_pool`, derived from the ball itself.
    """
    return sorted({sl.norm(p) for _v, p in ctx.store.collocations(e.a)})




# --------------------------------------------------------------------------
# Group G -- primeri uporabe
# --------------------------------------------------------------------------
MAX_EXAMPLE_WORDS = 60


def gen_T19(ctx, e, rng):
    ex = [t for _v, t in ctx.store.examples(e.a)
          if t and len(t.split()) <= MAX_EXAMPLE_WORDS]
    if not ex:
        return None
    # "First" is the example child with the lowest node id -- store.examples()
    # returns them in that order.  The graph's edge order is not a ranking.
    return {"L": e.lemma}, [ex[0]]


_WORD = sl.WORD


def occurrences(surface, sent):
    """How many times `surface` appears in `sent` as a whole word, case-blind.

    Public because T20's *negatives* need it too: the borrowed form has to be
    verified absent from the sentence, and doing that with a second, slightly
    different regex is how the positive and negative halves of one type drift
    apart.
    """
    return len(re.findall(_WORD.format(re.escape(surface)), sent, re.I))


def gen_T20(ctx, e, rng):
    """Form analysis with a sentence, restricted to UNAMBIGUOUS forms (D15).

    The sentence comes from the graph, so the item is fully contained in the
    ball; the form must occur in it exactly once and carry exactly one reading.
    """
    store = ctx.store
    if not healthy_paradigm(store, e.a):
        return None
    by_surface = paradigm_readings(store, e.a)
    unambiguous = {s: r for s, r in by_surface.items() if len(r) == 1}
    if not unambiguous:
        return None
    cand = []
    for _v, sent in store.examples(e.a):
        if not sent or len(sent.split()) > MAX_EXAMPLE_WORDS:
            continue
        hits = [s for s in unambiguous if occurrences(s, sent) == 1]
        # Reject if any OTHER form of the entry also appears: the question names
        # a surface, and two candidate surfaces in one sentence make it ambiguous
        # which one is meant.
        hits = [s for s in hits
                if not any(o != s and occurrences(o, sent) for o in by_surface)]
        for s in hits:
            cand.append((s, sent))
    if not cand:
        return None
    # Prefer a candidate that is NOT the base form.  Taking the first example and
    # the alphabetically first hit made `imenovalnik ednine` **66.1 %** of the
    # type's test items -- a constant answer would have scored two thirds, which
    # makes the type nearly uninformative (check C13).  A word in a dictionary
    # example is most often cited in its base form, so the skew is real; but among
    # the readings this entry genuinely offers, nothing obliges us to take that
    # one.  Every candidate here is already a correct, unambiguous, in-graph item
    # -- this reorders a valid set, it does not relabel anything.
    off_base = [(s, sent) for s, sent in cand
                if {(r[0], r[1]) for r in unambiguous[s]}
                != {("imenovalnik", "ednina")}]
    form, sent = sorted(off_base or cand, key=lambda p: (sl.sl_key(p[0]), p[1]))[0]
    return ({"F": form, "S": sent, "L": e.lemma},
            _reading_items(unambiguous[form]))


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Group H -- the first two types (step 7)
# --------------------------------------------------------------------------
YES_NO = ("da", "ne")


def relation_values(store, a, rel):
    """Everything anchor `a` records under `rel`, as the strings a ball shows."""
    if rel == "sopomenka":
        return store.partners(a, K_SYN)
    if rel == "protipomenka":
        return store.partners(a, K_ANT)
    if rel == "zgled":
        return [t for _v, t in store.examples(a) if t]
    if rel == "kolokacija":
        return [p for _v, p in store.collocations(a) if p]
    if rel == "pomen":
        return [b for _v, _o, b in seeds.defined_senses(store, a)]
    if rel == "oblika":
        return [s for _v, s, _f in store.forms(a) if s]
    if rel == "prevod":
        return [t for _v, t in store.translations(a) if t]
    raise KeyError(rel)


def relation_state(store, anchors, rel):
    """(present, ambiguous) for `rel` over every anchor the ball will hold.

    `ambiguous` means a `da`/`ne` answer would collide with a VALUE of the
    relation: `oblika: da` reads two ways -- "yes, it has forms" and "one of its
    forms is *da*" -- and *da* really is a form of `dati`.  Rare (2 of 13,015
    synonym partners, 1 of 16,494 senses) and cheap to refuse, which is the same
    treatment T1 gives its animacy doublets.
    """
    present = ambiguous = False
    for a in anchors:
        for v in relation_values(store, a, rel):
            present = True
            if v.strip().casefold() in YES_NO:
                ambiguous = True
                return present, ambiguous
    return present, ambiguous


def relation_present(ctx, lemma, own, rel):
    """Does ANY anchor of this surface record `rel`?  The union, not the seed.

    T23 asks the same question through `relation_state`, which also reports the
    `da`/`ne` collision; the counting types (T24, T27) need only the boolean, and
    need it over a relation their generator did not choose.
    """
    for a in (sibling_anchors(ctx, lemma) or [own]):
        if relation_values(ctx.store, a, rel):
            return True
    return False


def collocation_owners(ctx, phrase, skip=None, cache=None):
    """Every anchor that records `phrase` as one of its collocations.

    Reached the way the PIPELINE reaches it -- through the words of the phrase
    and the surface index -- rather than through a reverse edge, because that is
    the route D3c actually takes and the route T35's answer has to be unique
    along.

    `cache` maps an anchor to its normalised collocation set.  Pass one per
    generator call and not one per run: the anchors this reaches are the busy
    ones (up to 2,386 collocations each), so holding them all would cost more
    memory than the whole seed pool, while the tries WITHIN one call hit the
    same handful of anchors over and over.
    """
    out = set()
    want = sl.norm(phrase)
    store = ctx.store
    for w in phrase.split():
        for a in ctx.surface.get(w.casefold(), ()):
            a = int(a)
            if a == skip:
                continue
            if cache is None:
                found = any(sl.norm(q) == want for _v, q in store.collocations(a))
            else:
                norms = cache.get(a)
                if norms is None:
                    norms = cache[a] = {sl.norm(q) for _v, q in store.collocations(a)}
                found = want in norms
            if found:
                out.add(a)
    return out


def colloc_owner_anchors(store, v, skip=None):
    """The anchors a collocation NODE hangs off, minus `skip`.

    A collocation is a relation between two lexical units and the store reifies
    it as one node hanging off a sense of each: over 785 sampled nodes, every
    one has exactly two sense parents and exactly two anchor owners -- the two
    words of the collocation (`dekan blagoslovi` off `dekan` and `blagosloviti`).
    So "who else lists this phrase" is a reverse hop and not a search.
    """
    out = set()
    for u in store.nbrs(v):
        u = int(u)
        if store.kind[u] != K_SENSE:
            continue
        for w in store.nbrs(u):
            w = int(w)
            if store.kind[w] == K_ANCHOR and w != skip:
                out.add(w)
    return out


def reachable_owner(ctx, v, phrase, skip=None):
    """Does a SECOND owner of collocation node `v` sit in the ball for `phrase`?

    Reachability is asked of the surface index and not of `store.forms`, because
    that is the index D3c resolves a span through: an owner whose forms include a
    word of the phrase but whose index entry does not is an owner the model never
    sees, and an item is unambiguous exactly when its INPUT is.
    """
    if v is None:
        return False
    owners = colloc_owner_anchors(ctx.store, int(v), skip=skip)
    if not owners:
        return False
    reach = set()
    for w in phrase.split():
        reach.update(int(a) for a in ctx.surface.get(w.casefold(), ()))
    return bool(owners & reach)


def sibling_anchors(ctx, lemma):
    """Every anchor the item's own ball will be built from.

    D3 unions all anchors whose surface matches, so a homograph brings a second
    entry's subtree into the SAME ball.  A yes/no answer has to be true of the
    ball the model is shown, not of one anchor inside it: `ne` while the ball
    renders a `sopomenka:` node is an item whose gold contradicts its input.
    """
    hits = ctx.surface.get(lemma.casefold())
    return sorted(int(a) for a in hits) if hits else []


def gen_T23(ctx, e, rng):
    """Is relation R recorded for word L?  `da` or `ne`, both read off the graph.

    `ne` is an ANSWER and not the 0.2 sentinel: the database answers the question,
    and what it answers is that the relation is absent.  That is the fact this
    type teaches, and no other type in the corpus teaches it -- every other type
    asks the graph for content and can only fail to find it.

    The relation is chosen to keep `da` and `ne` near balanced ITEM BY ITEM rather
    than by post-hoc filtering: a coin picks which answer is wanted, and the
    relation is drawn from the ones that give it.  Left to a uniform draw over
    relations the type comes out heavily `da` -- most pool entries have senses and
    examples -- and a model can score well on it without reading the ball at all.

    Tier C: `protipomenka` may only be asked of a TEST entry (D12/C6).  A training
    question that merely names the held-out relation leaks it whether its answer
    is `da` or `ne`, so the constraint is on the item and not on the answer.
    """
    store = ctx.store
    anchors = sibling_anchors(ctx, e.lemma) or [e.a]
    cand = list(spec.T23_RELATIONS)
    if e.split == "test":
        cand += list(spec.T23_HELD_OUT)
    present = {}
    for r in cand:
        p, ambiguous = relation_state(store, anchors, r)
        if not ambiguous:
            present[r] = p
    if not present:
        return None
    want = rng.random() < 0.5          # True -> look for a `ne`
    pool = [r for r in present if present[r] is not want] or list(present)
    rel = pool[rng.randrange(len(pool))]
    slots = {"L": e.lemma}
    slots.update(sl.relation_slots(rel))
    return slots, [f"{rel}: {'da' if present[rel] else 'ne'}"]


def gen_T30(ctx, e, rng):
    """The headwords a multi-word phrase is composed of.

    The answer is the CONSTITUENT LEMMAS, not the phrase's own surface words:
    `pod drobnogledom` answers `iztočnica: pod | iztočnica: drobnogled`, and the
    two differ in 93.1 % of phrases -- which is what makes the type a lookup
    rather than a tokenisation exercise.

    Two entry-level rules, both refusals rather than repairs:

      * a phrase with fewer than two distinct constituent headwords has no
        composition to state, and
      * every constituent that is itself a word seed must share this phrase's
        split (H.3, one level down).  T30's answer NAMES word lemmas and C11 is a
        check over every lemma an item names, so a train phrase built on a test
        seed breaks lemma-disjointness exactly as a duplicated seed would.  Only
        39.9 % of phrases pass, which is a filter and not a blocker.

    And one at the constituent level: C26.  Every constituent must actually occur
    in the phrase, by its lemma or one of its forms.  The KG's constituent edges
    are not always right -- see `seeds.constituent_occurs` -- and a wrong one is
    a well-formed lemma in the right shape, which nothing downstream would query.
    Refused rather than repaired: dropping the bad constituent would ship an
    answer that is incomplete about a phrase whose composition we know we cannot
    read.  Costs a further 26.4 %.
    """
    store = ctx.store
    if not seeds.constituents_agree(store, e, ctx.word_split):
        return None
    out, seen = [], set()
    for c in seeds.constituents(store, e.a):
        if not seeds.constituent_occurs(store, c, e.lemma, ctx.form_cache):
            return None
        w = store.lemma(c)
        k = w.casefold()
        if not w or k in seen:
            continue
        seen.add(k)
        out.append(w)
    if len(out) < 2:
        return None
    # Canonical order is Slovene alphabetical, and C15 asserts it (VALUE_SORTED).
    # NOT the order the words appear in the phrase: `constituents` walks graph
    # edges, whose order is node id, so a "phrase order" read off it would be a
    # tie-break dressed up as a fact -- exactly what the set rule forbids.
    return {"L": e.lemma}, sorted(out, key=sl.sl_sort_key)


# ==========================================================================
# Group H -- composition, comparison and phrases
# ==========================================================================
# Every type below emits its labels GLUED to their values, `oznaka: vrednost`
# in one string, the way T7 and T12 already do.  That is not a style choice: a
# Group H label is chosen from the data -- which cell was asked for, which
# relation, which sense -- so unlike T1's it cannot be rebuilt from the type
# and the position.  `qa/pairs.py:_split_leader` takes it apart again, so the
# label still lands in `gold_items` through the one function that owns labels.


def _glue(label, value):
    return f"{label}: {value}"


#: T34's blank, and the cap on how far it will scan a constituent's memberships
#: to prove a fragment has exactly one completion.  2,000 is above the 99th
#: percentile of a content word's membership count and far below a function
#: word's (`biti`: 423,510), so the cap refuses the hub rather than truncating
#: an honest answer.
GAP_MARK = "___"
T34_SCAN_CAP = 2000

#: how many of a seed's collocation phrases T35 tests before giving up on it.
T35_TRIES = 12

#: relation name -> the store node kind holding it, for the types that take the
#: relation as a slot (H.2, H.3).
RELATION_KIND = {"sopomenka": K_SYN, "protipomenka": K_ANT}


# ---- H.1 selection over the paradigm -------------------------------------

def selection_cells(store, a, gender, rng):
    """T22's requested subset: a run along ONE axis of the grid, never both.

    Uniform over the non-empty subsets of one axis rather than over all 2**18
    (QA_TASKS.md T22).  Sizes 18, 6 and 1 are excluded because they are T1, T2
    and T3 -- what T22 contributes is every OTHER size, so that no per-type
    arity is memorisable and the model has to read which cells were named.

    Returns (cases, numbers, cells) with `cells` in the canonical grid order.
    """
    probe = nominal_grid(store, a, gender)
    if not probe:
        return None
    filled = {(c, n) for n in sl.NUMBERS for c in sl.CASES
              if _cell(probe, c, n) not in (None, False)}
    opts = []
    for n in sl.NUMBERS:
        cs = [c for c in sl.CASES if (c, n) in filled]
        if len(cs) >= 2:
            opts.append(("case", n, cs))
    for c in sl.CASES:
        ns = [n for n in sl.NUMBERS if (c, n) in filled]
        if len(ns) >= 2:
            opts.append(("number", c, ns))
    if not opts:
        return None
    axis, fixed, along = opts[rng.randrange(len(opts))]
    if axis == "case":
        # never 6 -- that is T2's whole number
        k = rng.randint(2, min(len(along), 5))
        cases = sorted(rng.sample(along, k), key=sl.CASE_IDX.get)
        numbers = [fixed]
    else:
        k = rng.randint(2, len(along))
        numbers = sorted(rng.sample(along, k), key=sl.NUMBER_IDX.get)
        cases = [fixed]
    cells = [(c, n) for n in numbers for c in cases]
    return cases, numbers, cells


def gen_T22(ctx, e, rng):
    """An arbitrary subset of T1's grid, named in the question.

    Shares T1's source, T1's seed filter (`healthy_grid`, C23) and T1's gender
    axis (A.0) exactly -- only the requested subset moves.  This is the type
    that stops T1, T2 and T3 being three things: they are the subset sizes 18,
    6 and 1 of one contract.
    """
    store = ctx.store
    gender = pick_gender(store, e.a, rng)
    sel = selection_cells(store, e.a, gender, rng)
    if sel is None:
        return None
    cases, numbers, cells = sel
    definiteness = pick_definiteness(store, e.a, gender, rng, set(cells))
    grid = nominal_grid(store, e.a, gender, definiteness)
    items = []
    for c, n in cells:
        v = _cell(grid, c, n)
        if v in (None, False):
            return None
        items.append(_glue(sl.cell_label(c, n, gender, definiteness), v))
    slots = {"L": e.lemma, "IZBOR": sl.selection_phrase(cases, numbers),
             **axis_slots(gender, definiteness)}
    return slots, items


def t22_absent_selection(ctx, lemma, own, rng):
    """A selection NO anchor of this surface fills -- T22's negative (or None).

    Decided over the union D3 will hand the model, never over the seed anchor:
    a pluralia tantum asked for the dual is a true negative only while no
    co-extracted homograph fills the dual under the same lemma.

    An entry with no nominal paradigm at all (a verb, or a lemma the base does
    not hold) makes ANY selection absent, which is the same negative one step
    coarser -- so it returns a random run rather than refusing.
    """
    anchors = sibling_anchors(ctx, lemma) or [own]
    filled = set()
    for a in anchors:
        for g in (paradigm_genders(ctx.store, a) or (None,)):
            grid = nominal_grid(ctx.store, a, g, require_healthy=False)
            filled |= {k for k, v in grid.items() if v}
    if not filled:
        if rng.random() < 0.5:
            n = sl.NUMBERS[rng.randrange(3)]
            return sorted(rng.sample(list(sl.CASES), rng.randint(2, 5)),
                          key=sl.CASE_IDX.get), [n]
        c = sl.CASES[rng.randrange(6)]
        return [c], sorted(rng.sample(list(sl.NUMBERS), rng.randint(2, 3)),
                           key=sl.NUMBER_IDX.get)
    opts = [("case", n) for n in sl.NUMBERS
            if not any((c, n) in filled for c in sl.CASES)]
    opts += [("number", c) for c in sl.CASES
             if not any((c, n) in filled for n in sl.NUMBERS)]
    if not opts:
        return None
    axis, fixed = opts[rng.randrange(len(opts))]
    if axis == "case":
        return sorted(rng.sample(list(sl.CASES), rng.randint(2, 5)),
                      key=sl.CASE_IDX.get), [fixed]
    return [fixed], sorted(rng.sample(list(sl.NUMBERS), rng.randint(2, 3)),
                           key=sl.NUMBER_IDX.get)


# ---- H.2 quantity over any relation --------------------------------------

def gen_T24(ctx, e, rng):
    """How many X are recorded for L -- T14 generalised off senses.

    Counted over the SEED ANCHOR, exactly as T14, T15 and T16 count, and
    deliberately not over the union: T24 and T14 both answer "how many senses"
    and a disagreement between them is C17's defect with two types instead of
    one.  The union question is real and it is T14's to answer first.

    Only `spec.COUNTABLE` -- a sampled or capped relation would grade the model
    on D5's seed rather than on the graph.
    """
    store = ctx.store
    cand = list(spec.COUNTABLE)
    if e.split == "test":
        cand += list(spec.COUNTABLE_HELD_OUT)
    counts = {}
    for rel in cand:
        n = len(dedup_by_norm(relation_values(store, e.a, rel)))
        if n:
            counts[rel] = n
    if not counts:
        return None
    rels = sorted(counts)
    rel = rels[rng.randrange(len(rels))]
    slots = {"L": e.lemma, **sl.relation_slots(rel)}
    return slots, [_glue(sl.count_label(rel), str(counts[rel]))]


# ---- H.3 two anchors -----------------------------------------------------

def _pair_partner(ctx, e, rng, exclude=()):
    """A distractor for H.3: same split, same band, same word class.

    Same split because C11 is a statement about LEMMAS and this item names two.
    Same band and POS so that *plausible* is not *related* -- a negative whose
    two words are obviously unalike is answerable without reading anything.
    """
    bad = {e.lemma.casefold(), *(x.casefold() for x in exclude)}
    cands = [o for o in ctx.by_split[e.split]
             if o.band == e.band and o.pos == e.pos
             and o.lemma.casefold() not in bad]
    if not cands:
        return None
    return cands[rng.randrange(len(cands))]


def gen_T25(ctx, e, rng):
    """Are X and Y in relation R?  One pair, the relation tag and `da`/`ne`.

    The negative is the type's real content and it is NOT a sentinel: both
    words are in the base, both balls are present, and no node links them.
    Answering it means checking the ABSENCE of an edge between two named
    things, which no single-anchor type in this corpus exercises.

    As in T23 the coin picks the answer first and the partner second, or the
    type comes out overwhelmingly `ne` -- most word pairs are unrelated -- and
    a model scores well by always saying so.
    """
    rels = list(spec.PAIR_RELATIONS)
    if e.split == "test":
        rels += list(spec.PAIR_HELD_OUT)
    rel = rels[rng.randrange(len(rels))]
    kind = RELATION_KIND[rel]
    linked = ctx.partners(kind).get(e.a) or []
    want_yes = rng.random() < 0.5 and linked
    if want_yes:
        other = linked[rng.randrange(len(linked))]
    else:
        other = _pair_partner(ctx, e, rng,
                              exclude=[o.lemma for o in linked])
        if other is None:
            return None
        # The verdict is over the UNION each ball will hold, not over one
        # anchor: a homograph of either word can carry the link, and `ne`
        # while the ball renders it is gold contradicting its own input.
        if _related(ctx, e.lemma, other.lemma, kind):
            return None
    answer = "da" if want_yes else "ne"
    slots = {"L": e.lemma, "L2": other.lemma, **sl.relation_slots(rel)}
    return slots, [_glue(sl.relation_tag(rel), answer)]


def _related(ctx, x, y, kind):
    """Does ANY anchor of surface x record surface y as a `kind` partner?"""
    yk = y.casefold()
    for a in sibling_anchors(ctx, x):
        if any(p.casefold() == yk for p in ctx.store.partners(a, kind)):
            return True
    for a in sibling_anchors(ctx, y):
        if any(p.casefold() == x.casefold()
               for p in ctx.store.partners(a, kind)):
            return True
    return False


def gen_T26(ctx, e, rng):
    """What do X and Y have in common?  T8's key set, intersected.

    The empty intersection returns None rather than the sentinel, which is a
    stated deviation from the note.  §0.2's sentinel means *the database has
    nothing to say about this*; two words that share no property are a case
    where it has plenty to say, and there is no label under 0.1 to hang "they
    share nothing" on.  Emitting the sentinel for it would make a positive item
    byte-identical to a negative one and quietly corrupt every baseline that
    counts sentinels (C13).  The pairs that share nothing are simply not items.
    """
    other = _pair_partner(ctx, e, rng)
    if other is None:
        return None
    pos_a, extra_a = lexeme_property(ctx.store, e.a)
    pos_b, extra_b = lexeme_property(ctx.store, other.a)
    shared = []
    if pos_a and pos_a == pos_b:
        shared.append(_glue("besedna vrsta", pos_a))
        # The second property is the one T8 renders for this word class, so it
        # can only agree when the class does -- checked inside this branch
        # rather than beside it, or a noun's gender could be compared against a
        # verb's aspect and come out equal by both being None.
        if extra_a and extra_a == extra_b:
            shared.append(_glue("spol", extra_a[:-len(" spol")])
                          if extra_a.endswith(" spol") else
                          _glue("vid", extra_a))
    if not shared:
        return None
    slots = {"L": e.lemma, "L2": other.lemma}
    return slots, shared


def gen_T27(ctx, e, rng):
    """Which of X and Y has more (or fewer) R?  The ANSWER names that word.

    The direction is drawn and asked, not read off the data: the question fixes
    it (`{PRIM}`) and the label repeats it, so `manj` is a third of the items
    rather than the accident of which word happened to be the seed.  Keying the
    label on the seed instead produced `manj pomenov: Shakespearov` under a
    frame asking which word had MORE -- an answer that does not answer.

    Ties are NOT filtered away: two random core anchors tie on sense count
    33.4 % of the time, and a seed filter demanding a strict difference would
    teach *there is always a winner*.  A tie answers with two pairs sharing a
    label rather than one pair holding a list -- 0.1 has no comma-separated
    values anywhere, and a lemma may contain a space.
    """
    store = ctx.store
    other = _pair_partner(ctx, e, rng)
    if other is None:
        return None
    cand = list(spec.COUNTABLE)
    if e.split == "test":
        cand += list(spec.COUNTABLE_HELD_OUT)
    usable = []
    for rel in cand:
        m = len(dedup_by_norm(relation_values(store, e.a, rel)))
        t = len(dedup_by_norm(relation_values(store, other.a, rel)))
        if m or t:
            usable.append((rel, m, t))
    if not usable:
        return None
    rel, m, t = usable[rng.randrange(len(usable))]
    asked = "več" if rng.random() < 0.5 else "manj"
    slots = {"L": e.lemma, "L2": other.lemma, "PRIM": asked,
             **sl.relation_slots(rel)}
    if m == t:
        return slots, [_glue(sl.compare_label("enako", rel), e.lemma),
                       _glue(sl.compare_label("enako", rel), other.lemma)]
    winner = e if (m > t) == (asked == "več") else other
    return slots, [_glue(sl.compare_label(asked, rel), winner.lemma)]


# ---- H.4 sense-scoped relations ------------------------------------------

def sense_relation(store, a, sense, kind):
    """One SENSE's own children of `kind`, as the ball spells them.

    `store.sense_children` unions every sense of the anchor, which is precisely
    the scope this type exists to narrow, so the walk is repeated here over one
    sense rather than filtered afterwards.
    """
    tag = TAG[kind]
    out = set()
    for v in store.nbrs(sense):
        v = int(v)
        if store.kind[v] == kind:
            t = store.text(v)
            out.add(t[len(tag):] if t.startswith(tag) else t)
    return out


def sense_partners(store, a, sense, kind):
    """One sense's partner LEMMAS, not the reified `A ~ B` node text.

    `store.partners` does this for the whole anchor and T15 answers with its
    output, so a sense-scoped type answering with the raw node text would emit
    `sopomenka: deček ~ otrok` where T15 emits `sopomenka: deček` -- two
    spellings of one relation, which is exactly what 0.1 exists to prevent.
    """
    me = store.lemma(a).casefold()
    out = set()
    for t in sense_relation(store, a, sense, kind):
        parts = [p.strip() for p in t.split(" ~ ")]
        if len(parts) != 2:
            continue
        out.update(p for p in parts if p and p.casefold() != me)
    return out


def _scoped_senses(store, a, kind):
    """[(sense node, ordinal, body, own set)] where the scope DISCRIMINATES.

    A sense whose set is the entry's whole set teaches nothing about scoping,
    so it is not a seed: the distractor has to be the entry's OTHER senses'
    children, sitting in the same ball.
    """
    sn = store.senses(a)
    if len(sn) < 2:
        return []
    # Three roles, and conflating them cost most of this type's pool.
    #
    # The ASKED sense must be NAMEABLE, which is not the same as defined.  A
    # definition can be quoted (`v pomenu 'Y'`); an ordinal names the sense just
    # as exactly, because the ball numbers every sense it renders.  What cannot
    # be asked about is a sense with neither -- the store renders a placeholder
    # as `pomen None: Shakespeare`, whose body is the headword and whose ordinal
    # is missing, so naming it asks the model to scope by nothing.
    #
    # The DISTRACTING senses need not be nameable at all.  What makes the scope
    # real is that the other senses' children sit in the same ball, and the ball
    # renders every sense's children.  Judging `full` over the defined senses
    # alone both under-counted the distractors and demanded a second definition
    # the item never uses.
    defined = {v for v, _o, _b in seeds.defined_senses(store, a)}
    per = [(v, o, b, sense_partners(store, a, v, kind)) for v, o, b in sn]
    full = set().union(*(p[3] for p in per))
    return [p for p in per
            if p[3] and p[3] != full
            and ((p[0] in defined and p[2]) or p[1])]


def gen_T28(ctx, e, rng):
    """Which synonyms does L have IN the sense 'Y'?  T15's contract, narrowed.

    Restricted to `sopomenka`, and the restriction is structural rather than
    timid: `mode` is per TYPE, and the other two sense-scoped relations are
    graded `membership` (T17, T19) because their ball content is a selection.
    One type cannot be both, so a sense-scoped collocation type is a second
    type and not a slot value here.
    """
    store = ctx.store
    scoped = _scoped_senses(store, e.a, K_SYN)
    if not scoped:
        return None
    v, ordinal, body, own = scoped[rng.randrange(len(scoped))]
    values = sorted(dedup_by_norm(own), key=sl.sl_sort_key)[:10]
    if not values:
        return None
    # A definition is quotable and an ordinal is not a fallback for one: the two
    # name the sense in different frame pools (spec.template_key), so an item
    # without a definition must not carry a POMEN slot at all -- a frame quoting
    # a body that identifies nothing is exactly what the `defined` rule forbids.
    defined = {s for s, _o, _b in seeds.defined_senses(store, e.a)}
    if v in defined and body:
        return ({"L": e.lemma, "POMEN": body, "ORD": ordinal or 1},
                [_glue("sopomenka", x) for x in values])
    if not ordinal:
        return None
    return ({"L": e.lemma, "ORD": ordinal},
            [_glue("sopomenka", x) for x in values])


def gen_T29(ctx, e, rng):
    """Which sense of L does the example/collocation Z belong to?

    Well defined only where the attachment is unique, so a child sitting on two
    senses of the same entry is excluded by seed filter rather than resolved by
    a tie-break.  Answered with the sense itself, so the label is `pomen N` and
    the contract is T12's -- the question gives what T12 asks for and asks for
    what T12 gives.
    """
    store = ctx.store
    # Defined senses only, and for T28's reason from the other side: the ANSWER
    # is the sense, so a placeholder body would answer `pomen 1: Pierre` -- the
    # headword restated, which is T12's `defined` contract being broken by a
    # type that shares its labels.
    sn = seeds.defined_senses(store, e.a)
    if len(sn) < 2:
        return None
    owner, dup = {}, set()
    for v, o, b in sn:
        if not b:
            continue
        for kind in (K_COLLOC, K_EXAMPLE):
            for txt in sense_relation(store, e.a, v, kind):
                if not txt or len(txt.split()) > MAX_EXAMPLE_WORDS:
                    continue
                if txt in owner and owner[txt][0] != v:
                    dup.add(txt)
                owner.setdefault(txt, (v, o, b))
    cands = sorted(t for t in owner if t not in dup)
    if not cands:
        return None
    txt = cands[rng.randrange(len(cands))]
    _v, ordinal, body = owner[txt]
    slots = {"L": e.lemma, "Z": txt}
    return slots, [_glue(sense_label(ordinal), body)]


# ---- H.5 phrases ---------------------------------------------------------

def phrase_count_slots(n):
    """The counted noun phrases T31's `exact` band needs, agreeing with n."""
    return {"N_ZVEZ": sl.counted(n, "besedna zveza"),
            "N_ZVEZA": sl.counted(n, "zveza"),
            "N_STALZVEZ": sl.counted(n, "stalna besedna zveza")}


def gen_T31(ctx, e, rng):
    """Which phrases does the word L appear in?  Graded `membership`.

    Membership for a different reason than T17's: the ball holds D5's top ten
    by a deterministic content-independent rank rather than a seeded sample, so
    the set is reproducible and still a SELECTION -- the anchor's true
    membership reaches 162,004.  The question therefore asks for SOME phrases
    and never for all.
    """
    mwes = dedup_by_norm(p for _v, p in ctx.store.memberships(e.a))
    if not mwes:
        return None
    r = rng.random()
    acc = 0.0
    band = "none"
    for b, share in BAND_SHARE:
        acc += share
        if r < acc:
            band = b
            break
    n_all = len(mwes)
    if band == "exact":
        n_asked = rng.randrange(2, 11)
        want = min(n_asked, len(mwes))
    else:
        n_asked = None
        want = min(BAND_LEN[band], len(mwes))
    if want < 1:
        return None
    slots = {"L": e.lemma, "band": band, "n_all": n_all}
    if n_asked is not None:
        slots["N"] = n_asked
        slots.update(phrase_count_slots(n_asked))
    return slots, mwes[:want]


def gen_T32(ctx, e, rng):
    """What does the phrase X mean?  T12's contract with an MWE anchor.

    Narrow by construction -- only 1.4 % of the MWE pool carries a defined
    sense -- and worth having as the only type that asks a PHRASE for a
    definition.
    """
    ds = seeds.defined_senses(ctx.store, e.a)
    if not ds:
        return None
    return {"L": e.lemma}, [_glue(sense_label(o), b) for _v, o, b in ds]


def gen_T33(ctx, e, rng):
    """A sentence containing the phrase X.  T19's contract, and H.5's big unlock.

    93.4 % of the 3.94 M MWEs carry an example against the 71,729 reachable
    from single-word entries, because the KG's 14.7 M usage examples hang off
    MWE senses -- which stops being unreachable the moment the MWE is itself
    the anchor and its senses are hop 1.
    """
    ex = [t for _v, t in ctx.store.examples(e.a)
          if t and len(t.split()) <= MAX_EXAMPLE_WORDS and "|" not in t]
    if not ex:
        return None
    return {"L": e.lemma}, [ex[0]]


def gen_T34(ctx, e, rng):
    """Complete the phrase: `beli ___`.

    Two properties have to hold at once and both are checked against the same
    constituent -- the RAREST one, by MWE membership count:

      * reachability.  The ball is built from the fragment, so the phrase has
        to survive D5's rank for a surviving word.  The rarest constituent is
        the one whose top-ten the phrase is likeliest to be in.
      * unambiguity.  A fragment can have several legal completions, and the
        gold names one.  Enumerating the rarest constituent's memberships is
        what makes "exactly one completion" checkable rather than assumed --
        and it is affordable precisely because that constituent is rare.

    Dropping the rarest word instead would leave the fragment hanging off a
    function word (`biti` is a constituent of 423,510 phrases), which is the
    hub D5c exists to cap.
    """
    store = ctx.store
    if not seeds.constituents_agree(store, e, ctx.word_split):
        return None
    words = e.lemma.split()
    if len(words) < 2:
        return None
    cons = seeds.constituents(store, e.a)
    if len(cons) < 2:
        return None
    # Counted, not listed: one constituent of a phrase is routinely a function
    # word belonging to hundreds of thousands of MWEs, and only the smallest
    # count is read (see store.n_memberships).
    ranked = sorted(((store.n_memberships(c), c) for c in cons),
                    key=lambda t: (t[0], t[1]))
    n_min, rare = ranked[0]
    if not n_min or n_min > T34_SCAN_CAP:
        return None
    # Which word of the phrase to hide: any one, but never the rare
    # constituent's own surface, or the fragment loses its only anchor.
    keep_rare = [i for i, w in enumerate(words)
                 if seeds.constituent_occurs(store, rare, w, ctx.form_cache)]
    if not keep_rare:
        return None
    hidden = [i for i in range(len(words)) if i not in keep_rare]
    if not hidden:
        return None
    h = hidden[rng.randrange(len(hidden))]
    frag = list(words)
    frag[h] = GAP_MARK
    fragment = " ".join(frag)
    pattern = tuple(w.casefold() for w in words[:h] + words[h + 1:])
    hits = 0
    for _v, other in store.memberships(rare):
        ow = other.split()
        if len(ow) != len(words):
            continue
        for j in range(len(ow)):
            if tuple(w.casefold() for w in ow[:j] + ow[j + 1:]) == pattern:
                hits += 1
                break
        if hits > 1:
            return None
    if hits != 1:
        return None
    return {"L": fragment, "ZVEZA": fragment}, [_glue("iztočnica", e.lemma)]


def gen_T35(ctx, e, rng):
    """Whose collocation is this phrase?  The anchor IS the answer.

    The one direction nothing else in this corpus asks, and the one type that
    cannot be answered by emitting the anchor's most obvious relation.  It is
    answerable only through D3c: the extractor returns the phrase as a single
    span, no index key contains a space for a COLLOCATION phrase (the MWE entry
    of the same string is a different node, and the two coincide 29.6 % of the
    time), and the constituent fallback is what reaches the anchor.

    Two filters, both about the answer being unique and reachable:

      * the phrase must contain a surface of this anchor, or the fallback has
        no route back to it (99.7 % do), and
      * no OTHER anchor of a word in the phrase may record the same phrase as
        a collocation, or the question has two right answers.
    """
    store = ctx.store
    by_norm = {}
    for v, p in store.collocations(e.a):
        by_norm.setdefault(sl.norm(p), v)
    phrases = dedup_by_norm(p for _v, p in store.collocations(e.a))
    if not phrases:
        return None
    lemma_cf = e.lemma.casefold()
    # The uniqueness filter is the expensive one, and the anchors that carry the
    # most phrases are exactly the ones that carry thousands of them: filtering
    # the whole list cost 9 s a seed, nearly all of it spent proving phrases
    # 13..2386 usable after the first one already was.  An item needs ONE, so
    # shuffle and stop at the first survivor -- the cost is then set by the
    # number of tries and not by how busy the anchor is.
    order = list(range(len(phrases)))
    rng.shuffle(order)
    seen = {}
    for i in order[:T35_TRIES]:
        p = phrases[i]
        if not seeds.constituent_occurs(store, e.a, p, ctx.form_cache):
            continue
        # The cheap half of the same question, and the half that answers it most
        # of the time.  A collocation node hangs off a sense of BOTH words of the
        # collocation, always exactly two anchors (measured: 785 of 785 sampled
        # nodes), so the second owner is one reverse hop away -- and if the
        # phrase's own words reach it in the surface index, the ball the model is
        # shown holds two headwords listing this phrase and the question has two
        # right answers.  That is the usual rejection, and settling it here means
        # the expensive scan below runs on a phrase that has already survived.
        if reachable_owner(ctx, by_norm.get(sl.norm(p)), p, skip=e.a):
            continue
        owners = {store.lemma(a).casefold()
                  for a in collocation_owners(ctx, p, skip=e.a, cache=seen)}
        if owners - {lemma_cf}:
            continue
        return {"L": p, "ZVEZA": p}, [_glue("iztočnica", e.lemma)]
    return None


# ---- H.6 the second held-out relation ------------------------------------

def gen_T36(ctx, e, rng):
    """The Hungarian equivalents of L -- Tier C, held out of training entirely.

    Needs no new machinery beyond a tag: the label is the node's own
    `prevod (madžarsko)`, so 0.1 makes it constructible unseen in exactly the
    way `protipomenka:` is.  9.6 % of core anchors carry one, which is far too
    thin to train a relation on and the right size for a second held-out probe
    -- Tier C was one relation at n = 115, where a single item moved the score
    by 0.009.
    """
    tr = dedup_by_norm(t for _v, t in ctx.store.translations(e.a) if t)
    if not tr:
        return None
    return {"L": e.lemma}, [_glue("prevod (madžarsko)", t) for t in tr[:10]]


GENERATORS = {
    "T1": gen_T1, "T2": gen_T2, "T3": gen_T3, "T4": gen_T4, "T21": gen_T21,
    "T5": gen_T5, "T6": gen_T6, "T7": gen_T7,
    "T8": gen_T8, "T9": gen_T9, "T10": gen_T10, "T11": gen_T11,
    "T12": gen_T12, "T14": gen_T14,
    "T15": gen_T15, "T16": gen_T16,
    "T17": gen_T17,
    "T19": gen_T19, "T20": gen_T20,
    # ── Group H ───────────────────────────────────────────────────────────
    "T22": gen_T22,
    "T23": gen_T23, "T24": gen_T24,
    "T25": gen_T25, "T26": gen_T26, "T27": gen_T27,
    "T28": gen_T28, "T29": gen_T29,
    "T30": gen_T30, "T31": gen_T31, "T32": gen_T32, "T33": gen_T33,
    "T34": gen_T34, "T35": gen_T35,
    "T36": gen_T36,
}
