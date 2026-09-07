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
from .store import K_SYN, K_ANT

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
NOMINAL_SLOTS = {"case", "number"}
# Only these decline by case and number ALONE in this KG.  Adjectives,
# participles and most pronouns decline by gender too, so a (case, number) cell
# would hold three surfaces and a positional 18-cell line would not be well
# defined -- QA_TASKS.md T1 reaches the same place through its doublet filter;
# this reaches it through the reason behind the filter.  The cost is small:
# the entire KG holds 2,434 numeral and 284 pronoun entries against 310,362 nouns.
NOMINAL_POS = {"samostalnik", "števnik", "zaimek"}


def nominal_grid(store, a, require_healthy=True):
    """{(case, number): {surface, ...}} over forms carrying case+number ONLY.

    Empty unless the paradigm is actually inflected -- see `healthy_grid`.  The
    check lives here rather than at the five call sites so that no type can
    forget it: a lemma-filled paradigm is wrong gold for T1/T2/T3 (a table of one
    repeated surface), wrong gold for T21 (an eighteen-way disjunction) and, worst
    of all, *plausible* gold for T20.
    """
    if store.pos(a) not in NOMINAL_POS:
        return {}
    grid = collections.defaultdict(set)
    for _v, s, f in store.forms(a):
        if set(f) != NOMINAL_SLOTS:
            continue
        grid[(f["case"], f["number"])].add(s)
    if not grid:
        return {}
    # The lemma form is MERGED INTO THE ANCHOR by the builder, so it has no
    # `oblika:` leaf of its own -- without this the nominative singular of every
    # noun comes out as a gap.  The anchor's own parenthetical names the cell it
    # is rendered in (`iztočnica: gora (samostalnik, ženski spol, imenovalnik,
    # ednina)`), so the cell is read off the node, not assumed.  Guarded on the
    # grid being non-empty, which is the evidence that this entry really does
    # inflect by case and number alone.
    surface = store.lemma(a)
    feat = store.anchor_features(a)
    if surface and feat.get("case") and feat.get("number") \
            and not ({"degree", "definiteness", "vform"} & set(feat)):
        grid[(feat["case"], feat["number"])].add(surface)
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
    """
    if not grid:
        return False
    sg = grid.get(("imenovalnik", "ednina"))
    pl = grid.get(("imenovalnik", "množina"))
    if sg and pl and sg == pl:
        return False
    return len({s for v in grid.values() for s in v}) >= MIN_DISTINCT_FORMS


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
    grid = nominal_grid(ctx.store, e.a)
    cells, filled = nominal_line(grid)
    if cells is None or filled < 12:
        return None
    return {"L": e.lemma}, cells


def gen_T2(ctx, e, rng):
    grid = nominal_grid(ctx.store, e.a)
    number = rng.choice(list(sl.NUMBERS))
    cells, filled = nominal_line(grid, numbers=(number,))
    if cells is None or filled < 5:
        return None
    cells = list(cells)
    cells[0] = f"{number}: {cells[0]}"
    return {"L": e.lemma, "number": number, **number_slots(number)}, cells


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
    grid = nominal_grid(ctx.store, e.a)
    # Sample the cell uniformly over the 18, not from the reference file's
    # nominative/singular-heavy distribution -- the cheapest robustness win here.
    cells = [(c, n) for n in sl.NUMBERS for c in sl.CASES
             if _cell(grid, c, n) not in (None, False)]
    if not cells:
        return None
    c, n = cells[rng.randrange(len(cells))]
    return ({"L": e.lemma, "case": c, "number": n,
             **case_slots(c), **number_slots(n)},
            [_cell(grid, c, n)])


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


def _readings(store, a, surface):
    """The (case, number) readings a surface carries inside ONE entry."""
    out = set()
    for _v, s, f in store.forms(a):
        if s.casefold() == surface.casefold() and set(f) == NOMINAL_SLOTS:
            out.add((f["case"], f["number"]))
    return out


def _reading_items(readings):
    """`tožilnik ednine ali orodnik ednine`, canonically ordered."""
    order = sorted(readings, key=lambda r: (sl.CASE_IDX[r[0]], sl.NUMBER_IDX[r[1]]))
    return [f"{c} {sl.NUMBER_GEN[n]}" for c, n in order]


def gen_T21(ctx, e, rng, want_ambiguous=True):
    grid = nominal_grid(ctx.store, e.a)
    by_surface = collections.defaultdict(set)
    for (c, n), ss in grid.items():
        for s in ss:
            by_surface[s].add((c, n))
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
    grid = nominal_grid(store, e.a)
    by_surface = collections.defaultdict(set)
    for (c, n), ss in grid.items():
        for s in ss:
            by_surface[s].add((c, n))
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
                if unambiguous[s] != {("imenovalnik", "ednina")}]
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


GENERATORS = {
    "T1": gen_T1, "T2": gen_T2, "T3": gen_T3, "T4": gen_T4, "T21": gen_T21,
    "T5": gen_T5, "T6": gen_T6, "T7": gen_T7,
    "T8": gen_T8, "T9": gen_T9, "T10": gen_T10, "T11": gen_T11,
    "T12": gen_T12, "T14": gen_T14,
    "T15": gen_T15, "T16": gen_T16,
    "T17": gen_T17,
    "T19": gen_T19, "T20": gen_T20,
    "T23": gen_T23, "T30": gen_T30,
}
