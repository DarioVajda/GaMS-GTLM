#!/usr/bin/env python3
"""The per-type answer contract, in one table.

Under the labelled-pair rule (QA_TASKS.md 0.1) there is no per-type *shape* left
to declare.  Every answer is one line of ` | `-separated `oznaka: vrednost`, so
the `arity`, `sep` and `regex` this table used to key on the type id are gone,
and with them the property that made the corpus learnable as a 19-way classifier:
the question named a type, the type named an output template, and only the slot
fillers ever came from the graph.

What is left per type is genuinely per type:

    name          the type's path name, as the item record reports it
    mode          `pairs` or `membership` -- the only two comparisons left
    keys          the label vocabulary the type's answer may use.  Documentation
                  and the argument of C25, not something the grader reads: which
                  keys an ITEM carries is fixed by its question (0.9), so the
                  type can only say which are admissible.
    member_kind   membership types only: the ball node prefix holding the
                  candidate set, so stage 4 can finalise `all_items`.
    band/n_asked  a default for an item-level field the type fixes for every
                  item (T19 asks for exactly one example).

`qa/pairs.py` owns how a type's cells acquire their labels; this module owns how
they are compared.
"""

SENTINEL = "ni podatka v bazi"
PREFIX = "ODGOVOR: "
GAP = "/"

# ── The declared label budget (QA_TASKS.md 0.1) ────────────────────────────
# The labels no ball can carry, because the generator composes the fact as well
# as the value.  It is a BUDGET, not an exemption: its length is how many labels
# the model must still memorise rather than read, and a ninth cannot be added
# without a diff that shows it.
#
#   preteklik / prihodnjik   M2 -- the KG stores no past or future form, so the
#       value is composed too.  `deležnik na -l` IS in every conjugation ball
#       and would be derived, but it names a participle while the value is a
#       periphrastic form: a true label of the wrong thing.
#   besedna vrsta / vid      the graph renders the VALUES (`samostalnik`,
#       `dovršni`) and never names the property.  `spol` is NOT here: node text
#       says `ženski spol`, so the label is present and only the suffix moves.
#   število                  a count is a fact about the ball, not a node in it.
#       Was the literal `število pomenov` while T14 was the only counting type.
#       T24 counts every complete relation, so the constant is the HEAD and the
#       tail (`sopomenk`, `zgledov`) is the relation's own genitive plural --
#       which `is_declared` already handles, since it matches a constant as one
#       component of a compound label.  The budget is unchanged at eight: this
#       is one constant either way, and writing out six would have been six.
#   več / manj / enako       Group H's T27 compares two anchors, and a comparison
#       is a relation between two ball facts rather than a third one.  Same
#       head-plus-relation shape as `število`.
#
# Two checks read this and they must not drift: `qa/check_labels.py` (C25) asks
# whether each label is in the ball, and `qa/check_balls.py` (C18 d) asks whether
# each gold item is -- so a declared label has to be excused from the second as
# well, or the two would disagree about the same eight strings.
DECLARED_LABELS = frozenset({
    "preteklik", "prihodnjik", "besedna vrsta", "vid", "število",
    "več", "manj", "enako",
})

#: The one shape, replacing 19 regexes (QA_TASKS.md 0.9).  A label may not
#: contain a colon (so the FIRST `: ` always splits the field) and no field may
#: contain the ` | ` that separates them.
SHAPE = r"^ODGOVOR: [^:|]+: [^|]+(?: \| [^:|]+: [^|]+)*$"

# The label vocabularies, named once and shared, because 21 of the 34 types draw
# from six of them (QA_TASKS.md section 1).  That sharing is the point: T3 and
# T21 are indistinguishable in output space, so a model that routes by type gains
# nothing on either.
GRID = "the 18 case x number cells, as `tožilnik ednine`"
CONJ = "the 27 tense x person x number cells, as `preteklik 1. osebe ednine`"
LEXEME = "besedna vrsta, spol, vid"

SPEC = {
    "T1":  dict(name="sklanjanje/celotna_sklanjatev", mode="pairs", keys=GRID),
    "T2":  dict(name="sklanjanje/sklanjatev_po_stevilu", mode="pairs", keys=GRID),
    "T3":  dict(name="sklanjanje/posamezen_sklon", mode="pairs", keys=GRID),
    "T4":  dict(name="sklanjanje/osnovna_oblika_leme", mode="pairs",
                keys="iztočnica"),
    "T5":  dict(name="spreganje/celotno_spreganje", mode="pairs", keys=CONJ),
    "T6":  dict(name="spreganje/spreganje_v_casu", mode="pairs", keys=CONJ),
    "T7":  dict(name="spreganje/neosebne_oblike", mode="pairs",
                keys="nedoločnik, namenilnik, velelnik"),
    "T8":  dict(name="besedna_vrsta/osnovne_lastnosti", mode="pairs", keys=LEXEME),
    "T9":  dict(name="besedna_vrsta/spol_samostalnika", mode="pairs", keys=LEXEME),
    "T10": dict(name="besedna_vrsta/vrsta_in_vid_glagola", mode="pairs", keys=LEXEME),
    "T11": dict(name="stopnjevanje/vse_stopnje", mode="pairs",
                keys="osnovnik, primernik, presežnik"),
    "T12": dict(name="pomen/razlaga_pomena", mode="pairs", keys="pomen N"),
    "T14": dict(name="pomen/stevilka_pomenov", mode="pairs", keys="število pomenov"),
    "T15": dict(name="sopomenke/navedi_sopomenke", mode="pairs", keys="sopomenka"),
    "T16": dict(name="protipomenke/navedi_protipomenke", mode="pairs",
                keys="protipomenka"),
    "T17": dict(name="kolokacije/navedi_kolokacije", mode="membership",
                keys="kolokacija", member_kind="kolokacija"),
    # `membership`, NOT `pairs` -- see MEMBER_KIND below.  The question asks "in
    # what sentence does L appear?" and a lemma commonly has several recorded
    # examples, every one of which answers it; the band pins the count at one.
    "T19": dict(name="primeri_uporabe/povedi_z_besedo", mode="membership",
                keys="zgled", member_kind="zgled", band="exact", n_asked=1),
    "T20": dict(name="primeri_uporabe/analiza_oblike_v_povedi", mode="pairs",
                keys=GRID),
    "T21": dict(name="sklanjanje/analiza_oblike", mode="pairs", keys=GRID),
    # ── Group H ────────────────────────────────────────────────────────────
    # H.1 selection over the paradigm
    "T22": dict(name="izbor/izbrane_oblike", mode="pairs", keys=GRID),
    # H.2 polarity and quantity, with the RELATION as a slot
    "T23": dict(name="preveri/ali_obstaja", mode="pairs",
                keys="the relation's own tag"),
    "T24": dict(name="stevilo/koliko", mode="pairs",
                keys="število <relation, genitive plural>"),
    # H.3 two anchors
    "T25": dict(name="primerjava/ali_sta_v_relaciji", mode="pairs",
                keys="the relation's own tag"),
    "T26": dict(name="primerjava/skupne_lastnosti", mode="pairs", keys=LEXEME),
    "T27": dict(name="primerjava/vec_ali_manj", mode="pairs",
                keys="več/manj/enako <relation, genitive plural>"),
    # H.4 sense-scoped relations
    "T28": dict(name="pomen/relacija_pomena", mode="pairs", keys="sopomenka"),
    "T29": dict(name="pomen/kateri_pomen", mode="pairs", keys="pomen N"),
    # H.5 phrases
    "T30": dict(name="zveze/sestava_zveze", mode="pairs", keys="iztočnica",
                seed_pool="phrase"),
    # `membership` for the reason H.5 gives, which is NOT T17's: the ball holds
    # D5's top-10 by a deterministic content-independent rank rather than a
    # seeded sample, so the set is reproducible and still a SELECTION -- the
    # anchor's true membership reaches 162,004.  `member_strict` is what keeps
    # the allow-list honest: `iztočnica:` is not this relation's own prefix (it
    # is every anchor's), so the ball's word anchors -- the subject included --
    # would otherwise be admitted as answers.  See build_balls.member_pool.
    "T31": dict(name="zveze/zveze_z_besedo", mode="membership", keys="iztočnica",
                member_kind="iztočnica", member_strict=True),
    "T32": dict(name="zveze/pomen_zveze", mode="pairs", keys="pomen N",
                seed_pool="phrase"),
    "T33": dict(name="zveze/zgled_za_zvezo", mode="membership", keys="zgled",
                member_kind="zgled", band="exact", n_asked=1,
                seed_pool="phrase"),
    "T34": dict(name="zveze/dopolni_zvezo", mode="pairs", keys="iztočnica",
                seed_pool="phrase"),
    "T35": dict(name="kolokacije/beseda_kolokacije", mode="pairs",
                keys="iztočnica"),
    # H.6 the second held-out relation
    "T36": dict(name="prevodi/madzarski_prevod", mode="pairs",
                keys="prevod (madžarsko)"),
}

#: Types that are DEFINED but not built.  They keep their spec row, generator,
#: frames and checks -- what they lose is a place in `build_dataset.TYPES`, so no
#: item of theirs reaches a split.  Parking rather than deleting is deliberate:
#: the work is sound and the reason it is out is a property of the pipeline, not
#: of the code, so the code should still be here when that property changes.
#:
#: T34 (`dopolni_zvezo`) is parked because it is the only type whose SUBJECT IS
#: NOT IN ITS QUESTION.  The item shows `sredozemna ___` and is about the phrase
#: `sredozemna medvedjica`, so entity linking -- which sees the question and
#: nothing else -- can reach it only through "every phrase containing
#: *sredozemna*", a hub of up to 212,286 anchors.  Measured: 0 of 57 items
#: resolved; unbounded, the membership hop gives a mean of 11,066 anchors an
#: item; capped at 2,000 memberships it buys 73 % resolution for 373-anchor
#: balls, against a corpus norm of 1-4.  There is no honest retrieval that lands
#: on the one phrase, and building its ball from the seed anchor instead would
#: train the model on a lookup that cannot happen at serving time.
#:
#: What would unpark it is a question shape that NAMES the phrase it asks about.
PARKED = frozenset({"T34"})

#: Membership types whose `member_kind` prefix is NOT specific to their relation,
#: so the ball's nodes under it must be intersected with the store's own truth
#: before they may enter the allow-list.  Only T31: every anchor renders as
#: `iztočnica:`, so without this the phrase question would accept its own subject.
MEMBER_STRICT = frozenset(k for k, v in SPEC.items() if v.get("member_strict"))

#: Types seeded from the PHRASE pool rather than the word pool.  `availability`
#: runs every generator over every seed, and a phrase type over word seeds
#: returns nothing for all of them -- it does not fail, it just reports the type
#: as unavailable, which is the silent kind of wrong.  Naming the pool here is
#: what makes the generation loop ask the right population.
PHRASE_SEEDED = frozenset(k for k, v in SPEC.items()
                          if v.get("seed_pool") == "phrase")

#: T23's slot.  The two held-out relations (`protipomenka` from T16, `prevod
#: (madžarsko)` from T36) are TEST-ONLY -- a question that merely names a held-out
#: relation leaks it whether the answer is `da` or `ne`, which is what C6 checks
#: over the whole item rather than over the answer.
#:
#: Four of the coverage table's seven training relations, not seven.  The three
#: absent ones are absent for reasons, not for want of typing:
#:
#:   `prevod (madžarsko)`  is the OTHER held-out relation, and T36 -- which holds
#:       it out -- has not landed.  Adding it here would put a Tier C slot value
#:       into training with no tag words registered in TIER_C_TAG_WORDS and so no
#:       check watching it.  It goes in with T36, in the same diff as its tag
#:       words.
#:   MWE membership       has no node prefix of its own.  Every other value here
#:       is the literal tag the ball spells (`sopomenka:`, `oblika:`), and
#:       "does this word belong to a phrase" would need a tag invented for the
#:       question -- which is a decision about the label vocabulary (0.1), not a
#:       detail of this type.
#:   `pomen`              cannot answer `ne` truthfully.  Presence would have to
#:       mean a DEFINED sense (what T12 lists), because every core anchor has a
#:       sense node and the answer would otherwise be `da` for all of them -- but
#:       measured over 6,000 pool entries, **100 %** of those with no defined
#:       sense still render a `pomen` node in the ball (`pomen None: Shakespeare`,
#:       a placeholder whose body is the headword).  So every `pomen: ne` ships an
#:       input that appears to contradict it, and the model would be supervised to
#:       ignore a node it can plainly see.  The fix is a label that names the
#:       DEFINITION rather than the sense, which is a §0.1 decision and not this
#:       type's to make.
T23_RELATIONS = ("sopomenka", "zgled", "kolokacija", "oblika")
#: `prevod` joins the held-out side now that T36 has landed and registered its
#: tag words above -- which is the condition the note said it was waiting on.
T23_HELD_OUT = ("protipomenka", "prevod")

# ── T24 / T27: the COUNTABLE relations ─────────────────────────────────────
# Counting grades the model on a total, so the relation has to be COMPLETE in
# the ball.  Two of the coverage table's relations are not:
#
#   `kolokacija`      D5b draws a seeded SAMPLE, so a count over it would grade
#       the model on the sampler's seed rather than on the graph.
#   MWE membership    capped at ten per anchor by D5, and the true membership
#       reaches 162,004 -- which is exactly why T31 is graded `membership`.
#
# That both are listable (T17, T31) but not countable is itself a thing the
# model has to read rather than assume.  Two of the six that remain are held
# out, so training sees four and test sees all six (H.0).
COUNTABLE = ("sopomenka", "zgled", "pomen", "oblika")
COUNTABLE_HELD_OUT = ("protipomenka", "prevod")

#: H.3's relation slot, on the same held-out rule as T23's.
PAIR_RELATIONS = ("sopomenka",)
PAIR_HELD_OUT = ("protipomenka",)

# ── The set rule ───────────────────────────────────────────────────────────
# ORDER IS NOT A PROPERTY OF THIS DATA.  The KG stores the examples of a sense,
# the collocations of a lemma and the synonyms of a word as SETS; the order they
# come out in is an artefact of serialisation (node id, then insertion).
#
# 0.1 settles this for good: nothing is positional any more, so `pairs` is
# order-blind by construction and a type CANNOT be graded on an arbitrary
# tie-break even by accident.  What survives is the narrower question this rule
# was really about -- whether the gold is the whole set or a DRAW from it.  A
# draw cannot be compared against one list, because the ball holds a seeded
# sample and two extractions of the same anchor legitimately support different
# answers; those types are `membership` (0.8.3).
#
# `member_kind` names the ball node prefix holding the candidate set, so stage 4
# can finalise `all_items` from the ball the model is ACTUALLY shown.  That
# matters because D3 unions several anchors into one ball: an allow-list built
# from a single anchor omits phrases the model can see, which is the same defect
# wearing different clothes.  `qa/check_balls.py` asserts the invariant.
MEMBER_KIND = {k: v["member_kind"] for k, v in SPEC.items() if v.get("member_kind")}
SET_VALUED = frozenset(MEMBER_KIND)

#: Types whose gold is canonically ordered by VALUE rather than by a paradigm
#: (C15).  T17 is here for the generated corpus, where its draw is still the
#: sampler's sorted one; stage 4 re-verbalises it from the ball's pool, whose
#: order is the ball's, so C15 is asserted before that stage and not after.
VALUE_SORTED = ("T4", "T15", "T16", "T17", "T30")


def check_set_rule():
    """Raise if a set-valued type is not graded `membership`.  Called by the checkers."""
    bad = [k for k in SET_VALUED if SPEC[k]["mode"] != "membership"]
    if bad:
        raise AssertionError(
            f"types {sorted(bad)} draw their answer from a seeded SAMPLE of a "
            f"set but are graded against one fixed list, so a correct answer "
            f"fails whenever the sample moves. Use `membership` -- see the set "
            f"rule in qa/spec.py.")
    return True


# Tier C: held out of training entirely (D12).  Every item of these types is a
# test item, and no training item anywhere may contain their tag words (C6).
TIER_C = ("T16", "T36")

# Two lists, because two different things can leak and only one of them is ours
# to control.
#
# TAG WORDS are the relation's name.  They identify the task, they appear in no
# curated definition or corpus sentence in this KG, and a training item
# containing one is a genuine leak.  Asserted absent from training, everywhere.
#
# SOFT WORDS are ordinary Slovene for "opposite".  They occur inside real
# lexicographic data -- the KG defines `obratno` as *na nasproten način*, and a
# corpus sentence about a dessert contains *sladko nasprotje* -- so banning them
# from ANSWERS would delete real data to protect against nothing: knowing the
# word `nasprotje` is not knowing that `protipomenka:` nodes exist or what
# question shape asks for them.  They are banned from training QUESTIONS, which
# we author, and reported but allowed in answers.
TIER_C_TAG_WORDS = ("protipomenka", "protipomenke", "protipomenko",
                    # T23 puts the relation name in the QUESTION, where it
                    # declines: the genitive plural is a tag word C6 never had
                    # to catch while the name only ever appeared as a label.
                    "protipomenk", "protipomenki",
                    "antonim")
TIER_C_SOFT_WORDS = ("nasprotje", "nasproten", "nasprotno", "nasprotna")

# T36, the second held-out relation, with the same two-part boundary.
#
# TAG WORDS.  `prevod` and the language name identify the task.  Unlike
# `protipomenka` these are ORDINARY Slovene, so the cost is real and worth
# stating: a training sentence or definition containing `prevod` is dropped, not
# just a question we authored.  Measured on the current corpus before landing
# this, so the size of the loss is known rather than discovered later.
#
# SOFT WORDS are the ordinary verb for "translate", banned from training
# QUESTIONS only, on TIER_C_SOFT_WORDS' reasoning exactly: knowing the word
# `prevesti` is not knowing that `prevod (madžarsko):` nodes exist.
TIER_C_TAG_WORDS += ("prevod", "prevoda", "prevodi", "prevode", "prevodov",
                     "prevodu", "prevodom",
                     "madžarsko", "madžarski", "madžarska", "madžarskem",
                     "madžarščina", "madžarščino", "madžarščini",
                     "madžarščine", "madžarsko-slovenski")
TIER_C_SOFT_WORDS += ("prevesti", "prevede", "prevedi", "prevajati", "prevaja",
                      "prevedeno", "preveden")

TIER_C_LEAK_WORDS = TIER_C_TAG_WORDS + TIER_C_SOFT_WORDS


def tier_c_tagged(text):
    """True if `text` carries a Tier C tag word -- C6's test, as a substring.

    Substring on purpose, and so deliberately coarse: `sprevod` and `neprevodni`
    match `prevod`.  It is the rule C6 enforces, so anything that decides what a
    train/dev answer may contain has to apply this one and not a smarter one.
    """
    t = text.casefold()
    return any(w in t for w in TIER_C_TAG_WORDS)


# The template pool a type draws its question from.  T6 splits on whether the
# gender slot is used; T17 splits on the quantity band.
def template_key(type_key, slots):
    if type_key == "T6" and slots.get("SPOL"):
        return "T6G"
    # The two quantity-banded types.  T31 bands for a different reason than T17
    # (a rank, not a sample -- see its SPEC entry) but asks the same question of
    # the frame pool: how many did the user ask for?
    if type_key in ("T17", "T31"):
        return f"{type_key}/{slots['band']}"
    # T28 names the sense it scopes to.  With a definition it can quote one;
    # without, the ordinal names it and every frame must use that instead --
    # a frame reaching for a slot the item does not carry is dropped silently.
    if type_key == "T28" and not slots.get("POMEN"):
        return "T28/ord"
    return type_key


def sentinel_line():
    return PREFIX + SENTINEL
