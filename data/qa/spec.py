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
#   število pomenov          a count is a fact about the ball, not a node in it.
#   več / manj / enako       Group H's T27 compares two anchors, and a comparison
#       is a relation between two ball facts rather than a third one.
#
# Two checks read this and they must not drift: `qa/check_labels.py` (C25) asks
# whether each label is in the ball, and `qa/check_balls.py` (C18 d) asks whether
# each gold item is -- so a declared label has to be excused from the second as
# well, or the two would disagree about the same eight strings.
DECLARED_LABELS = frozenset({
    "preteklik", "prihodnjik", "besedna vrsta", "vid", "število pomenov",
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
}

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
VALUE_SORTED = ("T4", "T15", "T16", "T17")


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
TIER_C = ("T16",)

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
TIER_C_TAG_WORDS = ("protipomenka", "protipomenke", "protipomenko", "antonim")
TIER_C_SOFT_WORDS = ("nasprotje", "nasproten", "nasprotno", "nasprotna")
TIER_C_LEAK_WORDS = TIER_C_TAG_WORDS + TIER_C_SOFT_WORDS

# Group H adds a second held-out relation (T36, Hungarian translations) and four
# types that take the relation as a SLOT, so a held-out relation can enter
# training as a question's subject rather than as its answer.  C6 is over the
# whole item and already catches that; the tag words go here when T36 lands.


# The template pool a type draws its question from.  T6 splits on whether the
# gender slot is used; T17 splits on the quantity band.
def template_key(type_key, slots):
    if type_key == "T6" and slots.get("SPOL"):
        return "T6G"
    if type_key == "T17":
        return f"T17/{slots['band']}"
    return type_key


def sentinel_line():
    return PREFIX + SENTINEL
