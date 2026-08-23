#!/usr/bin/env python3
"""The per-type answer contract, in one table.

QA_TASKS.md 0.9 says the Output template block is the implementation contract:
the generator emits exactly this shape and the grader parses exactly this shape.
This module IS that contract, in the form the code reads, so the two cannot be
edited apart.

`sep` may be a list, for the types whose line is grouped ("...; ...").  Splitting
on any of the listed separators recovers a flat positional list, which is what
the grader compares; the grouping stays for the reader.  A labelled block leader
("sedanjik: delam") is compared as one item, so emitting the wrong tense label is
wrong -- which is correct.
"""

SENTINEL = "ni podatka v bazi"
PREFIX = "ODGOVOR: "
GAP = "/"

# type -> contract
SPEC = {
    "T1":  dict(name="sklanjanje/celotna_sklanjatev", arity=18,
                sep=["; ", ", "], mode="sequence",
                regex=r"^ODGOVOR: (?:[^,;]+(?:, [^,;]+){5}; ){2}[^,;]+(?:, [^,;]+){5}$"),
    "T2":  dict(name="sklanjanje/sklanjatev_po_stevilu", arity=6,
                sep=", ", mode="sequence",
                regex=r"^ODGOVOR: (ednina|dvojina|množina): [^,]+(?:, [^,]+){5}$"),
    "T3":  dict(name="sklanjanje/posamezen_sklon", arity=1,
                sep=None, mode="sequence", regex=r"^ODGOVOR: \S.*$"),
    "T4":  dict(name="sklanjanje/osnovna_oblika_leme", arity=None,
                sep=", ", mode="multiset", regex=r"^ODGOVOR: [^,]+(?:, [^,]+)*$"),
    "T5":  dict(name="spreganje/celotno_spreganje", arity=27,
                sep=["; ", ", "], mode="sequence",
                regex=(r"^ODGOVOR: sedanjik: (?:[^,;]+, ){8}[^,;]+; "
                       r"preteklik: (?:[^,;]+, ){8}[^,;]+; "
                       r"prihodnjik: (?:[^,;]+, ){8}[^,;]+$")),
    "T6":  dict(name="spreganje/spreganje_v_casu", arity=9,
                sep=", ", mode="sequence",
                regex=(r"^ODGOVOR: (sedanjik|preteklik|prihodnjik)"
                       r"(?: \((moški|ženski|srednji) spol\))?: [^,]+(?:, [^,]+){8}$")),
    "T7":  dict(name="spreganje/neosebne_oblike", arity=3,
                sep="; ", mode="sequence",
                regex=(r"^ODGOVOR: nedoločnik: [^;]+; namenilnik: [^;]+; "
                       r"velelnik: [^;]+$")),
    "T8":  dict(name="besedna_vrsta/osnovne_lastnosti", arity=None,
                sep=", ", mode="sequence",
                regex=(r"^ODGOVOR: (samostalnik|pridevnik|glagol|prislov|števnik"
                       r"|medmet|zaimek|okrajšava|predlog|členek|veznik"
                       r"|določilnik|lastno ime)"
                       r"(?:, (moški spol|ženski spol|srednji spol|dovršni"
                       r"|nedovršni|dvovidski))?$")),
    "T9":  dict(name="besedna_vrsta/spol_samostalnika", arity=1,
                sep=None, mode="sequence",
                regex=r"^ODGOVOR: (moški|ženski|srednji) spol$"),
    "T10": dict(name="besedna_vrsta/vrsta_in_vid_glagola", arity=1,
                sep=None, mode="sequence",
                regex=r"^ODGOVOR: (dovršni|nedovršni|dvovidski)$"),
    "T11": dict(name="stopnjevanje/vse_stopnje", arity=3,
                sep=", ", mode="sequence",
                regex=r"^ODGOVOR: [^,]+, [^,]+, [^,]+$"),
    "T12": dict(name="pomen/razlaga_pomena", arity=None,
                sep=" | ", mode="sequence", regex=r"^ODGOVOR: .+(?: \| .+)*$"),
    "T14": dict(name="pomen/stevilka_pomenov", arity=1,
                sep=None, mode="sequence", regex=r"^ODGOVOR: [1-9][0-9]*$"),
    "T15": dict(name="sopomenke/navedi_sopomenke", arity=None,
                sep=", ", mode="multiset",
                regex=r"^ODGOVOR: [^,]+(?:, [^,]+){0,9}$"),
    "T16": dict(name="protipomenke/navedi_protipomenke", arity=None,
                sep=", ", mode="multiset",
                regex=r"^ODGOVOR: [^,]+(?:, [^,]+){0,9}$"),
    "T17": dict(name="kolokacije/navedi_kolokacije", arity=None,
                sep=" | ", mode="membership", regex=r"^ODGOVOR: .+(?: \| .+)*$",
                member_kind="kolokacija"),
    # `membership`, NOT `sequence` -- see MEMBER_KIND below.  The question asks
    # "in what sentence does L appear?" and a lemma commonly has several recorded
    # examples, every one of which answers it; the band pins the count at one.
    "T19": dict(name="primeri_uporabe/povedi_z_besedo", arity=1,
                sep=None, mode="membership", regex=r"^ODGOVOR: \S.*$",
                member_kind="zgled", band="exact", n_asked=1),
    "T20": dict(name="primeri_uporabe/analiza_oblike_v_povedi", arity=None,
                sep=" ali ", mode="multiset",
                regex=(r"^ODGOVOR: \w+ (ednine|dvojine|množine)"
                       r"(?: ali \w+ (?:ednine|dvojine|množine))*$")),
    "T21": dict(name="sklanjanje/analiza_oblike", arity=None,
                sep=" ali ", mode="multiset",
                regex=(r"^ODGOVOR: \w+ (ednine|dvojine|množine)"
                       r"(?: ali \w+ (?:ednine|dvojine|množine))*$")),
}

# ── The set rule ───────────────────────────────────────────────────────────
# ORDER IS NOT A PROPERTY OF THIS DATA.  The KG stores the examples of a sense,
# the collocations of a lemma and the synonyms of a word as SETS; the order they
# come out in is an artefact of serialisation (node id, then insertion).
#
# So a type whose answer is DRAWN FROM a set must be graded `membership` (pick
# some of them) or `multiset` (name all of them, order-blind) -- never
# `sequence`.  Grading such a type on order rewards an architecture for
# preserving an accident of file layout and penalises one for discarding it, and
# discarding it is *permutation equivariance*, which a graph model is supposed to
# have.  `sequence` is correct only where position carries meaning the data
# itself defines: a paradigm cell, a tense, a dictionary sense ordinal.
#
# T19 violated this until 2026-08-23 -- it was `sequence`/arity-1 against
# whichever example had the lowest node id, which cost the GTLM arm 2.44 points
# of a 4-arm comparison before anyone noticed (see train/README.md).
#
# `member_kind` names the ball node prefix holding the candidate set, so stage 4
# can finalise `all_items` from the ball the model is ACTUALLY shown.  That
# matters because D3 unions several anchors into one ball: an allow-list built
# from a single anchor omits phrases the model can see, which is the same defect
# wearing different clothes.  `qa/check_balls.py` asserts the invariant.
MEMBER_KIND = {k: v["member_kind"] for k, v in SPEC.items() if v.get("member_kind")}
SET_VALUED = frozenset(MEMBER_KIND)


def check_set_rule():
    """Raise if a set-valued type is graded on order.  Called by the checkers."""
    bad = [k for k in SET_VALUED if SPEC[k]["mode"] == "sequence"]
    if bad:
        raise AssertionError(
            f"types {sorted(bad)} draw their answer from a SET but are graded "
            f"`sequence`, i.e. on an arbitrary tie-break. Use `membership` or "
            f"`multiset` -- see the set rule in qa/spec.py.")
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
