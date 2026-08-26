#!/usr/bin/env python3
"""Slovene-language primitives shared by the generator and the grader.

Everything here is a fixed table.  Nothing in this module reads the graph, and
nothing in it depends on a locale: `sl_SI` is not installed on this cluster
(data/QA_TASKS.md 0.8.2), so `locale.strxfrm` would silently fall back to
byte order and make the training data non-canonical without failing anything.
"""
import re
import unicodedata

# --------------------------------------------------------------------------
# collation
# --------------------------------------------------------------------------
_SL = "abcčdefghijklmnoprsštuvzž"
_RANK = {ch: i for i, ch in enumerate(_SL)}


def sl_key(s):
    """Slovene collation key.  Unknown chars sort after the alphabet, by codepoint."""
    return [_RANK.get(ch, len(_SL) + ord(ch)) for ch in s.casefold()]


# --------------------------------------------------------------------------
# the grader's normalization (QA_TASKS.md 0.8) -- deliberately shallow
# --------------------------------------------------------------------------
def norm(s):
    """Fold case, collapse whitespace, drop ONE trailing period.  Nothing else.

    No diacritic stripping (it would merge real Slovene distinctions), no
    stemming, no reordering.
    """
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s).strip().rstrip(".")
    return s.casefold()


# --------------------------------------------------------------------------
# the closed vocabularies the node text uses (builder VALUE_SL / POS_SL)
# --------------------------------------------------------------------------
CASES = ("imenovalnik", "rodilnik", "dajalnik", "tožilnik", "mestnik", "orodnik")
NUMBERS = ("ednina", "dvojina", "množina")
GENDERS = ("moški spol", "ženski spol", "srednji spol")
PERSONS = ("1. oseba", "2. oseba", "3. oseba")
DEGREES = ("osnovnik", "primernik", "presežnik")
ASPECTS = ("dovršni", "nedovršni", "dvovidski")
TENSES = ("sedanjik", "preteklik", "prihodnjik")
VFORMS = ("sedanjik", "prihodnjik", "pogojnik", "velelnik", "deležnik na -l",
          "nedoločnik", "namenilnik")
POS = ("samostalnik", "pridevnik", "glagol", "prislov", "števnik", "medmet",
       "zaimek", "okrajšava", "predlog", "členek", "veznik", "določilnik",
       "lastno ime")

CASE_IDX = {c: i for i, c in enumerate(CASES)}
NUMBER_IDX = {n: i for i, n in enumerate(NUMBERS)}
GENDER_IDX = {g: i for i, g in enumerate(GENDERS)}
PERSON_IDX = {p: i for i, p in enumerate(PERSONS)}

# --------------------------------------------------------------------------
# the METALANGUAGE words decline too
# --------------------------------------------------------------------------
# A question frame does not always want the citation form: "v ednini" is
# locative, "za ednino" accusative, "tožilnik ednine" genitive.  Getting this
# wrong produces ungrammatical Slovene in the training data -- "sklanjatev za
# dvojini" -- which is a defect in the input distribution, not a cosmetic one.
#
# Every case name is a masculine -nik noun and every number name a feminine -a
# noun, so both paradigms are regular and can be written once rather than
# tabulated by hand.


def _decline_masc_ik(w):
    return {"nom": w, "gen": w + "a", "dat": w + "u", "acc": w,
            "loc": w + "u", "ins": w + "om"}


def _decline_fem_a(w):
    s = w[:-1]
    return {"nom": w, "gen": s + "e", "dat": s + "i", "acc": s + "o",
            "loc": s + "i", "ins": s + "o"}


CASE_FORMS = {c: _decline_masc_ik(c) for c in CASES}
NUMBER_FORMS = {n: _decline_fem_a(n) for n in NUMBERS}
TENSE_FORMS = {t: _decline_masc_ik(t) for t in TENSES}

# shorthands used by the templates
NUMBER_GEN = {n: f["gen"] for n, f in NUMBER_FORMS.items()}
NUMBER_LOC = {n: f["loc"] for n, f in NUMBER_FORMS.items()}
NUMBER_ACC = {n: f["acc"] for n, f in NUMBER_FORMS.items()}
CASE_LOC = {c: f["loc"] for c, f in CASE_FORMS.items()}
CASE_ACC = {c: f["acc"] for c, f in CASE_FORMS.items()}
TENSE_LOC = {t: f["loc"] for t, f in TENSE_FORMS.items()}
TENSE_ACC = {t: f["acc"] for t, f in TENSE_FORMS.items()}
GENDER_LOC = {"moški spol": "moškem spolu", "ženski spol": "ženskem spolu",
              "srednji spol": "srednjem spolu"}
GENDER_ADJ = {"moški spol": "moški", "ženski spol": "ženski",
              "srednji spol": "srednji"}

# ordinal of a case, as the reference CSV writes it ("4. Tožilnik sklon")
CASE_ORD = {c: i + 1 for i, c in enumerate(CASES)}


# --------------------------------------------------------------------------
# number agreement -- the reference file gets this wrong in 100/100 rows
# --------------------------------------------------------------------------
def pomen_agreement(n, frame="baza"):
    """Slovene agreement for a count of `pomen`.

    QA_TASKS.md T14: the verb and the participle agree too, not only the noun,
    so this is a table lookup on n mod 100 returning the WHOLE phrase, never a
    heuristic on the noun alone.

    Nothing emits this yet -- the gradeable line is the bare integer and the UI
    part is postponed (0.1).  It is kept and unit-tested (C7) because the UI
    layer will need it unchanged.
    """
    m = _slot(n)
    if frame == "baza":
        return {1: f"v bazi je zabeležen {n} pomen",
                2: f"v bazi sta zabeležena {n} pomena",
                3: f"v bazi so zabeleženi {n} pomeni",
                0: f"v bazi je zabeleženih {n} pomenov"}[m]
    if frame == "ima":                      # accusative frame
        return {1: f"ima {n} pomen", 2: f"ima {n} pomena",
                3: f"ima {n} pomene", 0: f"ima {n} pomenov"}[m]
    raise ValueError(frame)


def _slot(n):
    """1 = singular, 2 = dual, 3 = plural nominative, 0 = genitive plural.

    The rule is on n mod 10, with the teens as the exception -- 21 takes the
    singular (`21 pomen`) and 11 does not (`11 pomenov`).  QA_TASKS.md T14 states
    it as a lookup on n mod 100, which is right for the teens and wrong for
    everything above 20; T14's counts never exceed 36 in this KG, so the two
    agree on every value the dataset actually emits, but the correct rule costs
    nothing and the UI layer will use this function on arbitrary counts.
    """
    if 11 <= n % 100 <= 14:
        return 0
    r = n % 10
    if r == 1:
        return 1
    if r == 2:
        return 2
    if r in (3, 4):
        return 3
    return 0


# --------------------------------------------------------------------------
# counted noun phrases -- "2 kolokaciji", "3 kolokacije", "5 kolokacij"
# --------------------------------------------------------------------------
# T17's `exact` band puts a number in the QUESTION, and Slovene makes the noun
# agree with it: dual after 2, nominative plural after 3-4, genitive plural from
# 5 on.  The reference file writes "<N> kolokacij" for every N, which is right
# only from 5 up -- the same class of error T14 flags in 100/100 of its rows,
# and one the model would learn to imitate if it appeared in the question.
#
# Only N in 2..10 is generated, so each phrase needs three forms: dual, plural,
# genitive plural.
COUNTED = {
    "kolokacija":            ("kolokaciji", "kolokacije", "kolokacij"),
    "besedna zveza":         ("besedni zvezi", "besedne zveze", "besednih zvez"),
    "zveza":                 ("zvezi", "zveze", "zvez"),
    "kolokativna zveza":     ("kolokativni zvezi", "kolokativne zveze",
                              "kolokativnih zvez"),
    "kolokativni primer":    ("kolokativna primera", "kolokativne primere",
                              "kolokativnih primerov"),
    "tipična besedna zveza": ("tipični besedni zvezi", "tipični besedni zvezi",
                              "tipičnih besednih zvez"),
    "sopojavitev":           ("sopojavitvi", "sopojavitve", "sopojavitev"),
    "zabeležena sopojavitev": ("zabeleženi sopojavitvi", "zabeležene sopojavitve",
                               "zabeleženih sopojavitev"),
    "pogosta kolokacija":    ("pogosti kolokaciji", "pogoste kolokacije",
                              "pogostih kolokacij"),
}


def counted(n, noun):
    """'2 kolokaciji' / '3 kolokacije' / '5 kolokacij'."""
    dual, plural, genitive = COUNTED[noun]
    if n == 2:
        return f"{n} {dual}"
    if n in (3, 4):
        return f"{n} {plural}"
    return f"{n} {genitive}"


# There is deliberately no misspelling generator here: negatives of flavour (a)
# are real words the lookup cannot reach, drawn by qa/unlisted.py, which carries
# the reasoning.
