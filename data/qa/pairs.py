#!/usr/bin/env python3
"""The labelled-pair rule (QA_TASKS.md 0.1), as the one function that applies it.

Every answer in this corpus is one line of ` | `-separated `oznaka: vrednost`,
and this module is the single place where a type's cells acquire their labels.
Three callers share it -- `qa/build_dataset.py` when an item is first made,
`qa/build_balls.py` when a membership item is re-verbalised against its ball, and
`qa/migrate_pairs.py` when the pre-reformat corpus is converted -- so the three
cannot drift into three spellings of the same label.

**Input is the generator's positional list, unchanged.**  `to_pairs` takes what
`qa/gen.py` already returns (and what the old `gold_items` already holds on
disk): the cells in canonical order, with the block leader glued onto the first
cell of a group for T2/T5/T6 and the label glued on for T7.  That is deliberate:
it makes the conversion of the existing corpus and the generation of a new one
*the same code path*, so the migration proves the production path rather than
merely resembling it.

**The label is positional here, and read off the graph later.**  For T1 the true
label of cell 4 is the `case`/`number` of the form leaf that filled it, and the
generator has that bundle in hand; it walked the grid in a fixed order and threw
the bundle away.  Reconstructing it from the position is exactly equivalent --
the position *was* the bundle -- and it is what lets this land without touching
nineteen generators.  When the generators are rewritten to emit pairs natively
they should pass the bundle instead, and the labels must not change: this
module's output is the fixture that says so.

**Gaps disappear.**  A cell the KG does not fill was `/` in a positional line, to
keep the following cells addressable (0.9).  Under 0.1 a missing cell is an
absent key, so `/` is dropped rather than emitted -- 16 cells across T7 and T11
in the current corpus.  An item with nothing at all is the 0.2 sentinel, which
never reaches this module.
"""
from . import sl
from .spec import GAP, PREFIX, SENTINEL

#: T8's second cell is a gender for a noun and an aspect for a verb, and the
#: label differs (`spol` vs `vid`) while the position does not.  `spol` also
#: sheds the noun -- node text says `ženski spol`, so `spol: ženski` is the pair
#: whose label is in the ball and whose value is not a repetition of it.
_ASPECTS = frozenset(sl.ASPECTS)


def _property_pair(value):
    """`ženski spol` -> (`spol`, `ženski`);  `dovršni` -> (`vid`, `dovršni`)."""
    if value in _ASPECTS:
        return ["vid", value]
    if value.endswith(" spol"):
        return ["spol", value[:-len(" spol")]]
    raise ValueError(f"not a lexeme property: {value!r}")


def _split_leader(cell):
    """`množina: tečnosti` -> (`množina`, `tečnosti`).  The leader is the group's."""
    label, sep, value = cell.partition(": ")
    if not sep:
        raise ValueError(f"expected a block leader in {cell!r}")
    return label, value


def _tense_of(leader):
    """`preteklik (ženski spol)` -> (`preteklik`, `ženski spol`)."""
    if leader.endswith(")") and " (" in leader:
        tense, _, gender = leader.partition(" (")
        return tense, gender[:-1]
    return leader, None


def _person_pairs(tense, cells, gender=None):
    return [[sl.person_label(tense, p, n, gender), v]
            for (p, n), v in zip(sl.PERSON_CELLS, cells)]


def to_pairs(type_key, slots, items):
    """The item's gold as `[[oznaka, vrednost], ...]`, in canonical order.

    `items` is the generator's positional list; `slots` is the item's slot dict,
    which carries the question's own scope (which case T3 asked for, which form
    T20/T21 is analysing).  Raises on a type it does not know: a new type must
    declare how its answer is labelled, and inheriting some default silently is
    how a per-type output contract grows back.
    """
    slots = slots or {}
    fn = _BY_TYPE.get(type_key)
    if fn is None:
        raise KeyError(
            f"{type_key} has no labelling rule in qa/pairs.py. Every type's "
            f"answer is labelled pairs (QA_TASKS.md 0.1); add the rule here "
            f"rather than letting the type fall back to a positional line.")
    out = [[str(k), str(v)] for k, v in fn(slots, list(items))]
    return [p for p in out if p[1] != GAP]


# --------------------------------------------------------------------------
# one rule per type
# --------------------------------------------------------------------------
def _t1(slots, items):
    return list(zip(sl.GRID_LABELS, items))


def _t2(slots, items):
    number, items[0] = _split_leader(items[0])
    return [[sl.cell_label(c, number), v] for c, v in zip(sl.CASES, items)]


def _t3(slots, items):
    return [[sl.cell_label(slots["case"], slots["number"]), items[0]]]


def _t4(slots, items):
    return [["iztočnica", v] for v in items]


def _analysis(slots, items):
    """T20/T21: the reading IS the label and the analysed form is the value.

    `imenovalnik dvojine: fermentorja`.  The old line named the readings and left
    the form implicit in the question; under 0.1 the pair says which cell the
    form fills, which is the same fact in the shape every other type uses -- and
    it is what makes T3 and T21 indistinguishable in output space (section 1).
    """
    return [[reading, slots["F"]] for reading in items]


def _t5(slots, items):
    out = []
    for i, tense in enumerate(sl.TENSES):
        cells = items[9 * i:9 * (i + 1)]
        leader, cells[0] = _split_leader(cells[0])
        if leader != tense:
            raise ValueError(f"T5 group {i} is led by {leader!r}, not {tense!r}")
        out += _person_pairs(tense, cells)
    return out


def _t6(slots, items):
    leader, items[0] = _split_leader(items[0])
    tense, gender = _tense_of(leader)
    return _person_pairs(tense, items, gender)


def _t7(slots, items):
    return [_split_leader(c) for c in items]


def _t8(slots, items):
    return [["besedna vrsta", items[0]]] + [_property_pair(v) for v in items[1:]]


def _t9(slots, items):
    return [_property_pair(items[0])]


def _t10(slots, items):
    return [["vid", items[0]]]


def _t11(slots, items):
    return list(zip(sl.DEGREES, items))


def _t12(slots, items):
    """The generator glues the graph's own ordinal on, exactly as T7 does.

    Never `enumerate`: the ordinal is the graph's, and a 1..n renumbering makes
    the label name a node that is not there -- see `qa/gen.py:gen_T12`.
    """
    return [_split_leader(c) for c in items]


def _t14(slots, items):
    return [["število pomenov", items[0]]]


def _t23(slots, items):
    """`sopomenka: da`.  The relation asked about is the label, `da`/`ne` the value.

    The label cannot come from `slots` alone: the generator glues it on exactly as
    T7 and T12 do, so the pair survives the round trip through `gold_items` on disk
    without this module having to know which slot name each frame used.

    `ne` is an ANSWER, not the 0.2 sentinel.  The sentinel means the question
    cannot be answered from the database; `ne` means it can, and the answer is
    that the relation is absent -- which is the only fact this type teaches.
    """
    out = [_split_leader(c) for c in items]
    bad = [v for _, v in out if v not in ("da", "ne")]
    if bad:
        raise ValueError(f"T23 answers `da` or `ne`, got {bad!r}")
    return out


def _tagged(tag):
    return lambda slots, items: [[tag, v] for v in items]


_BY_TYPE = {
    "T1": _t1, "T2": _t2, "T3": _t3, "T4": _t4, "T21": _analysis,
    "T5": _t5, "T6": _t6, "T7": _t7,
    "T8": _t8, "T9": _t9, "T10": _t10, "T11": _t11,
    "T12": _t12, "T14": _t14,
    "T15": _tagged("sopomenka"), "T16": _tagged("protipomenka"),
    "T17": _tagged("kolokacija"), "T19": _tagged("zgled"),
    "T20": _analysis,
    # Group H.  T30 shares T4's rule and its label: a phrase's constituents are
    # headwords, named the way every other headword in this corpus is named.
    "T23": _t23, "T30": _t4,
}


# --------------------------------------------------------------------------
def render(pairs):
    """The gradeable line.  The inverse of `qa/grade.py:parse`, up to `norm`."""
    return PREFIX + " | ".join(f"{k}: {v}" for k, v in pairs)


def answer_of(type_key, slots, items):
    """(line, pairs) for a positive item -- what `make_item` writes."""
    p = to_pairs(type_key, slots, items)
    return render(p), p


def is_sentinel(line):
    return line.strip() == (PREFIX + SENTINEL)
