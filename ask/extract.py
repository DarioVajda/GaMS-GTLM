#!/usr/bin/env python3
"""Stage 1 -- question in, the strings the extractor names out (D7).

The extractor is an ordinary instruct model (today `cjvt/GaMS3-12B-Instruct`,
`data/analysis/measure_extraction.py:DEFAULT_MODEL`) prompted to return a Python
list of the words the question is about.  This module owns exactly two things
the measurement script does not: the retry policy, and the shape of the value
handed downstream.

**One repair turn, and only for unparseable output.**  Measured over the 12,490
dataset items: 95.48 % resolved, 4.52 % parsed but did not resolve, 0.01 %
(one item) unparseable, 0 % an empty list.  The one failure was a repetition
loop truncated by `max_new_tokens`, leaving an unterminated `[` that `LIST_RE`
cannot salvage -- exactly what a second turn fixes.  A *resolution* failure is
never retried: asking again for a word the base does not hold would reward
inventing one that resolves, and D6 already has a truthful answer for that case.
"""
from dataclasses import dataclass, field

# The retry, verbatim.  `{output}` is the model's unparseable reply, `{question}`
# the original question.  Slovene because the extractor is prompted in Slovene.
REPAIR_PROMPT = """Tvoj prejšnji odgovor ni bil v zahtevani obliki:
{output}

Poskusi znova. Vrni samo seznam nizov v Python sintaksi, brez razlage in brez
dodatnega besedila. Uporabi dvojne narekovaje, ne enojnih. Vsako besedo navedi
samo enkrat in seznam zaključi z oglatim zaklepajem.

Primeri pravilne oblike:
["avto"]
["avto", "pes"]
["črni pes"]

VPRAŠANJE:
{question}"""


class ExtractionError(RuntimeError):
    """The extractor's output could not be parsed, twice.  Caught per D19."""


@dataclass
class Extraction:
    """What the extractor named, plus how hard it was to get."""
    strings: list
    raw: str = ""                 # the model's reply, untrimmed
    repaired: bool = False        # did the repair turn fire?
    attempts: list = field(default_factory=list)   # raw text of each turn


def extract(question, extractor, ui):
    """Run the extractor over `question`, repairing once if it must.

    `extractor` is the loaded model bundle from `backbone.load_extractor()`;
    parsing is `measure_extraction.parse()`, imported rather than reimplemented,
    so the serving path and the corpus that trained it cannot drift.
    """
    raise NotImplementedError(
        "ask.extract.extract is a stub; run with --demo, or see build step 5 "
        "in ask/PLAN.md")
