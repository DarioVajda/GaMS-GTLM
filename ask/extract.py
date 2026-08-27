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


def given(strings):
    """The extraction a caller supplied directly (`--words`), not a model's.

    Same shape as a real one, so nothing downstream has to know the difference,
    and `attempts` stays empty -- there was no turn to record.
    """
    return Extraction(strings=[s for s in strings if s], raw="", repaired=False)


#: The corpus's own extraction run used 64.  The repair turn gets more, because
#: the one unparseable output in 12,490 items was a repetition loop that this
#: cap truncated -- retrying under the same cap would truncate it again.
MAX_NEW_TOKENS = 64
REPAIR_NEW_TOKENS = 128


def extract(question, extractor, ui=None):
    """Run the extractor over `question`, repairing once if it must.

    `extractor` is the loaded bundle from `backbone.load_extractor()`; the
    prompt template, the decoding and the parsing are all
    `analysis.measure_extraction`'s, imported rather than reimplemented, so the
    serving path and the corpus that trained the model cannot drift apart.

    The repair turn announces itself while it is running (D20) -- only this
    function knows the moment, and a warning printed after the fact reads as
    though the tool discovered the problem too late to act on it.
    """
    from analysis.measure_extraction import parse, generate_with

    prompt = extractor["template"].replace("{question}", question)
    raw = generate_with(extractor, [prompt], 1, MAX_NEW_TOKENS,
                        progress=False)[0]
    got, ok = parse(raw)
    if ok:
        return Extraction(strings=got, raw=raw, attempts=[raw])

    if ui is not None:
        ui.warn("ekstraktor ni vrnil veljavnega seznama; poskušam znova")
        ui.relabel("luščim iztočnice (2/2)…")
    again = generate_with(
        extractor,
        [REPAIR_PROMPT.format(output=raw.strip(), question=question)],
        1, REPAIR_NEW_TOKENS, progress=False)[0]
    got, ok = parse(again)
    if not ok:
        raise ExtractionError(
            f"the extractor returned nothing parseable, twice.\n"
            f"  first:  {raw.strip()[:200]!r}\n"
            f"  repair: {again.strip()[:200]!r}")
    return Extraction(strings=got, raw=again, repaired=True,
                      attempts=[raw, again])
