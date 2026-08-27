#!/usr/bin/env python3
"""Stage 2 -- the extractor's strings in, the ball the model reads out (D6, D9).

The chain is `strings -> surface_index() -> anchor indices -> build()`.  Every
piece of it is imported from `data/qa/`, not reimplemented: `build_balls.build()`
is already the reusable seam the dataset builder itself calls, so a served ball
and a trained ball differ only in which store they open.

**Two node identities live here, and mixing them is a silent corruption.**
`surface_index()` returns *anchor indices* -- positions in this store's arrays,
which change whenever the store is rebuilt.  `build()` consumes indices.  Node
*codes* are durable identity, are what `relabel.py` persists, and are what
leaves this module in `Ball.codes` for the JSON report.  Inside `ask` the chain
never round-trips through codes; the dataset only does because it writes targets
to disk.

**A miss is retrieved, not refused.**  When nothing resolves, the ball is the
single node `iztočnica: <beseda> (ni v bazi)` -- the same truthful input
`build_balls.py:368` builds for the 445 dataset items in that position, and the
input the sentinel answer was trained against.
"""
from dataclasses import dataclass, field

MISSING_MARK = "(ni v bazi)"

# `qa` and `lib` are imported inside the functions, not at module scope: the
# dataclasses below are what `demo.py` needs, and `--demo` has to keep running
# with nothing but the standard library on the path.


@dataclass
class Ball:
    """The retrieved neighbourhood, in the shape `train/data.py:_graph` wants."""
    texts: list                       # node texts, store order
    edges: list                       # (src, dst) over positions in `texts`
    anchors: list                     # positions in `texts` of the matched units
    codes: list = field(default_factory=list)      # durable ids of the anchors
    strings: list = field(default_factory=list)    # what resolved to them
    n_tokens: int = 0                 # sum over nodes, ~4 chars/token estimate

    @property
    def missing(self):
        return len(self.texts) == 1 and MISSING_MARK in self.texts[0]

    def anchor_texts(self):
        return [self.texts[a] for a in self.anchors]


def resolve(strings, index):
    """Casefolded strings -> the anchor INDICES they name, sorted and unique.

    Union of every match, per the dataset's D3 -- 3.8 % of the 911,404 surfaces
    map to more than one anchor, and training saw all of them.  Sorted because
    `build()` promises a ball that is a pure function of (store, targets) and
    not of the order the extractor happened to return its strings in.
    """
    return sorted({int(a) for s in strings
                   for a in index.get(s.casefold(), ())})


def missing_ball(strings):
    """The one-node ball for "nothing resolved" (D6).

    Built from the extractor's own string, which is the only word serving has --
    see V7 in ask/PLAN.md for why that is faithful to what training saw rather
    than a divergence from it.  Several unresolved strings still make ONE node:
    the dataset's miss case never had more than one, and inventing a shape the
    model has not seen is worse than dropping a word from a node text nobody
    could resolve anyway.  The full list survives on `Ball.strings`.
    """
    word = strings[0] if strings else "beseda"
    text = f"iztočnica: {word} {MISSING_MARK}"
    # `len // 4` is `build_balls.py:372`'s own estimate for this node, kept
    # identical so a served miss reports the token count training recorded.
    return Ball(texts=[text], edges=[], anchors=[0], codes=[],
                strings=list(strings), n_tokens=len(text) // 4)


def retrieve(strings, store, ui=None):
    """Resolve `strings` against `store` and build the union ball.

    `store` is the bundle `backbone.load_store` returns: the `QAStore`, its
    surface index, and a ball cache that lives as long as the session -- one
    anchor's hop-2 expansion is the expensive part and asking twice about the
    same word is the normal case in an interactive shell.
    """
    from qa.build_balls import build, K_MWE, K_COLLOC_CAP

    qs, index = store["store"], store["index"]
    anchors = resolve(strings, index)
    if not anchors:
        return missing_ball(strings)

    cache, stats = store["cache"], store["stats"]
    texts, edges, positions = build(qs, anchors, K_MWE, K_COLLOC_CAP,
                                    cache, stats)
    # The same node set `build` sorted into `texts`, which is what the corpus
    # counted tokens over (`build_balls.py:365`).
    tok = store["token_len"]
    n_tokens = sum(int(tok[n]) for n in {x for a in anchors for x in cache[a]})
    return Ball(texts=texts, edges=edges, anchors=positions,
                codes=[int(qs.codes[a]) for a in anchors],
                strings=list(strings), n_tokens=n_tokens)
