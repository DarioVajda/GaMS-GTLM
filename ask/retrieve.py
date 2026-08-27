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


def retrieve(strings, store, ui):
    """Resolve `strings` against `store` and build the union ball.

    Union of every match, per the dataset's D3 -- 3.8 % of the 911,404 surfaces
    map to more than one anchor, and training saw all of them.
    """
    raise NotImplementedError(
        "ask.retrieve.retrieve is a stub; run with --demo, or see build step 2 "
        "in ask/PLAN.md")
