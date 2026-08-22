#!/usr/bin/env python3
"""Real Slovene words the D3 lookup cannot reach -- the source for flavour (a).

0.2's flavour (a) used to be a **seeded single-character perturbation of a real
lemma**, verified absent from the surface index.  `sl.perturb`'s substitution
table was a voicing/vowel confusion map (a<->e, b<->p, d<->t, s<->z), so every
string it produced was edit-distance 1 from a real word and shaped exactly like a
native speaker's typo.  Measured: 100 % of them sat one edit from a real entry,
and 31 % were one edit from *two or more*.  Training "say you don't know" on that
teaches typo-intolerance, and it contradicts the extractor, which repairs typos
on purpose (prompt v2 onward) -- the two halves of the pipeline disagreed about
the same input.

The capability flavour (a) exists for is different and still needed: **a
well-formed word with no entry the lookup can reach**.  Without it the model has
never seen an empty ball and will invent a paradigm for a word the dictionary
does not cover, which is the worse failure because it is silent.

This module is that source.  The KG holds 4.24 M single-word lexical units
outside D8's core frame (`id >= 1M`, non-MWE) -- proper nouns, neologisms,
technical terms -- and `QAStore.surface_index` is built over the core pool only,
so none of them resolve.

**What the sentinel therefore claims.**  "ni podatka v bazi" means *not reachable
by the D3 lookup*, not *absent from the KG*: `triskajdekafobija` has 17 forms and
2 senses in the graph.  That is the honest statement about what the deployed
pipeline can answer, and it is the same reading that makes the T8 MWE negatives
correct.  If the surface index is ever widened past the core pool, every flavour
(a) item has to be regenerated -- this docstring is the warning.
"""
import os
import re
import json

import numpy as np

from qa.store import K_ANCHOR, TYPE_SHIFT, PAYLOAD, T_LU

# Slovene letters only, at least four of them.  The export carries mojibake
# ('sporoÄ', 'pooblaĺąäti') from an upstream encoding fault; a question about one
# would be a question about a broken string, not about an unlisted word.
_CLEAN = re.compile(r"^[A-Za-zČčŠšŽžĆćĐđ][A-Za-zČčŠšŽžĆćĐđ-]{3,}$")

# The pool is 75 % proper nouns.  Left alone, every flavour (a) item would be
# capitalised and the model would learn "capital letter -> say you don't know",
# which is the orthographic tell D7 warns about, rebuilt from the other side.
CAPITALISED_SHARE = 0.20


def _candidates(store):
    a = np.flatnonzero(store.kind[:store.n_real] == K_ANCHOR)
    c = store.codes[a]
    keep = (((c >> TYPE_SHIFT) == T_LU) & ((c & PAYLOAD) >= 1_000_000)
            & ~store.mwe[a])
    return a[keep]


def build(store, limit=20000, seed=20260821):
    """[(lemma, is_capitalised)] -- real words absent from the core index.

    Sorted, then interleaved to the target case mix, so the result is a pure
    function of the store and the seed.
    """
    idx = store.surface_index()
    rng = np.random.default_rng(seed)
    cand = _candidates(store)
    order = rng.permutation(len(cand))
    lower, upper = [], []
    want_up = int(limit * CAPITALISED_SHARE)
    for j in order:
        i = int(cand[int(j)])
        lem = store.lemma(i)
        if not lem or " " in lem or not _CLEAN.match(lem):
            continue
        if lem.casefold() in idx:
            continue
        (upper if lem[:1].isupper() else lower).append(lem)
        if len(lower) >= limit - want_up and len(upper) >= want_up:
            break
    out = sorted(set(lower))[:limit - want_up] + sorted(set(upper))[:want_up]
    return sorted(set(out))


def load(store, cache=True, limit=20000, seed=20260821):
    """The cached list, built on first use.  Cached beside the surface index."""
    path = os.path.join(store.path, "qa_unlisted.json")
    if cache and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    words = build(store, limit=limit, seed=seed)
    if cache:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(words, f, ensure_ascii=False)
        except OSError:
            pass
    return words
