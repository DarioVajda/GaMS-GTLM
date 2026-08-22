#!/usr/bin/env python3
"""Read-side view of a GTLM store, shaped for QA generation.

The store (data/lib/graph_store.py) is a CSR graph of 37M nodes whose text is
self-describing and whose edges are UNTYPED -- the relation lives in the node
text.  Every generator in this package needs the same handful of derived views:
which anchors are core word entries, what forms/senses/collocations hang off
one, and how to read a node's parenthetical back into features.  This module is
that layer, and it is the ONLY place that knows the text convention.

Deliberately not cached to disk: building the seed-pool index takes ~2 minutes
on 72 k anchors and a cache file is one more thing that can go stale against a
rebuilt store.  The one exception is the surface reverse index, which needs a
full 37M-node scan and is cached under the store directory.
"""
import os
import re
import sys
import json
import collections

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import graph_store                                            # noqa: E402

# node kinds, from the builder
K_ANCHOR, K_FORM, K_SENSE, K_EXAMPLE, K_TRANS = 0, 1, 2, 3, 4
K_COLLOC, K_SYN, K_ANT, K_OTHER = 5, 6, 7, 8

# node-code packing
TYPE_SHIFT = 56
PAYLOAD = (1 << TYPE_SHIFT) - 1
T_LU = 1

TAG = {K_ANCHOR: "iztočnica: ", K_FORM: "oblika: ", K_EXAMPLE: "zgled: ",
       K_COLLOC: "kolokacija: ", K_SYN: "sopomenka: ", K_ANT: "protipomenka: "}

_SENSE_RE = re.compile(r"^pomen(?: (\d+))?: (.*)$", re.S)
_SNIPPET_RE = re.compile(r" \(zgled: .*\)$", re.S)

# The features a parenthetical can hold, in builder render order.  Order matters:
# it is what lets a parenthetical be read back positionally when a value is
# ambiguous between slots (it never is today -- the vocabularies are disjoint --
# but the check below asserts that rather than assuming it).
from . import sl                                              # noqa: E402

_SLOT = {}
for _v in sl.POS:
    _SLOT[_v] = "pos"
for _v in sl.ASPECTS:
    _SLOT[_v] = "aspect"
for _v in sl.VFORMS:
    _SLOT[_v] = "vform"
for _v in sl.PERSONS:
    _SLOT[_v] = "person"
for _v in sl.CASES:
    _SLOT[_v] = "case"
for _v in sl.NUMBERS:
    _SLOT[_v] = "number"
for _v in sl.GENDERS:
    _SLOT[_v] = "gender"
for _v in sl.DEGREES:
    _SLOT[_v] = "degree"
_SLOT["določna oblika"] = "definiteness"
_SLOT["nedoločna oblika"] = "definiteness"
_SLOT["naslonska oblika"] = "clitic"
# `sedanjik` is both a vform value and a tense name; `dovršni`/`nedovršni` are
# aspects only.  No other string is shared between two slots -- asserted in
# selftest.py (C7).


def split_paren(text):
    """'oblika: X (a, b)' -> ('X', ['a', 'b']).  No parenthetical -> ('X', [])."""
    i = text.find(": ")
    body = text[i + 2:] if i >= 0 else text
    j = body.rfind(" (")
    if j < 0 or not body.endswith(")"):
        return body, []
    return body[:j], body[j + 2:-1].split(", ")


def features(text):
    """Parse a node's parenthetical into {slot: value}.  Unknown labels are kept
    under the key 'other' so nothing is silently dropped."""
    surface, items = split_paren(text)
    out = {}
    for it in items:
        slot = _SLOT.get(it)
        if slot is None:
            out.setdefault("other", []).append(it)
        else:
            out[slot] = it
    return surface, out


class QAStore:
    """A store plus the indexes the QA generators need."""

    def __init__(self, path, verbose=True):
        self.path = path
        self.G = graph_store.load_graph(path, verbose=verbose)
        self.text = self.G["text"].__getitem__
        self.kind = np.asarray(self.G["kind"])
        self.indptr = self.G["indptr"]
        self.indices = self.G["indices"]
        self.codes = np.asarray(self.G["node_codes"])
        self.mwe = np.asarray(self.G["mwe_set"])
        self.n = int(self.G["n"])
        self.n_real = int(self.G["n_real"])
        self.manifest = self.G["manifest"]
        self.convention = (self.manifest.get("meta") or {}).get("text_convention")
        self._core = None
        self._surface_index = None
        self._colloc_deg = None

    # -- basics ------------------------------------------------------------
    def nbrs(self, i):
        return self.indices[self.indptr[i]:self.indptr[i + 1]]

    def lu_id(self, i):
        return int(self.codes[i] & PAYLOAD)

    def lemma(self, i):
        return split_paren(self.text(int(i)))[0]

    def anchor_features(self, i):
        return features(self.text(int(i)))[1]

    def pos(self, i):
        return self.anchor_features(i).get("pos")

    # -- the core word-entry anchors --------------------------------------
    def core_anchors(self):
        """Indices of single-word lexical-unit anchors with id < 1M (D8's frame).

        MWE anchors are excluded (`mwe_set`), as is anything whose node code is
        not a lexical-unit: an anchor recovered from an inflected form still
        carries T_LU, so the type test is what keeps merged word-forms out.
        """
        if self._core is None:
            a = np.flatnonzero(self.kind[:self.n_real] == K_ANCHOR)
            c = self.codes[a]
            keep = (((c >> TYPE_SHIFT) == T_LU) & ((c & PAYLOAD) < 1_000_000)
                    & ~self.mwe[a])
            self._core = a[keep]
        return self._core

    # -- the pieces hanging off one anchor ---------------------------------
    def forms(self, a):
        """[(node, surface, {slot: value})] for the anchor's `oblika:` leaves."""
        out = []
        for v in self.nbrs(a):
            v = int(v)
            if self.kind[v] == K_FORM:
                s, f = features(self.text(v))
                out.append((v, s, f))
        return out

    def senses(self, a):
        """[(node, ordinal, body)] in dictionary-ordinal order, ordinal-less last.

        `body` is the sense text with the disambiguating ` (zgled: ...)` snippet
        stripped: that snippet is a rendering aid the builder adds to sense nodes
        with no definition, never lexicographic content.
        """
        out = []
        for v in self.nbrs(a):
            v = int(v)
            if self.kind[v] != K_SENSE:
                continue
            m = _SENSE_RE.match(self.text(v))
            if not m:
                continue
            ordinal = int(m.group(1)) if m.group(1) else None
            body = _SNIPPET_RE.sub("", m.group(2)).strip()
            out.append((v, ordinal, body))
        out.sort(key=lambda t: (t[1] is None, t[1] if t[1] is not None else 0,
                                self.codes[t[0]] & PAYLOAD if t[0] < self.n_real else t[0]))
        return out

    def sense_children(self, a, k):
        """[(node, text-after-the-tag)] of kind `k` reachable at sense depth 1.

        Deduplicated on the node id, then returned sorted by node id so any
        downstream sampling is a pure function of the store and not of CSR
        adjacency order (D5b's first implementation constraint).
        """
        tag = TAG[k]
        seen = {}
        for s, _o, _b in self.senses(a):
            for v in self.nbrs(s):
                v = int(v)
                if self.kind[v] == k and v not in seen:
                    t = self.text(v)
                    seen[v] = t[len(tag):] if t.startswith(tag) else t
        return [(v, seen[v]) for v in sorted(seen)]

    def examples(self, a):
        return self.sense_children(a, K_EXAMPLE)

    def collocations(self, a):
        return self.sense_children(a, K_COLLOC)

    def synonyms(self, a):
        return self.sense_children(a, K_SYN)

    def antonyms(self, a):
        return self.sense_children(a, K_ANT)

    def partners(self, a, k):
        """Partner lemmas from reified `A ~ B` nodes, minus this anchor's lemma."""
        me = self.lemma(a).casefold()
        out = []
        for _v, t in self.sense_children(a, k):
            parts = [p.strip() for p in t.split(" ~ ")]
            if len(parts) != 2:
                continue
            for p in parts:
                if p.casefold() != me and p:
                    out.append(p)
        return sorted(set(out), key=sl.sl_key)

    # -- the frequency proxy (D9) -----------------------------------------
    def proxy(self, a):
        """collocation memberships + MWE memberships, as D9 defines it.

        MWE memberships are the anchor's `sestavina` edges upward, i.e. real
        neighbours that are MWE anchors.
        """
        n_col = len(self.collocations(a))
        n_mwe = 0
        for v in self.nbrs(a):
            v = int(v)
            if v < self.n_real and self.kind[v] == K_ANCHOR and self.mwe[v]:
                n_mwe += 1
        return n_col + n_mwe

    def colloc_degree(self):
        """int32[n]: how many `kolokacija:` nodes each node touches.

        This is D5b's partner proxy.  QA_DATASET_DESIGN.md 3.1c measures the
        partner-frequency distribution on exactly this quantity -- its max,
        14,233, is `pomen 1: imeti`'s collocation fan-out -- so weighting by
        log1p of it is the measured thing, not a stand-in for it.

        Computed vectorised over the whole CSR in one pass (~3 s, ~0.5 GB)
        because the per-anchor alternative is quadratic: a hub anchor's pool has
        thousands of partners and each partner's own count is another traversal.
        """
        if self._colloc_deg is None:
            mask = (self.kind[np.asarray(self.indices)] == K_COLLOC)
            cs = np.zeros(len(mask) + 1, dtype=np.int32)
            np.cumsum(mask, dtype=np.int32, out=cs[1:])
            ip = np.asarray(self.indptr)
            self._colloc_deg = (cs[ip[1:]] - cs[ip[:-1]]).astype(np.int32)
        return self._colloc_deg

    # -- the surface-form reverse index -----------------------------------
    def surface_index(self, cache=True):
        """{casefolded surface -> (anchor, ...)} over core entries' lemmas and forms.

        Used for two things only: T4's lemmatisation targets, and verifying that a
        word really is unreachable by the lookup (0.2 flavour (a)).  It is built over
        the CORE pool rather than the whole KG, because that is the population the
        dataset draws from and a whole-KG index is 3.1M strings.
        """
        if self._surface_index is not None:
            return self._surface_index
        path = os.path.join(self.path, "qa_surface_index.json")
        if cache and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            self._surface_index = {k: tuple(v) for k, v in raw.items()}
            return self._surface_index
        idx = collections.defaultdict(set)
        for a in self.core_anchors():
            a = int(a)
            idx[self.lemma(a).casefold()].add(a)
            for _v, s, _f in self.forms(a):
                idx[s.casefold()].add(a)
        self._surface_index = {k: tuple(sorted(v)) for k, v in idx.items()}
        if cache:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self._surface_index, f)
            except OSError:
                pass
        return self._surface_index


def open_store(path=None, verbose=True):
    """Open the current store.  Defaults to the newest kg_graph_v*_gemma3."""
    if path is None:
        root = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "stores")
        cands = sorted(d for d in os.listdir(root)
                       if d.startswith("kg_graph_v") and d.endswith("_gemma3"))
        if not cands:
            raise SystemExit(f"no store found under {root}")
        path = os.path.join(root, cands[-1])
    return QAStore(path, verbose=verbose)
