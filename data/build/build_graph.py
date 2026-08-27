#!/usr/bin/env python3
"""Build the GTLM text graph from the raw CJVT N-Triples.

Streams the export, reshapes it into the untyped self-describing text graph
`gtlm` reads, and either saves it as a store (`--save-graph`) or runs the k-hop
ball-sizing analysis over it.  `data/README.md` describes the resulting graph and
the reshaping decisions behind it; this file implements them.

The five that shape the code:

  * **edges are untyped.**  The relation is recoverable from the endpoint texts,
    because every node's text carries its own type tag (`iztočnica:`, `oblika:`,
    `pomen:`, `zgled:`, `prevod (madžarsko):`).  The two relations that are NOT a
    function of the endpoint types -- synonym and antonym, both sense-to-sense --
    are reified as nodes whose text says which they are, and so are collocations.

  * **node codes are exact, never hashed buckets.**  A code is
    `(type_id << 56) | payload`, where payload is the numeric IRI suffix for a
    known prefix, an exact `(D << 28) | H` packing for collocations, and a 56-bit
    blake2b of the local name otherwise.

  * **empty connector nodes are collapsed.**  MWE decomposition
    (`MWE -> part -> word`) becomes one edge, and the three-node translation
    chain collapses onto the `translation-form` that carries the text and the
    link back to its Slovenian sense.

  * **a collocation node carries the curated phrase.**  A `frac:Collocation` node
    takes only `rdfs:member` / `frac:head` / `rdf:type` and nothing ever points at
    one, so the inflected phrase is reachable only through the IRI's naming
    convention: the dependent-sense id names the multi-word entry that spells the
    pairing out.  `--colloc-text pair` falls back to the bare lemma pair.

  * **one collocation node per distinct phrase.**  A pairing is reified once per
    participant, so several IRIs share a member set while naming different
    curated phrases; the dedup key is therefore `(member set, folded phrase)`,
    not the member set alone.

Sizing analysis: 2 (form_mode) x 2 (examples) x 2 (collocations) = 8 variants,
reported with percentiles and split by seed kind (single word vs MWE).  Tokens
are node-text tokens plus the prompt; there are no relation-label tokens.
"""
import os
import re
import glob
import json
import time
import argparse
import hashlib
import threading
import unicodedata
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from lib import graph_store
from lib.paths import HF_CACHE, KG_RAW_DIR, RESULTS_DIR

ONTOLEX = "http://www.w3.org/ns/lemon/ontolex#"
LEXINFO = "http://www.lexinfo.net/ontology/3.0/lexinfo#"
LEXICOG = "http://www.w3.org/ns/lemon/lexicog#"
DECOMP  = "http://www.w3.org/ns/lemon/decomp#"
VARTRANS= "http://www.w3.org/ns/lemon/vartrans#"
FRAC    = "http://www.w3.org/ns/lemon/frac#"
SKOS    = "http://www.w3.org/2004/02/skos/core#"
RDF     = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS    = "http://www.w3.org/2000/01/rdf-schema#"

# raw predicates we care about
P_CANON      = ONTOLEX+"canonicalForm"
P_OTHER      = ONTOLEX+"otherForm"
P_SENSE      = ONTOLEX+"sense"
P_SYN        = LEXINFO+"synonym"
P_ANT        = LEXINFO+"antonym"
P_POS        = LEXINFO+"partOfSpeech"
P_USAGE      = LEXICOG+"usageExample"
P_CONSTIT    = DECOMP+"constituent"
P_CORR       = DECOMP+"correspondsTo"
P_VSRC       = VARTRANS+"source"
P_MEMBER     = RDFS+"member"
P_WRITTENREP = ONTOLEX+"writtenRep"
P_DEFINITION = SKOS+"definition"
P_VALUE      = RDF+"value"
# deliberately NOT parsed: frac:head (indexing only), vartrans:target (chain
# collapsed), ontolex:isSenseOf (inverse of ontolex:sense), rdf:type, lime:entry
# (global hub), phoneticRep, and the lexinfo morphology *vocabulary* IRIs.

# ---- node type ids (packed into the high byte of a code) --------------------
T_LU        = 1    # lexical-unit          -> anchor (merged with its lemma form)
T_PART      = 2    # lexical-unit-part     -> collapsed away
T_WORDFORM  = 3    # word-form             -> inflected form leaf, or merged anchor
T_FORMLU    = 4    # form-lexical-unit     -> MWE canonical form, merged into anchor
T_SENSE     = 5    # sense
T_SENSETR   = 6    # sense-translation     -> dropped (chain collapsed)
T_TRANSFORM = 7    # translation-form      -> translation leaf (carries the text)
T_EXAMPLE   = 8    # example
T_RESOURCE  = 9    # resource
T_ENTRYTR   = 10   # lexical-entry-translation -> dropped (chain collapsed)
T_COLLOC    = 11   # dependent-sense-D-lexical-unit-H (frac:Collocation)
T_OTHER     = 12
KNOWN_PREFIX = {
    "lexical-unit": T_LU, "lexical-unit-part": T_PART, "word-form": T_WORDFORM,
    "form-lexical-unit": T_FORMLU, "sense": T_SENSE,
    "sense-translation": T_SENSETR, "translation-form": T_TRANSFORM,
    "example": T_EXAMPLE, "resource": T_RESOURCE,
    "lexical-entry-translation": T_ENTRYTR,
}
TYPE_SHIFT = 56
PAYLOAD_MASK = (1 << TYPE_SHIFT) - 1
TYPE_NAME = {T_LU: "lexical-unit", T_PART: "lexical-unit-part",
             T_WORDFORM: "word-form", T_FORMLU: "form-lexical-unit",
             T_SENSE: "sense", T_SENSETR: "sense-translation",
             T_TRANSFORM: "translation-form", T_EXAMPLE: "example",
             T_RESOURCE: "resource", T_ENTRYTR: "lexical-entry-translation",
             T_COLLOC: "collocation", T_OTHER: "other"}

# ---- synthetic node kinds (minted, no IRI) ---------------------------------
K_ANCHOR = 0
K_FORM   = 1
K_SENSE  = 2
K_EXAMPLE= 3
K_TRANS  = 4
K_COLLOC = 5
K_SYN    = 6
K_ANT    = 7
K_OTHER  = 8

HOST_WWW  = "https://www.cjvt.ddds.si/"
HOST_BARE = "https://cjvt.ddds.si/"
_num_re    = re.compile(r"^(.*)-(\d+)$")
_colloc_re = re.compile(r"^dependent-sense-(\d+)-lexical-unit-(\d+)$")

# ---- Slovenian labels ------------------------------------------------------
# Order matters: this is the order the parenthetical is rendered in.  vform and
# person lead, so a verb form reads "(sedanjik, 1. oseba, ednina)".  Appending a
# label here only ever GROWS a form string, leaving the existing items in place;
# the same is not true of anchors, where entry-level gender is inserted after
# the POS rather than appended.
FEATURE_PROPS = ("vform", "person", "case", "number", "gender", "degree",
                 "definiteness")
# Properties that sit on the lexical-unit rather than on a word-form, rendered
# into the ANCHOR parenthetical right after the POS.
#
# `gender` is in both sets because this KG carries it at both levels on disjoint
# parts of speech: on a word-form it is adjective/participle agreement, on a
# lexical-unit the inherent gender of a noun.  Every one of the 310,362 nouns
# carries exactly one entry-level gender and no other POS carries any
# (`analysis/scan_gender.py`), so the two roles never collide on one node.  Noun
# gender lands on the ANCHOR only; form leaves are one hop away and inherit it.
UNIT_PROPS = ("aspect", "gender", "clitic")
VALUE_SL = {
    "nominative":"imenovalnik","genitive":"rodilnik","dative":"dajalnik",
    "accusative":"tožilnik","locative":"mestnik","instrumental":"orodnik",
    "singular":"ednina","dual":"dvojina","plural":"množina",
    "masculine":"moški spol","feminine":"ženski spol","neuter":"srednji spol",
    # person: the KG emits first/second/third.  v3 listed only the *Person
    # spellings, which occur nowhere in this dump, so feat_string() -- which
    # drops what it cannot map -- silently discarded person on all 253,497
    # forms that carry it.  See data/README.md Finding 6.
    "first":"1. oseba","second":"2. oseba","third":"3. oseba",
    # vform.  This KG has NO lexinfo:tense and NO lexinfo:mood predicate; the
    # synthetic present, the imperative, the -l participle, the infinitive and
    # the supine are all vform values.  Past and future are periphrastic and
    # are not stored at all.
    "present":"sedanjik","future":"prihodnjik","conditional":"pogojnik",
    "imperative":"velelnik","participle":"deležnik na -l",
    "infinitive":"nedoločnik","supine":"namenilnik",
    "positive":"osnovnik","comparative":"primernik","superlative":"presežnik",
    # definiteness: without it the definite and indefinite forms of an
    # adjective carry byte-identical labels and differ only in surface, so no
    # consumer can tell which is the citation form.
    "definiteness:yes":"določna oblika","definiteness:no":"nedoločna oblika",
    # unit-level
    "perfective":"dovršni","progressive":"nedovršni","biaspectual":"dvovidski",
    "clitic:bound":"naslonska oblika","clitic:yes":"naslonska oblika",
}
# Values whose local name is ambiguous across properties ("yes" is both a
# definiteness and a clitic value) are looked up as "prop:value" first.
_AMBIGUOUS_PROPS = ("definiteness", "clitic")
POS_SL = {
    "noun":"samostalnik","verb":"glagol","adjective":"pridevnik",
    "adverb":"prislov","pronoun":"zaimek","numeral":"števnik",
    "preposition":"predlog","adposition":"predlog","conjunction":"veznik",
    "particle":"členek","interjection":"medmet","determiner":"določilnik",
    "properNoun":"lastno ime","abbreviation":"okrajšava",
}
LANG_SL = {"hun":"madžarsko","en":"angleško","de":"nemško","it":"italijansko"}

TAG_ANCHOR = "iztočnica: "
TAG_FORM   = "oblika: "
TAG_SENSE  = "pomen: "
TAG_SENSE_N = "pomen {}: "        # polysemous entry: dictionary ordinal
TAG_SENSE_EX = " (zgled: {})"     # disambiguating snippet, no definition
SENSE_SNIPPET_CHARS = 60          # default budget for that snippet

# Default tokenizer for token_len.  Every Gemma 3 checkpoint and GaMS3-12B share
# one tokenizer -- byte-identical `tokenizer.model`, byte-identical ids on real
# node texts -- so one build serves the whole ladder and this name only records
# which of the interchangeable repos was loaded.  Gemma 2 (vocab 256,000) is NOT
# in that family: +0.8 % tokens over this graph, agreeing node for node 46 % of
# the time, so a store built with it is not interchangeable.
DEFAULT_TOKENIZER = "cjvt/GaMS3-12B-Instruct"
TAG_EX     = "zgled: "
TAG_COLLOC = "kolokacija: "
TAG_SYN    = "sopomenka: "
TAG_ANT    = "protipomenka: "


def code_of(iri):
    """Stable 64-bit node code: (type_id << 56) | payload.  No collisions."""
    if iri.startswith(HOST_WWW):
        ln = iri[len(HOST_WWW):]
    elif iri.startswith(HOST_BARE):
        ln = iri[len(HOST_BARE):]
    else:
        return None
    m = _colloc_re.match(ln)
    if m:
        # exact packing: both ids are < 2^28 in this KG
        d = int(m.group(1)); h = int(m.group(2))
        if d < (1 << 28) and h < (1 << 28):
            return (T_COLLOC << TYPE_SHIFT) | (d << 28) | h
        return (T_COLLOC << TYPE_SHIFT) | _hash56(ln)
    m = _num_re.match(ln)
    if m:
        t = KNOWN_PREFIX.get(m.group(1))
        if t is not None:
            num = int(m.group(2))
            if num <= PAYLOAD_MASK:
                return (t << TYPE_SHIFT) | num
    return (T_OTHER << TYPE_SHIFT) | _hash56(ln)


def _hash56(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=7).digest(), "big")


def type_of(codes):
    return (codes >> TYPE_SHIFT).astype(np.int32)


def _localname(s):
    return s.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


# N-Triples literal escapes.  The dump triple-quotes every writtenRep / value /
# definition, so in practice only \" and \\ occur, but \uXXXX and the C-style set
# are handled too and an unrecognised escape passes through untouched rather than
# being silently eaten.
_ESC_RE = re.compile(r"\\(u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|.)", re.S)
_ESC_MAP = {'"': '"', "\\": "\\", "n": "\n", "r": "\r", "t": "\t",
            "b": "\b", "f": "\f", "/": "/", "'": "'"}


def _esc_sub(m):
    g = m.group(1)
    if len(g) > 1 and g[0] in "uU":
        try:
            return chr(int(g[1:], 16))
        except ValueError:
            return m.group(0)
    return _ESC_MAP.get(g, m.group(0))


def unescape_nt(s):
    """Decode N-Triples escapes in a literal.  Fast path: most have none."""
    return _ESC_RE.sub(_esc_sub, s) if "\\" in s else s


EDGE_KEYS = ("canon", "other", "sense", "syn", "ant", "usage",
             "constit", "corr", "tsrc", "member")


def parse_file(path):
    """Stream a .nt file. Return dict of edge arrays + text/feature tuples."""
    E = {k: ([], []) for k in EDGE_KEYS}
    wr = []    # (code, str)        writtenRep @sl
    wrf = []   # (code, lang, str)  writtenRep, other languages (translations)
    dfn = []   # (code, str)        skos:definition @sl
    val = []   # (code, str)        rdf:value @sl (example sentences)
    feat = []  # (form_code, prop, value_localname)
    unit = []  # (lu_code, prop, value_localname)   aspect / clitic
    pos = []   # (lu_code, pos_localname)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line or line[0] != "<":
                continue
            p1 = line.find("> <")
            if p1 < 0:
                continue
            subj = line[1:p1]
            rest = line[p1+3:]
            p2 = rest.find("> ")
            if p2 < 0:
                continue
            pred = rest[:p2]
            obj = rest[p2+2:]
            if obj and obj[0] == "<":
                oend = obj.find(">")
                oiri = obj[1:oend]
                if   pred == P_CANON:   key = "canon"
                elif pred == P_OTHER:   key = "other"
                elif pred == P_SENSE:   key = "sense"
                elif pred == P_SYN:     key = "syn"
                elif pred == P_ANT:     key = "ant"
                elif pred == P_USAGE:   key = "usage"
                elif pred == P_CONSTIT: key = "constit"
                elif pred == P_CORR:    key = "corr"
                elif pred == P_VSRC:    key = "tsrc"
                elif pred == P_MEMBER:  key = "member"
                elif pred == P_POS:
                    sc = code_of(subj)
                    if sc is not None:
                        pos.append((sc, _localname(oiri)))
                    continue
                elif pred.startswith(LEXINFO):
                    local = pred[len(LEXINFO):]
                    # Dispatch on the SUBJECT TYPE, not on whichever set the
                    # property name is found in first: `gender` is in both, so an
                    # `if FEATURE_PROPS / elif UNIT_PROPS` chain would route every
                    # noun-entry gender into the form branch, where the
                    # T_WORDFORM guard drops it and the elif can never run.
                    # T_PART stays excluded -- MWE components carry the bulk of
                    # these triples and are collapsed away, so their values are
                    # unreachable in the built graph anyway.
                    if local in FEATURE_PROPS or local in UNIT_PROPS:
                        sc = code_of(subj)
                        if sc is not None:
                            t = sc >> TYPE_SHIFT
                            if t in (T_WORDFORM, T_FORMLU):
                                if local in FEATURE_PROPS:
                                    feat.append((sc, local, _localname(oiri)))
                            elif t == T_LU and local in UNIT_PROPS:
                                unit.append((sc, local, _localname(oiri)))
                    continue
                else:
                    continue
                sc = code_of(subj); oc = code_of(oiri)
                if sc is None or oc is None:
                    continue
                E[key][0].append(sc); E[key][1].append(oc)
            else:
                if   pred == P_WRITTENREP: kind = "wr"
                elif pred == P_DEFINITION: kind = "dfn"
                elif pred == P_VALUE:      kind = "val"
                else:
                    continue
                if obj.startswith('"""'):
                    e = obj.find('"""', 3)
                    if e < 0: continue
                    txt = obj[3:e]; tail = obj[e+3:]
                elif obj.startswith('"'):
                    e = obj.find('"', 1)
                    if e < 0: continue
                    txt = obj[1:e]; tail = obj[e+1:]
                else:
                    continue
                sc = code_of(subj)
                if sc is None:
                    continue
                txt = unescape_nt(txt)
                is_sl = "@sl" in tail
                if kind == "wr":
                    if is_sl:
                        wr.append((sc, txt))
                    else:
                        # keep foreign writtenRep: translations are @hun
                        at = tail.find("@")
                        lang = tail[at+1:].strip().rstrip(" .").strip() if at >= 0 else ""
                        wrf.append((sc, lang, txt))
                elif is_sl:
                    (dfn if kind == "dfn" else val).append((sc, txt))
    out = {}
    for k, (a, b) in E.items():
        if a:
            arr = np.empty((len(a), 2), dtype=np.int64); arr[:, 0] = a; arr[:, 1] = b
        else:
            arr = np.empty((0, 2), dtype=np.int64)
        out[k] = arr
    out["wr"] = wr; out["wrf"] = wrf; out["dfn"] = dfn
    out["val"] = val; out["feat"] = feat; out["pos"] = pos; out["unit"] = unit
    return out


def _label(prop, value):
    """Slovenian label for a feature value, or None if the value is unmapped."""
    if prop in _AMBIGUOUS_PROPS:
        return VALUE_SL.get(prop + ":" + value)
    return VALUE_SL.get(value)


def feat_string(props, pos_local=None, unit_props=None):
    """-> ' (samostalnik, imenovalnik, ednina)' or ''.

    Order is POS, then the lexical-unit properties (aspect / gender / clitic),
    then the word-form features in FEATURE_PROPS order.
    """
    parts = []
    if pos_local:
        sl = POS_SL.get(pos_local)
        if sl:
            parts.append(sl)
    if unit_props:
        for p in UNIT_PROPS:
            v = unit_props.get(p)
            if v is None:
                continue
            # A property in both sets (gender) must not render twice if a node
            # ever carries it at both levels.  It never does on this dump, so the
            # guard costs a lookup and never fires.
            if p in props:
                continue
            sl = _label(p, v)
            if sl:
                parts.append(sl)
    for p in FEATURE_PROPS:
        v = props.get(p)
        if v is None:
            continue
        sl = _label(p, v)
        if sl:
            parts.append(sl)
    return " (" + ", ".join(parts) + ")" if parts else ""


def sense_snippet(s, limit):
    """Bounded, word-boundary example snippet for a definition-less sense."""
    s = " ".join(s.split())
    if not s or limit <= 0:
        return ""
    if len(s) <= limit:
        return s
    cut = s.rfind(" ", 0, limit)
    head = s[:cut] if cut > limit // 2 else s[:limit]
    return head.rstrip(" ,;:.!?-") + " ..."


def _dedup_pairs(a, b):
    """Undirected dedup of a pair list. Returns (m,2) sorted-unique array."""
    if len(a) == 0:
        return np.empty((0, 2), dtype=np.int64)
    lo = np.minimum(a, b); hi = np.maximum(a, b)
    keep = lo != hi
    lo = lo[keep]; hi = hi[keep]
    if len(lo) == 0:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.stack([lo, hi], axis=1)
    return np.unique(pairs, axis=0)


def _dedup_pairs_keyed(a, b, key):
    """_dedup_pairs, keeping one `key` alongside each surviving pair.

    Returns (pairs, keys) in the SAME row order as _dedup_pairs(a, b), so the
    minted node ids are unchanged by carrying the key through.

    Used for collocations under `--colloc-text pair`, where the key is the
    frac:Collocation IRI code and every reification of one member set renders the
    same lemma-pair string, so which representative survives cannot matter.

    NOT usable in phrase mode: 15.3 % of member sets name several dependent
    senses, i.e. several distinct curated phrases, and keeping one drops the
    rest.  Use `_dedup_colloc_by_phrase` there.
    """
    if len(a) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)
    lo = np.minimum(a, b); hi = np.maximum(a, b)
    keep = lo != hi
    lo = lo[keep]; hi = hi[keep]; key = np.asarray(key)[keep]
    if len(lo) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)
    order = np.lexsort((key, hi, lo))       # primary lo, then hi -> np.unique order
    lo = lo[order]; hi = hi[order]; key = key[order]
    pairs = np.stack([lo, hi], axis=1)
    first = np.flatnonzero(np.r_[True, (pairs[1:] != pairs[:-1]).any(axis=1)])
    return pairs[first], key[first]


def _colloc_key(s):
    """Fold a phrase to what any reader of the node text can actually tell apart.

    This is the QA grader's norm() (data/QA_TASKS.md 0.8) minus the casefold's
    only purpose here -- case, runs of whitespace and one trailing period.  Two
    phrases that agree under it are duplicates for every downstream purpose: the
    grader matches collocations case-insensitively, so a member set carrying both
    "Cvilece gume" and "cvilece gume" would produce a gold list that contains the
    same item twice and fails its own dedup invariant.

    Measured: keying on the raw surface instead yields 3,744,612 pairings against
    3,569,711 here, i.e. 174,901 nodes (4.7%) that differ from a sibling only in
    case or spacing.
    """
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip() \
             .rstrip(".").casefold()


def _dedup_colloc_by_phrase(a, b, iri, phrase_of):
    """Dedup collocation reifications on (member set, PHRASE).

    `a`, `b` are the two member senses of one frac:Collocation IRI and `iri` is
    that IRI's node code; `phrase_of(iri_code) -> str` dereferences the naming
    convention to the curated phrase, returning '' when it does not resolve.

    Returns (pairs, iri, phrases) with one row per distinct (pair, folded phrase),
    in lexsort order (lo, hi, phrase, capitalisation, surface, iri) so the minted
    node ids are a pure function of the parsed triple SET and not of file or
    worker order.

    Which SURFACE survives a fold is decided the same way build() decides between
    several writtenRep values for one form: fewest capitals, then lexicographically
    smallest, so the dictionary-style spelling wins over a sentence-initial or
    all-caps variant, and the choice does not depend on IRI order.

    Unresolved rows key on '' and therefore still collapse to one node per member
    set, which is right: their text is the lemma pair, a function of the members
    alone, so a second node would be a byte-identical duplicate.
    """
    if len(a) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64), []
    lo = np.minimum(a, b); hi = np.maximum(a, b)
    keep = lo != hi
    lo = lo[keep]; hi = hi[keep]; iri = np.asarray(iri)[keep]
    if len(lo) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64), []
    # Intern the phrases: comparing 4.7M strings through numpy would mean an
    # object array and an O(n log n) string sort; interning makes the sort keys
    # int64 columns.  First-encounter order is deterministic because the caller
    # feeds rows in sorted-IRI order -- and the tie-break below does not depend on
    # it anyway.
    kid = {}                      # folded phrase -> key id
    sid = {}                      # surface       -> surface id
    surfaces = []
    pid = np.empty(len(lo), dtype=np.int64)     # folded-phrase id, the dedup key
    rank = np.empty(len(lo), dtype=np.int64)    # capitals, for the tie-break
    sur = np.empty(len(lo), dtype=np.int64)     # surface id, for the tie-break
    for t in range(len(lo)):
        s = phrase_of(int(iri[t]))
        j = sid.get(s)
        if j is None:
            j = len(surfaces); sid[s] = j; surfaces.append(s)
        sur[t] = j
        rank[t] = sum(1 for ch in s if ch.isupper())
        k = _colloc_key(s)
        v = kid.get(k)
        if v is None:
            v = len(kid); kid[k] = v
        pid[t] = v
    del kid, sid
    # Rank the surfaces lexicographically once, so the tie-break is a real
    # alphabetical order and not first-encounter order.
    srank = np.empty(len(surfaces), dtype=np.int64)
    for r, i in enumerate(sorted(range(len(surfaces)), key=surfaces.__getitem__)):
        srank[i] = r
    # np.lexsort keys are LAST-primary: (lo, hi, pid) select the group and
    # (rank, surface, iri) pick its representative.
    order = np.lexsort((iri, srank[sur], rank, pid, hi, lo))
    lo = lo[order]; hi = hi[order]; iri = iri[order]; pid = pid[order]
    sur = sur[order]
    pairs = np.stack([lo, hi], axis=1)
    changed = ((pairs[1:] != pairs[:-1]).any(axis=1) | (pid[1:] != pid[:-1]))
    first = np.flatnonzero(np.r_[True, changed])
    return pairs[first], iri[first], [surfaces[int(j)] for j in sur[first]]


# ---------------------------------------------------------------------------
class Parsed:
    """What one pass over the raw N-Triples yields.

    `edges` holds an (m, 2) array of node codes per EDGE_KEYS relation; the rest
    are node-code-keyed maps of the literals and features the node text is built
    from.
    """

    __slots__ = ("edges", "wr", "wrf", "dfn", "val",
                 "feat", "unit", "pos", "n_multi_wr")

    def __init__(self):
        self.edges = {}
        self.wr = {}        # code -> Slovene writtenRep
        self.wrf = {}       # code -> (lang, writtenRep) for other languages
        self.dfn = {}       # code -> skos:definition
        self.val = {}       # code -> rdf:value (example sentences)
        self.feat = defaultdict(dict)   # word-form code -> {prop: value}
        self.unit = defaultdict(dict)   # lexical-unit code -> {prop: value}
        self.pos = {}       # lexical-unit code -> part of speech
        self.n_multi_wr = 0

    def __getitem__(self, key):
        return self.edges[key]


def _better_written_rep(new, old):
    """Is `new` the better spelling of a form that carries several?

    A form may carry several writtenRep values -- `word-form-1911547` has "BOJ",
    "Boj" and "boj".  Fewest capitals wins, then the lexicographically smallest,
    so the dictionary lemma beats sentence-initial and all-caps variants and the
    choice does not depend on which file was read first.
    """
    return (sum(1 for ch in new if ch.isupper()), new) < \
           (sum(1 for ch in old if ch.isupper()), old)


def _gated(items, sem):
    """Yield `items`, blocking until the consumer has released a slot.

    `Pool.imap_unordered` pulls its whole input as fast as workers free up and
    buffers every finished result until the consumer asks for it -- there is no
    bound on how far ahead it runs.  That matters here because the workers parse
    far faster than this module's merge loop can absorb (the merge does ~50 M
    dict inserts), so finished results pile up in the parent.

    Measured on the 2,594-file dump: 2, 4 and 8 workers all peak at 33 GB, and 16
    peak at 91 GB -- the extra 58 GB is nothing but results waiting to be merged,
    and 16 workers only buys 44 seconds of parse.  Gating the INPUT bounds the
    number in flight, which decouples memory from the worker count and lets the
    build keep all 16.

    The pool's task-feeder thread is what iterates this, so blocking here stalls
    submission rather than the merge.
    """
    for x in items:
        sem.acquire()
        yield x


def parse_all(files, workers, t0, in_flight=3):
    """Parse every file in parallel and merge the results into one `Parsed`.

    `in_flight` is the prefetch depth, as a multiple of `workers`: at most
    `workers * in_flight` files may be parsed but not yet merged.  Enough to
    keep every worker busy through the jitter in file sizes, small enough that
    the queue is not where the memory goes.
    """
    print(f"[parse] {len(files)} files x {workers} workers "
          f"(<= {workers * in_flight} in flight)", flush=True)
    out = Parsed()
    agg = {k: [] for k in EDGE_KEYS}
    sem = threading.Semaphore(max(1, workers * in_flight))
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(
                parse_file, _gated(files, sem), chunksize=1)):
            for k in agg:
                if res[k].shape[0]:
                    agg[k].append(res[k])
            for c, s in res["wr"]:
                old = out.wr.get(c)
                if old is None:
                    out.wr[c] = s
                elif s != old:
                    out.n_multi_wr += 1
                    if _better_written_rep(s, old):
                        out.wr[c] = s
            for c, l, s in res["wrf"]:  out.wrf[c] = (l, s)
            for c, s in res["dfn"]:     out.dfn[c] = s
            for c, s in res["val"]:     out.val[c] = s
            for c, p, v in res["feat"]: out.feat[c][p] = v
            for c, p, v in res["unit"]: out.unit[c][p] = v
            for c, p in res["pos"]:     out.pos[c] = p
            del res                  # release before waiting on the next one
            sem.release()            # ... and only then admit another file
            if (i + 1) % 200 == 0:
                print(f"[parse] {i+1}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    for k in EDGE_KEYS:
        out.edges[k] = (np.concatenate(agg[k], axis=0) if agg[k]
                        else np.empty((0, 2), dtype=np.int64))
    return out


def drop_translation_chain(canon, sense):
    """Remove the two textless nodes of the translation chain.

    `lexical-entry-translation-N` reaches `sense-translation-N` by
    `ontolex:sense` and `translation-form-N` by `canonicalForm`; keeping either
    edge would re-introduce a node that carries no text.  The form itself is
    attached to its Slovenian sense through `vartrans:source` instead.
    """
    n = 0
    out = []
    for arr in (sense, canon):
        if len(arr):
            keep = (arr[:, 0] >> TYPE_SHIFT) != T_ENTRYTR
            n += int((~keep).sum())
            arr = arr[keep]
        out.append(arr)
    if n:
        print(f"[trans] dropped {n:,} translation-chain edges "
              f"(collapsed onto translation-form)", flush=True)
    sense, canon = out
    return canon, sense


# The order the parse totals are printed and recorded in, which is neither
# EDGE_KEYS order nor alphabetical: relations that describe an entry first, then
# the two reified sense-to-sense ones.
REPORT_ORDER = ("canon", "other", "sense", "usage", "constit", "corr",
                "tsrc", "member", "syn", "ant")


def report_parse(p, edges, stats, t0):
    """Print the parse totals and record them in `stats["raw"]`."""
    counts = {k: len(edges[k]) for k in REPORT_ORDER}
    print(f"[parse] done {time.time()-t0:.0f}s  "
          + " ".join(f"{k}={v:,}" for k, v in counts.items())
          + f" wr={len(p.wr):,} wrf={len(p.wrf):,} dfn={len(p.dfn):,} "
            f"val={len(p.val):,} feat={len(p.feat):,} unit={len(p.unit):,} "
            f"pos={len(p.pos):,}", flush=True)

    def having(m, prop):
        return sum(1 for d in m.values() if prop in d)

    # Gender is counted at BOTH levels because it exists at both and the two mean
    # different things: on a word-form it is adjective/participle agreement, on a
    # lexical-unit the inherent gender of a noun.
    feats = {p_: having(p.feat, p_)
             for p_ in ("vform", "person", "definiteness", "gender")}
    units = {p_: having(p.unit, p_) for p_ in ("aspect", "clitic", "gender")}
    print(f"[feat] reachable on word entries: vform={feats['vform']:,} "
          f"person={feats['person']:,} definiteness={feats['definiteness']:,} "
          f"aspect={units['aspect']:,} clitic={units['clitic']:,} "
          f"gender(form)={feats['gender']:,} gender(entry)={units['gender']:,}",
          flush=True)
    stats["raw"] = {k: int(v) for k, v in dict(
        **counts,
        writtenrep_sl=len(p.wr), writtenrep_foreign=len(p.wrf),
        definitions=len(p.dfn), values=len(p.val), pos=len(p.pos),
        feat_forms=len(p.feat), unit_props=len(p.unit),
        vform=feats["vform"], person=feats["person"],
        definiteness=feats["definiteness"],
        aspect=units["aspect"], clitic=units["clitic"],
        writtenrep_multivalued=p.n_multi_wr).items()}


def collapse_mwe(constit, corr):
    """MWE -> word edges, with the empty `lexical-unit-part` connector removed.

    The raw shape is `MWE --constituent--> part --correspondsTo--> word`, where
    the part node carries no text of its own.
    """
    if not (len(constit) and len(corr)):
        return np.empty((0, 2), dtype=np.int64)
    order = np.argsort(corr[:, 0], kind="stable")
    parts_s = corr[order, 0]; words_s = corr[order, 1]
    idx = np.clip(np.searchsorted(parts_s, constit[:, 1]), 0, len(parts_s) - 1)
    hit = parts_s[idx] == constit[:, 1]
    mwe_word = np.unique(np.stack([constit[hit, 0], words_s[idx[hit]]], axis=1),
                         axis=0)
    print(f"[collapse] {len(mwe_word):,} MWE->word constituent edges", flush=True)
    return mwe_word


# The low 28 bits of a collocation code hold the indexing head; the next 28 the
# dependent sense that names the phrase.  See `colloc_phrase_of_iri`.
COLLOC_ID_MASK = (1 << 28) - 1


def undirected_csr(si, di, n):
    """(indptr, indices) over `n` nodes, with every edge stored both ways.

    The graph is built directed and traversed undirected, so each edge appears
    once per endpoint.

    Neighbours are sorted WITHIN each node's slice, which is what makes the store
    bit-reproducible.  Sorting only by source would leave the order inside a
    slice set by the order the edge arrays arrived in -- and the parse runs under
    `imap_unordered`, so that is worker-completion order and varies from run to
    run.  Nothing downstream reads the order (consumers sort by node id
    themselves), but a store that cannot be rebuilt byte-for-byte cannot be
    checked byte-for-byte either.
    """
    src2 = np.concatenate([si, di]); dst2 = np.concatenate([di, si])
    oo = np.lexsort((dst2, src2))          # primary src2, secondary dst2
    src2 = src2[oo]; dst2 = dst2[oo]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, src2 + 1, 1); np.cumsum(indptr, out=indptr)
    return indptr, dst2


# ---------------------------------------------------------------------------
def build(files, workers, stats, snippet_chars=SENSE_SNIPPET_CHARS,
          sense_index=True, colloc_text="phrase"):
    t0 = time.time()
    parsed = parse_all(files, workers, t0)
    wr, wrf, dfn, val = parsed.wr, parsed.wrf, parsed.dfn, parsed.val
    feat_map, unit_map, pos_map = parsed.feat, parsed.unit, parsed.pos

    canon, sense = drop_translation_chain(parsed["canon"], parsed["sense"])
    parsed.edges["canon"], parsed.edges["sense"] = canon, sense
    other = parsed["other"]
    syn = parsed["syn"]; ant = parsed["ant"]; usage = parsed["usage"]
    constit = parsed["constit"]; corr = parsed["corr"]
    tsrc = parsed["tsrc"]; member = parsed["member"]
    report_parse(parsed, parsed.edges, stats, t0)
    del parsed

    mwe_word = collapse_mwe(constit, corr)
    del constit, corr

    # ---- sense -> lexical unit, and lemma text per lexical unit ------------
    canon_of = {}
    for lu, fcode in canon:
        canon_of.setdefault(int(lu), int(fcode))
    sense_lu = {}
    for lu, se in sense:
        sense_lu.setdefault(int(se), int(lu))

    def lemma_of_lu(lu):
        f = canon_of.get(lu)
        return wr.get(f, "") if f is not None else ""

    def lemma_of_sense(se):
        lu = sense_lu.get(int(se))
        return lemma_of_lu(lu) if lu is not None else ""

    def colloc_phrase_of_iri(iri):
        """Curated phrase behind one frac:Collocation IRI code, or '' if unresolved.

        The phrase is not on the collocation node and not behind any edge -- it
        is reachable only through the IRI's naming convention:

            <dependent-sense-D-lexical-unit-H>  rdf:type  frac:Collocation
            <sense-D>          ontolex:isSenseOf     <lexical-unit-M>
            <lexical-unit-M>   ontolex:canonicalForm <form-lexical-unit-M>
            <form-lexical-unit-M> ontolex:writtenRep  'kisova voda'@sl

        `code_of` packs that IRI as `(T_COLLOC << 56) | (D << 28) | H` whenever
        both ids fit in 28 bits, which they do throughout this KG (max sense id
        ~1.0e7, max lexical-unit id ~1.2e7, against 2^28 = 2.7e8).  On the
        `_hash56` fallback the unpacked D names no sense and this returns ''.
        """
        if (iri >> TYPE_SHIFT) != T_COLLOC:
            return ""
        d = (iri >> 28) & COLLOC_ID_MASK
        return lemma_of_sense((T_SENSE << TYPE_SHIFT) | d)

    # ---- collocations from rdfs:member -------------------------------------
    colloc_pairs = np.empty((0, 2), dtype=np.int64)
    colloc_iri = np.empty(0, dtype=np.int64)        # parallel to colloc_pairs
    colloc_phrases = None                           # parallel too, phrase mode only
    member_hist = {}
    if len(member):
        member = np.unique(member, axis=0)          # the dump repeats triples
        o = np.argsort(member[:, 0], kind="stable")
        ms = member[o, 0]; mo = member[o, 1]
        bounds = np.flatnonzero(np.r_[True, ms[1:] != ms[:-1]])
        sizes = np.diff(np.r_[bounds, len(ms)])
        u, c = np.unique(sizes, return_counts=True)
        member_hist = {int(a): int(b) for a, b in zip(u, c)}
        two = sizes == 2
        starts = bounds[two]
        a = mo[starts]; b = mo[starts + 1]
        # ms[starts] is the frac:Collocation IRI code itself, kept because the
        # sense id it encodes is the only route to the phrase's surface string.
        if colloc_text == "phrase":
            colloc_pairs, colloc_iri, colloc_phrases = _dedup_colloc_by_phrase(
                a, b, ms[starts], colloc_phrase_of_iri)
            n_member_sets = len(_dedup_pairs(a, b))
        else:
            colloc_pairs, colloc_iri = _dedup_pairs_keyed(a, b, ms[starts])
            n_member_sets = len(colloc_pairs)
        n_colloc_iris = len(bounds)
        print(f"[colloc] {n_colloc_iris:,} collocation IRIs, member-count hist={member_hist}, "
              f"{int(two.sum()):,} binary -> {n_member_sets:,} member sets "
              f"-> {len(colloc_pairs):,} distinct pairings (key={colloc_text})",
              flush=True)
        stats["collocations"] = {"iris": int(n_colloc_iris),
                                 "member_count_hist": member_hist,
                                 "binary_iris": int(two.sum()),
                                 "member_sets": int(n_member_sets),
                                 "dedup_key": ("member_set+phrase"
                                               if colloc_text == "phrase"
                                               else "member_set"),
                                 "distinct_pairings": int(len(colloc_pairs))}
    del member

    # ---- reify the two sense->sense relations ------------------------------
    # Synonym and antonym are the only relations not recoverable from the
    # endpoint types, so each pairing becomes a node whose text names which.
    syn_pairs = _dedup_pairs(syn[:, 0], syn[:, 1]) if len(syn) else np.empty((0, 2), np.int64)
    ant_pairs = _dedup_pairs(ant[:, 0], ant[:, 1]) if len(ant) else np.empty((0, 2), np.int64)
    print(f"[reify] syn={len(syn_pairs):,} ant={len(ant_pairs):,} pairings", flush=True)
    del syn, ant

    # `translation-form` carries both the foreign text and `vartrans:source`
    # back to the Slovenian sense, so this one edge is the whole chain.
    trans_edges = tsrc[:, [0, 1]] if len(tsrc) else np.empty((0, 2), np.int64)
    del tsrc

    # ---- assemble the UNTYPED directed edge list over real IRI nodes -------
    real_edges = [e for e in (other, sense, usage, mwe_word, trans_edges) if len(e)]
    src = np.concatenate([e[:, 0] for e in real_edges])
    dst = np.concatenate([e[:, 1] for e in real_edges])

    node_codes = np.unique(np.concatenate([
        src, dst,
        np.fromiter(canon_of.keys(), dtype=np.int64, count=len(canon_of)),
    ]))
    n_real = len(node_codes)
    ntype = type_of(node_codes)
    si = np.searchsorted(node_codes, src)
    di = np.searchsorted(node_codes, dst)
    print(f"[nodes] {n_real:,} IRI-backed nodes, {len(src):,} IRI-backed edges", flush=True)

    # ---- node text (self-describing, type-tagged) -------------------------
    text = [""] * n_real
    kind = np.full(n_real, K_OTHER, dtype=np.int8)

    def form_text(fcode):
        s = wr.get(fcode, "")
        if not s:
            return ""
        return s + feat_string(feat_map.get(fcode, {}))

    # anchors: lemma + POS + morphology of the canonical form
    for lu, fcode in canon_of.items():
        j = idx = np.searchsorted(node_codes, lu)
        if idx >= n_real or node_codes[idx] != lu:
            continue
        s = wr.get(fcode, "")
        if not s:
            continue
        text[j] = TAG_ANCHOR + s + feat_string(feat_map.get(fcode, {}),
                                               pos_map.get(lu),
                                               unit_map.get(lu))
        kind[j] = K_ANCHOR
    # anchors whose canonical form has no @sl writtenRep: fall back to any of the
    # entry's inflected forms, so the lexical entry is still addressable.
    n_fallback = 0
    for lu, fcode in other:
        lu = int(lu)
        j = np.searchsorted(node_codes, lu)
        if j >= n_real or node_codes[j] != lu or text[j]:
            continue
        s = wr.get(int(fcode), "")
        if s:
            text[j] = TAG_ANCHOR + s + feat_string(feat_map.get(int(fcode), {}),
                                                   pos_map.get(lu),
                                                   unit_map.get(lu))
            kind[j] = K_ANCHOR
            n_fallback += 1
    if n_fallback:
        print(f"[text] {n_fallback:,} anchors recovered from an inflected form", flush=True)
    # inflected-form leaves
    for lu, fcode in other:
        fcode = int(fcode)
        j = np.searchsorted(node_codes, fcode)
        if j >= n_real or node_codes[j] != fcode or text[j]:
            continue
        t = form_text(fcode)
        if t:
            text[j] = TAG_FORM + t
            kind[j] = K_FORM
    # ---- tables that make sibling senses distinguishable -------------------
    # Only 2.7 % of senses carry a definition; the rest fall back to their
    # entry's lemma, which on a polysemous anchor makes every sibling sense node
    # byte-identical.  A dictionary ordinal fixes that unconditionally, and a
    # snippet of the first usage example fixes it meaningfully where there is
    # one.  Both are kept as sorted numpy arrays rather than dicts: at 8.5M
    # senses a Python dict of either costs ~1 GB on top of a 65 GB peak.
    empty64 = np.empty(0, dtype=np.int64)
    ex_se = ex_code = empty64                    # sense -> first example w/ text
    if snippet_chars > 0 and len(usage):
        u = np.unique(usage, axis=0)             # sorted by (sense, example)
        if len(val):
            vk = np.fromiter(val.keys(), dtype=np.int64, count=len(val))
            vk.sort()
            p = np.searchsorted(vk, u[:, 1])
            pc = np.clip(p, 0, max(len(vk) - 1, 0))
            u = u[(p < len(vk)) & (vk[pc] == u[:, 1])]
        if len(u):
            head = np.flatnonzero(np.r_[True, u[1:, 0] != u[:-1, 0]])
            ex_se = u[head, 0].copy(); ex_code = u[head, 1].copy()
        del u
    ord_se = empty64; ord_k = np.empty(0, dtype=np.int32)   # sense -> ordinal
    if sense_index and len(sense):
        su = np.unique(sense, axis=0)            # sorted by (lexical unit, sense)
        starts = np.flatnonzero(np.r_[True, su[1:, 0] != su[:-1, 0]])
        sizes = np.diff(np.r_[starts, len(su)])
        multi = np.repeat(sizes > 1, sizes)      # senses of polysemous entries
        ordv = np.arange(len(su), dtype=np.int32) - np.repeat(starts, sizes).astype(np.int32) + 1
        # su is sorted by (lexical unit, sense), so column 1 is grouped by entry
        # and NOT globally sorted -- it must be re-sorted before the searchsorted
        # lookups below, or they silently miss or mis-number.  A sense reachable
        # from two entries keeps the first ordinal, for determinism.
        ord_se = su[multi, 1]; ord_k = ordv[multi]
        o = np.argsort(ord_se, kind="stable")
        ord_se = ord_se[o]; ord_k = ord_k[o]
        keep = np.r_[True, ord_se[1:] != ord_se[:-1]] if len(ord_se) else np.empty(0, bool)
        ord_se = ord_se[keep].copy(); ord_k = ord_k[keep].copy()
        assert len(ord_se) < 2 or bool((np.diff(ord_se) > 0).all()), "ord_se unsorted"
        del su, ordv, multi, o, keep
    print(f"[sense] {len(ex_se):,} senses have a usable first example, "
          f"{len(ord_se):,} are in a polysemous entry", flush=True)

    # senses: definition, else the lemma; plus the ordinal and, where there is
    # no definition, a bounded first-example snippet.
    n_snip = n_ord = 0
    for se, lu in sense_lu.items():
        j = np.searchsorted(node_codes, se)
        if j >= n_real or node_codes[j] != se:
            continue
        d = dfn.get(se, "")
        body = d or lemma_of_lu(lu)
        if not body:
            continue
        if not d and len(ex_se):
            p = np.searchsorted(ex_se, se)
            if p < len(ex_se) and ex_se[p] == se:
                snip = sense_snippet(val.get(int(ex_code[p]), ""), snippet_chars)
                if snip:
                    body += TAG_SENSE_EX.format(snip)
                    n_snip += 1
        tag = TAG_SENSE
        if len(ord_se):
            p = np.searchsorted(ord_se, se)
            if p < len(ord_se) and ord_se[p] == se:
                tag = TAG_SENSE_N.format(int(ord_k[p])); n_ord += 1
        text[j] = tag + body
        kind[j] = K_SENSE
    print(f"[sense] {n_ord:,} numbered, {n_snip:,} carry an example snippet",
          flush=True)
    stats["sense_text"] = {"snippet_chars": int(snippet_chars),
                           "index": bool(sense_index),
                           "with_first_example": int(len(ex_se)),
                           "in_polysemous_entry": int(len(ord_se)),
                           "numbered": int(n_ord), "snippeted": int(n_snip)}
    del ex_se, ex_code, ord_se, ord_k
    # examples, translations, leftovers
    for i in range(n_real):
        if text[i]:
            continue
        c = int(node_codes[i]); t = ntype[i]
        if t == T_EXAMPLE:
            s = val.get(c) or dfn.get(c) or ""
            if s:
                text[i] = TAG_EX + s; kind[i] = K_EXAMPLE
        elif t == T_TRANSFORM:
            lt = wrf.get(c)
            if lt:
                lang = LANG_SL.get(lt[0], lt[0])
                text[i] = f"prevod ({lang}): {lt[1]}"; kind[i] = K_TRANS
            elif wr.get(c):
                text[i] = "prevod: " + wr[c]; kind[i] = K_TRANS
        else:
            s = dfn.get(c) or val.get(c) or wr.get(c) or ""
            if s:
                text[i] = s
    n_textless_real = sum(1 for t in text if not t)
    empty = np.array([not t for t in text], dtype=bool)
    tl_by_type = {}
    for t in np.unique(ntype[empty]):
        tl_by_type[TYPE_NAME.get(int(t), str(int(t)))] = int((ntype[empty] == t).sum())
    print(f"[text] {n_real - n_textless_real:,} with text, "
          f"{n_textless_real:,} textless ({100.0*n_textless_real/max(n_real,1):.2f}%) "
          f"by type: {tl_by_type}", flush=True)
    stats["textless_by_type"] = tl_by_type

    # ---- mint the reified nodes (collocation / synonym / antonym) ----------
    # Row t of `colloc_pairs` carries its text in `colloc_phrases[t]`.
    mint_src = []; mint_dst = []; mint_text = []; mint_kind = []
    next_id = n_real
    n_phrase = [0]

    def mint(pairs, tag, joiner, k, namer, phrase_of=None):
        """Reify each pairing as one node whose text names both endpoints.

        With `phrase_of`, the node carries the curated phrase instead whenever
        one resolves, falling back to the pair form when it does not.  Either
        way a pairing is skipped unless both members are real, named nodes; the
        node COUNT is set by the caller's dedup key, not here.
        """
        nonlocal next_id
        if not len(pairs):
            return 0
        # vectorised membership: both endpoints must be real nodes
        ia = np.searchsorted(node_codes, pairs[:, 0])
        ib = np.searchsorted(node_codes, pairs[:, 1])
        ok = ((ia < n_real) & (ib < n_real))
        ia = np.clip(ia, 0, n_real - 1); ib = np.clip(ib, 0, n_real - 1)
        ok &= (node_codes[ia] == pairs[:, 0]) & (node_codes[ib] == pairs[:, 1])
        idx = np.flatnonzero(ok)
        made = 0
        for t in idx:
            la = namer(pairs[t, 0]); lb = namer(pairs[t, 1])
            if not la or not lb:
                continue
            surface = phrase_of(t) if phrase_of is not None else ""
            if surface:
                mint_text.append(f"{tag}{surface}")
                n_phrase[0] += 1
            else:
                mint_text.append(f"{tag}{la} {joiner} {lb}")
            mint_kind.append(k)
            mint_src.append(next_id); mint_dst.append(int(ia[t]))
            mint_src.append(next_id); mint_dst.append(int(ib[t]))
            next_id += 1; made += 1
        return made

    n_syn = mint(syn_pairs, TAG_SYN, "~", K_SYN, lemma_of_sense)
    n_ant = mint(ant_pairs, TAG_ANT, "~", K_ANT, lemma_of_sense)
    n_col = mint(colloc_pairs, TAG_COLLOC, "+", K_COLLOC, lemma_of_sense,
                 phrase_of=((lambda t: colloc_phrases[t])
                            if colloc_text == "phrase" else None))
    print(f"[mint] {n_syn:,} sopomenka + {n_ant:,} protipomenka + {n_col:,} kolokacija "
          f"nodes ({len(mint_src):,} edges)", flush=True)
    if colloc_text == "phrase":
        print(f"[colloc] {n_phrase[0]:,} of {n_col:,} collocation nodes verbalised "
              f"({100.0*n_phrase[0]/max(n_col,1):.2f}%), "
              f"{n_col - n_phrase[0]:,} fell back to the lemma pair", flush=True)
    stats["minted"] = {"synonym": n_syn, "antonym": n_ant, "collocation": n_col,
                       "collocation_text": colloc_text,
                       "collocation_verbalised": int(n_phrase[0])}

    n = next_id
    text = text + mint_text
    kind = np.concatenate([kind, np.array(mint_kind, dtype=np.int8)]) if mint_kind else kind
    ntype = np.concatenate([ntype, np.full(len(mint_text), -1, dtype=np.int32)])
    if mint_src:
        si = np.concatenate([si, np.array(mint_src, dtype=np.int64)])
        di = np.concatenate([di, np.array(mint_dst, dtype=np.int64)])
    n_textless = sum(1 for t in text if not t)
    print(f"[nodes] final {n:,} nodes / {len(si):,} directed edges  "
          f"({n_textless:,} textless, {100.0*n_textless/max(n,1):.2f}%)", flush=True)

    indptr, dst2 = undirected_csr(si, di, n)
    print(f"[csr] {len(dst2):,} half-edges  {time.time()-t0:.0f}s", flush=True)

    is_form_leaf = (kind == K_FORM)
    is_example   = (kind == K_EXAMPLE)
    is_colloc    = (kind == K_COLLOC)
    mwe_set = np.zeros(n, dtype=bool)
    if len(mwe_word):
        mi = np.searchsorted(node_codes, np.unique(mwe_word[:, 0]))
        mi = mi[(mi < n_real)]
        mwe_set[mi] = True

    stats["global"] = {"nodes": int(n), "edges": int(len(si)),
                       "nodes_iri": int(n_real), "textless": int(n_textless),
                       "textless_pct": round(100.0 * n_textless / max(n, 1), 3)}
    return dict(n=n, n_real=n_real, node_codes=node_codes, ntype=ntype, kind=kind,
                indptr=indptr, indices=dst2, text=text,
                is_form_leaf=is_form_leaf, is_example=is_example,
                is_colloc=is_colloc, mwe_set=mwe_set)


# ---------------------------------------------------------------------------
def ragged_gather(frontier, indptr, indices):
    starts = indptr[frontier]; lengths = indptr[frontier + 1] - starts
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=indices.dtype)
    inc = np.ones(total, dtype=np.int64); inc[0] = starts[0]
    off = np.cumsum(lengths)[:-1]
    inc[off] += starts[1:] - (starts[:-1] + lengths[:-1])
    return indices[np.cumsum(inc)]


PCTS = [10, 25, 50, 75, 90, 95, 99]
def summarize(vals):
    a = np.array(vals, dtype=np.float64)
    if len(a) == 0:
        return {"n": 0}
    out = {"n": len(a), "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}
    for p in PCTS:
        out[f"p{p}"] = float(np.percentile(a, p))
    return out


METRICS = ("nodes", "edges", "tokens")

def analyze_variant(G, seeds, max_hops, token_len, prompt_tokens, active):
    n = G["n"]; indptr = G["indptr"]; indices = G["indices"]
    depth = np.full(n, -1, dtype=np.int8)
    per = {k: defaultdict(list) for k in range(1, max_hops + 1)}
    for seed in seeds:
        touched = [seed]; depth[seed] = 0; frontier = np.array([seed], dtype=np.int64)
        for d in range(1, max_hops + 1):
            nbr = ragged_gather(frontier, indptr, indices)
            if nbr.size:
                nbr = np.unique(nbr)
                nbr = nbr[active[nbr]]
                new = nbr[depth[nbr] < 0]
            else:
                new = np.empty(0, dtype=np.int64)
            depth[new] = d; touched.extend(new.tolist()); frontier = new
        touched = np.array(touched, dtype=np.int64)
        for k in range(1, max_hops + 1):
            ball = touched[depth[touched] <= k]
            nbrs = ragged_gather(ball, indptr, indices)
            induced = int(((depth[nbrs] >= 0) & (depth[nbrs] <= k)).sum()) // 2
            per[k]["nodes"].append(ball.shape[0])
            per[k]["edges"].append(induced)
            per[k]["tokens"].append(int(token_len[ball].sum()) + prompt_tokens)
        depth[touched] = -1
    return per


def tokenize_texts(texts, n, tokenizer, no_tokenizer=False):
    """Fill `token_len` for every node text.  Returns (token_len, name).

    Tokenizes each DISTINCT text once -- the graph has ~37.5 M nodes over far
    fewer distinct strings, and the tokenizer is the slowest part of the build.
    """
    if no_tokenizer:
        return (np.array([max(1, len(s) // 4) if s else 0 for s in texts],
                         dtype=np.int32),
                "char/4 proxy")

    os.environ.setdefault("HF_HOME", HF_CACHE)
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer)
    print(f"[tok] {tokenizer} loaded (vocab {len(tok):,})", flush=True)
    token_len = np.zeros(n, dtype=np.int32)
    uniq = defaultdict(list)
    for i, s in enumerate(texts):
        if s:
            uniq[s].append(i)
    keys = list(uniq.keys())
    print(f"[tok] {len(keys):,} unique node texts", flush=True)
    B = 20000
    for j in range(0, len(keys), B):
        chunk = keys[j:j+B]
        enc = tok(chunk, add_special_tokens=False)["input_ids"]
        for s, ids in zip(chunk, enc):
            L = len(ids)
            for idx in uniq[s]:
                token_len[idx] = L
    del uniq, keys
    return token_len, tokenizer


def save_store(out_dir, G, token_len, stats, *, kg_dir, n_files, files_limit,
               sense_snippet, sense_index, colloc_text, tok_name):
    """Persist a built graph, recording what built it.  Returns the manifest."""
    return graph_store.save_graph(
        out_dir, G, token_len, stats=stats,
        meta={"tokenizer": tok_name, "kg_dir": kg_dir,
              "n_files": n_files, "files_limit": files_limit,
              "sense_snippet": sense_snippet,
              "sense_index": sense_index,
              "colloc_text": colloc_text,
              # A store declares which text convention built it, so a reader
              # never has to infer it from the directory name.  Two switches
              # move it: whether collocation nodes carry their curated phrase
              # or a bare lemma pair, and whether noun entries render their
              # gender on the anchor.  Derived from those switches rather
              # than hard-coded, so a store built with either one turned off
              # cannot claim to be current.
              "text_convention": "+".join(
                  ["collocation-phrases" if colloc_text == "phrase"
                   else "collocation-pairs"]
                  + (["entry-gender"] if "gender" in UNIT_PROPS else [])),
              "feature_props": list(FEATURE_PROPS),
              "unit_props": list(UNIT_PROPS),
              "builder": os.path.basename(__file__),
              "builder_sha256": hashlib.sha256(
                  open(os.path.abspath(__file__), "rb").read()).hexdigest()})


def build_store(out_dir, tokenizer=DEFAULT_TOKENIZER, kg_dir=KG_RAW_DIR,
                workers=None, colloc_text="phrase",
                sense_snippet=SENSE_SNIPPET_CHARS, sense_index=True,
                files_limit=0, no_tokenizer=False):
    """Parse the raw KG, tokenize it and save it as a store.  The manifest.

    What `--save-graph ... --no-analysis` does from the command line, as one
    call, so `data/pipeline/` can build the store without shelling out.  Both
    paths run the same three steps below; neither reimplements the other.
    """
    if workers is None:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    files = sorted(glob.glob(os.path.join(kg_dir, "*.nt")))
    if files_limit:
        files = files[:files_limit]
    if not files:
        raise FileNotFoundError(
            f"no .nt files in {kg_dir} -- the raw KG is a separate 83 GB "
            f"download, see the repo README")
    print(f"KG dir: {kg_dir}  ({len(files)} .nt files)", flush=True)

    stats = {}
    G = build(files, workers, stats, snippet_chars=sense_snippet,
              sense_index=sense_index, colloc_text=colloc_text)
    token_len, tok_name = tokenize_texts(G["text"], G["n"], tokenizer,
                                         no_tokenizer)
    return save_store(out_dir, G, token_len, stats, kg_dir=kg_dir,
                      n_files=len(files), files_limit=files_limit,
                      sense_snippet=sense_snippet, sense_index=sense_index,
                      colloc_text=colloc_text, tok_name=tok_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg-dir", default=KG_RAW_DIR)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "sizing_study.json"))
    ap.add_argument("--n-seeds", type=int, default=400)
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    ap.add_argument("--files-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-tokens", type=int, default=24)
    ap.add_argument("--sense-snippet", type=int, default=SENSE_SNIPPET_CHARS,
                    help="chars of the first usage example folded into a "
                         "definition-less sense's text (0 disables)")
    ap.add_argument("--no-sense-index", action="store_true",
                    help="do not number the senses of a polysemous entry")
    ap.add_argument("--colloc-text", choices=("phrase", "pair"), default="phrase",
                    help="collocation node text: 'phrase' resolves the inflected "
                         "surface string through the multi-word entry named in the "
                         "frac:Collocation IRI (kolokacija: kisova voda); 'pair' is "
                         "the bare lemma-pair form (kolokacija: kisov + voda). "
                         "'phrase' falls back to 'pair' per node when it does not "
                         "resolve, so the graph structure is identical either way.")
    ap.add_argument("--variants", default="",
                    help="comma-separated subset of variant names (default: all 8)")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                    help=f"HF repo whose tokenizer fills token_len "
                         f"(default: {DEFAULT_TOKENIZER}; every Gemma 3 size "
                         f"gives identical ids, so the choice among them is "
                         f"cosmetic). Recorded in the store manifest.")
    ap.add_argument("--no-tokenizer", action="store_true",
                    help="skip the tokenizer; use character/4 as a token proxy "
                         "(smoke tests)")
    ap.add_argument("--dump-samples", type=int, default=0,
                    help="print N sample node texts per kind and exit after building")
    ap.add_argument("--save-graph", default="",
                    help="persist the built graph (CSR + node text + token_len) "
                         "to this directory, so later runs can skip the ~12min build")
    ap.add_argument("--load-graph", default="",
                    help="load the graph from a --save-graph directory instead of "
                         "parsing the raw KG; --kg-dir and --workers are then unused")
    ap.add_argument("--no-analysis", action="store_true",
                    help="stop after building (and optionally saving) the graph; "
                         "skips the 8 sizing variants")
    args = ap.parse_args()

    if args.load_graph and args.save_graph:
        ap.error("--load-graph and --save-graph are mutually exclusive")

    if args.load_graph:
        files = []
        G = graph_store.load_graph(args.load_graph)
        stats = G["stats"]
        token_len = np.asarray(G["token_len"])
        _meta = G["manifest"].get("meta", {})
        tok_name = _meta.get("tokenizer", "unknown (from store)")
        n_files = int(_meta.get("n_files", 0))
    else:
        files = sorted(glob.glob(os.path.join(args.kg_dir, "*.nt")))
        if args.files_limit:
            files = files[:args.files_limit]
        n_files = len(files)
        print(f"KG dir: {args.kg_dir}  ({n_files} .nt files)", flush=True)
        stats = {}
        G = build(files, args.workers, stats,
                  snippet_chars=args.sense_snippet,
                  sense_index=not args.no_sense_index,
                  colloc_text=args.colloc_text)
    n = G["n"]; kind = G["kind"]; texts = G["text"]

    if args.dump_samples:
        names = {K_ANCHOR: "anchor", K_FORM: "form", K_SENSE: "sense",
                 K_EXAMPLE: "example", K_TRANS: "translation",
                 K_COLLOC: "collocation", K_SYN: "synonym", K_ANT: "antonym",
                 K_OTHER: "other"}
        for k, nm in names.items():
            idx = np.flatnonzero(kind == k)[:args.dump_samples]
            print(f"\n--- {nm} ({int((kind == k).sum()):,} nodes) ---", flush=True)
            for i in idx:
                print(f"    {texts[i][:160]!r}", flush=True)
        return

    if args.load_graph:
        pass                      # token_len + tok_name came from the store
    else:
        token_len, tok_name = tokenize_texts(texts, n, args.tokenizer,
                                             args.no_tokenizer)

    if args.save_graph:
        save_store(args.save_graph, G, token_len, stats,
                   kg_dir=args.kg_dir, n_files=n_files,
                   files_limit=args.files_limit,
                   sense_snippet=args.sense_snippet,
                   sense_index=not args.no_sense_index,
                   colloc_text=args.colloc_text, tok_name=tok_name)

    if args.no_analysis:
        print("[done] --no-analysis: stopping after the build", flush=True)
        return

    lu_nodes = np.flatnonzero((kind == K_ANCHOR) & (token_len > 0))
    print(f"[seeds] {lu_nodes.shape[0]:,} candidate anchor nodes", flush=True)
    rng = np.random.default_rng(args.seed)
    ns = min(args.n_seeds, lu_nodes.shape[0])
    seeds = rng.choice(lu_nodes, size=ns, replace=False)
    mwe_set = G["mwe_set"]
    seeds_word = seeds[~mwe_set[seeds]]
    seeds_mwe = seeds[mwe_set[seeds]]
    print(f"[seeds] {ns} total ({len(seeds_word)} word, {len(seeds_mwe)} MWE)", flush=True)

    keep_form = ~G["is_form_leaf"]; keep_ex = ~G["is_example"]; keep_col = ~G["is_colloc"]
    ones = np.ones(n, dtype=bool)
    all_variants = {}
    for fm, fmask in (("expand", ones), ("collapse", keep_form)):
        for ex, emask in (("examples", ones), ("noexamples", keep_ex)):
            for co, cmask in (("colloc", ones), ("nocolloc", keep_col)):
                all_variants[f"{fm}_{ex}_{co}"] = fmask & emask & cmask
    if args.variants:
        want = [v.strip() for v in args.variants.split(",") if v.strip()]
        all_variants = {k: v for k, v in all_variants.items() if k in want}

    result = {"meta": {"version": 3, "n_seeds": int(ns),
                       "n_seeds_word": int(len(seeds_word)),
                       "n_seeds_mwe": int(len(seeds_mwe)),
                       "max_hops": args.max_hops,
                       "prompt_tokens": args.prompt_tokens,
                       "tokenizer": tok_name, "n_files": n_files,
                       "seed": args.seed,
                       "note": "untyped edges; no Levi reification; "
                               "levi_nodes would be nodes+edges"},
              "stats": stats, "variants": {}}
    for vname, active in all_variants.items():
        t0 = time.time()
        per_all = analyze_variant(G, seeds, args.max_hops, token_len, args.prompt_tokens, active)
        per_w = analyze_variant(G, seeds_word, args.max_hops, token_len, args.prompt_tokens, active) if len(seeds_word) else None
        per_m = analyze_variant(G, seeds_mwe, args.max_hops, token_len, args.prompt_tokens, active) if len(seeds_mwe) else None
        vres = {}
        for k in range(1, args.max_hops + 1):
            entry = {"all": {m: summarize(per_all[k][m]) for m in METRICS}}
            if per_w: entry["word"] = {m: summarize(per_w[k][m]) for m in METRICS}
            if per_m: entry["mwe"] = {m: summarize(per_m[k][m]) for m in METRICS}
            vres[str(k)] = entry
        result["variants"][vname] = vres
        print(f"[variant] {vname} done {time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print("[done] wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
