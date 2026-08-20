#!/usr/bin/env python3
"""
v3 builder + sizing analysis for the GTLM-compatible CJVT graph.

Fixes the four defects recorded in data/README.md "Known flaws" for v2:

  1. NO BLANKET LEVI REIFICATION.  Edges are untyped (GTLM's native TextGraph
     format).  The relation is recoverable from the endpoint texts, because
     every node's text is self-describing and carries its type tag:
        iztocnica: pes (samostalnik, imenovalnik, ednina)
        oblika: psa (rodilnik, ednina)
        pomen: <definition>
        zgled: <example sentence>
        prevod (madzarsko): <hun text>
     The only relations that were NOT a function of (src type, dst type) --
     synonym vs antonym, both sense->sense -- are reified as their own nodes
     with self-describing text ("sopomenka: pes ~ kuza"), as are collocations.

  2. COLLOCATIONS ARE WIRED IN.  v2 parsed only frac:head and left 4.8M
     frac:Collocation IRIs as textless degree-1 leaves.  v3 parses rdfs:member.
     Verified shape in the raw KG:
        <dependent-sense-D-lexical-unit-H> rdfs:member  <sense-X> .
        <dependent-sense-D-lexical-unit-H> rdfs:member  <sense-Y> .
        <dependent-sense-D-lexical-unit-H> frac:head    <lexical-unit-H> .
     The same pairing is reified once per participant (identical member set,
     different frac:head), so pairings are DEDUPLICATED BY MEMBER SET.  frac:head
     is an indexing head, not a grammatical one, so it is dropped and the
     collocation node is symmetric.  Members are SENSES (not lexical units), so
     the collocation node attaches to the two senses.
     v3/v4 gave the node the two lemmas joined by a plus ("kolokacija:
     aplikativen + etnologija") on the belief that the inflected phrase was not
     stored anywhere.  THAT WAS WRONG -- see flaw 8.

  3. NO NODE-ID COLLISIONS.  v2's prefix_id_for hashed unknown IRI prefixes into
     400 buckets, so every `dependent-sense-D-lexical-unit` prefix collided.
     v3 packs a code as (type_id << 56) | payload, where payload is the numeric
     suffix for known prefixes, an exact (D << 28) | H packing for collocations,
     and a 56-bit blake2b of the full local name otherwise.

  4. TRANSLATIONS HAVE TEXT.  v2 filtered writtenRep to @sl, which silently
     dropped every translation (they are @hun) and left the whole vartrans chain
     textless.  v3 keeps foreign writtenRep, and COLLAPSES the chain: the
     textless `sense-translation` and `lexical-entry-translation` nodes are
     dropped and `translation-form` (which carries both the text and the link to
     the Slovenian source sense) is attached directly to that source sense.

Also new: POS is parsed (lexinfo:partOfSpeech on lexical units) and folded into
the anchor text, and `collocations` joins `form_mode` and `examples` as a toggle
so the cost of flaw #2 can be measured rather than assumed.

  5. SIBLING SENSES ARE DISTINGUISHABLE.  Only 225,618 of 8,468,227 senses carry
     a skos:definition (2.7%); the rest fell back to their entry's lemma, so on
     96.9% of polysemous anchors EVERY sibling sense node had byte-identical
     text ("pomen: pes", "pomen: pes").  With untyped edges and GTLM's node-
     permutation equivariance those nodes are genuinely interchangeable whenever
     the extractor prunes their subtrees -- label noise produced by the builder,
     not a coverage gap.  A sense now gets:
        - its dictionary ordinal, when its entry has more than one sense
          ("pomen 2: pes"), which makes siblings distinct unconditionally; and
        - a bounded snippet of its first usage example when it has no definition
          ("pomen 1: pes (zgled: Sosedov pes je spet lajal ...)"), which makes
          them distinct MEANINGFULLY for the 47.7% of such senses that have one.
     Definition-bearing and example-bearing senses are largely disjoint
     populations, which is exactly why the example fallback pays.  Controlled by
     --sense-snippet (0 disables) and --no-sense-index.

  6. LITERALS ARE UNESCAPED.  v3 passed the N-Triples literal through verbatim,
     so 3.48% of example nodes carried a literal backslash-quote:
        zgled: ... jih imam pravico tozniti,\\" pravi.
     writtenRep / value / definition now go through unescape_nt().  Only \\" and
     \\\\ occur in this dump; the rest of the escape set is handled anyway and
     unknown escapes pass through unchanged.

  7. MORPHOLOGY IS ACTUALLY MAPPED (v4 text convention).  Three separate faults
     kept verb morphology out of the node text entirely, so a whole verb
     paradigm collapsed to a handful of distinct strings:
        oblika: popraskam (ednina)      <- 1st person present
        oblika: popraskaj (ednina)      <- 2nd person IMPERATIVE, same text mod surface
        oblika: popraskata (dvojina)    <- 2nd AND 3rd person dual: BYTE-IDENTICAL
     (a) VALUE_SL listed person as firstPerson/secondPerson/thirdPerson; the KG
         emits first/second/third, and feat_string() drops what it cannot map,
         so person vanished from all 253,497 forms carrying it.
     (b) FEATURE_PROPS listed "tense" and "mood".  Neither predicate exists in
         this KG.  What carries the tense/mood distinction is lexinfo:vform
         (present / imperative / participle / infinitive / supine), which was
         not in FEATURE_PROPS at all -- 452,782 forms.
     (c) aspect and clitic hang off the LEXICAL-UNIT, but the feature branch
         only accepted word-form subjects, so they were dropped even when
         listed.  They are now collected separately (UNIT_PROPS) and rendered
         into the anchor parenthetical after the POS.
     definiteness is mapped too: without it an adjective's definite and
     indefinite forms carry byte-identical labels.  Measured effect: form nodes
     byte-identical to a sibling fall 89,405 -> 8,651 (-90.3%).
     NOTE on reachability: the corpus-wide counts for these predicates (aspect
     1.7M, vform 2.4M, clitic 424k) are dominated by lexical-unit-part subjects
     -- MWE components, which flaw #1's collapse discards.  Reachable on word
     entries: vform 452,782, person 253,497, definiteness 164,812, aspect
     18,157, clitic 25.

  8. COLLOCATIONS ARE VERBALISED (v5 text convention).  Flaw 2 concluded that the
     inflected phrase "is genuinely not stored".  It is stored -- just not on the
     collocation node, and not behind any edge.  A frac:Collocation node takes
     only rdfs:member / frac:head / rdf:type in all 42 GB and nothing ever points
     AT one (0 triples with a collocation in the object position), so no traversal
     from either side can reach the text.  The link is a NAMING CONVENTION:

         <dependent-sense-D-lexical-unit-H> rdf:type frac:Collocation
         <sense-D> ontolex:isSenseOf <lexical-unit-M>   (M a MultiWordExpression)
         <lexical-unit-M> canonicalForm/writtenRep -> "kisova voda"

     D is the sense of the multi-word entry that spells the pairing out.  The
     trap is that the OTHER id in that name, H, is restated as a real triple
     (frac:head), so the convention looks like pure redundancy -- and code_of()
     was already parsing D out to pack the node id, using it as an identifier
     ingredient and never dereferencing it.
     Measured over the raw dump: 4,717,090 of 4,717,090 collocation nodes
     resolve, 3,744,473 distinct phrases, 75.3% of them differing from their
     constituent lemmas concatenated -- agreement, word order, and the
     prepositions/copula that the pair form drops outright.  Independently, 91.5%
     of the 1,307 collocation phrases in data/datasets/reference/ come back
     verbatim from this path alone, which also settles where that file's
     phrases came from (this export, not the DDDS API).
     Controlled by --colloc-text {phrase,pair}; the skip condition stays on the
     members, so structure is byte-identical to a v4 store of the same tokenizer.

Analysis: 2 (form_mode) x 2 (examples) x 2 (collocations) = 8 variants, reported
with percentiles, split by seed kind (single word vs MWE).  Tokens = node-text
tokens + prompt (no relation-label tokens: there are no relation labels).  For
comparison with v2 we also report the induced edge count, so levi_nodes would be
nodes + edges.
"""
import os, re, sys, glob, json, time, argparse, hashlib
from collections import defaultdict
import numpy as np
from multiprocessing import Pool

# graph_store lives in data/lib/, one level up and over: it is shared with the
# lookup CLI and the analysis scripts, so it is not owned by the builder.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import graph_store

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
# Order matters: it is the order the parenthetical is rendered in.  vform and
# person lead so a verb form reads "(sedanjik, 1. oseba, ednina)"; case/number/
# gender/degree keep their v3 relative order, so every NOMINAL string is
# byte-identical to what v3.1 produced (asserted by run_save_v4_*.sbatch).
FEATURE_PROPS = ("vform", "person", "case", "number", "gender", "degree",
                 "definiteness")
# Properties that sit on the lexical-unit rather than on a word-form.  They are
# rendered into the ANCHOR parenthetical, right after the POS.
UNIT_PROPS = ("aspect", "clitic")
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

# Default tokenizer for token_len.  Every Gemma 3 checkpoint -- 270m, 1b, 4b,
# 12b, 27b, pt and it alike -- and GaMS3-12B-Instruct share one tokenizer:
# tokenizer.model is byte-identical across all of them, and encoding 5,000 real
# node texts gives byte-identical *ids*, not merely equal counts (verified
# 2026-08-20).  So one build serves the whole iteration ladder, and the name
# below is a label for which of the interchangeable repos was loaded.
# The older cjvt/GaMS-2B (Gemma 2, vocab 256,000) is NOT in that family: it
# costs +0.8% tokens in aggregate on this graph (920,680,698 vs 913,315,688 over
# all 36.7M nodes), and agrees node for node only 45.97% of the time.
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
# definition, so the only escapes that actually occur are \" (1,461 per 400k
# lines) and \\ (12), but \uXXXX and the C-style set are handled too, and an
# unrecognised escape passes through untouched rather than being silently eaten.
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
                    if local in FEATURE_PROPS:
                        sc = code_of(subj)
                        if sc is not None and (sc >> TYPE_SHIFT) in (T_WORDFORM, T_FORMLU):
                            feat.append((sc, local, _localname(oiri)))
                    elif local in UNIT_PROPS:
                        # aspect/clitic hang off the lexical-unit, not off a
                        # word-form, so the guard above would drop them.  T_PART
                        # is deliberately excluded: MWE components carry the
                        # bulk of these triples and are collapsed away, so their
                        # values are unreachable in the built graph anyway.
                        sc = code_of(subj)
                        if sc is not None and (sc >> TYPE_SHIFT) == T_LU:
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

    Order is POS, then the lexical-unit properties (aspect / clitic), then the
    word-form features in FEATURE_PROPS order.
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

    Used for collocations, where the key is the frac:Collocation IRI code.  The
    same pairing is reified once per participant -- identical member set,
    different frac:head -- and the duplicates differ only in the frac:head half
    of the IRI, so they all carry the same sense id in the other half.  Which
    representative survives therefore does not affect the verbalisation; the
    lexsort makes the choice deterministic anyway.
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


# ---------------------------------------------------------------------------
def build(files, workers, stats, snippet_chars=SENSE_SNIPPET_CHARS,
          sense_index=True, colloc_text="phrase"):
    t0 = time.time()
    print(f"[parse] {len(files)} files x {workers} workers", flush=True)
    agg = {k: [] for k in EDGE_KEYS}
    wr = {}; wrf = {}; dfn = {}; val = {}
    feat_map = defaultdict(dict)
    unit_map = defaultdict(dict)
    pos_map = {}
    n_wr_multi = [0]
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(parse_file, files, chunksize=4)):
            for k in agg:
                if res[k].shape[0]:
                    agg[k].append(res[k])
            for c, s in res["wr"]:
                # A form may carry SEVERAL writtenRep values -- e.g. word-form-1911547
                # has "BOJ", "Boj" and "boj".  A plain dict assignment keeps an
                # arbitrary one (this is a latent v2 bug: it is why anchors came out
                # as "BOJ").  Prefer the least-capitalised variant, then the
                # lexicographically smallest, so the choice is deterministic and the
                # dictionary lemma wins over sentence-initial / acronym variants.
                old = wr.get(c)
                if old is None:
                    wr[c] = s
                elif s != old:
                    n_wr_multi[0] += 1
                    if (sum(1 for ch in s if ch.isupper()), s) < \
                       (sum(1 for ch in old if ch.isupper()), old):
                        wr[c] = s
            for c, l, s in res["wrf"]:  wrf[c] = (l, s)
            for c, s in res["dfn"]:     dfn[c] = s
            for c, s in res["val"]:     val[c] = s
            for c, p, v in res["feat"]: feat_map[c][p] = v
            for c, p, v in res["unit"]: unit_map[c][p] = v
            for c, p in res["pos"]:     pos_map[c] = p
            if (i + 1) % 200 == 0:
                print(f"[parse] {i+1}/{len(files)}  {time.time()-t0:.0f}s", flush=True)

    def cat(k):
        return (np.concatenate(agg[k], axis=0) if agg[k]
                else np.empty((0, 2), dtype=np.int64))
    canon = cat("canon"); other = cat("other"); sense = cat("sense")
    # FLAW 4 (cont.): `lexical-entry-translation-N ontolex:sense sense-translation-N`
    # would re-introduce the two textless chain nodes we mean to collapse away.
    n_chain = 0
    if len(sense):
        keep = (sense[:, 0] >> TYPE_SHIFT) != T_ENTRYTR
        n_chain += int((~keep).sum())
        sense = sense[keep]
    if len(canon):
        # ...and `lexical-entry-translation-N canonicalForm translation-form-N`
        # would re-add the entry node via the canonical-form map.
        keep = (canon[:, 0] >> TYPE_SHIFT) != T_ENTRYTR
        n_chain += int((~keep).sum())
        canon = canon[keep]
    if n_chain:
        print(f"[trans] dropped {n_chain:,} translation-chain edges "
              f"(collapsed onto translation-form)", flush=True)
    syn = cat("syn"); ant = cat("ant"); usage = cat("usage")
    constit = cat("constit"); corr = cat("corr")
    tsrc = cat("tsrc"); member = cat("member")
    del agg
    print(f"[parse] done {time.time()-t0:.0f}s  canon={len(canon):,} other={len(other):,} "
          f"sense={len(sense):,} usage={len(usage):,} constit={len(constit):,} "
          f"corr={len(corr):,} tsrc={len(tsrc):,} member={len(member):,} "
          f"syn={len(syn):,} ant={len(ant):,} wr={len(wr):,} wrf={len(wrf):,} "
          f"dfn={len(dfn):,} val={len(val):,} feat={len(feat_map):,} "
          f"unit={len(unit_map):,} pos={len(pos_map):,}",
          flush=True)
    n_vform = sum(1 for d in feat_map.values() if "vform" in d)
    n_person = sum(1 for d in feat_map.values() if "person" in d)
    n_defnt = sum(1 for d in feat_map.values() if "definiteness" in d)
    n_aspect = sum(1 for d in unit_map.values() if "aspect" in d)
    n_clitic = sum(1 for d in unit_map.values() if "clitic" in d)
    print(f"[feat] reachable on word entries: vform={n_vform:,} person={n_person:,} "
          f"definiteness={n_defnt:,} aspect={n_aspect:,} clitic={n_clitic:,}",
          flush=True)
    stats["raw"] = {k: int(v) for k, v in dict(
        canon=len(canon), other=len(other), sense=len(sense), usage=len(usage),
        constit=len(constit), corr=len(corr), tsrc=len(tsrc), member=len(member),
        syn=len(syn), ant=len(ant), writtenrep_sl=len(wr), writtenrep_foreign=len(wrf),
        definitions=len(dfn), values=len(val), pos=len(pos_map),
        feat_forms=len(feat_map), unit_props=len(unit_map),
        vform=n_vform, person=n_person, definiteness=n_defnt,
        aspect=n_aspect, clitic=n_clitic,
        writtenrep_multivalued=n_wr_multi[0]).items()}

    # ---- collapse MWE decomposition ---------------------------------------
    if len(constit) and len(corr):
        order = np.argsort(corr[:, 0], kind="stable")
        parts_s = corr[order, 0]; words_s = corr[order, 1]
        idx = np.clip(np.searchsorted(parts_s, constit[:, 1]), 0, len(parts_s) - 1)
        hit = parts_s[idx] == constit[:, 1]
        mwe_word = np.stack([constit[hit, 0], words_s[idx[hit]]], axis=1)
        mwe_word = np.unique(mwe_word, axis=0)
    else:
        mwe_word = np.empty((0, 2), dtype=np.int64)
    del constit, corr
    print(f"[collapse] {len(mwe_word):,} MWE->word constituent edges", flush=True)

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

    # ---- FLAW 2: collocations from rdfs:member ----------------------------
    # Each pairing is reified once per participant with an identical member set,
    # so group members by collocation IRI and dedup by the member set itself.
    colloc_pairs = np.empty((0, 2), dtype=np.int64)
    colloc_iri = np.empty(0, dtype=np.int64)        # parallel to colloc_pairs
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
        # ms[starts] is the frac:Collocation IRI code itself.  v3/v4 discarded it
        # once the members were read; flaw 8 needs it, because the sense id it
        # encodes is the only route to the phrase's surface string.
        colloc_pairs, colloc_iri = _dedup_pairs_keyed(a, b, ms[starts])
        n_colloc_iris = len(bounds)
        print(f"[colloc] {n_colloc_iris:,} collocation IRIs, member-count hist={member_hist}, "
              f"{int(two.sum()):,} binary -> {len(colloc_pairs):,} distinct pairings",
              flush=True)
        stats["collocations"] = {"iris": int(n_colloc_iris),
                                 "member_count_hist": member_hist,
                                 "binary_iris": int(two.sum()),
                                 "distinct_pairings": int(len(colloc_pairs))}
    del member

    # ---- FLAW 1: reify only syn/ant (the sense->sense ambiguous class) -----
    syn_pairs = _dedup_pairs(syn[:, 0], syn[:, 1]) if len(syn) else np.empty((0, 2), np.int64)
    ant_pairs = _dedup_pairs(ant[:, 0], ant[:, 1]) if len(ant) else np.empty((0, 2), np.int64)
    print(f"[reify] syn={len(syn_pairs):,} ant={len(ant_pairs):,} pairings", flush=True)
    del syn, ant

    # ---- FLAW 4: collapse the translation chain ---------------------------
    # translation-form carries the @hun text AND vartrans:source -> sl sense.
    # sense-translation / lexical-entry-translation carry nothing: drop them.
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
    # ---- FLAW 5: make sibling senses distinguishable -----------------------
    # Two lookup tables, both kept as sorted numpy arrays rather than dicts: at
    # 8.5M senses a Python dict of either costs ~1 GB, and the build already
    # peaks at 65 GB.
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

    # ---- FLAW 8: the collocation's surface string --------------------------
    # A frac:Collocation node carries no text of its own -- rdfs:member, frac:head
    # and rdf:type are the ONLY predicates it ever takes, in all 42 GB, and nothing
    # in the dump ever points AT one (0 triples with a collocation in the object
    # position).  But its IRI names the sense of the multi-word entry that spells
    # the phrase out, and that entry does carry the text:
    #
    #     <dependent-sense-D-lexical-unit-H>  rdf:type  frac:Collocation
    #     <sense-D>          ontolex:isSenseOf     <lexical-unit-M>
    #     <lexical-unit-M>   rdf:type              ontolex:MultiWordExpression
    #     <lexical-unit-M>   ontolex:canonicalForm <form-lexical-unit-M>
    #     <form-lexical-unit-M> ontolex:writtenRep """kisova voda"""@sl
    #
    # so the phrase is three ordinary lookups away and sense_lu / canon_of / wr
    # already hold every one of them.  The link exists only as a naming
    # convention, never as a triple -- which is why v2/v3/v4 all missed it while
    # code_of() was parsing D out of that very IRI to pack the node id.
    #
    # Measured on the raw dump: 4,717,090 of 4,717,090 collocation nodes resolve,
    # 3,744,473 distinct phrases, and 75.3% of them differ from their constituent
    # lemmas concatenated (agreement, word order, and the prepositions/copula the
    # pair form drops entirely).  See data/README.md Finding 8.
    COLLOC_ID_MASK = (1 << 28) - 1

    def colloc_phrase(t):
        """Surface string for row t of colloc_pairs, or '' if it does not resolve."""
        iri = int(colloc_iri[t])
        if (iri >> TYPE_SHIFT) != T_COLLOC:
            return ""
        # code_of() packs the IRI as (T_COLLOC << 56) | (D << 28) | H whenever both
        # ids fit in 28 bits, which they do throughout this KG (max sense id ~1.0e7,
        # max lexical-unit id ~1.2e7, vs 2^28 = 2.7e8).  On the _hash56 fallback the
        # unpacked D is meaningless, but then it names no sense and we return ''.
        d = (iri >> 28) & COLLOC_ID_MASK
        return lemma_of_sense((T_SENSE << TYPE_SHIFT) | d)

    # ---- mint the reified nodes (collocation / synonym / antonym) ----------
    mint_src = []; mint_dst = []; mint_text = []; mint_kind = []
    next_id = n_real
    n_phrase = [0]

    def mint(pairs, tag, joiner, k, namer, phrase_of=None):
        """Reify each pairing as one node whose text names both endpoints.

        With `phrase_of`, the node instead carries the phrase itself when one
        resolves, falling back to the pair form when it does not.  The SKIP
        condition stays on the members either way, so the node count and order
        are identical with and without it and the store's structure is unchanged.
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
                 phrase_of=colloc_phrase if colloc_text == "phrase" else None)
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

    # ---- undirected CSR (traverse both directions) ------------------------
    src2 = np.concatenate([si, di]); dst2 = np.concatenate([di, si])
    oo = np.argsort(src2, kind="stable")
    src2 = src2[oo]; dst2 = dst2[oo]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, src2 + 1, 1); np.cumsum(indptr, out=indptr)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg-dir", default="/shared/workspace/povejmo/gams_gtlm/data/kg_raw/OntoLex DSB")
    ap.add_argument("--out", default="/shared/workspace/povejmo/gams_gtlm/data/analysis/results/results_v3.json")
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
                         "the v3/v4 lemma-pair form (kolokacija: kisov + voda). "
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
    elif args.no_tokenizer:
        token_len = np.array([max(1, len(s) // 4) if s else 0 for s in texts], dtype=np.int32)
        tok_name = "char/4 proxy"
    else:
        os.environ.setdefault("HF_HOME", "/shared/workspace/povejmo/huggingface_cache")
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        print(f"[tok] {args.tokenizer} loaded (vocab {len(tok):,})", flush=True)
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
        tok_name = args.tokenizer

    if args.save_graph:
        graph_store.save_graph(
            args.save_graph, G, token_len, stats=stats,
            meta={"tokenizer": tok_name, "kg_dir": args.kg_dir,
                  "n_files": n_files, "files_limit": args.files_limit,
                  "sense_snippet": args.sense_snippet,
                  "sense_index": not args.no_sense_index,
                  "colloc_text": args.colloc_text,
                  # A store declares which text convention built it.  v3.1
                  # rendered no morphology and wrote collocations as lemma pairs;
                  # v4 added vform/person/definiteness on forms and aspect/clitic
                  # on anchors; v5 verbalises the collocation nodes (flaw 8).
                  "text_convention": "v5" if args.colloc_text == "phrase" else "v4",
                  "feature_props": list(FEATURE_PROPS),
                  "unit_props": list(UNIT_PROPS),
                  "builder": os.path.basename(__file__),
                  "builder_sha256": hashlib.sha256(
                      open(os.path.abspath(__file__), "rb").read()).hexdigest()})

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
