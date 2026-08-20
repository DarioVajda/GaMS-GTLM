#!/usr/bin/env python3
"""
Look things up in the CJVT lexicographical KG.

There are two graphs on disk, and "look at the graph" means a different thing in
each.  Pick the layer by the question you are actually asking:

  store   "What does OUR GRAPH know about X?"
          The built store in `data/stores/kg_graph_v4_gemma3`: collapsed, deduplicated,
          every node carrying self-describing text.  This is what the QA
          generator and the model will see, so it is the layer that settles
          coverage questions -- if a fact is not here, the model cannot use it
          however plainly the export states it.  Loads in 0.0s.

  raw     "What does the EXPORT literally say?"
          The untouched N-Triples in `data/kg_raw/OntoLex DSB`.  Shows the
          original predicates and the intermediate nodes the builder collapsed
          away, and is the only way to tell an id that is genuinely absent from
          the export apart from one the builder dropped.  ~8s per lookup.

The five lookups:

  store word <spelling> [--also TYPES]   ~10s     spelling -> ranked nodes
  store id   <id> [--hops N] [--max N]   instant  unit -> its text + neighbours
  raw   id   <id> [--max N]              ~8s      unit -> its raw triples
  raw   iri  <iri> [--max N]             ~8s      any IRI -> its raw triples
  raw   word <spelling> [--lang TAG]     ~8s      spelling -> the word-form IRIs

...and their two options:

  --hops N   how far to walk out from the node (default: 1).  The only option
             that costs anything: hop 1 around a common word is already ~200
             nodes and hop 3 around a multi-word entry can reach millions.
  --max N    how many lines to print (default: 25 in `store`, 40 in `raw`).
             Display limit only -- it never changes what is walked or scanned,
             and the true counts are always in the header line.

Every lookup exits 0 when it finds something and 1 when it does not, so `raw id`
doubles as a test for "is this id in the export at all".

Run `kg_lookup.py store -h` / `kg_lookup.py raw -h`, or `-h` on any of the five,
for the exact semantics and a worked example.

`bin/lookup` in the repo root wraps this script, so with that directory on PATH
every command below can be typed as `lookup ...` from any working directory.
"""
import argparse, os, re, subprocess, sys

# realpath, not abspath: `bin/lookup` may be reached through a symlink, and the
# store and corpus paths below are all derived from this location.
HERE = os.path.dirname(os.path.realpath(__file__))   # data/lookup
DATA = os.path.dirname(HERE)                         # data
LIB = os.path.join(DATA, "lib")                      # data/lib -- graph_store
# Override with KG_STORE=/path/to/store to query an older build -- e.g.
# kg_graph_v4_gams2b (same graph and text, token_len from the Gemma 2
# tokenizer), kg_graph_v3_1_gemma3 (before verb morphology was rendered) or
# kg_graph_v3 (before the v3.1 sense-indexing and unescaping).
STORE = os.environ.get("KG_STORE") or os.path.join(
    DATA, "stores", "kg_graph_v4_gemma3")
RAW = os.path.join(DATA, "kg_raw", "OntoLex DSB")

TYPE_SHIFT = 56
T_LU = 1          # lexical-unit -> headword anchor
T_WORDFORM = 3    # word-form    -> inflected form leaf

# `kind.npy` values, from build_gtlm_graph_v3.py.  Unlike node_codes these cover
# the minted nodes too, so they are what lets a search reach collocations.
K_ANCHOR, K_FORM, K_SENSE, K_EXAMPLE = 0, 1, 2, 3
K_TRANS, K_COLLOC, K_SYN, K_ANT, K_OTHER = 4, 5, 6, 7, 8

KIND_NAME = {K_ANCHOR: "iztočnica", K_FORM: "oblika", K_SENSE: "pomen",
             K_EXAMPLE: "zgled", K_TRANS: "prevod", K_COLLOC: "kolokacija",
             K_SYN: "sopomenka", K_ANT: "protipomenka", K_OTHER: "drugo"}

# what --also accepts; the Slovenian names match what the node text actually says
KIND_ALIAS = {"iztocnica": K_ANCHOR, "iztočnica": K_ANCHOR, "headword": K_ANCHOR,
              "oblika": K_FORM, "form": K_FORM,
              "pomen": K_SENSE, "sense": K_SENSE, "senses": K_SENSE,
              "zgled": K_EXAMPLE, "example": K_EXAMPLE, "examples": K_EXAMPLE,
              "prevod": K_TRANS, "translation": K_TRANS,
              "kolokacija": K_COLLOC, "collocation": K_COLLOC,
              "kolokacije": K_COLLOC, "collocations": K_COLLOC,
              "sopomenka": K_SYN, "synonym": K_SYN, "sopomenke": K_SYN,
              "protipomenka": K_ANT, "antonym": K_ANT,
              "drugo": K_OTHER, "other": K_OTHER}

# always searched: these are what "is this word in the dictionary" means
BASE_KINDS = (K_ANCHOR, K_FORM)

# match quality, best first
T_EXACT, T_WORD, T_SUB = 0, 1, 2
TIER_NAME = {T_EXACT: "exact match", T_WORD: "whole word within",
             T_SUB: "substring of"}

# IRI prefixes -> CURIE, longest first so the specific ones win
PREFIXES = [
    ("https://www.cjvt.ddds.si/", ""),
    ("https://cjvt.ddds.si/", ""),
    ("http://www.lexinfo.net/ontology/3.0/lexinfo#", "lexinfo:"),
    ("http://www.w3.org/ns/lemon/ontolex#", "ontolex:"),
    ("http://www.w3.org/ns/lemon/decomp#", "decomp:"),
    ("http://www.w3.org/ns/lemon/vartrans#", "vartrans:"),
    ("http://www.w3.org/ns/lemon/lime#", "lime:"),
    ("http://www.w3.org/ns/lemon/frac#", "frac:"),
    ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:"),
    ("http://www.w3.org/2000/01/rdf-schema#", "rdfs:"),
    ("http://www.w3.org/2004/02/skos/core#", "skos:"),
]


def shorten(tok):
    if tok.startswith("<") and tok.endswith(">"):
        iri = tok[1:-1]
        for long, short in PREFIXES:
            if iri.startswith(long):
                return short + iri[len(long):]
        return iri
    return tok


def split_triple(line):
    """N-Triples line -> (s, p, o).  The object may be a quoted literal
    containing spaces, so only the first two gaps are split on."""
    line = line.rstrip("\n").rstrip()
    if line.endswith(" ."):
        line = line[:-2]
    s, p, o = line.split(" ", 2)
    return s, p, o


# --------------------------------------------------------------------- raw

def raw_grep(pattern, fixed=True, limit=None):
    tool = "rg" if _has("rg") else "grep"
    if tool == "rg":
        cmd = ["rg", "-N", "-F" if fixed else "-e", pattern, "-g", "*.nt", "."]
        if not fixed:
            cmd = ["rg", "-N", pattern, "-g", "*.nt", "."]
    else:
        cmd = ["grep", "-rh", "-F" if fixed else "-E", pattern, "--include=*.nt", "."]
    p = subprocess.run(cmd, cwd=RAW, capture_output=True, text=True)
    if p.returncode not in (0, 1):
        sys.exit(f"{tool} failed: {p.stderr.strip()}")
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    return lines[:limit] if limit else lines


def _has(prog):
    return subprocess.run(["which", prog], capture_output=True).returncode == 0


def cmd_raw_iri(args):
    suffix = args.iri
    lines = raw_grep(f"/{suffix}>")
    if not lines:
        print(f"{suffix}: no triples in the export")
        return 1
    out, inc = [], []
    target = f"/{suffix}>"
    for l in lines:
        try:
            s, p, o = split_triple(l)
        except ValueError:
            continue
        (out if s.endswith(target) else inc).append((s, p, o))
    print(f"{suffix}: {len(lines)} triples  ({len(out)} outgoing, {len(inc)} incoming)\n")
    for title, rows in (("outgoing", out), ("incoming", inc)):
        if not rows:
            continue
        print(f"--- {title} ---")
        shown = rows[:args.max]
        for s, p, o in shown:
            if title == "outgoing":
                print(f"  {shorten(p):<28} {shorten(o)}")
            else:
                print(f"  {shorten(s):<28} {shorten(p)}")
        if len(rows) > len(shown):
            print(f"  … {len(rows) - len(shown)} more (raise --max)")
        print()
    return 0


def cmd_raw_id(args):
    args.iri = f"lexical-unit-{args.id}"
    return cmd_raw_iri(args)


def cmd_raw_word(args):
    """writtenRep is on word-form / form-lexical-unit nodes, not on the
    lexical-unit, so this finds the *form* IRIs carrying the spelling."""
    # Literals in this export are triple-quoted: '''"""sprijaznjen"""@sl'''
    lines = raw_grep(f'"""{args.word}"""@{args.lang}')
    if not lines:
        print(f'"""{args.word}"""@{args.lang}: not in the export')
        return 1
    hits = []
    for l in lines:
        try:
            s, p, o = split_triple(l)
        except ValueError:
            continue
        hits.append((shorten(s), shorten(p), o))
    print(f'"""{args.word}"""@{args.lang}: {len(hits)} triples\n')
    for s, p, o in hits[:args.max]:
        print(f"  {s:<34} {p:<22} {o}")
    if len(hits) > args.max:
        print(f"  … {len(hits) - args.max} more (raise --max)")
    return 0


# ------------------------------------------------------------------- store

def load_store():
    sys.path.insert(0, LIB)
    import numpy as np
    from graph_store import load_graph
    G = load_graph(STORE, verbose=False)
    return np.asarray(G["node_codes"]), G


def lu_index(codes, lu_id):
    """Index of lexical-unit-<lu_id>, or None.  node_codes is sorted."""
    import numpy as np
    code = (T_LU << TYPE_SHIFT) | int(lu_id)
    i = int(np.searchsorted(codes, code))
    return i if i < codes.size and int(codes[i]) == code else None


# Anchor and form-leaf text.  The morphology parenthetical is optional: MWE
# anchors carry none ("iztočnica: dolarski milijonarji"), so requiring it made
# every multi-word headword unfindable.
_ANCHOR = re.compile(r"^iztočnica: (.+?)(?: \(|$)")
_FORM = re.compile(r"^oblika: (.+?)(?: \(|$)")


def _scan_blob(query):
    """Byte offsets of every case-insensitive occurrence of `query` in the text
    blob.

    One C-level `mmap.find` sweep per case variant, which is ~4s over the 2.7 GB
    blob.  A single case-folding regex is correct for more spellings but 4x
    slower, and the extra spellings it finds are mixed-case noise (`BoJ`): over
    the whole blob the two agree exactly for `jabolk` and `čebul`, and differ by
    8 hits in 327,165 for `boj`.  The limit is documented in `store word -h`.
    """
    import mmap
    import numpy as np
    path = os.path.join(STORE, "text_blob.bin")
    offs = []
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for variant in {query, query.lower(), query.upper(),
                            query.capitalize(), query.title()}:
                nb = variant.encode("utf-8")
                i = mm.find(nb)
                while i != -1:
                    offs.append(i)
                    i = mm.find(nb, i + 1)
        finally:
            mm.close()
    return np.unique(np.array(offs, dtype=np.int64)) if offs \
        else np.empty(0, dtype=np.int64)


# The "name" a node is known by, per kind -- what an exact match must equal.
# Senses, examples and translations have no name, only content.
_NAME_RE = {
    K_ANCHOR: re.compile(r"^iztočnica: (.+?)(?: \(|$)"),
    K_FORM: re.compile(r"^oblika: (.+?)(?: \(|$)"),
}
_PAIR_RE = {
    K_COLLOC: re.compile(r"^kolokacija: (.+?) \+ (.+)$"),
    K_SYN: re.compile(r"^sopomenka: (.+?) ~ (.+)$"),
    K_ANT: re.compile(r"^protipomenka: (.+?) ~ (.+)$"),
}


def _names(text, kind):
    m = _NAME_RE.get(kind)
    if m:
        g = m.match(text)
        return [g.group(1)] if g else []
    m = _PAIR_RE.get(kind)
    if m:
        g = m.match(text)
        return [g.group(1), g.group(2)] if g else []
    return []


def _tier(text, kind, want, word_re):
    """T_EXACT if the node's own name is the query, T_WORD if the query stands as
    a whole word anywhere in the text, else T_SUB."""
    if any(nm.lower() == want for nm in _names(text, kind)):
        return T_EXACT
    return T_WORD if word_re.search(text) else T_SUB


# code type id -> the IRI stem it came from, so every printed label can be fed
# straight back to `raw iri`.  Mirrors TYPE_NAME in build_gtlm_graph_v3.py.
_STEM = {1: "lexical-unit", 2: "lexical-unit-part", 3: "word-form",
         4: "form-lexical-unit", 5: "sense", 6: "sense-translation",
         7: "translation-form", 8: "example", 9: "resource",
         10: "lexical-entry-translation", 11: "collocation", 12: "other"}


def _node_label(codes, i):
    """`lexical-unit-69611` for a real node, `minted-<i>` for a collocation or
    pairing node, which the builder created and the export has no IRI for."""
    if i < codes.size:
        c = int(codes[i])
        return f"{_STEM.get(c >> TYPE_SHIFT, 'type' + str(c >> TYPE_SHIFT))}-" \
               f"{c & ((1 << TYPE_SHIFT) - 1)}"
    return f"minted-{i}"


def cmd_store_word(args):
    import numpy as np
    codes, G = load_store()
    txt, kind = G["text"], np.asarray(G["kind"])
    want = args.word.lower()
    word_re = re.compile(r"(?<!\w)" + re.escape(args.word) + r"(?!\w)",
                         re.IGNORECASE)

    extra = _parse_also(args.also)
    if extra is None:
        return 2
    wanted = set(BASE_KINDS) | extra

    offs = _scan_blob(args.word)
    if offs.size:
        text_off = np.asarray(np.load(os.path.join(STORE, "text_off.npy"),
                                      mmap_mode="r"))
        nodes = np.unique(np.searchsorted(text_off, offs, side="right") - 1)
        nodes = nodes[(nodes >= 0) & (nodes < kind.size)]
    else:
        nodes = np.empty(0, dtype=np.int64)

    if not nodes.size:
        print(f"{args.word}: nowhere in the store -- not as a headword, a form, "
              f"a sense, an example or a collocation")
        return 1

    kinds = kind[nodes]
    # Counts are vectorised and therefore exact even for a query with millions
    # of hits; only the nodes we are going to rank get their text decoded.
    total_by_kind = {int(k): int(c) for k, c in zip(*np.unique(kinds,
                                                               return_counts=True))}

    ranked, truncated = {}, {}
    for k in sorted(wanted):
        sel = nodes[kinds == k]
        if not sel.size:
            continue
        if sel.size > args.rank_cap:
            truncated[k] = int(sel.size)
            sel = sel[:args.rank_cap]
        rows = []
        for i in sel.tolist():
            s = txt[i]
            rows.append((_tier(s, k, want, word_re), _node_label(codes, i), s))
        rows.sort(key=lambda r: (r[0], r[1]))
        ranked[k] = rows

    _print_ranked(args, ranked, truncated, total_by_kind)

    # "not a headword" is about the EXACT tier: `jabolke` matches the headword
    # `jabolke in rabarbara` as a whole word, which is not the same as being a
    # dictionary entry in its own right.
    def has_exact(k):
        return any(t == T_EXACT for t, _l, _s in ranked.get(k, ()))

    if has_exact(K_FORM) and not has_exact(K_ANCHOR):
        print(f"\n{args.word!r} is an inflected form, not a headword of its own.")
    _print_form_headwords(G, codes, ranked.get(K_FORM, ()), args)

    _print_unsearched(args, wanted, total_by_kind)
    return 0


def _parse_also(spec):
    """`--also` value -> set of kind ids, or None on a bad name."""
    if not spec:
        return set()
    if spec.strip().lower() in ("all", "vse"):
        return set(KIND_NAME) - set(BASE_KINDS)
    out = set()
    for raw in spec.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name not in KIND_ALIAS:
            print(f"unknown node type {raw.strip()!r}.  Choose from: "
                  f"{', '.join(sorted(set(KIND_NAME.values())))}, or all",
                  file=sys.stderr)
            return None
        out.add(KIND_ALIAS[name])
    return out - set(BASE_KINDS)


def _print_ranked(args, ranked, truncated, total_by_kind):
    for k in sorted(ranked):
        rows = ranked[k]
        total = total_by_kind.get(k, len(rows))
        print(f"\n=== {KIND_NAME[k]} — {total:,} node(s) ===")
        if k in truncated:
            print(f"    (ranking computed over the first {args.rank_cap:,}; "
                  f"the count above is exact)")
        shown, last_tier = 0, None
        for tier, label, text in rows:
            if shown >= args.max:
                print(f"    … {len(rows) - shown:,} more (raise --max)")
                break
            if tier != last_tier:
                print(f"\n  [{TIER_NAME[tier]} {args.word!r}]")
                last_tier = tier
            print(f"    {label:<22} {text}")
            shown += 1


def _print_unsearched(args, wanted, total_by_kind):
    rest = {k: n for k, n in total_by_kind.items() if k not in wanted and n}
    if not rest:
        return
    total = sum(rest.values())
    breakdown = " · ".join(f"{KIND_NAME[k]} {n:,}"
                           for k, n in sorted(rest.items(), key=lambda x: -x[1]))
    print(f"\n{total:,} further occurrence(s) of {args.word!r} in node types "
          f"not searched:")
    print(f"  {breakdown}")
    print(f"  show them with:  kg_lookup.py store word {args.word!r} --also all")


def _print_form_headwords(G, codes, form_rows, args):
    """For the matched form leaves, the lexical units they inflect -- the answer
    to "which headword do I actually look up?".

    Only the best tier present is followed.  Taking every matched form instead
    made this useless: searching `brin` matches `brinirati` and `Brinarjev` by
    substring, and their headwords crowded out `brin` itself.
    """
    if not form_rows:
        return
    best = min(t for t, _l, _s in form_rows)
    form_rows = [r for r in form_rows if r[0] == best]
    if best != T_EXACT:
        print(f"\n  (no form is exactly {args.word!r}; following the "
              f"{TIER_NAME[best]} matches instead)")
    txt, indptr, indices = G["text"], G["indptr"], G["indices"]
    heads = {}
    # form_rows carry printable labels, not node indices; recover the ids
    wanted = {int(lab.rsplit("-", 1)[1]) for _t, lab, _s in form_rows
              if lab.startswith("word-form-")}
    if not wanted:
        return
    import numpy as np
    payload = codes & ((1 << TYPE_SHIFT) - 1)
    idx = np.nonzero(((codes >> TYPE_SHIFT) == T_WORDFORM)
                     & np.isin(payload, np.array(sorted(wanted),
                                                 dtype=np.int64)))[0]
    for i in idx.tolist():
        for v in indices[indptr[i]:indptr[i + 1]]:
            v = int(v)
            if v < codes.size and int(codes[v] >> TYPE_SHIFT) == T_LU:
                heads[int(codes[v] & ((1 << TYPE_SHIFT) - 1))] = txt[v]
    if heads:
        print(f"\n  headwords those forms belong to:")
        for lu, s in sorted(heads.items())[:args.max]:
            print(f"    lexical-unit-{lu:<10} {s}")


def cmd_store_id(args):
    import numpy as np
    codes, G = load_store()
    txt, indptr, indices = G["text"], G["indptr"], G["indices"]
    i = lu_index(codes, args.id)
    if i is None:
        print(f"lexical-unit-{args.id}: not in the store")
        return 1
    print(f"lexical-unit-{args.id}  (node {i})\n  {txt[i]}\n")
    frontier, seen = {i}, {i}
    for hop in range(1, args.hops + 1):
        nxt = set()
        for u in frontier:
            nxt.update(int(v) for v in indices[indptr[u]:indptr[u + 1]])
        nxt -= seen
        seen |= nxt
        print(f"--- hop {hop}: {len(nxt)} nodes ---")
        for v in sorted(nxt)[:args.max]:
            print(f"  {txt[v]}")
        if len(nxt) > args.max:
            print(f"  … {len(nxt) - args.max} more (raise --max)")
        print()
        frontier = nxt
        if not frontier:
            break
    return 0


def main():
    RAWD = argparse.RawDescriptionHelpFormatter

    ap = argparse.ArgumentParser(
        prog="kg_lookup.py", description=__doc__, formatter_class=RAWD,
        epilog="""example -- spelling to id, then id to neighbourhood

  $ kg_lookup.py store word milijonar
  milijonar: 2 lexical-unit node(s)

    lexical-unit-69611      iztočnica: milijonar (samostalnik, imenovalnik, ednina)
    lexical-unit-11997406   iztočnica: Milijonar (samostalnik, imenovalnik, ednina)

  $ kg_lookup.py store id 69611 --hops 1 --max 2
  lexical-unit-69611  (node 69609)
    iztočnica: milijonar (samostalnik, imenovalnik, ednina)

  --- hop 1: 220 nodes ---
    iztočnica: dolarski milijonarji
    iztočnica: ekscentrični milijonar
    … 218 more (raise --max)

Two ids for one spelling is normal: the low one is the core entry, the high one a
duplicate from the >1M range.  Feed either to `raw id` instead to see the triples
it was built from.""")
    sub = ap.add_subparsers(dest="layer", required=True, metavar="{store,raw}")

    # ---------------------------------------------------------------- store
    st = sub.add_parser(
        "store", formatter_class=RAWD,
        help="the built graph -- fast, resolved node text",
        description="""Query the built graph in `data/stores/kg_graph_v4_gemma3`.

Nodes here carry self-describing text -- `iztočnica: milijonar (samostalnik,
imenovalnik, ednina)`, `pomen 1: …`, `zgled: …`, `kolokacija: boj + kriminaliteta`
-- and edges are untyped, so what you see printed is what the model would be
given verbatim.

  word <spelling> [--also TYPES]  spelling -> the nodes carrying it, ranked
  id   <id> [--hops N] [--max N]  id -> its text, then its neighbourhood

Lookup by id is a binary search over `node_codes.npy` and is instant; lookup by
word scans the 2.7 GB text blob and takes ~10s.

Set KG_STORE=/path/to/store to query a different build (e.g.
`data/stores/kg_graph_v4_gams2b`, identical except for token_len, or the older
`data/stores/kg_graph_v3_1_gemma3`, before verb morphology was rendered).""")
    sst = st.add_subparsers(dest="what", required=True, metavar="{word,id}")

    p = sst.add_parser(
        "word", formatter_class=RAWD,
        help="spelling -> the lexical units and forms carrying it",
        description="""Find every node whose text spells WORD.

Two kinds of node can, and the difference matters:

  headwords       `iztočnica: jabolko (samostalnik, …)` -- a lexical unit, the
                  thing that has senses, collocations and a whole paradigm
  inflected forms `oblika: jabolke (rodilnik, ednina)` -- a leaf hanging off a
                  headword, carrying only its own morphology

A spelling that is only ever an inflected form has no lexical unit of its own,
which is correct, not a gap: *jabolke* is the genitive singular of *jabolko*
and the dictionary has no separate entry for it.  This lookup reports both, so
"not a headword" never reads as "not in the graph".

RESULTS ARE RANKED, not filtered.  Matching is case-insensitive and by
substring, then sorted into three tiers, best first:

  exact match         the node's own name IS the query -- the headword `jabolka`,
                      the form `jabolke`, either half of `kolokacija: X + Y`
  whole word within   the query stands as a word inside a longer text, so `brin`
                      reaches the MWE headword `divji brin`
  substring of        anything else -- `jabolk` reaching `jabolka`, `jabolke`

So a partial query still works, and the entries that ARE the word come first.

WHAT IS SEARCHED.  Always the 4.34M headwords and 8.56M forms.  The other 23.8M
nodes -- 12.1M examples, 8.5M senses, 3.0M collocations, 181k synonym and 3.4k
antonym pairings, 78k translations -- are searched only when you name them with
`--also`, because a common word appears in tens of thousands of example
sentences and would bury the dictionary entries.  They are always COUNTED
though: if a query turns up there, the footer says how many and prints the
command that shows them.  Nothing is silently invisible.

One scan of the 2.7 GB text blob: ~5s for the scan, ~10s for the command.  It
runs one `find` pass per case variant (`jabolk`, `JABOLK`, `Jabolk`), so a
mixed-case spelling like `jaBolk` is not matched -- measured against a full
case-folding scan, which costs 4x more, the two agree exactly on `jabolk` and
`čebul` and differ by 8 hits in 327,165 on `boj`.""",
        epilog="""examples

  $ kg_lookup.py store word jabolke

  === iztočnica — 0 node(s) ===          (nothing: it is not a headword)

  === oblika — 6 node(s) ===

    [exact match 'jabolke']
      word-form-5345608    oblika: jabolke (rodilnik, ednina)
      word-form-5345619    oblika: jabolke (imenovalnik, množina)
      …
    headwords those forms belong to:
      lexical-unit-11987403   iztočnica: jabolka (samostalnik, …)

  1,247 further occurrence(s) of 'jabolke' in node types not searched:
    zgled 1,203 · pomen 12 · kolokacija 32
    show them with:  kg_lookup.py store word 'jabolke' --also all

  $ kg_lookup.py store word jabolk --also kolokacija,pomen
  ... headwords and forms as above, then the two named types, each with its
  own exact / whole-word / substring tiers.""")
    p.add_argument("word", help="the spelling to look for, e.g. milijonar")
    p.add_argument("--also", metavar="TYPES", default="",
                   help="""extra node types to search, comma-separated, ON TOP
of the headwords and forms that are always searched:
  pomen         sense definitions      (8.5M)
  zgled         example sentences      (12.1M)
  kolokacija    collocation pairings   (3.0M)
  sopomenka     synonym pairings       (181k)
  protipomenka  antonym pairings       (3.4k)
  prevod        Hungarian translations (78k)
  drugo         anything else
`all` takes the lot.  English names (sense, example,
collocation, synonym, antonym, translation) work too.
Types you do not name are still counted in the footer.""")
    p.add_argument("--max", type=int, default=25, metavar="N",
                   help="how many nodes to print per type (default: 25).\n"
                        "Display limit only; the per-type header count and\n"
                        "the footer counts are always exact.")
    p.add_argument("--rank-cap", type=int, default=200_000, metavar="N",
                   help="how many nodes per type to decode for ranking\n"
                        "(default: 200000).  Counting is vectorised and stays\n"
                        "exact past this; only the tier ordering is capped, so\n"
                        "a query like `je` cannot stall the tool.")
    p.set_defaults(fn=cmd_store_word)

    p = sst.add_parser(
        "id", formatter_class=RAWD,
        help="lexical-unit id -> its text, then its neighbourhood",
        description="""Print the text of lexical-unit-<ID>, then walk outwards.

The walk is a breadth-first search over the undirected graph: edges are followed
in both directions, and each node is reported once, at the hop where it is first
reached.  Exits 1 if the id is not in the store.""",
        epilog="""example

  $ kg_lookup.py store id 69611 --hops 1 --max 4
  lexical-unit-69611  (node 69609)
    iztočnica: milijonar (samostalnik, imenovalnik, ednina)

  --- hop 1: 220 nodes ---
    iztočnica: dolarski milijonarji
    iztočnica: ekscentrični milijonar
    iztočnica: evrski milijonar
    iztočnica: milijonar in filantrop
    … 216 more (raise --max)

220 neighbours for one common noun is the normal shape of this graph, not an
error: most of them are the multi-word entries that contain the word.""")
    p.add_argument("id", type=int, help="numeric part of lexical-unit-<ID>")
    p.add_argument(
        "--hops", type=int, default=1, metavar="N",
        help="""how many edges to walk out from the anchor (default: 1).
Hop 1 is everything directly attached -- senses, inflected
forms, examples, collocations, and every multi-word entry the
word is a constituent of.  That is already 220 nodes for
`milijonar`, so hop 1 is not a short list.
Beyond that the cost grows violently and unevenly.  Measured
over 400 random seeds: the full 2-hop ball is 19 nodes at the
median but 740 at p99 for single words, while at 3 hops a
multi-word entry reaches 137k nodes at the median and 1.9M at
worst.  Use 1 or 2; treat 3 as a deliberate choice.""")
    p.add_argument(
        "--max", type=int, default=25, metavar="N",
        help="""how many nodes to PRINT per hop (default: 25).
Display limit only.  The walk still visits every node, so this
never makes the command faster -- --hops is what costs time --
and the "hop N: … nodes" header always reports the true count.
The N shown are the first by internal node index, which is
roughly creation order and unrelated to relevance, so read them
as an arbitrary sample, not as the top N.""")
    p.set_defaults(fn=cmd_store_id)

    # ------------------------------------------------------------------ raw
    rw = sub.add_parser(
        "raw", formatter_class=RAWD,
        help="the raw N-Triples export -- slow, original triples",
        description="""Grep the raw N-Triples in `data/kg_raw/OntoLex DSB`.

Output is the original triples with long IRIs shortened to CURIEs (`ontolex:`,
`lexinfo:`, `skos:`, …); the CJVT host prefix is stripped entirely, so a node
prints as plain `lexical-unit-34748`.

  id   <id> [--max N]               every triple touching lexical-unit-<id>
  iri  <iri> [--max N]              the same for any IRI -- word-form, sense,
                                    example, MWE part, collocation
  word <spelling> [--lang TAG]      the triples whose literal is this spelling

`--max` is a display limit in all three: the grep scans the whole export either
way, and the header always reports the true counts.

Every lookup scans all 42 GB: the `N-words.nt` / `N-multi.nt` numbering is an
arbitrary export batch, not an id range, so there is nothing to narrow to.
Expect ~8s with `rg` installed, ~10s with plain grep.  For batch work -- checking
hundreds of ids -- read `node_codes.npy` directly instead of calling this in a
loop.""",
        epilog="""three traps, if a hand-written grep ever disagrees with this tool

  * `rg lexical-unit-34748` over-matches: collocation IRIs are named
    `dependent-sense-<D>-lexical-unit-34748`, so the id is a suffix of a
    different node.  That grep returns 166 lines where only 109 are about the
    entry.  This tool searches `/lexical-unit-34748>`.
  * literals are triple-quoted: `\"\"\"sprijaznjen\"\"\"@sl`, not
    `\"sprijaznjen\"@sl`.
  * the spelling is not on the lexical unit.  `lexical-unit-N` carries POS and
    structure; the string lives on a `word-form` reached via
    `ontolex:canonicalForm`, and one form may carry several spellings
    (`BOJ` / `Boj` / `boj`).""")
    srw = rw.add_subparsers(dest="what", required=True, metavar="{id,iri,word}")

    p = srw.add_parser(
        "id", formatter_class=RAWD,
        help="lexical-unit id -> every raw triple touching it",
        description="""Show every raw triple touching lexical-unit-<ID>.

Results are split by direction: outgoing, where the unit is the subject (its
POS, canonical form, senses), and incoming, where it is the object (senses
pointing back, multi-word entries containing it, collocations headed on it).

Exits 1 if the id appears nowhere in the export -- which is how you prove an id
is genuinely missing rather than dropped by the builder.""",
        epilog="""example

  $ kg_lookup.py raw id 34748 --max 3
  lexical-unit-34748: 109 triples  (65 outgoing, 44 incoming)

  --- outgoing ---
    rdf:type                     ontolex:LexicalEntry
    lexinfo:partOfSpeech         lexinfo:adjective
    ontolex:canonicalForm        word-form-1567346
    … 62 more (raise --max)

  --- incoming ---
    sense-34748                  ontolex:isSenseOf
    lexical-unit-part-27367301   decomp:correspondsTo
    dependent-sense-14008590-lexical-unit-34748  frac:head
    … 41 more (raise --max)

  $ kg_lookup.py raw id 542411
  lexical-unit-542411: no triples in the export        # exit status 1""")
    p.add_argument("id", type=int, help="numeric part of lexical-unit-<ID>")
    p.add_argument("--max", type=int, default=40, metavar="N",
                   help="how many triples to print in EACH of the two\n"
                        "directions (default: 40).  Display limit only: the\n"
                        "header reports the true counts, and the grep costs\n"
                        "the same either way.  The N shown are the first in\n"
                        "corpus scan order, which carries no meaning.")
    p.set_defaults(fn=cmd_raw_id)

    p = srw.add_parser(
        "iri", formatter_class=RAWD,
        help="any IRI -> every raw triple touching it",
        description="""Show every raw triple touching any CJVT IRI.

Same as `raw id`, but for the node types that have no dedicated lookup: forms,
senses, examples, MWE parts, collocations.  Takes the IRI with the host
stripped, exactly as this tool prints it, so you can follow a reference straight
out of a previous result.""",
        epilog="""example -- following canonicalForm from `raw id 34748` to the spelling

  $ kg_lookup.py raw iri word-form-1567346
  word-form-1567346: 11 triples  (10 outgoing, 1 incoming)

  --- outgoing ---
    rdf:type                     ontolex:Form
    ontolex:writtenRep           \"\"\"sprijaznjen\"\"\"@sl
    lexinfo:case                 lexinfo:nominative
    …

  --- incoming ---
    lexical-unit-34748           ontolex:canonicalForm

other IRIs worth following: `sense-34748`, `lexical-unit-part-27367301`,
`dependent-sense-14008590-lexical-unit-34748`.""")
    p.add_argument("iri", help="IRI without the host, e.g. word-form-1567346")
    p.add_argument("--max", type=int, default=40, metavar="N",
                   help="how many triples to print in EACH of the two\n"
                        "directions (default: 40).  Display limit only.")
    p.set_defaults(fn=cmd_raw_iri)

    p = srw.add_parser(
        "word", formatter_class=RAWD,
        help="spelling -> the form IRIs carrying it (prefer `store word`)",
        description="""Find the raw triples whose literal is exactly WORD.

Matching is exact and case-sensitive, against the triple-quoted literal this
export uses.  What comes back are `word-form` IRIs, not lexical units, because
the spelling lives on the form -- feed one to `raw iri` to walk back to its
lexical unit.

Slower and blunter than `store word`, which goes straight to the lexical unit.
Use this when you need to see the untouched literal, or a language the store
does not surface.""",
        epilog="""example

  $ kg_lookup.py raw word sprijaznjen --max 3
  \"\"\"sprijaznjen\"\"\"@sl: 7 triples

    word-form-1567346    ontolex:writtenRep    \"\"\"sprijaznjen\"\"\"@sl
    word-form-1567350    ontolex:writtenRep    \"\"\"sprijaznjen\"\"\"@sl
    word-form-1567352    ontolex:writtenRep    \"\"\"sprijaznjen\"\"\"@sl
    … 4 more (raise --max)

Seven forms share one spelling because distinct case/number/gender readings are
spelled alike -- the ambiguity is real, not a duplicate.""")
    p.add_argument("word", help="the exact spelling, e.g. sprijaznjen")
    p.add_argument("--lang", default="sl", metavar="TAG",
                   help="literal language tag to match (default: sl).\n"
                        "Slovenian text is @sl; translations in this export\n"
                        "are @hun; pronunciations are @sl-SI-fonipa.")
    p.add_argument("--max", type=int, default=40, metavar="N",
                   help="how many triples to print (default: 40).\n"
                        "Display limit only.")
    p.set_defaults(fn=cmd_raw_word)

    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
