#!/usr/bin/env python3
"""
K-hop neighborhood size / Levi-graph size / GTLM token-length analysis for the
CJVT (DDDS) OntoLex-Lemon lexicographical KG.

Pipeline
--------
1. Stream-parse all `*.nt` files into a content graph:
     nodes = cjvt instance IRIs (lexical-unit / word-form / sense / example /
             lexical-unit-part / translation-form / ...), identified by a compact
             int64 code = prefix_id * 1e9 + numeric_suffix.
     edges = triples whose predicate is an *object property* linking two
             instances (KEEP set below). rdf:type, morphology/POS vocab values
             (folded into node text), datatype literals, and the lime:entry
             lexicon hub are excluded.
   Text literals (writtenRep@sl, skos:definition@sl, rdf:value@sl, partOfSpeech)
   are recorded so node text can be tokenized with the real GaMS-2B tokenizer.
2. Build a deduplicated undirected CSR adjacency.
3. Sample random lexical-entry seeds; BFS to depth 3.
4. For k=1,2,3 report, per seed:
     - original ball node count + node-type composition
     - induced edge count  -> Levi nodes = ball_nodes + induced_edges
     - combined GTLM input token length (node texts + relation labels + prompt),
       tokenized with cjvt/GaMS-2B.
5. Write results.json with percentiles.

Run `--files-limit N` for a quick local sanity check; full run reads everything.
"""
import os, re, sys, glob, json, time, argparse, zlib
from collections import defaultdict
import numpy as np
from multiprocessing import Pool

# ---------------------------------------------------------------------------
# Namespaces / vocabulary
# ---------------------------------------------------------------------------
ONTOLEX = "http://www.w3.org/ns/lemon/ontolex#"
LEXINFO = "http://www.lexinfo.net/ontology/3.0/lexinfo#"
LEXICOG = "http://www.w3.org/ns/lemon/lexicog#"
DECOMP  = "http://www.w3.org/ns/lemon/decomp#"
VARTRANS= "http://www.w3.org/ns/lemon/vartrans#"
FRAC    = "http://www.w3.org/ns/lemon/frac#"
SKOS    = "http://www.w3.org/2004/02/skos/core#"
RDF     = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

# object-property predicate -> (pred_id, short label used as Levi relation-node text)
KEEP_PREDS = {
    ONTOLEX+"canonicalForm": (1, "canonical form"),
    ONTOLEX+"otherForm":     (2, "other form"),
    ONTOLEX+"sense":         (3, "sense"),
    ONTOLEX+"isSenseOf":     (3, "sense"),          # inverse of sense -> same undirected edge
    LEXINFO+"synonym":       (4, "synonym"),
    LEXINFO+"antonym":       (5, "antonym"),
    LEXICOG+"usageExample":  (6, "usage example"),
    DECOMP+"constituent":    (7, "constituent"),
    DECOMP+"correspondsTo":  (8, "corresponds to"),
    VARTRANS+"source":       (9, "translation source"),
    VARTRANS+"target":       (10, "translation target"),
    FRAC+"head":             (11, "head"),
}
PRED_LABEL = {}
for _uri, (_pid, _lab) in KEEP_PREDS.items():
    PRED_LABEL[_pid] = _lab

# text (datatype) predicates -> which node kind carries the string
TXT_WRITTENREP = ONTOLEX+"writtenRep"
TXT_DEFINITION = SKOS+"definition"
TXT_VALUE      = RDF+"value"
TXT_POS        = LEXINFO+"partOfSpeech"

HOST_WWW  = "https://www.cjvt.ddds.si/"
HOST_BARE = "https://cjvt.ddds.si/"

# fixed prefix -> id (must be consistent across worker processes)
KNOWN_PREFIX = {
    "lexical-unit": 1, "lexical-unit-part": 2, "word-form": 3,
    "form-lexical-unit": 4, "sense": 5, "sense-translation": 6,
    "translation-form": 7, "example": 8, "resource": 9,
}
PREFIX_NAME = {v: k for k, v in KNOWN_PREFIX.items()}
ENTRY_PREFIX_ID = KNOWN_PREFIX["lexical-unit"]
MULT = 1_000_000_000  # numeric suffixes observed < 5e7; ample headroom

_num_re = re.compile(r"^(.*)-(\d+)$")


def prefix_id_for(prefix):
    pid = KNOWN_PREFIX.get(prefix)
    if pid is not None:
        return pid
    return 500 + (zlib.crc32(prefix.encode()) % 400)  # deterministic fallback band


def localname_to_code(obj_iri):
    """obj_iri is the content between < >. Return int64 code or None if not a cjvt instance."""
    if obj_iri.startswith(HOST_WWW):
        ln = obj_iri[len(HOST_WWW):]
    elif obj_iri.startswith(HOST_BARE):
        ln = obj_iri[len(HOST_BARE):]
    else:
        return None
    m = _num_re.match(ln)
    if not m:
        return None
    return prefix_id_for(m.group(1)) * MULT + int(m.group(2))


def parse_file(path):
    """Return (edges Nx3 int64 [src,dst,pred], texts list[(code,kind,str)])."""
    src = []; dst = []; prd = []
    texts = []  # (code, kind, text) ; kind: 0 writtenRep 1 definition 2 value 3 pos
    ap_s = src.append; ap_d = dst.append; ap_p = prd.append; ap_t = texts.append
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            # <subj> <pred> <rest>
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
                # IRI object -> candidate edge
                kp = KEEP_PREDS.get(pred)
                if kp is None:
                    continue
                oend = obj.find(">")
                ocode = localname_to_code(obj[1:oend])
                if ocode is None:
                    continue
                scode = localname_to_code(subj)
                if scode is None:
                    continue
                ap_s(scode); ap_d(ocode); ap_p(kp[0])
            else:
                # literal object -> maybe node text
                if pred == TXT_WRITTENREP:
                    kind = 0
                elif pred == TXT_DEFINITION:
                    kind = 1
                elif pred == TXT_VALUE:
                    kind = 2
                elif pred == TXT_POS:
                    # partOfSpeech is actually an IRI value; handled above only if <..>.
                    continue
                else:
                    continue
                # literal text between triple-quotes (data uses \"\"\" ... \"\"\")
                if obj.startswith('"""'):
                    e = obj.find('"""', 3)
                    if e < 0:
                        continue
                    txt = obj[3:e]
                    tail = obj[e+3:]
                elif obj.startswith('"'):
                    e = obj.find('"', 1)
                    if e < 0:
                        continue
                    txt = obj[1:e]
                    tail = obj[e+1:]
                else:
                    continue
                if kind in (0, 1) and "@sl" not in tail:
                    continue  # writtenRep/definition: keep Slovenian only
                scode = localname_to_code(subj)
                if scode is None:
                    continue
                ap_t((scode, kind, txt))
    e = np.empty((len(src), 3), dtype=np.int64)
    if len(src):
        e[:, 0] = src; e[:, 1] = dst; e[:, 2] = prd
    return e, texts


# ---------------------------------------------------------------------------
def build_graph(files, workers):
    t0 = time.time()
    print(f"[parse] {len(files)} files, {workers} workers", flush=True)
    edges_parts = []
    texts_all = []  # (code,kind,text)
    with Pool(workers) as pool:
        for i, (e, texts) in enumerate(pool.imap_unordered(parse_file, files, chunksize=4)):
            if len(e):
                edges_parts.append(e)
            texts_all.extend(texts)
            if (i + 1) % 200 == 0:
                print(f"[parse] {i+1}/{len(files)} files  edges~{sum(x.shape[0] for x in edges_parts):,}  "
                      f"texts~{len(texts_all):,}  {time.time()-t0:.0f}s", flush=True)
    edges = np.concatenate(edges_parts, axis=0)
    del edges_parts
    print(f"[parse] done: {edges.shape[0]:,} directed kept-triples, {len(texts_all):,} text literals, "
          f"{time.time()-t0:.0f}s", flush=True)

    # unordered endpoints for dedup
    a = edges[:, 0]; b = edges[:, 1]; pr = edges[:, 2]
    lo = np.minimum(a, b); hi = np.maximum(a, b)
    # dedup undirected pairs, keep first predicate
    order = np.lexsort((hi, lo))
    lo = lo[order]; hi = hi[order]; pr = pr[order]
    keep = np.empty(lo.shape[0], dtype=bool)
    keep[0] = True
    keep[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
    lo = lo[keep]; hi = hi[keep]; pr = pr[keep]
    print(f"[dedup] {lo.shape[0]:,} unique undirected edges", flush=True)

    # compact node ids
    nodes = np.unique(np.concatenate([lo, hi]))
    n = nodes.shape[0]
    li = np.searchsorted(nodes, lo)
    hj = np.searchsorted(nodes, hi)
    print(f"[nodes] {n:,} content nodes", flush=True)

    # undirected CSR: each edge twice
    src2 = np.concatenate([li, hj])
    dst2 = np.concatenate([hj, li])
    pr2  = np.concatenate([pr, pr]).astype(np.int8)
    oo = np.argsort(src2, kind="stable")
    src2 = src2[oo]; dst2 = dst2[oo]; pr2 = pr2[oo]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, src2 + 1, 1)
    np.cumsum(indptr, out=indptr)
    indices = dst2
    epred = pr2
    print(f"[csr] built, {indices.shape[0]:,} half-edges, {time.time()-t0:.0f}s", flush=True)

    # node text: prefer definition(sense)/value(example)/writtenRep(form); pos small
    # store as list[str] over compact ids
    text_by_id = [""] * n
    KIND_PRIORITY = {1: 3, 2: 3, 0: 2, 3: 1}  # def/value > writtenRep > pos
    cur_pri = np.zeros(n, dtype=np.int8)
    for code, kind, txt in texts_all:
        idx = np.searchsorted(nodes, code)
        if idx >= n or nodes[idx] != code:
            continue
        pri = KIND_PRIORITY[kind]
        if pri > cur_pri[idx]:
            cur_pri[idx] = pri
            text_by_id[idx] = txt
    del texts_all
    print(f"[text] assigned node text, {time.time()-t0:.0f}s", flush=True)

    # node type ids (prefix) for composition
    ntype = (nodes // MULT).astype(np.int32)
    return dict(nodes=nodes, n=n, indptr=indptr, indices=indices, epred=epred,
                text_by_id=text_by_id, ntype=ntype)


# ---------------------------------------------------------------------------
def ragged_gather(frontier, indptr, indices, epred=None):
    """Vectorized concat of indices[indptr[v]:indptr[v+1]] for v in frontier."""
    starts = indptr[frontier]
    lengths = indptr[frontier + 1] - starts
    total = int(lengths.sum())
    if total == 0:
        z = np.empty(0, dtype=indices.dtype)
        return (z, np.empty(0, dtype=epred.dtype)) if epred is not None else z
    inc = np.ones(total, dtype=np.int64)
    inc[0] = starts[0]
    off = np.cumsum(lengths)[:-1]
    inc[off] += starts[1:] - (starts[:-1] + lengths[:-1])
    gidx = np.cumsum(inc)
    if epred is not None:
        return indices[gidx], epred[gidx]
    return indices[gidx]


def analyze(G, seeds, max_hops, token_len, label_len, prompt_tokens, main_types):
    n = G["n"]; indptr = G["indptr"]; indices = G["indices"]; epred = G["epred"]
    ntype = G["ntype"]
    depth = np.full(n, -1, dtype=np.int8)
    per_k = {k: defaultdict(list) for k in range(1, max_hops + 1)}
    comp_k = {k: defaultdict(lambda: np.zeros(len(main_types) + 1, dtype=np.float64))
              for k in range(1, max_hops + 1)}
    comp_cnt = {k: 0 for k in range(1, max_hops + 1)}
    tmap = {t: i for i, t in enumerate(main_types)}

    for si, seed in enumerate(seeds):
        touched = [seed]
        depth[seed] = 0
        frontier = np.array([seed], dtype=np.int64)
        balls = {0: [seed]}
        for d in range(1, max_hops + 1):
            nbr = ragged_gather(frontier, indptr, indices)
            if nbr.size:
                nbr = np.unique(nbr)
                new = nbr[depth[nbr] < 0]
            else:
                new = np.empty(0, dtype=np.int64)
            depth[new] = d
            touched.extend(new.tolist())
            balls[d] = new
            frontier = new
        touched_arr = np.array(touched, dtype=np.int64)

        for k in range(1, max_hops + 1):
            ball = touched_arr[depth[touched_arr] <= k]
            b = ball.shape[0]
            # induced edges within ball
            nbrs, prds = ragged_gather(ball, indptr, indices, epred)
            inmask = depth[nbrs] >= 0  # all touched are within some depth
            inmask &= depth[nbrs] <= k
            induced_half = int(inmask.sum())
            induced_edges = induced_half // 2
            rel_tok = int(label_len[prds[inmask]].sum()) // 2
            node_tok = int(token_len[ball].sum())
            levi_nodes = b + induced_edges
            total_tok = node_tok + rel_tok + prompt_tokens
            per_k[k]["orig_nodes"].append(b)
            per_k[k]["induced_edges"].append(induced_edges)
            per_k[k]["levi_nodes"].append(levi_nodes)
            per_k[k]["tokens"].append(total_tok)
            # composition
            cvec = comp_k[k][0] if False else None
            tt = ntype[ball]
            vec = np.zeros(len(main_types) + 1, dtype=np.float64)
            for t in main_types:
                vec[tmap[t]] = (tt == t).sum()
            vec[-1] = b - vec[:-1].sum()  # other
            comp_k[k][0] += vec
            comp_cnt[k] += 1

        # reset depth for touched
        depth[touched_arr] = -1
        if (si + 1) % 50 == 0:
            print(f"[bfs] {si+1}/{len(seeds)} seeds", flush=True)

    return per_k, comp_k, comp_cnt


PCTS = [10, 25, 50, 75, 90, 95, 99]


def summarize(vals):
    a = np.array(vals, dtype=np.float64)
    out = {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}
    for p in PCTS:
        out[f"p{p}"] = float(np.percentile(a, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg-dir", default="/shared/workspace/povejmo/gams_gtlm/data/kg_raw/OntoLex DSB")
    ap.add_argument("--out", default="/shared/workspace/povejmo/gams_gtlm/data/kg_analysis/results.json")
    ap.add_argument("--n-seeds", type=int, default=400)
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    ap.add_argument("--files-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-tokens", type=int, default=24)
    ap.add_argument("--no-decomp", action="store_true",
                    help="drop decomp:constituent/correspondsTo edges (MWE decomposition, "
                         "the mega-hub driver) -> 'semantic-core' graph")
    args = ap.parse_args()

    if args.no_decomp:
        for uri in (DECOMP + "constituent", DECOMP + "correspondsTo"):
            KEEP_PREDS.pop(uri, None)
        print("[cfg] no-decomp: dropped MWE decomposition edges", flush=True)

    files = sorted(glob.glob(os.path.join(args.kg_dir, "*.nt")))
    if args.files_limit:
        files = files[:args.files_limit]
    print(f"KG dir: {args.kg_dir}  ({len(files)} .nt files)", flush=True)

    G = build_graph(files, args.workers)
    nodes = G["nodes"]; n = G["n"]

    # tokenizer
    os.environ.setdefault("HF_HOME", "/shared/workspace/povejmo/huggingface_cache")
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("cjvt/GaMS-2B")
    print("[tok] GaMS-2B loaded", flush=True)

    # token length per node (batch encode unique texts)
    texts = G["text_by_id"]
    token_len = np.zeros(n, dtype=np.int32)
    uniq = {}
    for i, s in enumerate(texts):
        if s:
            uniq.setdefault(s, []).append(i)
    keys = list(uniq.keys())
    print(f"[tok] tokenizing {len(keys):,} unique node texts", flush=True)
    B = 20000
    for j in range(0, len(keys), B):
        chunk = keys[j:j+B]
        enc = tok(chunk, add_special_tokens=False)["input_ids"]
        for s, ids in zip(chunk, enc):
            L = len(ids)
            for idx in uniq[s]:
                token_len[idx] = L
        if (j // B) % 20 == 0:
            print(f"[tok] {j+len(chunk):,}/{len(keys):,}", flush=True)
    del uniq, keys

    # relation label token lengths (index by pred_id)
    max_pid = max(PRED_LABEL) + 1
    label_len = np.zeros(max_pid, dtype=np.int32)
    for pid, lab in PRED_LABEL.items():
        label_len[pid] = len(tok(lab, add_special_tokens=False)["input_ids"])

    # sample seeds among lexical-unit entries present in the graph
    ntype = G["ntype"]
    entry_ids = np.nonzero(ntype == ENTRY_PREFIX_ID)[0]
    print(f"[seeds] {entry_ids.shape[0]:,} lexical-entry nodes in graph", flush=True)
    rng = np.random.default_rng(args.seed)
    ns = min(args.n_seeds, entry_ids.shape[0])
    seeds = rng.choice(entry_ids, size=ns, replace=False)

    main_types = [KNOWN_PREFIX[t] for t in
                  ["lexical-unit", "word-form", "sense", "example",
                   "lexical-unit-part", "form-lexical-unit", "translation-form"]]
    type_name = {KNOWN_PREFIX[t]: t for t in
                 ["lexical-unit", "word-form", "sense", "example",
                  "lexical-unit-part", "form-lexical-unit", "translation-form"]}

    per_k, comp_k, comp_cnt = analyze(G, seeds, args.max_hops, token_len, label_len,
                                      args.prompt_tokens, main_types)

    result = {"meta": {"n_content_nodes": int(n),
                       "n_undirected_edges": int(G["indices"].shape[0] // 2),
                       "n_entry_nodes": int(entry_ids.shape[0]),
                       "n_seeds": int(ns), "max_hops": args.max_hops,
                       "prompt_tokens": args.prompt_tokens,
                       "tokenizer": "cjvt/GaMS-2B",
                       "n_files": len(files)},
              "per_hop": {}}
    comp_labels = [type_name[t] for t in main_types] + ["other"]
    for k in range(1, args.max_hops + 1):
        entry = {m: summarize(per_k[k][m]) for m in
                 ["orig_nodes", "induced_edges", "levi_nodes", "tokens"]}
        comp = comp_k[k][0] / max(comp_cnt[k], 1)
        entry["avg_composition"] = {lab: float(v) for lab, v in zip(comp_labels, comp)}
        result["per_hop"][str(k)] = entry

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print("[done] wrote", args.out, flush=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
