#!/usr/bin/env python3
"""
v2 builder + sizing analysis for the GTLM-compatible CJVT graph.

Differences from analyze_neighborhoods.py (v1):
  * MWE decomposition is COLLAPSED: the empty `lexical-unit-part` connector is
    removed and  MWE --constituent--> part --correspondsTo--> word  becomes a
    single directed edge  MWE --sestavina--> word.
  * Each headword (lexical-unit) is MERGED with its canonical (lemma) form; the
    anchor node's text is the lemma's writtenRep + a Slovenian morphology suffix,
    e.g. "pes (imenovalnik, ednina)".  No empty entity nodes.
  * form_mode:
      - collapse : only the anchor (lemma) node; inflected forms dropped.
      - expand   : every otherForm is a self-describing LEAF node
                   "psa (rodilnik, ednina)" star-linked to the anchor via `oblika`.
  * Every node is guaranteed non-empty text (senses fall back to their lemma).
  * Relation labels are Slovenian (reified as Levi relation-node text).
  * Edges are stored DIRECTED; k-hop neighborhoods traverse BOTH directions.
  * Usage examples toggled by a flag.

Analysis: 2 (form_mode) x 2 (examples) = 4 variants, reported with percentiles,
split by seed kind (single word vs MWE).  Levi nodes = ball_nodes + induced_edges,
tokens = node-text tokens + relation-label tokens + prompt, via cjvt/GaMS-2B.
"""
import os, re, sys, glob, json, time, argparse, zlib
from collections import defaultdict
import numpy as np
from multiprocessing import Pool

ONTOLEX = "http://www.w3.org/ns/lemon/ontolex#"
LEXINFO = "http://www.lexinfo.net/ontology/3.0/lexinfo#"
LEXICOG = "http://www.w3.org/ns/lemon/lexicog#"
DECOMP  = "http://www.w3.org/ns/lemon/decomp#"
VARTRANS= "http://www.w3.org/ns/lemon/vartrans#"
FRAC    = "http://www.w3.org/ns/lemon/frac#"
SKOS    = "http://www.w3.org/2004/02/skos/core#"
RDF     = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

# ---- edge-type ids in the FINAL (collapsed) graph + Slovenian Levi labels ----
REL_FORM   = 1   # anchor -> otherForm leaf
REL_SENSE  = 2   # lexical-unit -> sense
REL_SYN    = 3   # synonym
REL_ANT    = 4   # antonym
REL_USAGE  = 5   # sense -> example
REL_CONSTIT= 6   # MWE -> constituent word (collapsed)
REL_TRANS  = 7   # translation (vartrans)
REL_HEAD   = 8   # frac head
REL_LABEL_SL = {
    REL_FORM:    "oblika",
    REL_SENSE:   "pomen",
    REL_SYN:     "sopomenka",
    REL_ANT:     "protipomenka",
    REL_USAGE:   "zgled",
    REL_CONSTIT: "sestavina",
    REL_TRANS:   "prevod",
    REL_HEAD:    "glava",
}

# raw predicates we care about
P_CANON = ONTOLEX+"canonicalForm"
P_OTHER = ONTOLEX+"otherForm"
P_SENSE = ONTOLEX+"sense"
P_ISSENSEOF = ONTOLEX+"isSenseOf"
P_SYN   = LEXINFO+"synonym"
P_ANT   = LEXINFO+"antonym"
P_USAGE = LEXICOG+"usageExample"
P_CONSTIT = DECOMP+"constituent"
P_CORR  = DECOMP+"correspondsTo"
P_VSRC  = VARTRANS+"source"
P_VTGT  = VARTRANS+"target"
P_HEAD  = FRAC+"head"
P_WRITTENREP = ONTOLEX+"writtenRep"
P_DEFINITION = SKOS+"definition"
P_VALUE = RDF+"value"

# lexinfo morphology feature properties -> Slovenian value labels
FEATURE_PROPS = ("case", "number", "gender", "person", "tense", "mood", "degree")
VALUE_SL = {
    "nominative":"imenovalnik","genitive":"rodilnik","dative":"dajalnik",
    "accusative":"tožilnik","locative":"mestnik","instrumental":"orodnik",
    "singular":"ednina","dual":"dvojina","plural":"množina",
    "masculine":"moški spol","feminine":"ženski spol","neuter":"srednji spol",
    "firstPerson":"1. oseba","secondPerson":"2. oseba","thirdPerson":"3. oseba",
    "present":"sedanjik","past":"preteklik","future":"prihodnjik",
    "positive":"osnovnik","comparative":"primernik","superlative":"presežnik",
    "indicative":"povedni","imperative":"velelni",
}

HOST_WWW  = "https://www.cjvt.ddds.si/"
HOST_BARE = "https://cjvt.ddds.si/"
KNOWN_PREFIX = {
    "lexical-unit": 1, "lexical-unit-part": 2, "word-form": 3,
    "form-lexical-unit": 4, "sense": 5, "sense-translation": 6,
    "translation-form": 7, "example": 8, "resource": 9,
}
PID_LU   = 1
PID_PART = 2
PID_FORM = 3
PID_SENSE= 5
PID_EX   = 8
MULT = 1_000_000_000
_num_re = re.compile(r"^(.*)-(\d+)$")


def prefix_id_for(prefix):
    pid = KNOWN_PREFIX.get(prefix)
    return pid if pid is not None else 500 + (zlib.crc32(prefix.encode()) % 400)


def code_of(iri):
    if iri.startswith(HOST_WWW):
        ln = iri[len(HOST_WWW):]
    elif iri.startswith(HOST_BARE):
        ln = iri[len(HOST_BARE):]
    else:
        return None
    m = _num_re.match(ln)
    if not m:
        return None
    return prefix_id_for(m.group(1)) * MULT + int(m.group(2))


def _localname_hash(s):
    return s.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def parse_file(path):
    """Stream a .nt file. Return dict of edge arrays + text/feature tuples."""
    # typed edge endpoint lists
    E = {k: ([], []) for k in ("canon","other","sense","syn","ant","usage",
                                "constit","corr","trans","head")}
    wr = []    # (code, str) writtenRep@sl
    dfn = []   # (code, str) definition@sl
    val = []   # (code, str) value@sl
    feat = []  # (form_code, prop, value_localname)
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
                # edge predicates
                if pred == P_CANON:   key="canon"
                elif pred == P_OTHER: key="other"
                elif pred == P_SENSE: key="sense"
                elif pred == P_ISSENSEOF: key=None  # inverse; handled via sense
                elif pred == P_SYN:   key="syn"
                elif pred == P_ANT:   key="ant"
                elif pred == P_USAGE: key="usage"
                elif pred == P_CONSTIT: key="constit"
                elif pred == P_CORR:  key="corr"
                elif pred == P_VSRC or pred == P_VTGT: key="trans"
                elif pred == P_HEAD:  key="head"
                elif pred == P_ISSENSEOF:
                    key=None
                else:
                    key=None
                    # morphology feature on a word-form subject?
                    if pred.startswith(LEXINFO):
                        local = pred[len(LEXINFO):]
                        if local in FEATURE_PROPS:
                            sc = code_of(subj)
                            if sc is not None and sc // MULT == PID_FORM:
                                feat.append((sc, local, _localname_hash(oiri)))
                    continue
                if key is None:
                    continue
                sc = code_of(subj); oc = code_of(oiri)
                if sc is None or oc is None:
                    continue
                if key == "sense":
                    E["sense"][0].append(sc); E["sense"][1].append(oc)
                else:
                    E[key][0].append(sc); E[key][1].append(oc)
            else:
                # literal
                if pred == P_WRITTENREP: bucket = wr; sl_only = True
                elif pred == P_DEFINITION: bucket = dfn; sl_only = True
                elif pred == P_VALUE: bucket = val; sl_only = True
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
                if sl_only and "@sl" not in tail:
                    continue
                sc = code_of(subj)
                if sc is None:
                    continue
                bucket.append((sc, txt))
    # to arrays
    out = {}
    for k,(a,b) in E.items():
        if a:
            arr = np.empty((len(a),2), dtype=np.int64); arr[:,0]=a; arr[:,1]=b
        else:
            arr = np.empty((0,2), dtype=np.int64)
        out[k] = arr
    out["wr"]=wr; out["dfn"]=dfn; out["val"]=val; out["feat"]=feat
    return out


def feat_string(props):
    """props: dict prop->value_localname -> Slovenian '(imenovalnik, ednina)' or ''."""
    parts = []
    for p in FEATURE_PROPS:
        v = props.get(p)
        if v is None:
            continue
        sl = VALUE_SL.get(v)
        if sl:
            parts.append(sl)
    return " (" + ", ".join(parts) + ")" if parts else ""


# ---------------------------------------------------------------------------
def build(files, workers):
    t0=time.time()
    print(f"[parse] {len(files)} files x {workers} workers", flush=True)
    agg = {k: [] for k in ("canon","other","sense","syn","ant","usage",
                            "constit","corr","trans","head")}
    wr={}; dfn={}; val={}
    feat_map = defaultdict(dict)   # form_code -> {prop: value_localname}
    with Pool(workers) as pool:
        for i,res in enumerate(pool.imap_unordered(parse_file, files, chunksize=4)):
            for k in agg:
                if res[k].shape[0]:
                    agg[k].append(res[k])
            for c,s in res["wr"]:  wr[c]=s
            for c,s in res["dfn"]: dfn[c]=s
            for c,s in res["val"]: val[c]=s
            for c,p,v in res["feat"]:
                feat_map[c][p]=v
            if (i+1)%200==0:
                print(f"[parse] {i+1}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    def cat(k):
        return (np.concatenate(agg[k],axis=0) if agg[k]
                else np.empty((0,2),dtype=np.int64))
    canon=cat("canon"); other=cat("other"); sense=cat("sense")
    syn=cat("syn"); ant=cat("ant"); usage=cat("usage")
    constit=cat("constit"); corr=cat("corr"); trans=cat("trans"); head=cat("head")
    print(f"[parse] done {time.time()-t0:.0f}s  canon={len(canon):,} other={len(other):,} "
          f"sense={len(sense):,} usage={len(usage):,} constit={len(constit):,} "
          f"corr={len(corr):,} wr={len(wr):,} dfn={len(dfn):,} val={len(val):,} "
          f"feat_forms={len(feat_map):,}", flush=True)

    # ---- collapse MWE decomposition: join constit(mwe,part) with corr(part,word)
    if len(constit) and len(corr):
        parts = corr[:,0]; words = corr[:,1]
        order = np.argsort(parts, kind="stable")
        parts_s = parts[order]; words_s = words[order]
        idx = np.searchsorted(parts_s, constit[:,1])
        idx = np.clip(idx, 0, len(parts_s)-1)
        hit = parts_s[idx] == constit[:,1]
        mwe_word = np.empty((hit.sum(),2), dtype=np.int64)
        mwe_word[:,0] = constit[hit,0]
        mwe_word[:,1] = words_s[idx[hit]]
    else:
        mwe_word = np.empty((0,2), dtype=np.int64)
    print(f"[collapse] {len(mwe_word):,} MWE->word constituent edges", flush=True)

    # ---- canonical form per lexical-unit (lu -> form code)
    canon_of = {}
    for lu, fcode in canon:
        canon_of.setdefault(int(lu), int(fcode))

    # ---- assemble the final DIRECTED, LABELED edge list ------------------
    esrc=[]; edst=[]; elab=[]
    def add(arr, lab):
        if len(arr):
            esrc.append(arr[:,0]); edst.append(arr[:,1])
            elab.append(np.full(len(arr), lab, dtype=np.int8))
    add(other,   REL_FORM)     # anchor(lu) -> otherForm leaf
    add(sense,   REL_SENSE)
    add(syn,     REL_SYN)
    add(ant,     REL_ANT)
    add(usage,   REL_USAGE)
    add(mwe_word,REL_CONSTIT)
    add(trans,   REL_TRANS)
    add(head,    REL_HEAD)
    src=np.concatenate(esrc); dst=np.concatenate(edst); lab=np.concatenate(elab)
    print(f"[edges] {len(src):,} directed labeled edges", flush=True)

    # ---- node universe = all endpoints + all lus that have a canonical form
    node_codes = np.unique(np.concatenate([src, dst,
                    np.fromiter(canon_of.keys(), dtype=np.int64, count=len(canon_of))]))
    n = len(node_codes)
    idx_of = {int(c):i for i,c in enumerate(node_codes)}
    ntype = (node_codes // MULT).astype(np.int32)
    print(f"[nodes] {n:,} nodes", flush=True)

    # remap edges to compact ids
    si = np.searchsorted(node_codes, src)
    di = np.searchsorted(node_codes, dst)
    # (all endpoints are in node_codes by construction)

    # ---- node text -------------------------------------------------------
    # anchor (lu): writtenRep(canonicalForm) + features(canonicalForm)
    # form leaf : writtenRep + features
    # sense     : definition, else fallback to its lemma's anchor text
    # example   : value ; translation/other: def|val|wr
    text = [""]*n
    is_form_leaf = np.zeros(n, dtype=bool)   # otherForm leaves
    # form -> its lu (from other + canon), to know leaves vs canonical
    # mark otherForm leaves
    for f in other[:,1]:
        j = idx_of.get(int(f))
        if j is not None:
            is_form_leaf[j] = True
    # sense -> lu (for fallback text)
    sense_lu = {}
    for lu,se in sense:
        sense_lu.setdefault(int(se), int(lu))

    def form_text(fcode):
        s = wr.get(fcode, "")
        if not s:
            return ""
        return s + feat_string(feat_map.get(fcode, {}))

    # anchors
    n_anchor=0
    for lu, fcode in canon_of.items():
        j = idx_of.get(lu)
        if j is None:
            continue
        t = form_text(fcode)
        if not t:
            # fallback: any otherForm writtenRep, else empty
            t = wr.get(fcode, "")
        text[j] = t
        if t: n_anchor+=1
    # form leaves
    for lu, fcode in other:
        j = idx_of.get(int(fcode))
        if j is not None and not text[j]:
            text[j] = form_text(int(fcode))
    # senses
    for se, lu in sense_lu.items():
        j = idx_of.get(se)
        if j is None:
            continue
        d = dfn.get(se, "")
        if not d:
            aj = idx_of.get(lu)
            d = text[aj] if aj is not None and text[aj] else ""
        text[j] = d
    # examples + any remaining (translations etc.)
    for i,c in enumerate(node_codes):
        if text[i]:
            continue
        c=int(c)
        t = dfn.get(c) or val.get(c) or wr.get(c) or ""
        text[i]=t
    n_textless = sum(1 for t in text if not t)
    print(f"[text] anchors_with_text={n_anchor:,}  textless_nodes={n_textless:,}", flush=True)

    # ---- undirected CSR (both directions) for BFS ------------------------
    src2 = np.concatenate([si, di]); dst2 = np.concatenate([di, si])
    lab2 = np.concatenate([lab, lab]).astype(np.int8)
    oo = np.argsort(src2, kind="stable")
    src2=src2[oo]; dst2=dst2[oo]; lab2=lab2[oo]
    indptr=np.zeros(n+1, dtype=np.int64)
    np.add.at(indptr, src2+1, 1); np.cumsum(indptr, out=indptr)
    indices=dst2; epred=lab2
    print(f"[csr] {len(indices):,} half-edges  {time.time()-t0:.0f}s", flush=True)

    # MWE seed set = lus that are subject of a collapsed constituent edge
    mwe_set = np.zeros(n, dtype=bool)
    if len(mwe_word):
        mi = np.searchsorted(node_codes, np.unique(mwe_word[:,0]))
        mwe_set[mi] = True

    return dict(node_codes=node_codes, n=n, ntype=ntype, indptr=indptr,
                indices=indices, epred=epred, text=text,
                is_form_leaf=is_form_leaf, mwe_set=mwe_set)


# ---------------------------------------------------------------------------
def ragged_gather(frontier, indptr, indices, epred=None):
    starts = indptr[frontier]; lengths = indptr[frontier+1]-starts
    total=int(lengths.sum())
    if total==0:
        z=np.empty(0,dtype=indices.dtype)
        return (z,np.empty(0,dtype=epred.dtype)) if epred is not None else z
    inc=np.ones(total,dtype=np.int64); inc[0]=starts[0]
    off=np.cumsum(lengths)[:-1]
    inc[off]+=starts[1:]-(starts[:-1]+lengths[:-1])
    g=np.cumsum(inc)
    if epred is not None:
        return indices[g], epred[g]
    return indices[g]


PCTS=[10,25,50,75,90,95,99]
def summarize(vals):
    a=np.array(vals,dtype=np.float64)
    if len(a)==0:
        return {"n":0}
    out={"n":len(a),"mean":float(a.mean()),"min":float(a.min()),"max":float(a.max())}
    for p in PCTS: out[f"p{p}"]=float(np.percentile(a,p))
    return out


def analyze_variant(G, seeds, max_hops, token_len, label_len, prompt_tokens, active):
    n=G["n"]; indptr=G["indptr"]; indices=G["indices"]; epred=G["epred"]
    depth=np.full(n,-1,dtype=np.int8)
    per={k:defaultdict(list) for k in range(1,max_hops+1)}
    mwe_set=G["mwe_set"]
    for seed in seeds:
        touched=[seed]; depth[seed]=0; frontier=np.array([seed],dtype=np.int64)
        for d in range(1,max_hops+1):
            nbr=ragged_gather(frontier,indptr,indices)
            if nbr.size:
                nbr=np.unique(nbr)
                nbr=nbr[active[nbr]]
                new=nbr[depth[nbr]<0]
            else:
                new=np.empty(0,dtype=np.int64)
            depth[new]=d; touched.extend(new.tolist()); frontier=new
        touched=np.array(touched,dtype=np.int64)
        for k in range(1,max_hops+1):
            ball=touched[depth[touched]<=k]
            b=ball.shape[0]
            nbrs,prds=ragged_gather(ball,indptr,indices,epred)
            inmask=(depth[nbrs]>=0)&(depth[nbrs]<=k)
            induced=int(inmask.sum())//2
            rel_tok=int(label_len[prds[inmask]].sum())//2
            node_tok=int(token_len[ball].sum())
            per[k]["orig_nodes"].append(b)
            per[k]["levi_nodes"].append(b+induced)
            per[k]["tokens"].append(node_tok+rel_tok+prompt_tokens)
        depth[touched]=-1
    return per


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--kg-dir", default="/shared/workspace/povejmo/gams_gtlm/data/kg_raw/OntoLex DSB")
    ap.add_argument("--out", default="/shared/workspace/povejmo/gams_gtlm/data/kg_analysis/results_v2.json")
    ap.add_argument("--n-seeds", type=int, default=400)
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK","16")))
    ap.add_argument("--files-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-tokens", type=int, default=24)
    args=ap.parse_args()

    files=sorted(glob.glob(os.path.join(args.kg_dir,"*.nt")))
    if args.files_limit: files=files[:args.files_limit]
    print(f"KG dir: {args.kg_dir}  ({len(files)} .nt files)", flush=True)

    G=build(files, args.workers)
    n=G["n"]; ntype=G["ntype"]

    os.environ.setdefault("HF_HOME","/shared/workspace/povejmo/huggingface_cache")
    os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained("cjvt/GaMS-2B")
    print("[tok] GaMS-2B loaded", flush=True)

    texts=G["text"]; token_len=np.zeros(n,dtype=np.int32)
    uniq=defaultdict(list)
    for i,s in enumerate(texts):
        if s: uniq[s].append(i)
    keys=list(uniq.keys())
    print(f"[tok] {len(keys):,} unique node texts", flush=True)
    B=20000
    for j in range(0,len(keys),B):
        chunk=keys[j:j+B]
        enc=tok(chunk, add_special_tokens=False)["input_ids"]
        for s,ids in zip(chunk,enc):
            L=len(ids)
            for idx in uniq[s]: token_len[idx]=L
    del uniq,keys
    max_lab=max(REL_LABEL_SL)+1
    label_len=np.zeros(max_lab,dtype=np.int32)
    for pid,lab in REL_LABEL_SL.items():
        label_len[pid]=len(tok(lab, add_special_tokens=False)["input_ids"])

    # seeds among lexical-units that have text
    lu_nodes=np.nonzero((ntype==PID_LU) & (token_len>0))[0]
    print(f"[seeds] {lu_nodes.shape[0]:,} candidate lexical-unit nodes", flush=True)
    rng=np.random.default_rng(args.seed)
    ns=min(args.n_seeds, lu_nodes.shape[0])
    seeds=rng.choice(lu_nodes, size=ns, replace=False)
    mwe_set=G["mwe_set"]
    seeds_word=seeds[~mwe_set[seeds]]
    seeds_mwe =seeds[ mwe_set[seeds]]
    print(f"[seeds] {ns} total  ({len(seeds_word)} word, {len(seeds_mwe)} MWE)", flush=True)

    is_form_leaf=G["is_form_leaf"]; is_example=(ntype==PID_EX)
    variants={
        "expand_examples":   np.ones(n,dtype=bool),
        "collapse_examples": ~is_form_leaf,
        "expand_noexamples": ~is_example,
        "collapse_noexamples": ~is_form_leaf & ~is_example,
    }
    result={"meta":{"n_nodes":int(n),
                    "n_edges":int(G["indices"].shape[0]//2),
                    "n_seeds":int(ns),"n_seeds_word":int(len(seeds_word)),
                    "n_seeds_mwe":int(len(seeds_mwe)),
                    "max_hops":args.max_hops,"prompt_tokens":args.prompt_tokens,
                    "tokenizer":"cjvt/GaMS-2B","n_files":len(files),
                    "labels_sl":REL_LABEL_SL},
            "variants":{}}
    for vname,active in variants.items():
        t0=time.time()
        per_all=analyze_variant(G,seeds,args.max_hops,token_len,label_len,args.prompt_tokens,active)
        per_w  =analyze_variant(G,seeds_word,args.max_hops,token_len,label_len,args.prompt_tokens,active) if len(seeds_word) else None
        per_m  =analyze_variant(G,seeds_mwe ,args.max_hops,token_len,label_len,args.prompt_tokens,active) if len(seeds_mwe) else None
        vres={}
        for k in range(1,args.max_hops+1):
            entry={"all":{m:summarize(per_all[k][m]) for m in ("orig_nodes","levi_nodes","tokens")}}
            if per_w: entry["word"]={m:summarize(per_w[k][m]) for m in ("orig_nodes","levi_nodes","tokens")}
            if per_m: entry["mwe"] ={m:summarize(per_m[k][m]) for m in ("orig_nodes","levi_nodes","tokens")}
            vres[str(k)]=entry
        result["variants"][vname]=vres
        print(f"[variant] {vname} done {time.time()-t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w") as fh:
        json.dump(result,fh,indent=2,ensure_ascii=False)
    print("[done] wrote", args.out, flush=True)


if __name__=="__main__":
    main()
