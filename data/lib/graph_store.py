#!/usr/bin/env python3
"""
On-disk store for the v3 GTLM graph.

`build_gtlm_graph.build()` spends ~12 minutes (parse -> collapse -> reify ->
CSR -> tokenisation) reconstructing the same 36.7M-node graph from 42 GB
of N-Triples on every run, then throws it away.  This module persists that
result so downstream work (subgraph extraction, TextGraphDataset adaptation)
loads it in seconds and in ~3 GB of RAM instead of the 200 GB the builder needs.

Layout of a store directory:

    manifest.json      shapes, dtypes, provenance, the builder's `stats` dict
    node_codes.npy     int64  [n_real]  packed (type<<56)|payload, sorted
    ntype.npy          int32  [n]       raw KG type id, -1 for minted nodes
    kind.npy           int8   [n]       K_ANCHOR .. K_OTHER
    mwe_set.npy        bool   [n]       anchor is a multi-word expression
    indptr.npy         int64  [n+1]     undirected CSR
    indices.npy        int32  [2*edges] undirected CSR (ids < 2^31)
    token_len.npy      int32  [n]       token count of each node text, under the
                                        tokenizer named in manifest.meta
    text_blob.bin      uint8  [...]     all node texts, UTF-8, concatenated
    text_off.npy       int64  [n+1]     byte offsets into text_blob.bin

`is_form_leaf`, `is_example` and `is_colloc` are not stored: each is one
comparison against `kind`, and three bool arrays are 105 MB of redundancy.

Node text is stored as a blob rather than 36.7M pickled Python strings because
`np.save` on an object array pickles element-by-element -- slower to write, far
slower to read, and impossible to memory-map.  `TextStore` maps the blob and
decodes a node's text only when it is asked for.
"""
import os, io, json, time, mmap, hashlib
import numpy as np

FORMAT_VERSION = 1

# arrays: name -> (dtype on disk, length key in the manifest)
_ARRAYS = (
    ("node_codes", np.int64, "n_real"),
    ("ntype",      np.int32, "n"),
    ("kind",       np.int8,  "n"),
    ("mwe_set",    np.bool_, "n"),
    ("indptr",     np.int64, "n_plus_1"),
    ("indices",    np.int32, "n_half_edges"),
    ("token_len",  np.int32, "n"),
)

# kind ids that the derived masks test against (mirrors build_gtlm_graph)
K_FORM, K_EXAMPLE, K_COLLOC = 1, 3, 5


class TextStore:
    """Sequence-like view of the node texts, backed by a memory-mapped blob.

    Supports the two access patterns the builder and analysis code already use:
    `texts[i]` and `for i, s in enumerate(texts)`.  Both yield `str`; a textless
    node yields `""`.
    """

    def __init__(self, blob, offsets, _owns=()):
        self._blob = blob          # bytes or mmap.mmap
        self._off = offsets        # int64 [n+1]
        self._owns = _owns         # file handles kept alive for the mmap
        self._n = len(offsets) - 1

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self._n))]
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError(i)
        a = int(self._off[i]); b = int(self._off[i + 1])
        return self._blob[a:b].decode("utf-8") if b > a else ""

    def __iter__(self):
        off = self._off; blob = self._blob
        for i in range(self._n):
            a = int(off[i]); b = int(off[i + 1])
            yield blob[a:b].decode("utf-8") if b > a else ""

    @property
    def nbytes(self):
        return int(self._off[-1])

    def close(self):
        for h in self._owns:
            try:
                h.close()
            except Exception:
                pass
        self._owns = ()


def _sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            if limit and h.block_size and fh.tell() > limit:
                break
    return h.hexdigest()


def save_graph(out_dir, G, token_len, *, stats=None, meta=None,
               text_chunk=1_000_000, verbose=True):
    """Persist a built graph.  Writes atomically-ish: manifest.json last, so a
    directory without a manifest is an interrupted write, not a valid store."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)          # invalidate before overwriting parts

    n = int(G["n"]); n_real = int(G["n_real"])
    texts = G["text"]
    if len(texts) != n:
        raise ValueError(f"text has {len(texts)} entries, expected n={n}")

    # ---- node texts: one UTF-8 blob + byte offsets -------------------------
    # Encoded in chunks so peak memory stays ~chunk-sized rather than holding a
    # second copy of all 36.7M strings.
    off = np.zeros(n + 1, dtype=np.int64)
    blob_path = os.path.join(out_dir, "text_blob.bin")
    n_textless = 0
    with open(blob_path, "wb", buffering=1 << 22) as fh:
        for i0 in range(0, n, text_chunk):
            chunk = texts[i0:i0 + text_chunk]
            enc = [s.encode("utf-8") for s in chunk]
            off[i0 + 1:i0 + 1 + len(enc)] = np.fromiter(
                (len(b) for b in enc), dtype=np.int64, count=len(enc))
            n_textless += sum(1 for b in enc if not b)
            fh.write(b"".join(enc))
            if verbose and (i0 // text_chunk) % 10 == 9:
                print(f"[save] text {i0 + len(enc):,}/{n:,}", flush=True)
    np.cumsum(off, out=off)
    np.save(os.path.join(out_dir, "text_off.npy"), off)
    blob_bytes = int(off[-1])
    if verbose:
        print(f"[save] text_blob.bin {blob_bytes / 2**30:.2f} GB "
              f"({n_textless:,} textless)  {time.time() - t0:.0f}s", flush=True)

    # ---- the flat arrays --------------------------------------------------
    lengths = {"n": n, "n_real": n_real, "n_plus_1": n + 1,
               "n_half_edges": int(G["indices"].shape[0])}
    written = {}
    for name, dtype, lkey in _ARRAYS:
        a = token_len if name == "token_len" else G[name]
        a = np.asarray(a)
        want = lengths[lkey]
        if a.shape != (want,):
            raise ValueError(f"{name}: shape {a.shape}, expected ({want},)")
        if dtype in (np.int32,) and a.dtype != dtype:
            lo, hi = int(a.min()), int(a.max())
            if lo < np.iinfo(dtype).min or hi > np.iinfo(dtype).max:
                raise ValueError(f"{name}: range [{lo}, {hi}] does not fit {dtype}")
        a = a.astype(dtype, copy=False)
        p = os.path.join(out_dir, name + ".npy")
        np.save(p, a)
        written[name] = {"dtype": np.dtype(dtype).name, "shape": [int(want)],
                         "bytes": os.path.getsize(p)}
        if verbose:
            print(f"[save] {name}.npy {written[name]['bytes'] / 2**20:.0f} MB", flush=True)

    total = blob_bytes + off.nbytes + sum(v["bytes"] for v in written.values())
    manifest = {
        "format_version": FORMAT_VERSION,
        "graph_version": 3,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n": n, "n_real": n_real,
        "n_edges": int(G["indices"].shape[0]) // 2,
        "n_half_edges": int(G["indices"].shape[0]),
        "n_textless": n_textless,
        "text_bytes": blob_bytes,
        "total_bytes": int(total),
        "arrays": written,
        "text": {"blob": "text_blob.bin", "offsets": "text_off.npy",
                 "encoding": "utf-8"},
        "derived": {"is_form_leaf": f"kind == {K_FORM}",
                    "is_example": f"kind == {K_EXAMPLE}",
                    "is_colloc": f"kind == {K_COLLOC}"},
        "stats": stats or {},
        "meta": meta or {},
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    if verbose:
        print(f"[save] {total / 2**30:.2f} GB to {out_dir}  "
              f"{time.time() - t0:.0f}s", flush=True)
    return manifest


def load_graph(store_dir, *, mmap_arrays=True, mmap_text=True, verbose=True):
    """Load a store written by `save_graph`.

    Returns a dict shaped like `build_gtlm_graph.build()`'s return value --
    plus `token_len`, `manifest` and `stats` -- so analysis code can consume it
    unchanged.  With the defaults nothing is read into RAM until touched.
    """
    t0 = time.time()
    with open(os.path.join(store_dir, "manifest.json")) as fh:
        man = json.load(fh)
    if man.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"store format v{man.get('format_version')}, "
                         f"this module reads v{FORMAT_VERSION}")

    G = {"n": int(man["n"]), "n_real": int(man["n_real"])}
    mm = "r" if mmap_arrays else None
    for name, _dtype, _lkey in _ARRAYS:
        G[name] = np.load(os.path.join(store_dir, name + ".npy"), mmap_mode=mm)

    kind = G["kind"]
    G["is_form_leaf"] = (kind == K_FORM)
    G["is_example"] = (kind == K_EXAMPLE)
    G["is_colloc"] = (kind == K_COLLOC)

    off = np.load(os.path.join(store_dir, "text_off.npy"), mmap_mode=mm)
    blob_path = os.path.join(store_dir, man["text"]["blob"])
    if mmap_text:
        fh = open(blob_path, "rb")
        blob = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        G["text"] = TextStore(blob, off, _owns=(blob, fh))
    else:
        with open(blob_path, "rb") as fh:
            G["text"] = TextStore(fh.read(), off)

    G["manifest"] = man
    G["stats"] = man.get("stats", {})
    if verbose:
        print(f"[load] {man['n']:,} nodes / {man['n_edges']:,} edges from "
              f"{store_dir}  {time.time() - t0:.1f}s", flush=True)
    return G


def verify_graph(store_dir, *, n_samples=2000, seed=0, verbose=True):
    """Structural self-check of a store.  Returns a dict of findings; raises on
    anything that would make the store silently wrong."""
    G = load_graph(store_dir, verbose=verbose)
    man = G["manifest"]
    n = G["n"]; indptr = G["indptr"]; indices = G["indices"]
    problems = []

    if int(indptr[0]) != 0:
        problems.append(f"indptr[0] = {int(indptr[0])}, expected 0")
    if int(indptr[-1]) != indices.shape[0]:
        problems.append(f"indptr[-1] = {int(indptr[-1])}, "
                        f"expected {indices.shape[0]}")
    d = np.diff(np.asarray(indptr))
    if (d < 0).any():
        problems.append("indptr is not non-decreasing")
    if indices.shape[0]:
        lo, hi = int(np.asarray(indices).min()), int(np.asarray(indices).max())
        if lo < 0 or hi >= n:
            problems.append(f"indices range [{lo}, {hi}] outside [0, {n})")
    codes = np.asarray(G["node_codes"])
    if (np.diff(codes) <= 0).any():
        problems.append("node_codes is not strictly increasing (searchsorted "
                        "lookups would be wrong)")

    # symmetry: the CSR is undirected, so every u->v must have a matching v->u.
    rng = np.random.default_rng(seed)
    texts = G["text"]
    checked = 0
    for u in rng.choice(n, size=min(n_samples, n), replace=False):
        u = int(u)
        for v in np.asarray(indices[indptr[u]:indptr[u + 1]]):
            v = int(v)
            back = np.asarray(indices[indptr[v]:indptr[v + 1]])
            if u not in back:
                problems.append(f"edge {u}->{v} has no reverse")
                break
            checked += 1
        if checked > 50_000:
            break

    off = np.asarray(np.load(os.path.join(store_dir, "text_off.npy")))
    if int(off[-1]) != man["text_bytes"]:
        problems.append(f"text offsets end at {int(off[-1])}, "
                        f"manifest says {man['text_bytes']}")
    empties = 0
    for i in rng.choice(n, size=min(n_samples, n), replace=False):
        if not texts[int(i)]:
            empties += 1

    tl = np.asarray(G["token_len"])
    out = {"nodes": n, "edges": man["n_edges"], "half_edges": int(indices.shape[0]),
           "half_edges_checked": checked,
           "sampled_textless": empties, "manifest_textless": man["n_textless"],
           "token_len_zero": int((tl == 0).sum()),
           "token_len_mean": float(tl.mean()),
           "total_GB": round(man["total_bytes"] / 2**30, 2),
           "problems": problems}
    if verbose:
        print(json.dumps(out, indent=2), flush=True)
    if problems:
        raise AssertionError(f"{len(problems)} problem(s): {problems[:5]}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="inspect/verify a v3 graph store")
    ap.add_argument("store_dir")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--samples", type=int, default=0,
                    help="print N sample node texts")
    a = ap.parse_args()
    if a.verify:
        verify_graph(a.store_dir)
    else:
        G = load_graph(a.store_dir)
        print(json.dumps({k: v for k, v in G["manifest"].items()
                          if k not in ("stats", "arrays")}, indent=2,
                         ensure_ascii=False))
    if a.samples:
        G = load_graph(a.store_dir, verbose=False)
        kind = np.asarray(G["kind"])
        names = {0: "anchor", 1: "form", 2: "sense", 3: "example", 4: "translation",
                 5: "collocation", 6: "synonym", 7: "antonym", 8: "other"}
        for k, nm in names.items():
            idx = np.flatnonzero(kind == k)[:a.samples]
            print(f"\n--- {nm} ({int((kind == k).sum()):,} nodes) ---")
            for i in idx:
                print(f"    {G['text'][int(i)][:160]!r}")
