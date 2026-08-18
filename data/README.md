# CJVT Lexicographical KG — graph construction & GTLM input sizing

> **Status: v3, 2026-08-15.** This supersedes the v2 snapshot of 2026-07-22 and
> resolves all four defects that were listed as *Known flaws* there. The v2
> section is retained at the bottom purely as history — **do not quote v2
> figures**. Builder:
> [`kg_analysis/build_gtlm_graph_v3.py`](kg_analysis/build_gtlm_graph_v3.py).

This directory holds the raw knowledge graph (`kg_raw/`) and the construction +
sizing study (`kg_analysis/`) behind one design question:

> If we build a GTLM-ready graph from the CJVT (DDDS) lexicographical KG and
> extract **k-hop neighborhoods** around a lemma, how big is the resulting model
> input — in nodes and in **GaMS-2B tokens**?

Computed on a Slurm compute node (`frida`) over the **full** KG (all 2,594 `.nt`
files), with the real, offline **`cjvt/GaMS-2B`** tokenizer.

---

## What v3 changes

| # | v2 defect | v3 |
|---|---|---|
| 1 | Blanket **Levi reification** — every relation became its own text node | **Untyped edges** (GTLM's native `TextGraph`). The type lives in each node's *text*. Only the genuinely ambiguous `sense→sense` class (synonym / antonym) and collocations are reified, with self-describing text. |
| 2 | **Collocations missing** — `rdfs:member` never parsed, so 4.8 M `frac:Collocation` IRIs sat as textless degree-1 leaves | **Parsed and wired in**, deduplicated to 2.98 M distinct pairings, each a node `kolokacija: boj + kriminaliteta` |
| 3 | **Node-ID collisions** — unknown IRI prefixes hashed into 400 buckets | Codes are `(type << 56) | payload`; collocations pack exactly as `(D << 28) | H`. **No collisions.** |
| 4 | **9.6 % of nodes textless** | **0.001 %** (279 nodes) |

**Plus one defect found while building v3, which also affects v1/v2:** a single
form carries **several `writtenRep` values** — `word-form-1911547` has `"BOJ"`,
`"Boj"` *and* `"boj"` (402,606 such conflicts corpus-wide). Both older builders
did a plain `wr[c] = s`, so the surviving spelling depended on line order; that
is why anchors came out as `iztočnica: BOJ` and pairs as
`sopomenka: BIBLIOTEKAR ~ knjižničar`. v3 deterministically prefers the
least-capitalised variant, which yields `boj ~ sodelovanje` while correctly
leaving the proper noun `Beznik` capitalised.

### Corrections to the v2 write-up

- The v2 note that a multi-word phrase "is not stored anywhere" holds **only for
  collocations**. MWE *headwords* do store their surface form
  (`form-lexical-unit-8148598 → "divji brin"@sl`, full coverage), so those
  anchors carry the real phrase. Collocations remain lemma-joined, because a
  collocation is a pair of *senses* with no stored surface.
- Collocation `rdfs:member` targets are **senses, not lexical units**. The
  collocation node therefore attaches to two senses, and its text joins *their*
  lemmas.
- Translations were textless because they are **`@hun`**, and v2 filtered
  `writtenRep` to `@sl`. Not a modelling problem — a language-filter bug.

---

## The graph we build

| decision | raw RDF | what we build |
|---|---|---|
| **MWE decomposition** | `MWE → part → word` via an empty connector | connector **collapsed** into one edge `MWE → word` |
| **Headword vs lemma** | separate empty `lexical-unit` + `word-form` node | **merged**; anchor text = lemma + POS + morphology |
| **Inflected forms** | always separate nodes | **`form_mode`**: `collapse` (drop) or `expand` (self-describing leaf) |
| **Collocations** | double-reified, one IRI per participant, members = senses | **deduplicated by member set** into one symmetric node per pairing |
| **Synonym / antonym** | `sense → sense`, mutually indistinguishable | **reified** as a node with self-describing text |
| **Translations** | `translation-form → sense-translation → lexical-entry-translation` chain, all textless | **chain collapsed** onto `translation-form`, which holds both the `@hun` text and the link to the Slovenian sense |
| **`frac:head`** | present on every collocation | **dropped** — it is an *indexing* head, not a grammatical one (4.8 M edges removed) |
| **Edge types** | English predicate IRIs | **none** — edges are untyped; type is encoded in node text |
| **Direction** | directed triples | stored **directed**; k-hop traverses **both** directions |

Excluded throughout: `rdf:type`, morphology/POS controlled-vocabulary values
(these become node *text*), the global `lime:entry` lexicon hub, and
`phoneticRep`.

### Node text is self-describing

```
iztočnica: bežnica (samostalnik, imenovalnik, ednina)
oblika: Afričanu (dajalnik, ednina)
pomen: <definition, else the entry's lemma>
zgled: Vsak konjenik je dobil simbolno darilo: žganje z brinom ter malico.
prevod (madžarsko): rüh kezelése
kolokacija: boj + kriminaliteta
sopomenka: biblioteka ~ knjižnica
protipomenka: boj ~ sodelovanje
```

POS (`lexinfo:partOfSpeech`, 400,180 entries) is new in v3 — v2 dropped it
entirely.

**Sampling.** 400 random lexical-entry seeds, `--seed 42` — **34 single words
and 366 MWEs**, byte-identical to the v2 seed split, so the two runs are
comparable seed-for-seed.

---

## Global statistics

| | v2 | **v3** |
|---|--:|--:|
| Nodes | 37,035,372 | **36,735,791** |
| Directed edges | 47,561,737 | **48,534,031** |
| Lexical-entry nodes (seed pool) | 4,340,597 | 4,340,597 |
| Nodes with no text | 3,543,870 (9.6 %) | **279 (0.001 %)** |

Edge composition: `zgled` 14.72 M · `sestavina` 10.25 M · `oblika` 8.68 M ·
`pomen` 8.47 M · `kolokacija` 5.96 M · `sopomenka` 0.36 M · `prevod` 0.078 M ·
`protipomenka` 0.007 M.

Collocations: **4,717,087 IRIs → 2,981,731 distinct pairings** (99.8 % are
binary; member-count histogram `{1: 8875, 2: 4707758, 3: 406, 4: 44, 5: 4}`).

---

## Finding 1 — collocations are nearly free. The prediction was wrong.

v2 predicted that wiring collocations in "will add a **new hub class on common
words**, which is likely to destroy the currently-flat single-word case." It was
explicitly flagged as *must be measured, not assumed*. Measured, it does not
happen.

**Median neighborhood is completely unchanged**, and even the tails barely move
(`expand` + examples, nodes):

| seed kind | hop | p50 | p90 | p99 | max |
|---|---|--:|--:|--:|--:|
| **word**, with colloc | 2 | 19 | 76 | **740** | 1,039 |
| **word**, no colloc | 2 | 19 | 67 | **614** | 851 |
| **word**, with colloc | 3 | 19 | 192,807 | 1,640,421 | 1,914,215 |
| **word**, no colloc | 3 | 19 | 192,799 | 1,640,291 | 1,914,027 |
| **MWE**, with colloc | 3 | 137,172 | 1,065,676 | 1,750,771 | 1,752,161 |
| **MWE**, no colloc | 3 | 133,629 | 1,051,319 | 1,746,929 | 1,751,226 |

The largest effect anywhere is **+21 % at word p99 / hop 2** (614 → 740, i.e.
+126 nodes); at hop 3 the difference is under **0.01 %**, and for MWE seeds
**+2.7 %** at the median.

**Why the feared hub never forms.** Two structural reasons:

1. **Collocations hang off *senses*, not lemmas.** A common word's collocations
   are spread across its many senses, so no single node accumulates them.
2. **A collocation node has degree exactly 2.** It is a leaf-like connector, not
   a hub: reaching one costs a hop and it fans out to exactly one other sense.
   Contrast `sestavina`, where one common word is a constituent of tens of
   thousands of MWEs *from a single node*.

So the `sestavina` (MWE↔word) hub remains the **only** structure in this graph
that explodes, exactly as in v2.

## Finding 2 — dropping Levi halves the node count, but tokens are a wash

This corrects the v2 claim that blanket reification cost "roughly **a third of
the token budget**." It costs about **half the nodes** — but almost no tokens.

Induced-edge counts show what Levi would have added (v3, `expand` + examples,
median):

| seed kind | hop | nodes | induced edges | Levi nodes | factor |
|---|---|--:|--:|--:|--:|
| word | 1–3 | 19 | 18 | 37 | **×1.95** |
| MWE | 2 | 41,629 | 41,736 | 83,364 | **×2.00** |
| MWE | 3 | 137,172 | 214,880 | 352,052 | **×2.57** |

But the token budget does **not** fall, because the type tag that replaces the
relation label costs about the same — measured with GaMS-2B:

| v3 tag (per **node**) | tokens | | v2 label (per **edge**) | tokens |
|---|--:|---|---|--:|
| `iztočnica: ` | 6 | | `oblika` | 2 |
| `oblika: ` | 4 | | `pomen` | 2 |
| `pomen: ` | 4 | | `zgled` | 2 |
| `zgled: ` | 4 | | `sestavina` | 3 |
| `prevod (madžarsko): ` | 9 | | `protipomenka` | 4 |

Since this graph has ≈1.3 edges per node — and *within a ball* edges ≈ nodes —
tags at 4–6 tokens/node are not cheaper than labels at 2–4 tokens/edge. The
cleanest apples-to-apples comparison is a word seed at hop 1, where both builders
see the same 19 original nodes:

| | v2 | v3 |
|---|--:|--:|
| nodes fed to the model | 37 (Levi) | **19** |
| GaMS-2B tokens | 262 | **282** |

**49 % fewer nodes, 7.6 % more tokens.** The extra tokens buy POS, real text on
the 9.6 % of nodes that previously had none, and self-describing collocation /
synonym nodes.

This is still the right trade for GTLM — the magnetic-Laplacian attention bias is
built over the **node set**, so node count is the structural cost driver, while
tokens are merely context length — but the saving should be stated as *nodes*,
not tokens.

## Finding 3 — the MWE explosion is unchanged, and it is still the whole story

| seed kind | hop | nodes (p50) | **GaMS-2B tokens (p50)** |
|---|---|--:|--:|
| **single word** (n=34) | 1 | 19 | **282** |
| | 2 | 19 | **282** |
| | 3 | 19 | **282** |
| **MWE** (n=366) | 1 | 6 | **100** |
| | 2 | **41,629** | **453,784** |
| | 3 | **137,172** | **1,434,222** |

Unchanged from v2 in shape: single-word seeds are flat and tiny across all three
hops (the median word is not a constituent of any MWE); MWE seeds detonate at
hop 2 and catastrophically at hop 3. The *word* tail still explodes at hop 3
(p90 ≈ 193 k nodes, p99 ≈ 1.64 M) — those are the common words that *are*
constituents.

`form_mode` still matters only for single-word seeds (19 nodes / 282 tokens in
`expand` vs 2 / 50 in `collapse`), and examples are still numerically negligible
within 3 hops.

---

## Practical implications

- **Collocations can stay in unconditionally.** They cost ~0 % at the median and
  ≤21 % in the worst percentile measured, and they carry real lexicographic
  signal. This was the main open question from v2; it is now settled.
- **Raw k-hop around an MWE seed is unusable beyond hop 1**; around a
  single-word seed it is fine to hop 2 and usually hop 3.
- **The lever is still, and only, the `sestavina` hub.** Any real extractor must
  cap constituent degree or filter MWE membership before going past the danger
  hop. Nothing else in the graph needs taming.
- **`form_mode=collapse` unless morphological questions are in scope.**
- **Tags are worth a second look.** At 4–6 tokens each on every node they are the
  single largest avoidable overhead in v3. The design principle only requires
  disambiguating *lemma-anchor vs form-leaf* and *definition vs example*; shorter
  tags, or dropping them where the endpoint types are already unambiguous, is
  untested and could recover most of the 7.6 %.

---

## Reproduce

```bash
cd /shared/workspace/povejmo/gams_gtlm
sbatch data/kg_analysis/run_build_v3.sbatch     # -> kg_analysis/results_v3.json
sbatch data/kg_analysis/run_save_v3.sbatch      # -> data/kg_graph_v3/  (the store)
```

- Pipeline: stream-parse → collapse MWE parts → merge lemma → dedup collocations
  → reify syn/ant/colloc → Slovenian self-describing text → undirected CSR BFS →
  GaMS-2B tokenize → 2×2×2 variants × word/MWE split.
- Flags: `--n-seeds`, `--max-hops`, `--seed`, `--prompt-tokens`, `--workers`,
  `--variants`, `--files-limit`; persistence: `--save-graph DIR`, `--load-graph DIR`,
  `--no-analysis` (see "Persisted graph store" above); and two debugging aids:
  `--dump-samples N` (print sample node texts per kind and exit) and
  `--no-tokenizer` (char/4 proxy).
- **Run 126763: 23 min 46 s, 68.7 GB peak RSS on 32 CPU.** The sbatch asks for
  200 GB; **100 GB is sufficient** and will schedule faster.
- Uses `graph_model/.venv` on the shared filesystem. **Do not** use the local
  `.venv` symlink: it points at the per-node `/opt/deepops` venv, which lacks
  numpy on some `frida` nodes and fails in seconds.
- **The shared venv is not node-independent either.** `.venv/bin/python` is an
  absolute symlink to `/usr/bin/python3`, so on a node whose system Python is not
  3.10 the interpreter starts and then cannot see `.venv/lib/python3.10/site-packages`
  — the same `ModuleNotFoundError: No module named 'numpy'`, from a different cause.
  Job 128295 died this way on `axa` in 2 seconds.
- **Only `aga`, `ana` and `apl` can run these jobs** (probed 2026-08-18). They are
  the Ubuntu 22.04 / Python 3.10.12 nodes; `axa` and every `ix*` node are 24.04 and
  ship `/usr/bin/python3.12` with **no 3.10 at all**, so no `PYTHONPATH` trick
  rescues them — the venv's compiled wheels are `cp310`. `run_save_v3.sbatch`
  therefore pins `--nodelist=aga,ana,apl` and additionally calls
  `kg_analysis/pick_python.sh`, which prefers the venv launcher, falls back to a
  real 3.10 interpreter plus the venv's `site-packages` on `PYTHONPATH`, and exits
  with a diagnostic naming the node rather than a bare `ModuleNotFoundError`.
  `run_build_v3.sbatch` does neither and will fail in seconds if it lands on a
  24.04 node — it only ever succeeded because job 126763 happened to get `apl`.
- Tokenizer loaded offline from `HF_HOME=/shared/workspace/povejmo/huggingface_cache`.
- Raw output `kg_analysis/results_v3.json`; job log `kg_analysis/build_v3_*.out`.

> **Gotcha: `--files-limit` invalidates collocation statistics.** Files are
> globbed alphabetically, and a collocation's member senses routinely live in a
> far-away words file (`0-multi.nt` pairs with `341-words.nt`). A 700-file prefix
> yields **zero** resolvable collocations — which looks exactly like a bug. For a
> quick end-to-end check, point `--kg-dir` at a directory of symlinks that
> includes matching `*-multi`, `*-words` and `examples*` files.

---

## Persisted graph store

The builder used to discard the graph after every run, so each piece of downstream
work paid ~12 minutes and a ~70 GB node to rebuild the identical 36.7 M-node
object. `--save-graph DIR` writes it once; `kg_analysis/graph_store.py` reads it
back in seconds under a few GB.

```bash
sbatch data/kg_analysis/run_save_v3.sbatch          # build once -> data/kg_graph_v3/
python data/kg_analysis/graph_store.py data/kg_graph_v3 --verify
python data/kg_analysis/graph_store.py data/kg_graph_v3 --samples 3
```

```python
import graph_store
G = graph_store.load_graph("data/kg_graph_v3")
G["indptr"], G["indices"]        # undirected CSR
G["text"][12345]                 # "iztočnica: pes (samostalnik, imenovalnik, ednina)"
G["token_len"][12345]            # GaMS-2B token count, no tokenizer needed
G["kind"], G["mwe_set"], G["node_codes"]
```

### What is in the directory

| File | dtype | Length | Notes |
|---|---|---|---|
| `manifest.json` | — | — | shapes, provenance, and the builder's `stats` dict |
| `node_codes.npy` | int64 | `n_real` | packed `(type<<56)\|payload`, sorted — `searchsorted` maps an IRI code to a node index |
| `ntype.npy` | int32 | `n` | raw KG type id; `-1` for minted nodes |
| `kind.npy` | int8 | `n` | `K_ANCHOR … K_OTHER` |
| `mwe_set.npy` | bool | `n` | anchor is a multi-word expression |
| `indptr.npy` | int64 | `n+1` | undirected CSR |
| `indices.npy` | int32 | `2·edges` | int32 is exact: ids < 2^31 |
| `token_len.npy` | int32 | `n` | the ~8 minutes of GaMS-2B tokenization, banked |
| `text_blob.bin` | uint8 | — | every node text, UTF-8, concatenated |
| `text_off.npy` | int64 | `n+1` | byte offsets into the blob |

Three things are deliberately **not** stored. `is_form_leaf`, `is_example` and
`is_colloc` are each one comparison against `kind`, and are recomputed on load
rather than costing 105 MB of redundancy. The manifest is written **last**, so a
directory without one is an interrupted write rather than a subtly incomplete
store.

### Why node text is a blob

`text` is a Python list of 36.7 M `str`. `np.save` on that pickles element by
element: slow to write, far slower to read, and impossible to memory-map. One
concatenated UTF-8 blob plus an offsets array maps instead, and `TextStore`
decodes a node's text only when asked. It supports `texts[i]` and
`for i, s in enumerate(texts)`, which is exactly what the builder and analysis
code already do, so nothing downstream had to change.

### Measured

Built by **job 128328** on `apl`: **14 min 36 s**, **65.3 GiB peak RSS**, 16 CPU.
The save itself is 17 s of that — 16 s to encode and write the 2.28 GB text blob,
1 s for the arrays. Sizing note: the 68.7 GB figure recorded for run 126763 is the
real constraint, and it is a *build*-phase peak — before anything becomes numpy,
the parsed triples live as Python lists (12.9 M `writtenRep` + 12.1 M `rdf:value`
strings, 14.7 M usage pairs, ...), each small object carrying ~50-60 bytes of
CPython overhead. Only the file parse is parallel (94 s of the run at 16 workers,
60 s at 32), so CPUs past ~16 buy nothing.

| | bytes | |
|---|---:|---|
| `text_blob.bin` | 2,451,024,988 | 2.28 GiB, 66 chars/node mean |
| `indices.npy` | 388,272,376 | int32 halves this |
| `indptr.npy` | 293,886,464 | |
| `text_off.npy` | 293,886,464 | |
| `node_codes.npy` | 268,554,936 | |
| `ntype.npy`, `token_len.npy` | 146,943,292 each | |
| `kind.npy`, `mwe_set.npy` | 36,735,919 each | |
| **total** | **4,062,983,522** | **3.78 GB** |

Node text is ~60 % of the store and did not shrink the way dropping `phoneticRep`
suggested it would: the Slovenian tag prefixes, the morphology strings and the
3.2 M minted collocation/synonym texts add back what phonetics removed. Mean node
text is **22.4 GaMS-2B tokens**.

**What the store buys.** Job 128336 re-ran the full 400-seed, 8-variant study from
the store alone, on **4 CPU / 16 GB**, and wrote a file with the **same MD5** as the
published `results_v3.json` — `11f4b10c7fbc1e4195795e896a285820`. It peaked at
**1.04 GiB RSS** and loaded the graph in **0.0 s**.

| | build from raw | load from store |
|---|---|---|
| time to a usable graph | ~12 min | **0.0 s** |
| peak RSS | 65.3 GiB | **1.04 GiB** |
| nodes it can run on | `aga`/`ana`/`apl` only | any |

### Reusing a store

`--load-graph DIR` skips parse, build and tokenization entirely and runs the
sizing analysis straight off the store — reproducing `results_v3.json` byte for
byte at full scale (see "Measured"). On a 20-file subset every array and all
203,591 node texts also compare equal to a fresh build, element by element.
`--no-analysis` stops after the build, which is how the store is produced without
re-running the 8 variants.

A store is only valid for the inputs that made it, so `manifest.meta` records the
tokenizer, `--kg-dir`, the file count and a SHA-256 of the builder script. Check
`builder_sha256` before trusting a store against a modified builder.

---

## Remaining caveats

1. **279 textless nodes** (278 word-forms, 1 example) — negligible, but they are
   still nodes; a production export should drop them.
2. **Definition coverage is thin.** Only 225,618 `skos:definition` literals for
   8.47 M senses, so the vast majority of `pomen:` nodes fall back to their
   entry's lemma and carry no new information. Definition-bearing and
   example-bearing senses remain largely **disjoint populations**.
3. **Morphology mapping is partial.** `case/number/gender/person/tense/mood/degree`
   plus POS are mapped to Slovenian; `aspect`, `vform`, `definiteness`,
   `negative`, `clitic`, `animate` are parsed by the KG but not rendered.
4. **The writtenRep tie-break is a heuristic.** "Fewest capitals, then
   lexicographic" is right for `BOJ/Boj/boj` and harmless for `Beznik`, but a
   proper noun that also lists a lowercase variant would be lower-cased.
5. **Ordered MWE components are available but unused.** `rdf:_1 … rdf:_4` give
   constituent order; v3 ignores it, so `sestavina` edges are an unordered set.
6. Token counts use `add_special_tokens=False` per node; GTLM's own separators
   are not modelled, and the prompt allowance is a flat 24 tokens.

---

## Appendix — superseded v2 snapshot (2026-07-22)

Retained for provenance only. v2 built a 37.0 M-node / 47.6 M-edge graph and
reported MWE hop-2 medians of 42,226 original / 85,284 Levi nodes / 385,061
tokens, and word hop-1 of 19 / 37 / 262. Those figures assume blanket Levi
reification, omit collocations entirely, silently merge ≥1.3 M colliding
collocation IRIs, and count 9.6 % of nodes as contributing zero tokens. Builder
`kg_analysis/build_gtlm_graph.py`, results `kg_analysis/results_v2.json`, still
in the tree. The v1 raw-projection study (`analyze_neighborhoods.py`,
`results.json`, `results_semcore.json`) was judged heavily flawed and is not
documented here.

> **Note on disk.** `kg_raw/` currently keeps both the 41 GB `share_download.zip`
> and the extracted `OntoLex DSB/` (42 GB), plus a redundant nested
> `OntoLex DSB/examples-data.zip` (1 GB). The two zips can be deleted to reclaim
> ~42 GB once the extracted files are confirmed good.
