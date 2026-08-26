# `data/` — the knowledge graph and the QA dataset

Two artefacts, built in that order:

1. a **graph store** — the CJVT (DDDS) lexicographical KG, parsed and reshaped into
   the untyped text graph GTLM reads, persisted as memory-mappable arrays;
2. a **QA dataset** — Slovene lexicographical questions with their answers, each
   paired with the subgraph ("ball") that contains the evidence.

`../train/` reads the second. Nothing in the training path opens the store: a ball is
an artefact on disk, so the graph the model saw is part of the record.

```
data/
├── lib/         graph_store.py — loads a store; paths.py — where everything lives;
│                pick_python.sh — interpreter guard
├── build/       build_graph.py + check_store.py, its regression check -> stores/
├── qa/          the dataset package -> datasets/
├── lookup/      cli.py, the query CLI (`bin/lookup` wraps it)
├── analysis/    measurement scripts (extraction runs, hub costs, sizing)
├── prompts/     the extractor prompt, and the superseded revisions in archive/
├── stores/      the built stores (gitignored, ~4 GB each)
├── datasets/    generated/ the items · balls/ their subgraphs · reference/ two
│                reference QA files, used for question phrasing only
└── kg_raw/      the untouched N-Triples (83 GB, gitignored)
```

A directory names a **purpose, not an import**: anything imported by two or more of
build/analysis/lookup lives in `lib/`, and data products live under a directory that
says what they are.

Every package here is run with `-m` **from `data/`** — `python -m qa.build_dataset`,
`python -m build.build_graph`, `python -m lookup.cli` — which is what puts `data/` on
the import path. The job scripts `cd` there for you.

---

## 1. Build the graph store

```bash
sbatch data/build/run_save.sbatch gemma3     # -> data/stores/kg_graph_gemma3/
```

~20–35 min and ~70 GB RSS. **Only `aga`, `ana` and `apl` can run it**: they are the
Ubuntu 22.04 / Python 3.10 nodes the shared venv was built against, and the job
pins them.

The build ends with `graph_store.py --verify`, which is self-contained: shapes,
offsets and the manifest are checked against the store itself.

Pass an existing store as a second argument to add a **regression check**
(`build/check_store.py`): it asserts the new build lost nothing relative to that one
— IRI-backed layer and synonym/antonym layer untouched, collocation layer only ever
growing, no member set carrying one phrase twice. A first build has nothing to
compare against and skips it.

The result is `data/stores/kg_graph_gemma3` — what `lookup`, the QA generator and
everything downstream read.

| | |
|---|--:|
| nodes | 37,498,126 |
| directed edges | 50,058,701 |
| textless nodes | 279 (0.001 %) |
| on disk | 4.4 GB |

`token_len` is the one array that depends on a model choice, so the directory is
named for the tokenizer that filled it. `gemma3` is `cjvt/GaMS3-12B-Instruct`, which
every Gemma 3 size shares — one store serves the whole iteration ladder. Pass
`gams2b` instead for a `cjvt/GaMS-2B` store; the two differ in `token_len` and
nothing else.

Loading is instant and needs ~1 GB, on any node:

```python
from lib import graph_store
G = graph_store.load_graph("stores/kg_graph_gemma3")
G["indptr"], G["indices"]        # undirected CSR
G["text"][12345]                 # "iztočnica: pes (samostalnik, moški spol, ...)"
G["token_len"][12345]            # token count, no tokenizer needed
G["kind"], G["mwe_set"], G["node_codes"]
```

```bash
cd data && python -m lib.graph_store stores/kg_graph_gemma3 --verify
```

The directory holds `node_codes` (packed IRI ids, sorted — `searchsorted` maps an
IRI to a node index), `ntype`, `kind`, `mwe_set`, `indptr`/`indices` (undirected
CSR), `token_len`, and node text as one UTF-8 blob plus an offsets array so it can
be mapped rather than unpickled. `manifest.json` is written **last**, so a directory
without one is an interrupted write; its `meta` records the tokenizer, the source
directory, the text convention and a SHA-256 of the builder.

### What the graph looks like

Edges are **untyped** — GTLM's native `TextGraph` — and the relation is encoded in
the node's own text, so every node is self-describing:

```
iztočnica: megalomanka (samostalnik, ženski spol, imenovalnik, ednina)
oblika: popraskam (sedanjik, 1. oseba, ednina)
pomen 2: pes (zgled: Sosedov pes je spet lajal ...)
zgled: Vsak konjenik je dobil simbolno darilo: žganje z brinom ter malico.
prevod (madžarsko): rüh kezelése
kolokacija: kisova voda
sopomenka: biblioteka ~ knjižnica
protipomenka: boj ~ sodelovanje
```

The reshaping the builder does, and why:

| raw RDF | what we build |
|---|---|
| `MWE → part → word` via an empty connector | connector **collapsed** into one edge |
| separate empty `lexical-unit` + `word-form` | **merged** into one anchor: lemma + POS + morphology |
| `sense → sense`, synonym and antonym mutually indistinguishable | **reified** as a node whose text says which it is |
| collocations double-reified, textless, members are senses | **one node per distinct phrase**, carrying the curated phrase the dictionary stores |
| translation chain, all three nodes textless | **collapsed** onto the form that holds the `@hun` text |
| English predicate IRIs on every edge | **dropped** — the type is in the node text |
| directed triples | stored directed; traversal goes **both** ways |

Excluded: `rdf:type`, the morphology/POS vocabulary values (they become node text),
the global lexicon hub, `phoneticRep`, and `frac:head` (an indexing head, not a
grammatical one).

Node text is what the model reads, so each fact is written in the cheapest place
that keeps it unambiguous: a sense carries its dictionary ordinal and, absent a
definition, a snippet of its first example, because otherwise sibling senses are
byte-identical and a permutation-equivariant architecture cannot tell them apart.
Noun gender goes on the anchor only, since every form leaf is adjacent to its
anchor and repeating it would cost 16× the text.

---

## 2. Generate the QA dataset

Five stages. Submit from the repo root — every job `cd`s to `data/`, so **its path
arguments are relative to `data/`**. `v2_clean` is the current output; substitute
your own tag throughout.

```bash
# 1 the items
sbatch data/qa/run_generate.sbatch stores/kg_graph_gemma3 \
       datasets/generated/v2

# 2 entity linking (GPU)
sbatch data/analysis/run_extract_sharded.sbatch \
       datasets/generated/v2 analysis/results/extraction_v2

# 3 anchors from stage 2
sbatch data/qa/run_relabel.sbatch datasets/generated/v2 \
       analysis/results/extraction_v2 \
       datasets/generated/v2_relabelled

# 4 balls AND items
sbatch data/qa/run_build_balls.sbatch "" datasets/generated/v2_relabelled \
       datasets/balls/v2_clean datasets/generated/v2_clean

# 5 (optional) the two baselines
sbatch data/qa/run_build_variants.sbatch datasets/balls/v2_clean
```

* **Stage 1** walks the seed pool, computes the availability matrix by *running*
  every generator over every pool entry (so "this type is available for this lemma"
  is a fact, not a prediction), then generates, then selftests.
* **Stage 2** is the only GPU stage — a Slurm array in the pyxis container. It runs
  a real entity-linking pass, so an item's anchors are what a plain-text lookup
  resolves rather than the generator's privileged knowledge of which node it used.
  The extractor reads `prompts/extractor_prompt.txt`; changing it changes which
  anchors the corpus is built from, so the corpus has to be rebuilt with it.
* **Stage 4** writes the balls **and** the dataset in one pass. They must be built
  together: the T17 target is re-verbalised against the ball, so that the model is
  never supervised on a collocation its own input does not contain. It refuses any
  item stage 3 did not relabel.
* **Stage 5** derives the two baseline inputs from the ball file itself, so all
  three arms are the same items with the same answers.

Each script's header documents its own arguments and defaults.

### What comes out

| | |
|---|--:|
| items | 12,490 — train 9,266 / dev 1,040 / test 2,184 |
| types | 19 (`QA_TASKS.md`) |
| negatives | 10.5 % |
| ball size | p50 77 nodes / 1,314 tokens · p99 299 / 5,723 · max 705 / 14,055 |

Splits are **lemma-disjoint**, with held-out generalisation tiers; antonyms (T16) are
held out of training entirely, so their score is not evidence about any
architecture. Seeds are drawn across seven frequency bands, where frequency is
proxied by collocation + MWE membership.

An item carries the question, the gradeable `ODGOVOR:` line, and the grading
contract; the ball carries the same answer plus the node and edge lists:

```json
{"id": "T1-000580", "type": "T1", "lemma": "megalomanka", "band": "B0",
 "question": "Navedi vse sklone besede megalomanka.",
 "answer": "ODGOVOR: megalomanka, megalomanke, megalomanki, ..."}

{"id": "T1-000580", "n_nodes": 20, "n_tokens": 281, "anchors": [0],
 "nodes": ["iztočnica: megalomanka (samostalnik, ženski spol, imenovalnik, ednina)",
           "oblika: megalomanke (rodilnik, ednina)", "..."],
 "edges": [[0, 1], "..."]}
```

`generated/` is the authority for both the target and the contract; `train/data.py`
joins the two on `id` and asserts the answers are identical.

### Inspecting and grading

```bash
cd data
python -m qa.inspect  datasets/generated/v2_clean --stats
python -m qa.inspect  datasets/generated/v2_clean -n 3
python -m qa.selftest --dataset datasets/generated/v2_clean --store stores/kg_graph_gemma3
python -m qa.grade    datasets/generated/v2_clean/test.jsonl --predictions preds.jsonl
python -m qa.check_balls datasets/generated/v2_clean datasets/balls/v2_clean
```

`qa/grade.py` is the grader — a script, not a judge. Every gradeable string appears
verbatim in the model's input, so exact match after a fixed normalization is both
fair and meaningful. Three modes, one constant per type in `qa/spec.py`, never
inferred at run time: `sequence` (positions carry meaning), `multiset` (a set whose
order is our convention), `membership` (any allowed subset, plus a count rule).
Run against the gold it scores 100 %, on every split, type and band.

### How a ball is extracted

One policy for every question type — production accepts arbitrary questions, so
extraction must not branch on what was asked:

* **hop 2** from every matched lexical unit, unioned when a surface string owns
  several. Hop 2 is both floor and ceiling: synonyms, antonyms and collocations are
  reified one hop past the sense, and those nodes carry both lemmas in their text.
* **two hubs are capped**, or the ball explodes. `sestavina` (word → MWE) at K = 10,
  deterministically ranked; `sense → kolokacija` at K = 10 **per anchor**, drawn by
  seeded weighted sampling with `w ∝ log(1 + proxy(partner))`. Uncapped, the anchors
  this dataset samples on purpose cost a p50 of 22,704 tokens against ~2,000 capped.
  Collocations pile onto sense 1 rather than spreading over an anchor's senses, so
  the cap is pooled per anchor.

The ball is uniform across task types deliberately. Giving T3 the paradigm and T9
the anchor alone would be cheaper and would also make every score meaningless: the
model would be told which relation to read before it read anything.

---

## 3. Looking things up

```bash
lookup store word jabolke              # spelling -> headwords and forms, ranked
lookup store word boj --also all       # ...and senses, examples, collocations
lookup store id 69611 --hops 1         # a unit's text, then its neighbourhood
lookup raw id 34748                    # every raw triple touching it (~8 s)
lookup raw iri word-form-1567346       # the same for any IRI
```

`bin/lookup` runs from any directory and picks an interpreter that can import numpy
on the current node; put it on `PATH` once:

```bash
printf '\nexport PATH="/shared/workspace/povejmo/gams_gtlm/bin:$PATH"\n' >> ~/.bashrc && source ~/.bashrc
```

The leading `\n` is not cosmetic: a dotfile whose last line has no trailing newline
turns `echo … >>` into a silent corruption. `KG_STORE=/path` points lookups at
another store, `KG_PYTHON=/path` at another interpreter. `lookup -h` has the rest.

---

## Known limits

* **Definition coverage is thin** — 225,618 `skos:definition` literals for 8.47 M
  senses. Most `pomen:` nodes fall back to the entry's lemma plus an example
  snippet. Unfixable from this source, and it caps what T12 can be.
* **MWEs carry no grammar.** POS and morphology sit on single-word entries only, and
  MWEs are 91 % of anchors; the component collapse also discards each constituent's
  case and number. The graph holds no grammatical information about multi-word
  expressions anywhere.
* **`negative` and `animate` are parsed but not rendered**, and `clitic` is
  effectively dead at 25 reachable units.
* **The `writtenRep` tie-break is a heuristic** — fewest capitals, then
  lexicographic. Right for `BOJ/Boj/boj`, but a proper noun that also lists a
  lowercase variant would be lower-cased.
* **MWE constituent order is available and unused** (`rdf:_1 … rdf:_4`), so
  `sestavina` edges are an unordered set. Cheap to ignore since the collocation node
  carries the phrase in its real word order.
* **279 nodes have no text at all** — 278 word-forms and one example.

Per-type answer contracts are in [`QA_TASKS.md`](QA_TASKS.md); the dataset's design
decisions and the KG answerability census are in
[`QA_DATASET_DESIGN.md`](QA_DATASET_DESIGN.md).
