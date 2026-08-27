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
├── pipeline/    the whole build as one program -> run_pipeline.sbatch
├── lib/         graph_store.py — loads a store; paths.py — where everything lives;
│                errors.py — the one exception a stage raises;
│                pick_python.sh — interpreter guard
├── build/       build_graph.py, check_store.py (did a rebuild lose anything?)
│                and diff_stores.py (is a rebuild byte-identical?) -> stores/
├── qa/          the dataset package -> datasets/
├── lookup/      cli.py, the query CLI (`bin/lookup` wraps it)
├── analysis/    measurement scripts (extraction runs, hub costs, sizing)
├── prompts/     the extractor prompt, and the superseded revisions in archive/
├── stores/      the built stores (gitignored, ~4 GB each)
├── datasets/    generated/ the items · balls/ their subgraphs ·
│                balls_serialised/ + balls_noretrieval/ the two baseline inputs ·
│                work/ what the build made on the way, and run.json ·
│                previous/ the build before this one · reference/ two reference
│                QA files, used for question phrasing only
└── kg_raw/      the untouched N-Triples (83 GB, gitignored)
```

**There is one dataset, and it is not versioned.** `datasets/generated` and
`datasets/balls` are the current corpus; a rebuild replaces them and moves what
was there into `datasets/previous/`. Intermediates live under `datasets/work/`
and are yours to delete.

A directory names a **purpose, not an import**: anything imported by two or more of
build/analysis/lookup lives in `lib/`, and data products live under a directory that
says what they are.

Every package here is run with `-m` **from `data/`** — `python -m qa.build_dataset`,
`python -m build.build_graph`, `python -m lookup.cli` — which is what puts `data/` on
the import path. The job scripts `cd` there for you.

---

## Build everything

One job, one log, one thing to wait for:

```bash
sbatch data/run_pipeline.sbatch
```

That builds the graph store if it is missing, generates the items, runs the
entity linker on every GPU the job was given, relabels, extracts the balls,
derives the two baselines, runs every check, and publishes the result into
`datasets/`. **~35 minutes** when the store already exists, **~50** when it has
to be built first. The only prerequisite it cannot supply is the raw KG — 83 GB
from the link in the [repo README](../README.md), unpacked to
`data/kg_raw/OntoLex DSB/`.

| flag | |
|---|---|
| `--scale 0.02` | shrink every split proportionally — **this is the fast end-to-end test**, ~250 items and a few minutes |
| `--types T1,T3` | narrow the type **mix**. Note it does *not* shrink the build: each split's budget is divided among the types asked for, so a two-type corpus is the same size as a nineteen-type one. Combine with `--scale` |
| `--gpus N` | use at most N of the visible GPUs (default: all of them) |
| `--rebuild-store` | rebuild the store even if the one on disk matches |
| `--seed N` | the generator's seed (default: 20260821) |
| `--store DIR` | read (and if absent, build) a store elsewhere |

Any `--scale` other than 1.0, and any `--types`, makes the run a **subset**: it
works entirely inside `datasets/work/subset/` and is **never published**, so it
touches neither the corpus nor the intermediates that describe it. Expect
`selftest` to fail checks that quantify over the whole corpus — C18 asserts T17
items exist, and a `T1,T3` build has none. That is the subset, not the pipeline.

The job asks for **2 GPUs** by default; stage 3 shards across whatever it gets,
so `sbatch --gres=gpu:1 data/run_pipeline.sbatch` and `--gres=gpu:8` both work
unchanged. One GPU costs about 13 minutes more than two; eight save about three.

It runs on **B200, B300 or H100**. Nothing here needs Blackwell — the store
build is CPU-only and the extractor is 24 GB in bf16 — and the Blackwells are
usually the contended ones, so when the queue is deep the fastest route to a
finished build is often the smallest allocation that starts:

```bash
sbatch --constraint=GPU_BRD:H100 --gres=gpu:1 data/run_pipeline.sbatch
```

**The store is skipped when it already matches.** The pipeline compares the
store's `manifest.json` against what was asked for — tokenizer, source
directory, collocation text mode, sense snippet and index, text convention — and
reuses it when they agree. It deliberately does *not* key that decision on the
builder's SHA-256: that moves when a comment changes, and rebuilding 4 GB over a
comment is not a service to anyone. A difference there is printed instead.

**Checks report; they do not gate.** `selftest`, the grader over the gold, and
the ball and variant checks all run at their proper stage. A failing one is
recorded, named in a banner at the end and written to `datasets/work/run.json` —
and the build still finishes and still publishes, because what to do about a
failing check is a judgement call and this path exists to hand you the artefacts
plus the truth about them. A failing *stage* stops the run, since there is
nothing downstream to build; nothing is published then, and `datasets/` is left
untouched.

**Nothing lands in place until the whole run succeeds.** Stages write under
`datasets/work/`; the four finished directories are moved into `datasets/` at
the end, and the build they replace steps aside into `datasets/previous/`. A run
that dies in stage 5 therefore cannot leave `generated/` and `balls/` disagreeing
about what a T17 item is supervised on.

`datasets/work/run.json` records what happened: the git SHA, the arguments, the
store's identity, the extractor's model **and the SHA-256 of the prompt it
used**, every stage's duration and every check's verdict. The prompt matters
because changing it changes which anchors the corpus is built from, and nothing
in `datasets/` would otherwise say which one it was.

### What is and is not reproducible

Stages 1, 4, 5 and 6 are **bit-reproducible**: same store, same seed, same
bytes out. Verified by rebuilding the whole deterministic chain twice under
different `PYTHONHASHSEED` values and diffing all 19 artefacts. The store is
too — `python -m build.diff_stores NEW OLD` asserts it, which is the check to
run after touching the builder:

```bash
cd data && python -m build.diff_stores stores/kg_graph_gemma3 /path/to/rebuild
```

`check_store` asks the softer question — did a rebuild *lose* anything — and is
what to reach for when a rebuild legitimately differs.

That property is easy to lose, and was lost twice:

* `sl_key` is a *collation* key and casefolds, so `zastava` and `Zastava` compare
  equal. Sorting a **set** by it left their order to set iteration, which Python
  varies per process — the same rebuild emitted `zastava, Zastava` one day and
  `Zastava, zastava` the next. `sl.sl_sort_key` breaks the tie on the string
  itself; use it whenever a collated result is written to disk. The selftest's
  C15 could not catch this, because `sorted` is stable and a list in *either*
  order compared equal to `sorted(g, key=sl_key)`; it now checks the total key.
  One item in the corpus on disk (`T4-000439`, `zastava`) predates the fix and
  will be reported by C15 until the next rebuild — T4 is graded `multiset`, so
  no score ever depended on it.
* CSR adjacency used to come out in `imap_unordered` worker-completion order.
  `undirected_csr` now lexsorts within each node's slice.

**Stage 3 is not reproducible and cannot be.** It is LLM generation, and the
shard count changes batch composition, so a 2-GPU run and a 4-GPU run produce
different extractions — which then move a handful of items between `keep` and
`extract_miss`. A corpus is reproducible *given its extraction dump*, which is
why the dump is kept in `datasets/work/extraction/`.

---

## Build it stage by stage

The pipeline above runs exactly the jobs below, in this order. Reach for these
when you are iterating on one stage rather than rebuilding everything — each
script documents its own arguments and defaults, and the defaults point at the
same paths the pipeline uses, so the two interoperate.

### 1. Build the graph store

```bash
sbatch data/build/run_save.sbatch gemma3     # -> data/stores/kg_graph_gemma3/
```

~15–35 min and **~48 GB**. **Only `aga`, `ana` and `apl` can run it this way**:
they are the Ubuntu 22.04 / Python 3.10 nodes the shared venv was built against,
and the job pins them. The pipeline has no such limit — it runs in the pyxis
container, which is a Python 3.10 base, so the same build works on the whole
B200/B300 fleet, where it takes ~13 min.

### What the store build actually costs

Worth stating, because the number in this file was wrong by 2× for a while and
the job asked for 110 G on the strength of it:

| | |
|---|--:|
| the builder process itself | **29 GB** peak, 19 GB for the tokenizer tail |
| the job, as the cgroup counts it | **48 GB** |
| the job, before the prefetch bound | 105 GB |

The three differ for two reasons. Slurm's `MaxRSS` comes from the cgroup, which
charges **page cache** — and this build streams 41 GB of N-Triples — so the job
figure will always sit well above the process figure. And until it was bounded,
`Pool.imap_unordered` prefetched without limit: workers parse far faster than
the merge loop's ~50 M dict inserts can absorb, so finished results piled up and
the queue grew into whatever memory the job was given. Measured: 2, 4 and 8
workers all peak at 33 GB, 16 peaks at 91 GB, and 16 buys 44 seconds.

`_gated` in `build_graph.py` caps the pool's *input* with a semaphore, which
bounds results in flight and decouples memory from the worker count. The build
keeps all 16 workers, runs slightly **faster** (less allocation pressure), and
produces a byte-identical store — verified with `build.diff_stores` against a
reference build.

It is not a memory *limit*: a build capped at 48 G completed fine even before
the fix, because the kernel reclaims cache and the queue adapts to pressure. It
is that the job now asks for what it uses.

`analysis/profile_store_build.py` is the harness that measured all of this —
reach for it before believing a memory number here, including these.

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

### 2. Generate the QA dataset

Five stages, each one job. Submit from the repo root — every job `cd`s to
`data/`, so **its path arguments are relative to `data/`**. Every argument below
is also that script's default, so the bare `sbatch` does the same thing.

```bash
# 1 the items
sbatch data/qa/run_generate.sbatch stores/kg_graph_gemma3 \
       datasets/work/generated_raw

# 2 entity linking (GPU)
sbatch data/analysis/run_extract_sharded.sbatch \
       datasets/work/generated_raw datasets/work/extraction/items

# 3 anchors from stage 2
sbatch data/qa/run_relabel.sbatch datasets/work/generated_raw \
       datasets/work/extraction/items \
       datasets/work/relabelled

# 4 balls AND items
sbatch data/qa/run_build_balls.sbatch "" datasets/work/relabelled \
       datasets/balls datasets/generated

# 5 the two baselines
sbatch data/qa/run_build_variants.sbatch datasets/balls datasets
```

Note that stages 4 and 5 write **straight into `datasets/`**, overwriting the
live corpus in place — the staging-and-publish dance is the pipeline's, not
theirs. That is usually what you want when you are deliberately rebuilding a
stage, and worth knowing before you run one against a corpus you care about.

* **Stage 1** walks the seed pool, computes the availability matrix by *running*
  every generator over every pool entry (so "this type is available for this lemma"
  is a fact, not a prediction), then generates, then selftests.
* **Stage 2** is the only GPU stage — as a standalone job it is a Slurm array in
  the pyxis container, one task per shard; inside the pipeline it is one
  subprocess per GPU of a single job. Either way it runs a real entity-linking
  pass, so an item's anchors are what a plain-text lookup resolves rather than
  the generator's privileged knowledge of which node it used. The extractor reads
  `prompts/extractor_prompt.txt`; changing it changes which anchors the corpus is
  built from, so the corpus has to be rebuilt with it.
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
python -m qa.inspect  datasets/generated --stats
python -m qa.inspect  datasets/generated -n 3
python -m qa.selftest --dataset datasets/generated --store stores/kg_graph_gemma3
python -m qa.grade    datasets/generated/test.jsonl --predictions preds.jsonl
python -m qa.check_balls datasets/generated datasets/balls
python -m qa.check_variants datasets/balls datasets/balls_{noretrieval,serialised}
```

A full pipeline run performs all of these itself, at the stage each belongs to,
and records the verdicts in `datasets/work/run.json`. Run them by hand against a
corpus you built stage by stage, or to re-check one after a change.

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

## Looking things up

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
