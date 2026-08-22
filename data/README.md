# CJVT Lexicographical KG — graph construction & GTLM input sizing

> **Status: v3, 2026-08-15.** This supersedes the v2 snapshot of 2026-07-22 and
> resolves all four defects that were listed as *Known flaws* there. The v2
> section is retained at the bottom purely as history — **do not quote v2
> figures**. Builder:
> [`build/build_gtlm_graph.py`](build/build_gtlm_graph.py).
>
> **Companion document:** [`QA_DATASET_DESIGN.md`](QA_DATASET_DESIGN.md) records the design
> of the question–answer dataset built on top of this graph, including a full-corpus census
> of which question types the KG can and cannot answer.

This directory holds the raw knowledge graph (`kg_raw/`), the built graph stores
(`stores/`) and the construction + sizing study (`build/`, `analysis/`) behind
one design question:

> If we build a GTLM-ready graph from the CJVT (DDDS) lexicographical KG and
> extract **k-hop neighborhoods** around a lemma, how big is the resulting model
> input — in nodes and in **GaMS-2B tokens**?

Computed on a Slurm compute node (`frida`) over the **full** KG (all 2,594 `.nt`
files), with the real, offline **`cjvt/GaMS-2B`** tokenizer. The current store has
since moved to the Gemma 3 tokenizer, which is worth **−0.80 %** in aggregate — so
every token figure below is within ~1 % rather than restated. See
[Tokenizer](#tokenizer-and-what-the-store-directories-are-named).

---

## Layout

```
data/
├── README.md  QA_DATASET_DESIGN.md  QA_TASKS.md
├── lib/         graph_store.py, pick_python.sh — shared by everything below
├── build/       produces a graph store: the v3 builder, check_v4_text.py,
│                run_save_*.sbatch, run_verify_store_v3.sbatch
├── analysis/    produces a results JSON: the v1 sizing study, the superseded
│                v2 builder, run_analysis*.sbatch, run_build_v[23].sbatch,
│                results/*.json
├── lookup/      kg_lookup.py — the query CLI (`bin/lookup` wraps it)
├── qa/          produces the QA dataset from a store: build_dataset.py and the
│                pieces it composes — seeds/spec/templates/gen/d5b/grade/
│                selftest/inspect, plus run_generate.sbatch; and build_balls.py,
│                which turns items into the text graphs a GTLM reads
├── datasets/    reference/ the two reference QA sets; generated/ our items;
│                balls/ one text graph per item (nodes + edges, explicit)
│                        — consumed by ../train/, which fine-tunes GTLM on them
├── stores/      kg_graph_v3 … kg_graph_v7_gemma3 (gitignored, ~4 GB each)
├── kg_raw/      the untouched CJVT N-Triples (83 GB, gitignored)
└── logs/        Slurm job logs
```

Three rules keep it that way. A directory names a **purpose, not an import**:
`run_build_v3.sbatch` sits in `analysis/` although it invokes the builder in
`build/`, because what it produces is the census, not a store. Anything imported
by two or more of build/analysis/lookup goes in **`lib/`**. Data products live
under a directory that says what they are, never loose in `data/`.

`qa/` is the one package rather than a bag of scripts, because its pieces have to
agree with each other: `spec.py` is the per-type answer contract that both
`gen.py` and `grade.py` read, and `d5b.py` is the collocation sampler that both
the T17 generator and — later — the ball builder must call rather than
reimplement. `analysis/measure_d5b.py` imports from it for the same reason.

### Generating the dataset

```
sbatch data/qa/run_generate.sbatch                    # -> datasets/generated/v1
python -m qa.inspect datasets/generated/v1 --stats    # what came out
python -m qa.inspect datasets/generated/v1 -n 3       # read some of it
python -m qa.selftest --dataset datasets/generated/v1 --store data/stores/kg_graph_v7_gemma3
python -m qa.grade   datasets/generated/v1/test.jsonl --predictions preds.jsonl
```

`report.json` records the availability matrix (computed by *running* every
generator over every pool entry, not by predicting eligibility), the realised
type × band × split counts, and the negative mix.

### What v1 came out as (2026-08-22, against `kg_graph_v7_gemma3`)

**12,493 items** — train 9,268 / dev 1,040 / test 2,185 — over 19 types and 279
distinct question frames, from a seed pool of **72,334 lemmas** (196 excluded for
a source-ambiguous lemma, Finding 10's neighbourhood). Negatives are **8.5 %**
(mismatch 442 / nonexistent 318 / absent 308), tiers A 3.9 % and C 0.9 %. The
selftest (`QA_TASKS.md` §2) passes on all of C6, C7, C9, C10, C11, C13, C15,
C16, C17, C18, and the grader scores the gold at **100.00 %** on every split,
type and band.

Availability — every generator *run* over every pool entry, not predicted:

| | | | |
|---|---|---|---|
| T8 72,334 | T17 65,288 | T12/T14 40,781 | T9 38,741 |
| T15 35,517 | T3 33,553 | T21 33,550 | T2 33,537 |
| T1 33,267 | T4 32,781 | T19 10,175 | T7 8,061 |
| T10 8,060 | T5/T6 8,057 | T16 3,699 | T20 2,386 |
| **T11 575** — the thinnest, after C22 rejects degenerate gradation | | | |

Two numbers to read with the results, both reported by C13: **T20's
majority-class baseline is 53.9 %** (a dictionary example cites its headword in
the base form) and **T9's is 42.6 %**. A per-type score below its own baseline
row means nothing was learned.

Four defects were found by *reading* generated items rather than by any
assertion, and each produced a check: Finding 12 / C23 (lemma-filled paradigms),
C24 (frames asking for fewer slots than the answer contract fills), C20 (false
premises) and C22 (degenerate gradation). C17 caught a fifth on its own — T12
and T14 were drawing their negatives from separate random streams, so their
agreement was a coincidence that held in one generation and broke in the next.

---

## What v3 changes

| # | v2 defect | v3 |
|---|---|---|
| 1 | Blanket **Levi reification** — every relation became its own text node | **Untyped edges** (GTLM's native `TextGraph`). The type lives in each node's *text*. Only the genuinely ambiguous `sense→sense` class (synonym / antonym) and collocations are reified, with self-describing text. |
| 2 | **Collocations missing** — `rdfs:member` never parsed, so 4.8 M `frac:Collocation` IRIs sat as textless degree-1 leaves | **Parsed and wired in**, deduplicated to 2.98 M distinct pairings, each a node `kolokacija: boj + kriminaliteta` — *superseded by v5, which writes the real phrase `kolokacija: boj proti kriminaliteti`; see Finding 8* |
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

- ~~The v2 note that a multi-word phrase "is not stored anywhere" holds **only for
  collocations**.~~ **Wrong — it holds for nothing. Corrected in v5, see
  Finding 8.** MWE *headwords* do store their surface form
  (`form-lexical-unit-8148598 → "divji brin"@sl`, full coverage). So do
  collocations: the collocation node has no text of its own, but its IRI names
  the sense of the multi-word entry that spells the pairing out, and every one of
  the 4,717,090 of them resolves.
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
kolokacija: boj proti kriminaliteti
sopomenka: biblioteka ~ knjižnica
protipomenka: boj ~ sodelovanje
```

The collocation line read `kolokacija: boj + kriminaliteta` through v4. v5 writes
the phrase the dictionary actually stores — see **Finding 8**.

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

## Finding 1 — collocations are nearly free *at the median only* ⚠️

> **⚠️ PARTLY SUPERSEDED 2026-08-21 — see finding 9.** The per-seed cost table below
> is correct as measured. The conclusion drawn from it was not: v2's prediction was
> **right**, and this finding dismissed it on a sample that could not see it.

v2 predicted that wiring collocations in "will add a **new hub class on common
words**, which is likely to destroy the currently-flat single-word case." It was
explicitly flagged as *must be measured, not assumed*.

**v2 was right.** The hub is real — `pomen 1: imeti` carries 14,233 collocation
edges — and this finding missed it by sampling lexical units uniformly, where the
median anchor has 4 collocations and the hubs are a thin tail. Everything below is
accurate for the median seed and misleading for the seeds the QA dataset actually
draws. Finding 9 has the corrected picture.

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

> **⚠️ SUPERSEDED 2026-08-21 — this subsection was wrong.** See finding 9. Reason 1 is
> false as measured, reason 2 is true but measures the wrong direction, and the
> conclusion does not hold. Kept here because the reasoning error is instructive.

<details>
<summary>Original text — "Why the feared hub never forms"</summary>

**Why the feared hub never forms.** Two structural reasons:

1. **Collocations hang off *senses*, not lemmas.** A common word's collocations
   are spread across its many senses, so no single node accumulates them.
2. **A collocation node has degree exactly 2.** It is a leaf-like connector, not
   a hub: reaching one costs a hop and it fans out to exactly one other sense.
   Contrast `sestavina`, where one common word is a constituent of tens of
   thousands of MWEs *from a single node*.

So the `sestavina` (MWE↔word) hub remains the **only** structure in this graph
that explodes, exactly as in v2.

</details>

**What is actually true.** Reason 1 is empirically false: collocations are **not**
spread across a common word's senses, they pile onto sense 1. `imeti` has 14,249
collocations over all its senses and 14,233 of them — 99.9 % — hang off `pomen 1`.
`voda`: 3,003 of 3,026, 99.2 %.

Reason 2 is true but irrelevant. A collocation node does have degree exactly 2; the
hub is the **sense on the other end**, and the argument only ever measured the
`kolokacija → sense` direction. Traversed the other way, `sense → kolokacija` is
unbounded: p90 = 101, p99 = 791, max = 14,233, with 28.8 % of collocation-bearing
senses above 15.

So `sestavina` is **not** the only structure that explodes. Both hubs need a cap,
and both are capped — `sestavina` by D5, `sense → kolokacija` by D5b (per anchor,
K = 15). Full measurements in `QA_DATASET_DESIGN.md` §3.1b.

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

## Finding 3 — the MWE explosion is unchanged (but it is *not* the whole story) ⚠️

> **⚠️ TITLE CORRECTED 2026-08-21.** The MWE numbers below stand. "The whole story"
> does not — the `sense → kolokacija` hub is a second explosion of the same order.
> See finding 9.

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

## Finding 4 — sibling senses were byte-identical, and that is a builder defect

Caveat 2 below records that only 225,618 of 8,468,227 senses (2.7 %) carry a
`skos:definition`, and that the rest fall back to their entry's lemma. The
write-up called that "no new information." Measured on the v3 store over a
200,000-anchor sample, it is worse than that:

| | |
|---|--:|
| anchors with ≥2 senses | 91.3 % |
| …where **some** sibling sense texts are byte-identical | **97.7 %** |
| …where **all** sibling sense texts are byte-identical | **96.9 %** |
| mean senses per polysemous anchor | 2.04 |

So for essentially every polysemous entry, `pomen: pes` and `pomen: pes` were
two separate nodes with the same text. This is specifically bad for GTLM rather
than merely wasteful: edges are untyped and the architecture is
**node-permutation equivariant**, so two same-text sense nodes are distinguished
only by what hangs off them. Once the extractor prunes those subtrees — which
Finding 3 says it must — the nodes become genuinely interchangeable, and the
model is asked to attach different answers to inputs that are identical up to
permutation. That is label noise manufactured by the builder, not a gap in the
source data.

**v3.1 fixes it** (`build_gtlm_graph.py`, flaw 5). A sense node now gets:

- its **dictionary ordinal** when its entry has more than one sense
  (`pomen 2: pes`), which makes siblings distinct unconditionally; and
- a **bounded snippet of its first usage example** when it has no definition
  (`pomen 1: pes (zgled: Sosedov pes je spet lajal ...)`), which makes them
  distinct *meaningfully*.

The second only pays because of the disjointness recorded in caveat 2 —
definition-bearing and example-bearing senses are largely different populations.
Of the indistinguishable sense nodes, **47.7 %** have at least one `zgled`
neighbour (collocation neighbours are useless here: 0.1 %), and an
example-derived signature fully separates the siblings of **92.5 %** of
polysemous anchors. Where only one sibling has an example the pair becomes
distinct but only one of them becomes informative; the ordinal covers the rest.

Both are controlled by `--sense-snippet N` (0 disables, default 60 chars) and
`--no-sense-index`, and both are recorded in `manifest.meta` so a store declares
which text convention built it.

> **Status: built and verified, 2026-08-19** (job 128706, `apl`, **17 min 28 s**,
> **52.7 GB peak RSS**, 16 CPU). The store is `data/stores/kg_graph_v3_1_gams2b`
> (renamed 2026-08-20 from `kg_graph_v3_1`, for the tokenizer that filled its
> `token_len`; see "Tokenizer" below for the current `_gemma3` store), deliberately
> **beside** `data/stores/kg_graph_v3` rather than over it, so the verified v3 store —
> the one whose store-backed rerun reproduces `results_v3.json` at MD5
> `11f4b10c7fbc1e4195795e896a285820` — stays intact and the two can be diffed node
> for node. Builder SHA-256 `ad8d4284…`, matching the file on disk.

**Measured, v3 → v3.1.** Structure is untouched — the fixes are text-only:

| | v3 | **v3.1** |
|---|--:|--:|
| Nodes / directed edges | 36,735,791 / 48,534,031 | **identical** |
| Textless nodes | 279 | **279** |
| Senses numbered | 0 | **8,088,702** (all in polysemous entries) |
| Senses with an example snippet | 0 | **3,662,840** (43 %) |
| Polysemous anchors with duplicate sibling texts | 97.7 % | **0.00 %** |
| Example nodes with an unresolved escape | 3.53 % | **0.00 %** |
| Mean sense tokens | 8.9 | 20.5 |
| Corpus tokens | 822,362,127 | 920,680,698 (**+12.0 %**) |
| Store on disk | 3.78 GB | 4.06 GB |

The +12 % token cost lands almost entirely on `pomen:` nodes, which were 9.1 % of
the budget and are the nodes a lexicographic QA model must actually read. Both
acceptance checks run at the end of `run_save_v3_1.sbatch`, so a future rebuild
re-measures rather than assumes.

> **One bug, caught by that check.** The first attempt (job 128632) reported
> 97.7 % → 3.68 % rather than → 0 %. `np.unique(sense, axis=0)` sorts rows by
> *(lexical unit, sense)*, so column 1 is grouped by entry and **not** globally
> sorted; the `searchsorted` ordinal lookup against it silently missed 819,563
> senses and could mis-number others. It is a wrong answer, not an error, and the
> headline 97.7 % → 3.68 % improvement would have hidden it. `ord_se` is now
> re-sorted (and deduplicated, so a sense reachable from two entries keeps the
> first ordinal) behind an `assert`.

### Two corrections to the v3 write-up, found while measuring this

**Morphology is not on every anchor.** The example
`iztočnica: bežnica (samostalnik, imenovalnik, ednina)` reads as typical; it
describes 9.2 % of anchors. `lexinfo:partOfSpeech` and the form features sit on
`Word` entries only:

```
word anchors = 400,180      with a morphology/POS parenthetical: 99.9%
MWE  anchors = 3,940,417    with a morphology/POS parenthetical:  0.0%
```

MWEs are **91 % of all anchors** and carry no POS or morphology at all.
Together with caveat 5 (the `Component` collapse discards each constituent's
case/number), the graph holds **no grammatical information about multi-word
expressions anywhere**.

**The token budget is not tag-dominated.** Finding 2 calls tags "the single
largest avoidable overhead in v3." Corpus-wide, tags are ~19 % and examples are
the budget:

| kind | nodes | tokens | % | mean |
|---|--:|--:|--:|--:|
| example | 12,119,157 | 520,068,130 | **63.2 %** | 42.9 |
| form | 8,563,521 | 142,886,188 | 17.4 % | 16.7 |
| sense | 8,468,227 | 75,060,920 | 9.1 % | 8.9 |
| anchor | 4,340,597 | 51,256,360 | 6.2 % | 11.8 |
| collocation | 2,981,731 | 29,834,948 | 3.6 % | 10.0 |
| synonym / antonym | 184,709 | 2,250,096 | 0.3 % | 12.2 |
| **total** | 36,735,791 | **822,362,127** | | |

The Finding 2 claim is defensible in the narrow case it was measured (word seed,
hop 1, where examples are absent) but does not hold as stated. Shortening tags
is a real second-order win; Finding 4 is the one that changes what the model can
learn.

**Collocation join text is redundant but worth keeping.** For 94.4 % of
collocation nodes both endpoint lemmas already appear in 1-hop neighbour text —
but only while both neighbours survive extraction, which Finding 3 says they
often will not. At 10.0 tokens mean it is cheap insurance. Keep it.

---

## Finding 5 — literals were never unescaped

v3 took each N-Triples literal verbatim, so **3.48 % of `zgled:` nodes carried a
literal backslash-quote**:

```
zgled: Tudi če jo predelajo, jih imam pravico tožiti,\" pravi.
```

Every quoted passage in the corpus — reported speech, scare quotes, titles — was
affected, and the v3.1 sense snippets inherited it from the example text they
quote. `parse_file` now runs `writtenRep` / `rdf:value` / `skos:definition`
through `unescape_nt()`.

The dump triple-quotes every text literal, so the only escapes that actually
occur are `\"` (1,461 per 400k lines) and `\\` (12) — but `\uXXXX` and the
C-style set are decoded too, and an **unrecognised escape passes through
unchanged** rather than being silently eaten. Order is handled by a single
left-to-right regex pass, so `\\` immediately before `\"` resolves correctly
instead of the two-`replace()` trap. Verified on the subset: 0 of 190,258 example
nodes retain an escape, while genuine quote characters survive
(`... kljub formalnemu "thatcherizmu" odlagala ...`).

Since all text literals are triple-quoted, the single-quote branch of the literal
parser is dead code for these predicates — its terminator scan would truncate at
an embedded `\"`, but nothing reaches it.

---

## Finding 6 — verb morphology never reached the node text at all

> **Status: fixed in v4, built and verified 2026-08-20.** Stores
> `data/stores/kg_graph_v4_gemma3` (job 129328, `apl`, **35 min 17 s**, **67.3 GB
> peak RSS**) and `data/stores/kg_graph_v4_gams2b` (job 129329, `aga`,
> **21 min 16 s**, **63.4 GB**), both `ACCEPTANCE: PASS`. Builder flaw 7;
> acceptance `build/check_v4_text.py`; job script `build/run_save_v4.sbatch`.
> The two v4 stores are byte-identical in every array except `token_len`.
> Text cost of the change, same tokenizer: **917,858,318 vs 913,315,688 tokens,
> +0.50 %**; mean node text 24.985 vs 24.862 Gemma 3 tokens.

Caveat 3 below used to say morphology mapping was "partial" — `aspect`, `vform`,
`definiteness`, `negative`, `clitic` and `animate` parsed but not rendered. That
understated it. Measured on the v3.1 store, the entire verb paradigm of
*popraskati* collapsed to a handful of strings:

```
oblika: popraskam (ednina)        <- 1st person singular PRESENT
oblika: popraskaj (ednina)        <- 2nd person singular IMPERATIVE
oblika: popraskata (dvojina)      <- 2nd person dual  \  byte-identical,
oblika: popraskata (dvojina)      <- 3rd person dual  /  two separate nodes
oblika: popraskat                 <- supine, no label at all
```

Three independent faults, none of them a gap in the source data:

1. **`person` was silently dropped.** `VALUE_SL` listed
   `firstPerson`/`secondPerson`/`thirdPerson`; this KG emits `first`/`second`/
   `third`. `feat_string()` drops values it cannot map, without a warning, so
   person vanished from all **253,497** forms that carry it.
2. **`FEATURE_PROPS` listed `tense` and `mood`.** **Neither predicate exists in
   this KG.** What carries that distinction is `lexinfo:vform` — `present`,
   `imperative`, `participle`, `infinitive`, `supine` — on **452,782** forms,
   and it was not in `FEATURE_PROPS` at all. Preteklik and prihodnjik are
   periphrastic and are not stored in any form.
3. **`aspect` and `clitic` hang off the lexical-unit**, but the feature branch
   only accepted `word-form`/`form-lexical-unit` subjects, so they were dropped
   even though `aspect` was in scope. They are now collected separately
   (`UNIT_PROPS`) and rendered into the anchor parenthetical after the POS.

`definiteness` (**160,524** forms) is mapped in the same pass: without it an
adjective's definite and indefinite forms carry byte-identical labels and differ
only in surface, so nothing downstream can tell which is the citation form.

**Why this is a GTLM problem specifically, not just missing detail.** Same
argument as Finding 4: edges are untyped and the architecture is
node-permutation equivariant, so two same-text sibling forms are genuinely
interchangeable once the extractor prunes their subtrees. The builder was
manufacturing label noise.

**Measured, v3.1 → v4.** Structure is untouched; the fixes are text-only:

| | v3.1 | **v4** |
|---|--:|--:|
| Nodes / directed edges | 36,735,791 / 48,534,031 | **identical** |
| Form nodes byte-identical to a sibling | 86,848 | **418** (−99.5 %) |
| Anchors affected by that | 20,546 | **53** |
| Nodes whose text changed | — | **614,287** (70,873 anchors · 543,414 forms) |
| Forms carrying a `vform` label | 0 | **452,782** |
| Forms carrying a `person` label | 0 | **253,497** |
| Forms carrying a `definiteness` label | 0 | **160,524** |
| Anchors carrying `aspect` | 0 | **18,157** |
| Anchors carrying `clitic` | 0 | **25** |

```
iztočnica: popraskati (glagol)              ->  iztočnica: popraskati (glagol, dovršni, nedoločnik)
oblika: popraskat                           ->  oblika: popraskat (namenilnik)
oblika: popraskal (ednina, moški spol)      ->  oblika: popraskal (deležnik na -l, ednina, moški spol)
oblika: popraskam (ednina)                  ->  oblika: popraskam (sedanjik, 1. oseba, ednina)
oblika: popraskata (dvojina)                ->  oblika: popraskata (sedanjik, 2. oseba, dvojina)
oblika: popraskata (dvojina)                ->  oblika: popraskata (sedanjik, 3. oseba, dvojina)
oblika: popraskaj (ednina)                  ->  oblika: popraskaj (velelnik, 2. oseba, ednina)
iztočnica: medse (zaimek, tožilnik)         ->  iztočnica: medse (zaimek, naslonska oblika, tožilnik)
oblika: navrženi (…, osnovnik)              ->  oblika: navrženi (…, osnovnik, določna oblika)
```

`FEATURE_PROPS` is ordered `vform, person, case, number, gender, degree,
definiteness` so that **every pre-existing label keeps its position** — the diff
is purely additive. `check_v4_text.py` asserts that as a *subsequence* property:
no node's text may shrink, the surface may never change, and the old
parenthetical's items must be a subsequence of the new one's. It fails loudly if
any existing label was reworded, reordered or dropped. Measured on the real
build: **614,287 nodes changed (1.67 %), all of them anchors or forms, with zero
violations.**

> **On the numbers.** The pre-build estimate from the raw RDF was 89,405 → 8,651
> (−90.3 %). The measured store figures are better for two reasons: the store
> groups siblings by an anchor's actual form *neighbours*, which excludes the
> canonical form (merged into the anchor, so it is not a separate node), and the
> estimate was computed before `definiteness` was added — that label alone
> separates most of the adjective forms that were left colliding.

### Reachability: most of these triples are unreachable, and that matters

The corpus-wide census in `QA_DATASET_DESIGN.md` §3 counts `aspect` 1,735,245,
`vform` 2,353,283 and `clitic` 423,585. Those totals are dominated by
**`lexical-unit-part` subjects — MWE components, which the `Component` collapse
discards** (caveat 5). Reachable on word entries:

| predicate | corpus-wide | **reachable on word entries** |
|---|--:|--:|
| `vform` | 2,353,283 | **452,782** |
| `person` | — | **253,497** |
| `definiteness` | — | **160,524** |
| `aspect` | 1,735,245 | **18,157** |
| `clitic` | 423,585 | **25** |

Two consequences. All 18,157 single-word verbs carry an aspect, so verb-aspect
questions are fully supported. But **clitics are effectively dead at 25 units** —
`QA_DATASET_DESIGN.md` §6.3 calls a clitic/negation question type "arguably the
first thing to add back after the MWE family", and that is wrong for single-word
entries.

---

## Finding 8 — collocations *are* verbalised. The link is a naming convention, not a triple.

> **Status: fixed in v5, built and verified 2026-08-20.** Store
> `data/stores/kg_graph_v5_gemma3` (job 129512, `apl`, **21 min 07 s**),
> `ACCEPTANCE: PASS`. Builder flaw 8; acceptance `build/check_v5_text.py`; job
> script `build/run_save_v5.sbatch`.
> **2,981,731 of 2,981,731 collocation nodes verbalised (100.00 %), 0 fell back
> to the pair form.** Structure byte-identical to the v4 store; the only nodes
> whose text changed are the 2,981,731 collocations (8.12 % of the graph).
> Token cost of the change: **917,184,046 vs 917,858,318, −0.07 %** — the phrase
> is *cheaper* than the pair form, because it drops the ` + ` joiner and the two
> words usually tokenize together. Non-collocation tokens are identical.

Every write-up since v2 has recorded some version of "the inflected phrase is
genuinely not stored" (builder flaw 2, `QA_DATASET_DESIGN.md` §6.2,
`QA_TASKS.md` Group F, and the v2 correction above). **It is stored.** The
mistake was not a missed triple — it was assuming the fact had to be reachable by
traversal.

### What is actually true about the collocation node

A full predicate census over all 2,594 files / 42 GB:

| | |
|---|--:|
| predicates ever seen on a `frac:Collocation` subject | `rdfs:member`, `frac:head`, `rdf:type` — **and nothing else** |
| literals on a collocation node | **0** |
| triples with a collocation node in the **object** position | **0** |

So the narrow claim was right and the node really is textless — and because
nothing points *at* one either, no walk from the MWE reaches the collocation and
no walk from the collocation reaches the MWE. They sit in disjoint directions.

The whole KG has only four literal-bearing predicates: `phoneticRep` (22.2 M),
`rdf:value` (14.7 M usage examples), `writtenRep` (13.5 M) and `skos:definition`
(395 K).

### Where the phrase actually lives

On a *separate entry*, linked only by the collocation's own name:

```
<dependent-sense-10249562-lexical-unit-21062> rdf:type    frac:Collocation
<dependent-sense-10249562-lexical-unit-21062> rdfs:member <sense-21062>   → kisov
<dependent-sense-10249562-lexical-unit-21062> rdfs:member <sense-96587>   → voda
                ^^^^^^^^
<lexical-unit-9191303>       ontolex:sense          <sense-10249562>
                                                          ^^^^^^^^
<lexical-unit-9191303>       rdf:type               ontolex:MultiWordExpression
<lexical-unit-9191303>       ontolex:canonicalForm  <form-lexical-unit-9191303>
<form-lexical-unit-9191303>  ontolex:writtenRep     """kisova voda"""@sl
```

A collocation and a multi-word expression are the same fact filed twice: the MWE
is the phrase as a *headword* (it has the spelling), the `frac:Collocation` is
the same phrase as a *relation between two senses* (it has the participants). The
dictionary keeps both views and never cross-references them.

**Why it stayed hidden.** The *other* id in that name, `H`, **is** restated as a
real triple (`frac:head <lexical-unit-21062>`), so the convention looks like pure
redundancy — a name that repeats what the data already says. Having confirmed
that for `H`, assuming the same of `D` is the natural next step. It is the one
half that is load-bearing. Worse, `code_of()` was already parsing `D` out of that
IRI to pack the node id as `(D << 28) | H`: the number was in hand the whole time,
used as an identifier ingredient and never dereferenced.

### Coverage and quality, measured over the raw dump

| | |
|---|--:|
| `ontolex:MultiWordExpression` entries | 3,940,417 |
| …with a `canonicalForm → writtenRep` | **3,940,417 (100 %)** |
| collocation nodes | 4,717,090 |
| …resolving to a surface string | **4,717,090 (100 %), 0 failures** |
| distinct phrases | 3,744,473 |

The phrases are lexicographer-curated, not generated:

| | |
|---|--:|
| phrase differs from **all** its constituent lemmas concatenated | **75.3 %** |
| …one constituent slot inflected | 60.1 % |
| …two or more slots inflected | 15.2 % |
| identical to the lemma concatenation | 23.7 % (legitimately — *enostavno ignorirati*) |

Measured in the built store, against the **two member lemmas the pair form
actually showed**, the figure is higher: **92.45 %** of collocation nodes now
carry text that the v4 pair form could not have produced. (The two numbers differ
because a collocation has exactly two `rdfs:member` senses but its multi-word
entry may have three or more constituents, so the store-level comparison is the
one that says what v4 was losing.) 99.94 % of phrases share a stem with at least
one of the two lemmas they replaced — the check that rules out having resolved
through the wrong entry.

```
odnašati na glavah      ←  odnašati + na + glava
zaslužiti si odmora     ←  zaslužiti + se + odmor
sobe so enoposteljne    ←  soba + biti + enoposteljen
juhe in enolončnice     ←  juha + in + enolončnica
```

Note the last two: prepositions, conjunctions and the copula are constituents in
their own right, and the copula comes out conjugated. The pair form was dropping
inflection, function words **and** word order.

### Independent confirmation

Regenerating the 200 collocation rows of
`data/datasets/reference/Lexical-QA-SLO(in).csv` from this path alone:

| | |
|---|--:|
| reference phrases reproduced **verbatim** | 1,196 / 1,307 — **91.5 %** |
| same collocation, different number/case | 36 — 2.8 % |
| anchor has no collocations in the KG | 31 — 2.4 % |
| genuinely absent | 44 — 3.4 % |
| rows reproduced in full | **155 / 200** |

That file's `id` column is the KG `lexical-unit` id (verified 8/8 on a sample), so
this also settles its provenance: **the phrases came from this export, not from
the DDDS API** that `QA_DATASET_DESIGN.md` §6.2 went looking for. The residual is
concentrated in possessive adjectives from proper names (*Bergerjev, Žalgirisov,
Charlottin, Nicolin, Mirtov*), which have few or no MWE entries.

The other reference file, `Lexical-QA-SLO-test.json`, is **not** KG-derived: its
`raba/collocations` items hit the KG at 52.9 %, `mwe_component_search` at 28.5 %
and `idiom_completion` / `mwe_structure` at ~6.5 %, with several of those hits
incidental. It is generator output and should not be treated as ground truth.

### What v5 does

`--colloc-text phrase` (the default) dereferences `D` at mint time. The skip
condition stays on the members, so node count and order are unchanged and the
store's structure is byte-identical to the v4 store of the same tokenizer;
`--colloc-text pair` restores the v3/v4 text exactly. Asserted by
`check_v5_text.py`.

```
v4                                  v5
kolokacija: kisov + voda        →   kolokacija: kisova voda
kolokacija: Pierre + baron      →   kolokacija: baron Pierra
kolokacija: Shakespeare + citirati → kolokacija: citirati Shakespeara
kolokacija: Shakespeare + napisati → kolokacija: Shakespeare ne napiše
```

This is what unparks **Group F** of the QA dataset: the verbalisations need no
external API, and the phrase is inside the extracted subgraph, so the containment
invariant holds.

---

## Finding 9 — `sestavina` was never the only hub. `sense → kolokacija` is the second.

**Status: measured 2026-08-21 on `kg_graph_v5_gemma3`.** Structure is md5-identical to
v4, so this holds for every store from v3 onward — it is a property of the graph, not of
the v5 text change. Supersedes finding 1's conclusion, finding 3's title, and the
"why the feared hub never forms" subsection.

### What was believed

Findings 1 and 3 concluded that collocations are "nearly free" and that `sestavina` is
the **only** structure that explodes. Two reasons were given: collocations spread over a
lemma's many senses, and a collocation node has degree exactly 2.

### What is true

| claim | verdict |
|---|---|
| a collocation node has degree exactly 2 | **true**, but it measures `kolokacija → sense` — the harmless direction |
| collocations spread across a lemma's senses | **false** — they pile onto sense 1 |
| `sestavina` is the only hub | **false** |

`imeti` has 14,249 collocations across all senses; 14,233 sit on `pomen 1` — **99.9 %**.
`voda`: 3,003 of 3,026, **99.2 %**. Traversed `sense → kolokacija`, the fan-out is
unbounded:

| | value |
|---|--:|
| collocation-bearing senses | 120,871 |
| p50 / p90 / p99 | 4 / 101 / 791 |
| max | 14,233 (`pomen 1: imeti`) |
| senses above 15 | 34,827 (**28.8 %**) |

Top hubs: `imeti`, `biti`, `tako`, `iti`, `leto`, `bolj`, `priti`, `čas` — the same
function words that head the `sestavina` hub list.

### Why the original measurement missed it

Findings 1 and 3 sampled lexical units **uniformly**, where the median anchor has 4
collocations. The hubs are a thin tail under that sampling, so p50 and even p90 look
flat. But D9 defines the QA frequency proxy as `collocation memberships + MWE
memberships` and D10 bands on it — so the dataset samples *the tail on purpose*. The
study's sampling distribution and the dataset's are close to inverted.

Ball cost, 2 hops, D5 already applied, over the top 400 anchors by proxy:

| `sense → kolokacija` cap | p50 | p90 | p99 | max |
|---|--:|--:|--:|--:|
| 0 | 1,844 | 2,826 | 6,388 | 9,046 |
| **15 (D5b)** | **1,986** | **2,964** | **6,516** | **9,192** |
| 100 | 2,806 | 3,789 | 7,189 | 9,995 |
| uncapped | 22,704 | 41,779 | 80,429 | 135,515 |

### Worked example — the two `kisova voda` nodes

Seeded on each of the two nodes carrying that string (2 hops):

| seed | uncapped | D5 only | D5 + D5b |
|---|--:|--:|--:|
| `iztočnica: kisova voda` | 5,073 n / 50,526 t | **121 n / 1,997 t** | 121 n / 1,997 t |
| `kolokacija: kisova voda` | 3,013 n / 26,242 t | **3,013 n / 26,242 t** | **24 n / 231 t** |
| `iztočnica: voda` | 21,118 n / 343,001 t | 3,124 n / 27,842 t | **113 n / 1,654 t** |

The middle row is the point: D5 caps **zero** hubs on a collocation seed, because no
`sestavina` edge is traversed at all. The path is
`kolokacija → pomen 1: voda → its 3,003 other collocations`. Row 1 needs no D5b — once
D5 has capped the MWE fan-out, no collocation-heavy sense is reached.

### Consequence

D5b (per anchor, K = 15) is added to `QA_DATASET_DESIGN.md` §4, measured in §3.1b, and
T17/T18 draw their gold from the sampled set, so a gold phrase can never fall outside its
own ball.

**Which 15, though, is not a ranking problem.** Three deterministic keys were tried and
all three are biased: partner proxy ascending lands in the hapax tail
(`akratotermalna voda`), descending walks into `imeti nad vodo`, and the best-looking
variant — cut at a threshold, then sort descending — turns out to return whatever sits
just under the threshold, which had been picked by eye. Any fixed key also re-selects the
same slice of the distribution on every anchor.

D5b therefore **samples** rather than ranks: K = 15 drawn without replacement with
`w ∝ log(1 + proxy(partner))`, seeded from the anchor's node code. The partner-proxy
distribution is a power law (log-log slope **−1.418**, R² = **0.933**), which is what
justifies the log damping — linear weights put a third of the list on function words.
Sampling also makes T17/T18's gold in-ball by construction, since the gold is drawn from
the sampled set. Measurements in `QA_DATASET_DESIGN.md` §3.1c.

---

## Finding 10 — noun gender was never in the store, and the recorded one-line fix could not have put it there

> **Status: fixed in v6, built and verified 2026-08-21.** Store
> `data/stores/kg_graph_v6_gemma3` (job 129977, `apl`, **17 min 35 s**),
> `ACCEPTANCE: PASS`. Builder flaw 9, recorded as **M4** in `QA_TASKS.md` §0.7;
> acceptance `build/check_v6_text.py`; job script `build/run_save_v6.sbatch`.
> Text-only change: `node_codes` / `ntype` / `kind` / `mwe_set` / `indptr` are
> **md5-identical** to v5, `indices` identical as a per-row multiset (0.46 % of
> rows reordered by `imap_unordered`). **310,362 of 36,735,791 nodes changed
> text (0.84 %), every one of them an anchor**; noun-anchor gender coverage
> **100.000 %**; sampled noun form leaves carrying a gender: **0**.
>
> *Provenance note:* `manifest.meta.builder_sha256` is `f790dbf4…`, the builder as
> it stood at save time. Two edits landed after the job exited — the narrowed
> `FEATURE_PROPS` ordering comment, and `gender(form)=` / `gender(entry)=` in the
> `[feat]` diagnostic — so the file now hashes differently. Both are a comment and
> a `print`; neither touches node text, and a rebuild would produce a
> byte-identical store. The store was **not** rebuilt to resynchronise the hash.

Every noun in this KG has a gender, and no store through v5 contained a single one
of them. `iztočnica: miza (samostalnik, imenovalnik, ednina)` never said that *miza*
is feminine.

### Why it hid for three versions

The predicate looked rendered, because it *was* rendered — for the wrong half of the
data. This KG carries `lexinfo:gender` at **two levels, on disjoint parts of speech**:

| subject | what the gender means | triples | rendered |
|---|---|--:|---|
| `word-form` | agreement feature of an adjective or *-l* participle | 3,242,639 | since v3 |
| `lexical-unit` | **inherent gender of a noun** | **310,362** | **only since v6** |

So `oblika: popraskal (deležnik na -l, ednina, moški spol)` was in the store and
looked like proof the feature worked. Caveat 3 of this document said in so many
words that v4 "maps … `gender` … on forms", which is true and was exactly the
sentence that made the gap invisible. Finding 6's flaw 7c had already fixed this
*class* of bug for `aspect` and `clitic` — entry-level properties dropped by a
form-level subject guard — and left `gender` behind, because `gender` was the one
member of the class that also legitimately existed at the form level.

### The recorded fix does not work, and the reason generalises

`QA_TASKS.md` §0.7 proposed, and this README implied, a one-line change:

```python
UNIT_PROPS = ("aspect", "gender", "clitic")
```

**That changes nothing.** `gender` was already in `FEATURE_PROPS`, and the parse
branch tested the two sets in order:

```python
if local in FEATURE_PROPS:                              # gender matches HERE
    if (sc >> TYPE_SHIFT) in (T_WORDFORM, T_FORMLU):    # a lexical-unit fails this
        feat.append(...)                                # ... so it is dropped
elif local in UNIT_PROPS:                               # unreachable for gender, forever
    ...
```

Every noun-entry gender was routed into the **form** branch, discarded by the
subject-type guard, and the `elif` could never run — adding the name to the second
set leaves that path untouched. The lesson is not about gender: **an `if set-A /
elif set-B` dispatch silently assumes a property belongs to exactly one set**, and
breaks the moment one legitimately belongs to both. v6 dispatches on the **subject
type** and consults both sets.

### The gender goes on the anchor, and the forms inherit it

Rendered on the **anchor only** —

```
iztočnica: miza (samostalnik, imenovalnik, ednina)
iztočnica: miza (samostalnik, ženski spol, imenovalnik, ednina)
```

— with the entry's form leaves left untouched. Two reasons, both measured:

1. **It is an entry property, not a form property.** The source models it that way;
   a noun's word-forms carry only `case` and `number`. Repeating it down the
   paradigm would assert something the export does not.
2. **Every form leaf is adjacent to its anchor**, so no extracted ball can contain a
   noun form without the node carrying its gender. Measured on the v5 store
   (`analysis/check_gender_reachability.py`): **200,000 of 200,000** sampled form
   leaves have an anchor at hop 1. Repeating the label would have written it onto
   **5,176,265** noun form leaves instead of **310,362** anchors — **×16.7**, on the
   same argument as Finding 2: node text is what the LM reads, and the cheapest
   correct encoding wins.

`check_v6_text.py` asserts the inheritance rather than trusting it: noun form leaves
must contain **no** gender label, and **only anchors** may differ from v5. This is
the narrowing M4 called for — v4's "nominal strings are byte-identical to v3.1"
assertion moves onto nominal *form* nodes, which is where it still holds.

### Verified against the raw dump, not the store

`analysis/scan_gender.py` over all 401 `*-words.nt`:

| | |
|---|--:|
| noun lexical units | 310,362 |
| carrying a gender | **310,362 (100.00 %)** |
| carrying **more than one** | **0** |
| any other POS carrying an entry gender | **0** |
| distribution | masculine 223,357 · feminine 75,460 · neuter 11,545 |

Every noun has exactly one, nothing else has any — so the property's two roles never
collide on one node and both can be rendered unambiguously. The build confirms it at
scale: `unit=328,544` parsed entry-level properties is exactly
310,362 gender + 18,157 aspect + 25 clitic.

### The text blob delta is exact, which settles the blast radius without sampling

`text_blob.bin` grew **2,729,398,242 → 2,733,519,953 bytes, +4,121,711**. That figure
is not approximately the expected cost — it *is* the expected cost, to the byte,
computed from the raw-dump gender counts and the UTF-8 length of each inserted label
(including its `, ` separator):

| label | anchors | bytes each | total |
|---|--:|--:|--:|
| `moški spol, ` | 223,357 | 13 | 2,903,641 |
| `ženski spol, ` | 75,460 | 14 | 1,056,440 |
| `srednji spol, ` | 11,545 | 14 | 161,630 |
| | | | **4,121,711** |

Exact agreement over a 2.7 GB blob is a stronger statement than any sampled check:
**exactly** 310,362 nodes changed, each gained **exactly one** gender label, the
store's gender distribution matches the export's, and nothing else in the entire
text blob moved by a single byte. A stray label on a form leaf, a double-rendered
anchor, or a re-ordered parenthetical would all have shown up here as a mismatch.

`token_len` says the same thing independently. Over all 36,735,791 nodes:

| | v5 | **v6** |
|---|--:|--:|
| Total Gemma 3 tokens | 917,184,046 | **918,425,494** (**+0.135 %**) |
| Mean tokens per node | 24.967 | **25.001** |
| Nodes whose token count changed | — | **310,362** |
| Token delta on those nodes | — | **+4, on every one of them** |

The delta histogram is the single entry `{+4: 310,362}` — all three genders cost
four tokens, so the change is 4 tokens × the noun-anchor count and nothing else.
Compare Finding 6's v4 morphology pass at +0.50 %: this is a third of that, for a
feature that unblocks a whole question type.

### The acceptance check's own trap: 68 nouns that are not nouns

The first run of `check_v6_text.py` reported **310,430** noun anchors and
**99.978 %** coverage — 68 anchors apparently tagged `samostalnik` with no gender.
The store was complete; the *check* was wrong. It selected noun anchors with
`"samostalnik" in text`, which matches the **lemma**, and the export contains 68
multi-word grammatical-terminology entries whose lemma is literally about nouns:

```
iztočnica: abstraktni samostalnik
iztočnica: spol samostalnika
iztočnica: moški samostalniki
iztočnica: opis z dvema samostalnikoma namesto s pridevnikom in samostalnikom
```

These are MWEs. No multi file carries `partOfSpeech` at all, so they have no
parenthetical, no POS tag, and nothing to hold a gender. The fix is to test
membership in the parsed parenthetical rather than the whole string, after which
coverage is **310,362 / 310,362 = 100.000 %** and `COVERAGE_MIN` is set to 1.0 —
the honest threshold, since anything looser lets a future regression hide in the
slack. Worth remembering for any check keyed on a Slovenian POS tag: **the tag
vocabulary and the lemma vocabulary overlap**, and a dictionary of Slovenian
contains the words for its own parts of speech.

**What it unblocks.** T9 (`spol_samostalnika`) was ungeneratable from any store and
T8's noun row was POS-only; both run against a v6 store.

---

## Finding 11 — the store kept one collocation per *member set*, and threw away a fifth of the phrases

**Status: found and fixed 2026-08-21; the v7 store.** This is the first version bump
since v3 that changes **structure** rather than node text, and it is a bug fix, not a
feature. It is also a fix to a defect that **two earlier findings each half-saw and
neither caught**, which is the part worth reading.

### The defect

Each collocation is reified once per participant, so the dump holds several
`frac:Collocation` IRIs over one `{sense_a, sense_b}` member set. Finding 2 (v2) saw
that and deduplicated on the member set. That was **correct at the time**: the nodes
were textless, the builder rendered them as `kolokacija: boj + kriminaliteta`, and
every duplicate of one member set produced a byte-identical string. Keeping one was
keeping all the information there was.

Finding 8 (v5) then gave the nodes their curated phrase, by dereferencing the sense
id embedded in the IRI. It did not revisit the key — and the duplicates do **not**
name the same dependent sense. They name different ones, which is to say different
curated phrases. The dedup that had been lossless became lossy in the same commit
that made the nodes worth having.

`_dedup_pairs_keyed`'s docstring asserted the opposite outright:

> the duplicates differ only in the `frac:head` half of the IRI, so they all carry
> the same sense id in the other half. Which representative survives therefore does
> not affect the verbalisation

Measured against the v6 store, that is false:

| | |
|---|--:|
| binary collocation IRIs | 4,707,758 |
| distinct member sets | 2,981,731 |
| member sets naming **more than one** dependent sense | **439,370 (15.3 %)** |
| curated phrases dropped | **717,545 (20.0 %)** |

And the loss was **systematic, not random**: the surviving representative was the
lowest IRI code, i.e. the earliest-entered entry, so the store consistently preferred
one end of the editing history. `iziti` + `zbirka` has 8 curated phrases — *zbirka
izide*, *iziti v zbirki*, … — and the store kept 1.

### The fix

Key the dedup on **(member set, phrase)** — one node per distinct node *text*, which
is the only distinction the model can make anyway. Two details:

- **Fold before keying.** The phrase is normalised (case, whitespace, one trailing
  period) before it becomes a key, because the QA grader matches collocations
  case-insensitively; a member set carrying both `Cvileče gume` and `cvileče gume`
  would produce a gold list containing the same item twice and failing its own
  dedup invariant. Keying on the raw surface gives 3,744,612 pairings against
  3,744,066 — 546 nodes, small but exactly the ones that break a downstream
  contract. Which *surface* survives a fold is decided the same way `build()`
  decides between several `writtenRep` values for one form: fewest capitals, then
  alphabetical.
- **Phrase mode only.** Under `--colloc-text pair` every duplicate still renders
  identically, so keying on the text is precisely the old behaviour and a pair store
  stays byte-identical to v4. The old key was a defect in phrase mode and an option
  in neither, so **v6 is no longer reachable from this builder** in phrase mode.

### Measured effect

| | v6 | v7 |
|---|--:|--:|
| collocation nodes | 2,981,731 | **3,744,066** (+25.6 %) |
| …verbalised, not a lemma-pair fallback | 100.00 % | **100.00 %** (0 fallbacks) |
| store nodes | 36,735,791 | **37,498,126** (+2.1 %) |
| directed edges | 48,534,031 | **50,058,701** (+3.1 %) |
| synonym / antonym nodes | 181,260 / 3,449 | unchanged, same ids |
| IRI-backed nodes | 33,569,351 | unchanged |

Note that the v6 collocation-node count **equals** the member-set count exactly. That
identity is the defect stated as an arithmetic fact: the store held one node per
pairing rather than one per phrase.

**The pre-build simulation under-predicted this.** It estimated 3,569,711 nodes
(+19.7 %) against the 3,744,066 (+25.6 %) the rebuild produced. The cause is in the
simulation, not the build: it read each phrase out of the *store's* MWE anchor text
with a regex that strips a trailing parenthetical, so phrases that legitimately end
in one collapsed together. The builder reads `canonicalForm → writtenRep` directly
and keeps them apart. The simulated figure is recorded here because it was quoted in
`QA_TASKS.md` and `QA_DATASET_DESIGN.md` before the rebuild; **the measured figure is
the one to cite.**

### What it unblocks

`ALL(anchor)` — the set the QA grader accepts a collocation answer against
(`QA_TASKS.md` §0.8.3) — was missing a fifth of its members, and missing them
non-uniformly. Every T17 item graded against a v6 store would have scored a correct,
in-graph phrase as an invention whenever the model read a node the dedup had
dropped. That is not a small inaccuracy in a metric; it is the metric measuring the
wrong thing.

---

## Finding 12 — ~9 % of noun paradigms are lemma-filled, and one QA type turns that into a *plausible* wrong answer

**Status: found 2026-08-22 by reading generated items; filtered, not fixable.** Unlike
Findings 5–11 this is **not** a builder defect. The store renders faithfully what the
export contains; the export contains paradigm cells that were never inflected.

### The defect

Generated T20 (`primeri_uporabe/analiza_oblike_v_povedi`) asks what form a word takes in
a real corpus sentence. One item read:

```
Q: … beseda odstotkov v povedi … torej 40 odstotkov.
A: ODGOVOR: rodilnik dvojine          <- wrong: genitive PLURAL
```

The generator was not guessing. It labels a form only when that surface occupies exactly
one cell of the paradigm, so the label is correct by construction — that is the whole of
design decision D15. The paradigm it consulted:

```
rodilnik     dvojina   odstotkov     <- right
rodilnik     množina   odstotek      <- WRONG, should be `odstotkov`
imenovalnik  množina   odstotek      <- WRONG, should be `odstotki`
```

`odstotkov` looked unambiguous **only because the cell it should have shared was filled
with the lemma**. The ambiguity test worked exactly as specified; its premise — that the
paradigm is complete — was false and had never been written down.

This is the same class as T11's degenerate gradation, where the KG stores `oblika:
mikaven (imenovalnik, ednina, moški spol, primernik, določna oblika)` — a comparative
identical to the positive. The cell exists in the RDF, the surface was never inflected.

### How much of it there is

Measured over the **49,078** noun entries carrying ≥ 12 filled case×number cells:

| test | entries | share |
|---|---|---|
| nominative plural == nominative singular | 4,406 | **8.98 %** |
| lemma surface in all 18 cells | 3,335 | 6.8 % |

The population is mostly foreign proper nouns (*Baudelaire*, *Apollinaire*, *SMS*) plus
ordinary Slovene words whose plural column was simply never filled (*pilot*). Some
lemmas have **both** a healthy and a defective entry — *Moliere* and *Gilmore* appear on
both sides — so this must be decided per **entry**, never per lemma.

### The filter

`qa/gen.py:healthy_grid` requires the nominative plural to differ from the nominative
singular *and* the filled cells to hold ≥ 6 distinct surfaces (a real Slovene noun has
eight to ten; an uninflected one has one, repeated). It is applied **inside**
`nominal_grid`, not at the five call sites, so no type can forget it.

Recorded as check C23 in `QA_TASKS.md` and as an amendment to D15 in
`QA_DATASET_DESIGN.md`. Cost ~9 % of the noun pool for T1, T2, T3, T4, T20 and T21.

### Why it was worth finding

The other five affected types would have failed loudly: T1 renders an eighteen-cell
table of one repeated surface, T21 an eighteen-way disjunction — obviously broken to
anyone reading one item. **T20 is the only type where the defect is invisible in the
output**, because *"rodilnik dvojine"* is a well-formed, plausible answer to a
well-formed question about a real sentence. It was caught by reading items, not by any
assertion, which is the argument for the human half of verification existing at all.

---

## Practical implications

- **Collocations can stay in, but not unconditionally** *(revised 2026-08-21 —
  finding 9)*. They cost ~0 % at the median and ≤21 % in the worst percentile
  measured **here**, and they carry real lexicographic signal. But this sizing study
  seeded uniformly over lexical units, where the median anchor has 4 collocations.
  The QA dataset samples by a frequency proxy that *is* collocation count (D9/D10),
  so it draws precisely the anchors this study under-weights: uncapped, the top 400
  anchors by proxy cost a p50 of 22,704 tokens and a max of 135,515. Collocations
  stay in **with the D5b per-anchor cap applied**.
- **Raw k-hop around an MWE seed is unusable beyond hop 1**; around a
  single-word seed it is fine to hop 2 and usually hop 3.
- **There are two levers, not one** *(revised 2026-08-21 — finding 9)*. Any real
  extractor must cap `sestavina` constituent degree (D5) **and** the
  `sense → kolokacija` fan-out (D5b) before going past the danger hop. The earlier
  claim that `sestavina` was the only structure needing taming was wrong; nothing
  *else* does.
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
sbatch data/analysis/run_build_v3.sbatch        # -> data/analysis/results/results_v3.json
sbatch data/build/run_save_v3.sbatch            # -> data/stores/kg_graph_v3/
sbatch data/build/run_save_v6.sbatch gemma3     # -> data/stores/kg_graph_v6_gemma3/  (current)
sbatch data/build/run_save_v6.sbatch gams2b     # -> data/stores/kg_graph_v6_gams2b/
sbatch data/build/run_save_v5.sbatch gemma3     # -> data/stores/kg_graph_v5_gemma3/  (v6's acceptance reference)
sbatch data/build/run_save_v5.sbatch gams2b     # -> data/stores/kg_graph_v5_gams2b/
sbatch data/build/run_save_v4.sbatch gemma3     # -> data/stores/kg_graph_v4_gemma3/  (v5's acceptance reference)
sbatch data/build/run_save_v4.sbatch gams2b     # -> data/stores/kg_graph_v4_gams2b/
```

Each `run_save_vN` acceptance-checks the store it builds against the **same
tokenizer's** v(N−1) store, so build the reference first or the job stops before
the rebuild.

- Pipeline: stream-parse → collapse MWE parts → merge lemma → dedup collocations
  → reify syn/ant/colloc → Slovenian self-describing text → undirected CSR BFS →
  tokenize (`--tokenizer`, default `cjvt/GaMS3-12B-Instruct`) → 2×2×2 variants ×
  word/MWE split.
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
  `lib/pick_python.sh`, which prefers the venv launcher, falls back to a
  real 3.10 interpreter plus the venv's `site-packages` on `PYTHONPATH`, and exits
  with a diagnostic naming the node rather than a bare `ModuleNotFoundError`.
  `run_build_v3.sbatch` does neither and will fail in seconds if it lands on a
  24.04 node — it only ever succeeded because job 126763 happened to get `apl`.
- Tokenizer loaded offline from `HF_HOME=/shared/workspace/povejmo/huggingface_cache`.
- Raw output `analysis/results/results_v3.json`; job log `logs/build_v3_*.out`.

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
object. `--save-graph DIR` writes it once; `lib/graph_store.py` reads it
back in seconds under a few GB.

```bash
sbatch data/build/run_save_v3.sbatch    # build once -> data/stores/kg_graph_v3/
python data/lib/graph_store.py data/stores/kg_graph_v3 --verify
python data/lib/graph_store.py data/stores/kg_graph_v3 --samples 3
```

```python
import graph_store
G = graph_store.load_graph("data/stores/kg_graph_v3")
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

### Tokenizer, and what the store directories are named

`token_len` is the one array whose values depend on a model choice, so the store
directory is named for the tokenizer that filled it:

| directory | tokenizer | text convention | status |
|---|---|---|---|
| `data/stores/kg_graph_v7_gemma3` | `cjvt/GaMS3-12B-Instruct` (vocab 262,145) | v7 | **current** — what `lookup`, the QA generator and downstream work read |
| `data/stores/kg_graph_v6_gemma3` | `cjvt/GaMS3-12B-Instruct` | v6 | superseded by v7 — one collocation node per member set instead of per phrase (Finding 11); kept as v7's acceptance reference |
| `data/stores/kg_graph_v5_gemma3` | `cjvt/GaMS3-12B-Instruct` | v5 | superseded by v6 — no noun gender on anchors (Finding 10) |
| `data/stores/kg_graph_v4_gemma3` | `cjvt/GaMS3-12B-Instruct` | v4 | superseded by v5 — collocations as lemma pairs (Finding 8); kept as v5's acceptance reference |
| `data/stores/kg_graph_v4_gams2b` | `cjvt/GaMS-2B` (vocab 256,000) | v4 | for tokenizer diffing |
| `data/stores/kg_graph_v3_1_gemma3` | `cjvt/GaMS3-12B-Instruct` | v3.1 | superseded by v4 — no morphology on verb forms |
| `data/stores/kg_graph_v3_1_gams2b` | `cjvt/GaMS-2B` | v3.1 | superseded; kept for diffing |
| `data/stores/kg_graph_v3` | `cjvt/GaMS-2B` | v3 | pre-v3.1 sense conventions |

`manifest.meta.colloc_text` records `phrase` (v5–v7) or `pair` (v3/v4) — the builder
can still produce either via `--colloc-text`. Through v6 the two differed in node
**text** only; from **v7** they also differ in the dedup key, and therefore in
structure (Finding 11), because keying on the member set was a defect in phrase mode
and an option in neither.
`manifest.meta.text_convention` is derived rather than hard-coded: a store claims
**v7** only when `gender` is in `UNIT_PROPS` *and* collocations are verbalised, so a
store built with either switched back off cannot misreport itself as current. v6 is
no longer reachable from the builder in phrase mode, deliberately.
`manifest.meta.unit_props` records the entry-level properties in full.

`manifest.meta.text_convention` records which convention built a store, alongside
`feature_props` and `unit_props`, so a store declares what its node text contains
rather than leaving it to be inferred from the directory name.

The two v3.1 stores are **byte-identical in every array except `token_len`** —
`node_codes`, `ntype`, `kind`, `mwe_set`, `indptr`, `text_off` and the 2.7 GB
`text_blob` all match md5 for md5, and the rebuild asserts exactly that
(`run_save_v3_1_gemma3.sbatch`), so a difference anywhere else means the build was
not reproducible.

**`indices.npy` is the one exception, and it is benign.** The builder merges
parsed files with `pool.imap_unordered`, so files arrive in completion order and
the neighbours of a node that several files touch get appended in a run-dependent
order. Measured across the two stores: **92,945 of 36,735,791 rows (0.25 %)** are
reordered — 89,957 of them example nodes, 2,972 forms, 16 senses — with an
identical neighbour **set** in every one, verified by comparing the row-sorted
edge multiset. Node ids are untouched because codes come from the IRI, not from
insertion order; that is what keeps two independently built stores comparable node
for node. If byte-reproducible CSR is ever wanted, sort each row before saving —
nothing currently depends on neighbour order.

**Every Gemma 3 checkpoint shares one tokenizer, so one store serves all of them.**
Verified 2026-08-20 across `gemma-3-270m-it`, `-1b-it`, `-4b-it`, `-12b-it`,
`-27b-it`, `-4b-pt` and `cjvt/GaMS3-12B-Instruct`: `tokenizer.model` is
byte-identical in all seven, and encoding 5,000 real node texts gives identical
token **ids**, not merely equal counts. (`tokenizer.json` has two md5s — the
270m-it/pt variant differs in per-token flags on `added_tokens`, with identical
vocab, merges and id→content, which changes no output.) The iteration ladder in
the root README — develop on a small Gemma 3, scale to GaMS3-12B — therefore
needs no rebuild between rungs.

Switching tokenizers is worth **−0.80 %**, now measured over the whole graph
rather than a sample: **913,315,688** tokens under Gemma 3 against **920,680,698**
under GaMS-2B, mean 24.862 vs 25.062 per node. Only **45.97 %** of nodes get the
same count individually — the differences are real, they just very nearly cancel
in aggregate. So the GaMS-2B token figures elsewhere in this document are within
~1 % of the current store and have not been recomputed; treat the label as
approximate rather than the numbers as wrong.

Use `--tokenizer REPO` to build against something else; it is recorded in
`manifest.meta.tokenizer`, which is the authority on what a store contains.

### Looking things up: `lookup`

`lookup/kg_lookup.py` queries either the built store or the raw N-Triples
from one CLI. `bin/lookup` wraps it so it runs from any directory; put the
directory on `PATH` once:

```bash
printf '\nexport PATH="/shared/workspace/povejmo/gams_gtlm/bin:$PATH"\n' >> ~/.bashrc && source ~/.bashrc
```

The leading `\n` is not cosmetic. A dotfile whose last line has no trailing
newline turns a plain `echo … >>` into a silent corruption: the export is glued
onto the previous line, so it never runs *and* it mangles whatever was there.

```bash
lookup store word jabolke              # spelling -> headwords and forms, ranked
lookup store word boj --also all       # ...and senses, examples, collocations
lookup store id 69611 --hops 1         # a unit's text, then its neighbourhood
lookup raw id 34748                    # every raw triple touching it (~8s)
lookup raw iri word-form-1567346       # the same for any IRI
```

Store lookups read `data/stores/kg_graph_v6_gemma3`; `KG_STORE=/path` points them elsewhere.
The wrapper picks an interpreter that can actually import numpy **on the current
node**, falling back to `pick_python.sh`, so it does not hit the 3.10-only trap
that killed job 128295. `KG_PYTHON=/path/to/python` overrides that choice.
`lookup -h`, and `-h` on any subcommand, carries the details.

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
   example-bearing senses remain largely **disjoint populations** — which is
   what makes the v3.1 example-snippet fallback work; see **Finding 4**. The
   thinness itself is unfixed and unfixable from this source.
3. ~~**Morphology mapping is partial.**~~ **Fixed in v4, completed in v6** — see
   Finding 6, which also shows the caveat understated the problem: `person` was
   listed but silently dropped, and `tense`/`mood` were listed but do not exist
   as predicates in this KG. v4 maps `vform`, `person`, `case`, `number`,
   `gender`, `degree`, `definiteness` on forms and `aspect`, `clitic` on
   anchors. **That "`gender` on forms" hid the last gap for two versions**: this
   KG stores `gender` at *both* levels, and the entry-level half — every noun —
   reached no store until **v6** (Finding 10). v6 maps `gender` on anchors too.
   Still unrendered: `negative` (121,229, of which only 6,630 are `yes`) and
   `animate`.
4. **The writtenRep tie-break is a heuristic.** "Fewest capitals, then
   lexicographic" is right for `BOJ/Boj/boj` and harmless for `Beznik`, but a
   proper noun that also lists a lowercase variant would be lower-cased.
5. **Ordered MWE components are available but unused.** `rdf:_1 … rdf:_4` give
   constituent order; v3 ignores it, so `sestavina` edges are an unordered set.
   Less costly since v5: the collocation node now carries the phrase in its real
   word order, so ordering is no longer only recoverable from these ordinals.
6. Token counts use `add_special_tokens=False` per node; GTLM's own separators
   are not modelled, and the prompt allowance is a flat 24 tokens.
7. ~~Example text is not unescaped from N-Triples.~~ **Fixed in v3.1** — see
   Finding 5.

---

## Appendix — superseded v2 snapshot (2026-07-22)

Retained for provenance only. v2 built a 37.0 M-node / 47.6 M-edge graph and
reported MWE hop-2 medians of 42,226 original / 85,284 Levi nodes / 385,061
tokens, and word hop-1 of 19 / 37 / 262. Those figures assume blanket Levi
reification, omit collocations entirely, silently merge ≥1.3 M colliding
collocation IRIs, and count 9.6 % of nodes as contributing zero tokens. Builder
`analysis/build_gtlm_graph.py`, results `analysis/results/results_v2.json`, still
in the tree. The v1 raw-projection study (`analyze_neighborhoods.py`,
`results.json`, `results_semcore.json`) was judged heavily flawed and is not
documented here.

> **Note on disk.** `kg_raw/` currently keeps both the 41 GB `share_download.zip`
> and the extracted `OntoLex DSB/` (42 GB), plus a redundant nested
> `OntoLex DSB/examples-data.zip` (1 GB). The two zips can be deleted to reclaim
> ~42 GB once the extracted files are confirmed good.
