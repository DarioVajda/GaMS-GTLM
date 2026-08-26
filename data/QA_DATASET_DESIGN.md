# The lexicographical QA dataset — design

Why this dataset exists, what the knowledge graph can and cannot be asked, and the
rules the generator follows. Per-type answer contracts are in
[`QA_TASKS.md`](QA_TASKS.md); how to build the thing is in [`README.md`](README.md).

| | |
|---|---|
| **Target** | a public lexicographical Q&A service, general-public users, single-turn |
| **Size** | 12,490 items — train 9,266 / dev 1,040 / test 2,184 |
| **Types** | 19, over 279 question frames |
| **Seeds** | core lemmas passing a content filter, sampled across 7 fixed frequency bands |
| **Entity linking** | no delimiters — external extractor, plain-text lookup, **union ball** when several units match |
| **Extraction** | uniform hop 2 from every matched unit, with both hubs capped at K = 10 |
| **Answers** | one gradeable line, `ODGOVOR: …`; no MSD tags |
| **Split** | lemma-disjoint, with three held-out generalisation tiers |
| **Baselines** | no-retrieval and serialised-graph — the two gaps isolate retrieval and structure |

## 1. What we are building

**The deliverable is a service**, hosted on the lab website, where members of the public
ask about lexicographical properties of Slovene words. Not a paper artefact. Three
things follow directly:

* **Arbitrary phrasing, no input conventions.** We cannot ask general-public users to
  mark which word they are asking about — hence D3.
* **Serving a 12 B model is acceptable**, and no latency or concurrency target was set,
  so there is no throughput constraint to design around.
* **There is no external number to beat.** A parallel method by colleagues is a
  comparison point, not a threshold, so we define our own baselines (D19).

### 1.2 Why we generate our own items

The alternative was to label an existing set of synthetic questions against the graph.
Three findings make generation clearly better, and since the reference items were
synthetic to begin with, nothing is lost:

1. Roughly **half of the reference question types are unanswerable from this KG at
   all** (§3) — no register, dialect, frequency or error metadata, no meronymy,
   effectively no hypernymy, translations only into Hungarian.
2. The newer reference file's entity ids are **42 % dangling** against this dump.
3. Several of its subtypes are **defective at a 99–100 % rate** — gradation is
   degenerate in 99 of 100 rows, form analysis answers with a raw MSD tag in 100 of
   100, and Slovene number agreement is wrong in 100 of 100.

Generating also gives exact control over answer format, over which facts are guaranteed
present in the extracted subgraph, and over the split.

## 2. The reference files

`datasets/reference/` holds two QA files. **Neither supplies data.** They supply the
question-type inventory and phrasing style, and nothing else.

`Lexical-QA-SLO(in).csv` is KG-derived — its `id` column is the KG `lexical-unit` id,
and its collocation rows reproduce from our own store at 91.5 %, which settles its
provenance. `Lexical-QA-SLO-test.json` is generator output whose items hit the KG at
6–53 % depending on subtype, and is not ground truth for anything.

## 3. What the KG can and cannot answer

Full-corpus predicate census. **Not present anywhere:** register, style, dialect,
domain, frequency, typo metadata, meronym, holonym, valency frames, derivational
morphology.

| signal | count | supports |
|---|--:|---|
| `ontolex:writtenRep` | 13,499,411 | surface forms, the reverse index |
| `lexicog:usageExample` | 14,722,502 | usage examples — **but see (a)** |
| `decomp:constituent` (+ `lexinfo:category` on all) | 10,265,230 | MWE structure, component search |
| `rdf:_1 … _22` | — | constituent order |
| `ontolex:sense` | 8,545,797 | sense enumeration, counting |
| `lexinfo:degree` | 4,542,306 (comp 107,717 / super 72,856) | gradation |
| `lexinfo:vform` | 2,353,283 (supine only 20,878) | non-finite forms |
| `lexinfo:aspect` | 1,735,245 | verb aspect |
| `lexinfo:partOfSpeech` | 400,180 | POS — **single-word entries only** |
| `lexinfo:synonym` | 362,520 | synonyms |
| `skos:definition` (`@sl`) | **225,618** | definitions |
| `vartrans:*` | 77,570, all `directEquivalent`, all **`@hun`** | Hungarian only |
| `lexinfo:antonym` | **6,898** | antonyms |
| `lexinfo:hypernym` / `hyponym` | **5,415** each | unusable (0.12 %) |

**(a) Usage examples are overwhelmingly attached to MWE senses.** Of 14.7 M examples,
**99.5 % hang off MWE senses** and only **71,729** are reachable from a single-word
entry — spread over just **11,048 words**. Word-level example questions therefore need
no MWE traversal, but the eligible pool is small, and it is the binding constraint on
T19 and T20 in the low bands.

**(b) The KG has two populations**, differing 3–4× in content richness:

| | entries | sl definition | collocations | synonyms | usage example | **any** |
|---|--:|--:|--:|--:|--:|--:|
| **core** (id < 1 M) | 100,801 | 40.6 % | 65.0 % | 35.4 % | 10.2 % | **72.0 %** |
| **tail** (id > 1 M) | 299,379 | 7.3 % | 12.2 % | 4.4 % | 0.3 % | **20.4 %** |

The core is a curated lexicon; the tail a thin auto-generated extension. Seeding from
the core (D8) is what keeps *garsid* and *misailović* out without needing frequency
data.

### 3.1 The `sestavina` hub is directional

| direction | n | p50 | p90 | p99 | max |
|---|--:|--:|--:|--:|--:|
| **MWE → its constituent words** | 3,940,417 | 3 | 3 | 4 | **22** |
| **word → MWEs containing it** | 114,033 | 4 | 136 | 1,207 | **423,510** |

Downward traversal is bounded at 22 across the entire graph and needs no cap. Upward
traversal explodes, and its hubs are function words — *biti* (423,510), *in*, *kot*,
*se*, *ne*, *imeti*, and the prepositions. Only 114,033 of 400,180 words are
constituents of any MWE at all, which is why a median word seed stays flat at ~19 nodes
across three hops.

### 3.1b `sense → kolokacija` is a second hub

Collocations do **not** spread across a lemma's senses — they pile onto sense 1.
`imeti` has 14,249 collocations and 14,233 of them (99.9 %) hang off `pomen 1`; `voda`,
3,003 of 3,026. A collocation node itself has degree exactly 2, which is the harmless
direction; traversed `sense → kolokacija` the fan-out is unbounded:

| | value |
|---|--:|
| collocation-bearing senses | 120,871 |
| p50 / p90 / p99 | 4 / 101 / 791 |
| max | 14,233 (`pomen 1: imeti`) |
| senses above 15 | 34,827 (**28.8 %**) |

This matters more than a uniform sample of the graph suggests, because the dataset
bands on a proxy that **is** collocation count (D9/D10) and therefore samples the tail
on purpose. Ball cost over the top 400 anchors by proxy, 2 hops, with the `sestavina`
cap already applied:

| `sense → kolokacija` cap | p50 | p90 | p99 | max |
|---|--:|--:|--:|--:|
| capped (D5b) | **~2,000** | ~3,000 | ~6,500 | ~9,200 |
| 100 | 2,806 | 3,789 | 7,189 | 9,995 |
| uncapped | 22,704 | 41,779 | 80,429 | 135,515 |

### 3.1c Selecting *which* K collocations — sampling beats ranking

Deterministic ranking was tried and rejected. Every fixed key is biased: partner proxy
ascending lands in the hapax tail (*akratotermalna voda*), descending walks into
*imeti nad vodo*, and the best-looking variant — threshold, then sort descending —
returns whatever sits just under a cutoff picked by eye. Any fixed key also re-selects
the same slice of the distribution on every anchor.

D5b therefore **samples**: K drawn without replacement with `w ∝ log(1 + proxy(partner))`,
seeded from the anchor's node code. The partner-proxy distribution is a power law
(log-log fit R² = 0.93), which is what justifies damping it rather than weighting
linearly.

Measured over the **632** collocation anchors the dataset actually drew: the candidate
pool before the cap is p50 36, p90 526, max 10,559, so **61.4 %** of anchors hit the cap
and the weighting is doing real work on them. Mean hub slots out of 15 on those:

| partner degree ≥ | `log1p` | linear |
|---|--:|--:|
| 1,055 *(p99 of the global fan-out)* | 7.49 | 11.91 |
| 5,000 | **1.73** | **5.44** |
| 64,842 *(`biti`, the largest hub)* | 0.19 | 1.45 |

Linear weighting is worse at every threshold, by 3× at the top — two thirds of its list
becomes hub partners. `log1p` damps them without banning them: *iti po vodo* is a
perfectly good Slovene collocation. Mean distinct partners per ball is 11.1 of 15, so a
ball is rarely one partner repeated.

### 3.2 The surface-form reverse index

Built from `canonicalForm`/`otherForm` + `writtenRep`: **8,960,506 forms with a
resolvable owner, 3,148,457 distinct surface strings.**

| a surface string maps to | share |
|---|--:|
| exactly 1 lexical unit | **88.9 %** |
| ≤ 2 | 97.2 % |
| ≤ 3 | 99.1 % |
| worst case | 18 (*goli*, *meti*, *ti*, *peti*) |

Resolving a known token to a node is a dict lookup, not a model. The hard part is
deciding *which* token is the target, which D3 hands to an external extractor; the
11.1 % of strings that resolve to more than one unit are handed to the model as a union
of balls.

## 4. Design decisions

**D1 — Generate, don't relabel.** Fresh questions and answers from the graph; no
reference rows are reused (§1.2).

**D2 — Every item is anchored to a `lexical-unit` id**, in the store's own id space, so
item ↔ graph node is recoverable in both directions.

**D3 — No delimiters. Entity linking is an external extractor plus a plain-text lookup,
and the union it returns is part of the model's input.** The pipeline is: extract the
word(s) from the user's question → look each up verbatim in the reverse index → extract
the hop-2 ball of every exact match → hand the model their union. Four consequences,
all binding on generation:

* **Choosing among what linking returned is a learned skill, and it is supervised
  throughout.** Resolution is upstream; *selection* is the model's job. It is also the
  only mechanism that handles cross-POS homonymy (*kot* = angle / as), which delimiters
  could never have resolved.
* **Dataset extraction runs the identical pipeline**, never privileged knowledge of
  which unit the generator started from. Multi-entity balls are therefore not a quota'd
  slice — their share is whatever the pipeline yields, which is what makes the training
  distribution match inference.
* **All phrasings are plain prose.** A user who does type quotes costs nothing: the
  extractor consumes them as ordinary text.
* **Cost stays affordable.** 88.9 % of surfaces resolve to one unit and 97.2 % to two,
  against single-seed ball sizes of p50 19 / p99 740 nodes.

**D4 — One extraction policy for every question: hop 2, uniform.** Production accepts
arbitrary questions, so extraction must not branch on question type. Hop 2 is the floor
— synonyms, antonyms and collocations are reified one hop past the sense — and also the
ceiling, since those nodes carry both lemmas in their own text.

**D5 — `sestavina` traversal is directional and capped.** Downward always (bounded at
22); upward at **K = 10**, ranked deterministically — prefer MWEs whose senses carry a
definition or example, then fewest constituents. One content-independent rule, so D4
still holds.

**D5b — `sense → kolokacija` is capped per anchor at K = 10**, by the seeded weighted
sampling of §3.1c. **Pooled per anchor, not per sense**, because a polysemous anchor's
collocations are not spread over its senses — a per-sense cap would still admit 17 × K
for *voda*.

> **K does not affect gradability, and that is worth stating because it was got wrong
> once.** T17 grades by membership in `ALL(anchor)` — the anchor's *complete*
> collocation set, shipped inside the item — plus a count band, so the set of correct
> answers does not depend on K at all. What K controls is *which* correct phrases the
> model can see.
>
> **The supervision target did depend on it.** The generator verbalised T17's answer
> from its own draw before any ball existed, and 26.5 % of its target phrases named a
> collocation absent from the item's own ball — training the model to emit a
> plausible-sounding collocation whether or not the evidence is in front of it, which is
> exactly what the graph is there to prevent. Membership targets are therefore
> re-verbalised **after the ball is known**, from `ball ∩ all_items`, in the same pass
> that builds the ball.

**D6 — One gradeable line per answer**, fixed prefix, no markdown, no MSD tags. See
`QA_TASKS.md` §0.1.

**D7 — A "not recorded" slice is included by construction**, ~10 % of items, mostly
ordinary lemmas genuinely missing the relation rather than invented words — otherwise
the model learns "weird-looking word → say you don't know" and never learns to check
the graph. The graph supplies these in bulk: among core lemmas 59 % lack a definition,
65 % lack synonyms, 90 % lack a usage example. Flavours and rates: `QA_TASKS.md` §0.2.

**D8 — Seed pool: core entries passing a content filter.** `id < 1 M` plus at least one
of {`@sl` definition, synonym, collocation, usage example} → **72,528 lemmas**.

> **One exclusion: entries whose lemma is not determined by the source.** 958
> lexical-units (196 core) have a canonical form carrying two different words as
> `writtenRep`, with no signal for choosing between them — the reflexive clitic *se*
> renders as `iztočnica: prikazati`, and *v*, *pri*, *s/z*, *za*, *na* are affected too,
> i.e. disproportionately the highest-frequency function words, where a wrong lemma does
> the most damage. Some are legitimate variants (*molsti* / *molzti*) rather than
> contamination, but a variant still leaves "which one is the lemma" undecided, so all
> are excluded. Cost: 0.27 % of the pool.

**D9 — The frequency proxy is computed from the graph**, no external data:
`proxy(lemma) = collocation memberships + MWE memberships`. Both are corpus-derived,
both have real dynamic range, and the ranking they produce is a frequency ranking — its
top is *biti, in, kot, se, ne, imeti* and the prepositions. Usage-example count was
measured as an alternative and fails: p50 = p90 = p99 = 4, because the data is capped at
~4 examples per sense.

**D10 — Fixed log-spaced bands, not quantiles.** §5.

**D11 — Lemma-disjoint split.** Partition lemmas first, generate inside each partition,
then stratify so both sides share the band profile. Test lemmas are never seen in
training, so the score measures graph reading rather than memorisation.
Neighbourhood-level disjointness is not required — a test lemma's ball may overlap a
training lemma's.

**D12 — Three tiers of held-out generalisation.** Tier A, unseen *phrasings* of trained
types (2–3 templates per type withheld, which is why each type needs more frames than
it uses). Tier B, an unseen *question type* over seen relations. Tier C, an unseen
*relation*: **antonyms are held out entirely** — only 6,898 pairs exist, so the training
cost is negligible, and it tests whether a model that learned to read `sopomenka:` nodes
can read `protipomenka:` nodes it was never supervised on. These tiers are a headline
result, not a side experiment.

**D13 — Retrieval-focused.** Reasoning-heavier items (aggregation, filtering, multi-hop)
are deferred; Tiers B and C are the only place reasoning is probed.

**D14 — The type inventory is the reference set reworked, plus antonyms.** 16 types kept
as-is, 2 reworked (D15), antonyms added as Tier C, and two pairs merged — sense listing
into sense definition, and numbered collocations into collocations with a quantity band.
`stopnjevanje` is regenerated from `lexinfo:degree` rather than imitated, since the
reference version is degenerate in 99 of 100 rows.

**D15 — Morphological ambiguity is split across the two form-analysis types.** The graph
lists every reading a form can carry but cannot pick between them; only a sentence can.
So `analiza_oblike` (no sentence) takes the **ambiguous** forms and answers with the
disjunction, which is well-posed and fully graph-derivable, while
`analiza_oblike_v_povedi` (with sentence) is restricted to **unambiguous** forms, so
every label is correct by construction. No tagger dependency either way.

> **The ambiguity test is only as good as the paradigm.** It rests on the paradigm being
> complete, which D15 assumed without saying so: the store's `odstotek` has
> `rodilnik/množina → odstotek`, so the genitive plural occupied one cell and passed as
> unambiguous, yielding the label *rodilnik dvojine* on a real sentence. Same defect
> class as degenerate gradation, one word class over — the cell exists in the RDF, the
> surface was never inflected, and it is a property of the source lexicon rather than of
> the builder. The `healthy_grid` filter (`QA_TASKS.md` §1, C23) costs ~9 % of the noun
> pool and protects all six affected types.

**D16 — Scale: ~10 k train (≤1 k dev) + ~2 k test.** Deliberately modest; regenerating
more is cheap. One consequence: a per-(type × band) cell would hold ~12 items, so
results are reported **per type** and **per band**, never as a cross-tab.

**D17 — Slice sizes.** Sentinels ~10 % of items, ~70 % of them ordinary lemmas. Tier A
~5 %, Tier B ~3 %, Tier C 100 % of antonym items.

**D18 — The builder renders the morphology the types need** — `vform`, `person`,
`definiteness` on forms, `aspect`, `gender`, `clitic` on entries. What is and is not
renderable: `QA_TASKS.md` §0.6.

**D19 — Two baselines, isolating two different things.** *No-retrieval* — same model,
same questions, nothing retrieved — measures how much is already in the weights.
*Serialised graph* — the same extracted subgraph flattened to plain text — measures what
structure buys over merely having the facts in context. Both run on the identical item
set, so neither costs extra generation.

**D20 — Single-turn, no personas, no JSON.** Every item is one self-contained question
and one answer. Follow-up handling and off-topic refusal are unspecified, so neither is
trained for or evaluated.

**D21 — Capability retention is phase two.** Preserving the base model's general
abilities is highly preferable, but it comes after establishing that the model can
answer these questions from a subgraph at all. There is no general-instruction replay
slice — recorded so the absence reads as a decision rather than an oversight.

## 5. Frequency bands

Fixed log-spaced thresholds on the proxy value.

| band | proxy | lemmas | share of pool | **sampling quota** |
|---|---|--:|--:|--:|
| **B0** | 0 — no corpus signal | 6,264 | 8.6 % | **5 %** |
| **B1** | 1–4 | 9,700 | 13.4 % | **10 %** |
| **B2** | 5–16 | 13,855 | 19.1 % | **15 %** |
| **B3** | 17–64 | 15,385 | 21.2 % | **20 %** |
| **B4** | 65–256 | 14,357 | 19.8 % | **20 %** |
| **B5** | 257–1024 | 9,214 | 12.7 % | **20 %** |
| **B6** | 1025+ | 3,753 | 5.2 % | **10 %** |

The quotas tilt upward: B5+B6 rise from their natural 17.9 % of the pool to **30 %** of
the dataset, so ordinary-to-common vocabulary is well represented instead of being ~1 %
of a Zipf-shaped sample. B6 is held at 10 % rather than more, because the very top of
the proxy is function-like words (*med*, *raven*, and above them *biti*, *in*, *kot*)
whose lexicographic questions are less meaningful. The counterweight for the low bands
is that dictionary users look words up *because* they are unfamiliar — hence a tilt
rather than a reversal.

**Why fixed thresholds rather than quantiles.** They are **version-stable**:
`257–1024 memberships` means the same thing forever, where a quantile boundary moves
whenever the content filter or the snapshot changes, and per-band accuracy stops being
comparable across dataset versions. They are **interpretable**: an absolute band is a
claim about the language, a percentile band a claim about our filtering. And
**per-type stratification requires them** — 65,480 lemmas have collocations but only
11,048 have direct usage examples, so with quantiles each sub-pool would get its own
boundaries and "band 5" would mean a different frequency range per question type.

The two schemes nearly coincide here anyway: the proxy is close to log-uniform over the
pool, so powers-of-four thresholds already produce roughly equal band sizes.

**Why stratify at all.** The pool is Zipf-shaped, so uniform sampling would put ~80 % of
items in the four lowest bands — the reference file's failure mode. Beyond realism, the
proxy *is* a degree count, so it predicts subgraph size and therefore difficulty
(*oživčiti*, proxy 0, has a ~19-node ball; *med*, proxy 39,830, saturates the cap).
Stratifying by frequency stratifies by graph size, which lets results be read as a
function of how much graph the model had to read — the most informative axis available
for a model whose thesis is inference-time structure reading.

**Thin cells are handled, not ignored.** A "B0 × usage example" cell is nearly empty,
so the availability matrix is computed before generating — by *running* every generator
over every pool entry rather than predicting eligibility — and a thin cell's quota is
shrunk with the shortfall redistributed across bands with headroom. Per-type quotas
therefore cannot all be identical.

## 6. Not supported, and deferred

What the KG cannot support at all, and what is consciously postponed, is enumerated per
family in `QA_TASKS.md` §4. The largest deferred item is the **MWE question family** —
10.3 M constituent edges with categories and constituent order, and MWE headwords that
do store their surface phrase. It is deferred on implementation risk rather than on
value: it is the one family that depends on the upward `sestavina` cap behaving.

## 7. Open items

| # | item |
|---|---|
| **R4** | **The extraction model of D3 is unspecified** — which model, where it runs, what it costs. Its misses are end-to-end service errors: if it does not return the target word, the right ball is never extracted and no amount of training recovers it. Its recall is measurable for free against our own templates, since we author them and the gold target word is known. |
| **R5** | **The natural multi-entity rate is unmeasured.** D3 makes the union share whatever the pipeline produces rather than a quota, so it has to be measured on real question strings — the 88.9 % single-match figure is over *all* surface strings, and the templates sample lemmas, not forms. If the rate is near zero, the discrimination skill is untrained in practice and ambiguous seeds should be oversampled. |
| **R7** | **The fourth negative flavour is not generated** — the relation present in the ball but only on a co-extracted distractor unit, where answering from it would be wrong. It is the negative counterpart of D3's discrimination skill, and it is the one flavour that cannot be built without the ball, so it belongs with the ball builder. |

## 8. How these numbers were measured

Full-corpus figures are parallel scans of `kg_raw/OntoLex DSB/` with `xargs -P 32` over
awk, **each worker writing to its own output file** — a shared stdout interleaves and
corrupts ~0.3 % of lines. A predicate census takes ~30 s; a value census ~60 s. Use the
shared venv, not the local one. The scans behind this document:

- predicate census (all files, count by `$2`), and value censuses for the `lexinfo:*`
  feature predicates, `vartrans:category` and `rdf:type`;
- language-tag censuses for `writtenRep`, `skos:definition`, `rdf:value`;
- `lexical-unit → canonicalForm` and `→ partOfSpeech` maps;
- `decomp:constituent` out-degree by subject and `correspondsTo` in-degree by object;
- `usageExample` by sense, joined through `ontolex:sense` to entries; `rdfs:member` by
  sense for collocation membership;
- the surface index: `canonicalForm`/`otherForm` joined to `writtenRep`, lowercased.

Store-level figures (§3.1b, §3.1c) come from `analysis/measure_colloc_sampling.py` and
`analysis/measure_ball_mix.py`, which read the built store rather than the dump.
