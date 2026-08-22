# Lexicographical QA dataset — design decisions

> **Status: locked 2026-08-18; re-locked 2026-08-20 after the owner meeting of 2026-08-19,
> which superseded D3 and added D19–D21. Generation not yet started.** This document records
> the design of the question–answer dataset that pairs with the v3 GTLM input graph. It
> supersedes any assumption that the two reference QA files in `data/` will be used as
> training data directly. Companion documents: [`README.md`](README.md) for how the graph is
> built, and the root [`../README.md`](../README.md) for the model target.

All figures below were measured over the **full** raw KG (all 2,594 `.nt` files in
`kg_raw/OntoLex DSB/`) unless stated otherwise. Where a number contradicts an earlier
belief, the earlier belief is called out explicitly.

### v1 at a glance

| | |
|---|---|
| **Target** | a public lexicographical Q&A service on the lab website, general-public users, single-turn |
| **Size** | ~10 k train (≤1 k dev) + ~2 k test |
| **Types** | **19** — the reference inventory reworked, plus antonyms (T13→T12, T18→T17) |
| **Seeds** | core lemmas passing a content filter, minus 196 with a source-ambiguous lemma, sampled across 7 fixed frequency bands |
| **Store** | `kg_graph_v7_gemma3` — v4 morphology, v5 verbalised collocations, v6 noun gender, v7 one collocation node per phrase |
| **Entity linking** | no delimiters — external extractor, plain-text lookup, **union ball** when several units match |
| **Extraction** | uniform hop 2 from every matched unit; `sestavina` downward always, upward capped at K = 10; `sense → kolokacija` capped at K = 10 **per anchor** (both were K = 15 until 2026-08-22 — see D5/D5b) |
| **Answers** | the gradeable line `ODGOVOR: …` and, in v1, nothing else — UI prose postponed; no MSD tags |
| **Split** | lemma-disjoint; three held-out generalisation tiers, antonyms entirely unseen |
| **Baselines** | no-retrieval, and serialised-graph — the two gaps isolate retrieval and structure |
| **Deferred** | the MWE question family, verbalised collocations, reasoning-tier items — **Section 6** |

---

## 1. What we are building, and why we generate our own items

### 1.1 Target, users, and how success is judged

Confirmed with the project owner, 2026-08-19.

- **The deliverable is a service** hosted on the lab website, where users query the model
  about lexicographical properties of Slovene words. Not a paper artefact, not an internal
  demo.
- **The users are the general public** — no lexicographic training assumed, no controlled
  input, arbitrary phrasing. This is the fact behind D3: we cannot impose input conventions
  on them.
- **Serving a 12 B model is acceptable.** No latency or concurrency target was set, so there
  is no throughput constraint to design around in v1.
- **There is no external number to beat.** A parallel method by colleagues will be compared
  against this one, but it is a comparison point, not a threshold — so we define our own
  baselines (D19).
- **Interaction is single-turn** (D20), and **retaining the base model's general abilities is
  phase two, not phase one** (D21).

### 1.2 Why we generate our own items

The original plan was to label the existing synthetic questions in
`Lexical-QA-SLO-test.json` against the knowledge graph. We are instead **extracting the
question *types* and generating fresh questions and answers directly from the graph**.

The reference items were synthetic to begin with, so nothing is lost, and three
independent findings make regeneration clearly the better path:

1. Roughly **half of the reference question types are unanswerable from this KG at all**
   (Section 3) — the graph has no register, dialect, frequency or error metadata, no
   meronymy, effectively no hypernymy, and translations only into Hungarian.
2. The newer reference file's entity ids are **42 % dangling** against our KG dump
   (Section 2.2).
3. Several of its subtypes are **defective at a 99–100 % rate** (Section 2.2), so they
   would need regenerating regardless.

Generating ourselves also gives exact control over answer format, over which facts are
guaranteed present in the extracted subgraph, and over the train/test split.

---

## 2. The two reference datasets

### 2.1 `Lexical-QA-SLO-test.json` (older, 2,747 items)

ShareGPT-style conversations, `source: lexicographic_generator`, all `data_type:
automatic`. **6 categories / 28 sub_types**, ~100 items each. Metadata flags
`has_role` (286 items embed a role-play persona) and `is_json_format` (353 items demand
JSON output). Two prompt generations, `v1` (2,077) and `v2` (670).

No entity references — questions name the lemma in prose only. **65 %** delimit the
target word (1,504 with `»…«`, 422 with `"…"`, 89 with `'…'`).

Answers are conversational prose. Notably, this file handles in-sentence morphology
*better* than the newer one: it answers in words, gets *Vidim goro* → tožilnik/ednina
right, and in at least one case names an ambiguity rather than guessing
("*orodnik ednine moškega ali srednjega spola*").

### 2.2 `Lexical-QA-SLO(in).csv` (newer, 2,000 items)

Columns `id, lemma, vprašanje, odgovor, vrsta, podvrsta, split`. **8 `vrsta` / 20
`podvrsta`**, exactly 100 rows each; 1,700 train / 300 test; no duplicate Q/A pairs.
Answers are consistently formatted (markdown tables for paradigms, `**bold**` headwords,
`/` for gaps). Only **5 %** delimit the target word.

**The `id` column is genuine.** It is a CJVT `lexical-unit-N` id: for every id present in
our dump, the CSV `lemma` equals the KG `ontolex:canonicalForm` → `writtenRep` —
**1,140/1,140, 100 % match** — and POS/aspect agree (`43215` = *občutno*/adverb,
`10305` = *zgrinjati*/`aspect progressive`).

**But 42 % of ids are absent from our dump.** 837 of 1,977 unique ids are missing, and
every missing one lies in the band **442,546–740,313**. Our dump's `Word` lexical-units
occupy 1–100 k (99,998) and >1 M (299,379); that middle band is entirely absent. Our copy
is *not* truncated — the export is exactly 401 `-words.nt` files, 400 holding 1,000
entries and one holding 180, matching the 400,180 `partOfSpeech` count. So the CSV was
built against a newer snapshot or a live database.

Coverage by subtype: `kolokacije` 100 %, `primeri_uporabe` 95–96 %, `pomen` 84–93 %,
`sopomenke`/`spreganje` 53–68 %, `besedna_vrsta` 20–22 %, **`sklanjanje` 15–30 %**. The
declension subtypes are worst hit because they sample foreign proper nouns
(*misailović, haibach, garsid, spitall, ziege*) that only exist in the newer snapshot.

**Measured defects:**

| subtype | defect | rate |
|---|---|--:|
| `stopnjevanje/vse_stopnje` | comparative = superlative = lemma | **99/100** |
| " | "positive" cell is an unsorted dump of all inflected forms | 76/100 |
| `pomen/stevilka_pomenov` | agreement error ("*ima registrirana 1 pomen*") | **100/100** |
| `primeri_uporabe/analiza_oblike_v_povedi` | answers only a raw MSD tag | **100/100** |
| " | answer names the case in words | 1/100 |
| " | vacuous "form X of lemma X" | 36/100 |
| `pomen/*` | junk sense labels ("*raba obliki*", "*raba živali*") | 12 rows |

The MSD tags are well-formed JOS tags (category letter agrees with KG POS **91/91**, and
decoded number+case is a legal KG reading **58/58**) — but **50 of 58 noun forms are
morphologically ambiguous**, and on the cleanest ambiguity class the disambiguation is
wrong every time: *haljico, medicino, uniformo, marihuano, teorijo* are all tagged
instrumental where context forces accusative (**5/5 wrong**).

Split discipline is row-level, not lemma-level: **5 ids and 6 lemmas** appear on both
sides.

**Provenance, confirmed by the data owner (2026-08-18).** The rows were built against the
live DDDS **database**, not this RDF export — which is why the ids run past what we have.
Collocations came from a separate API endpoint (see Section 6.2), and **definitions were
drawn partly from WordNet and a bridge dictionary**, not only from the KG. That last point
matters: the COBUILD-flavoured full-sentence glosses in `pomen/razlaga_pomena`
("*Kadar sodišče dostavi pravno odredbo nekomu, mu jo pošlje*") are not `skos:definition`
values, so those gold answers are partly ungrounded in this graph — a further reason not to
reuse them. The owner also noted independently that many of the sampled words are very rare
and probably should not have been used, which is the same conclusion D8 reaches from the
numbers.

**One of its subtypes cannot be backed by this KG:**

- ~~`kolokacije/*` — 100 % of ids resolve, yet the answers are not reproducible.~~
  **Withdrawn 2026-08-20 — they are reproducible; see §6.2 and `README.md` Finding 8.**
  The `frac:Collocation` node is indeed textless, but the phrase sits on the multi-word
  entry whose sense id its IRI embeds. **1,196 of the 1,307 phrases (91.5 %) come back
  verbatim** from the export alone, 155/200 rows in full; the rest is mostly the same
  collocation in a different number/case, plus possessive adjectives of proper names. The
  uppercase items ("*Potrpežljivost je vrlina*", 175/1,307) are not foreign corpus
  fragments either — they are stored that way in the KG.
- `analiza_oblike_v_povedi` MSD strings — the KG stores `lexinfo` feature triples, not
  MULTEXT-East tags.

### 2.3 Type mapping between the two files

The newer file keeps **13 of the older file's 28 subtypes** (re-cut into 20) and drops 15.

| old category / sub_type | new counterpart |
|---|---|
| oblikoslovna/paradigm_generation | sklanjanje/posamezen_sklon, spreganje/spreganje_v_casu |
| oblikoslovna/paradigm_identification | sklanjanje/analiza_oblike, primeri_uporabe/analiza_oblike_v_povedi |
| oblikoslovna/lemma_lookup | sklanjanje/osnovna_oblika_leme |
| oblikoslovna/part_of_speech | besedna_vrsta/osnovne_lastnosti, spol_samostalnika |
| oblikoslovna/gradation | stopnjevanje/vse_stopnje |
| oblikoslovna/aspect_and_vform | besedna_vrsta/vrsta_in_vid_glagola, spreganje/neosebne_oblike |
| semanticna/definitions | pomen/razlaga_pomena |
| semanticna/polysemy | pomen/nastevanje_pomenov, stevilka_pomenov |
| semanticna/synonyms | sopomenke/navedi_sopomenke |
| raba/collocations | kolokacije/navedi_, stevilsko_opredeljene_ |
| raba/usage_example | primeri_uporabe/povedi_z_besedo |
| oblikoslovna/clitics_and_negation | *dropped* — but KG supports it |
| semanticna/antonyms | *dropped* — but KG supports it |
| skladenjska_besedotvorna/mwe_structure, mwe_component_search, idiom_completion | *dropped* — but KG supports these well |
| semanticna/hypernyms_hyponyms, meronyms_holonyms | *dropped* — KG cannot support |
| raba/context_disambiguation | *dropped* |
| prevajalska/* (3) | *dropped* — KG cannot support |
| skladenjska_besedotvorna/valency, word_formation | *dropped* — KG cannot support |
| statisticna_metapodatkovna/* (4) | *dropped* — KG cannot support |

The newer inventory is a much better fit for the graph, because it drops almost exactly
what the graph cannot answer. It over-trims in three places (antonyms, the MWE family,
clitics) and keeps two things the graph cannot back (collocation surface phrases, MSD).

---

## 3. What the KG can and cannot answer

Full-corpus predicate and value census. **Not present anywhere in the KG:** register,
style, dialect, domain, frequency, typo/error metadata, meronym, holonym, valency frames,
derivational morphology. `lexinfo:type = foreign` exists but numbers **702** instances.

| signal | count | supports |
|---|--:|---|
| `ontolex:writtenRep` | 13,499,411 | surface forms, the reverse index |
| `lexicog:usageExample` | 14,722,502 | usage examples — **but see below** |
| `decomp:constituent` (+ `lexinfo:category` on all) | 10,265,230 | MWE structure, component search |
| `rdf:_1 … _22` | — | constituent order |
| `ontolex:sense` | 8,545,797 | sense enumeration, counting |
| `lexinfo:degree` | 4,542,306 (pos 4,361,733 / comp 107,717 / super 72,856) | gradation |
| `lexinfo:vform` | 2,353,283 (inf 1,580,662 / supine 20,878 / …) | non-finite forms |
| `lexinfo:aspect` | 1,735,245 (perf 882,484 / prog 718,921 / biasp 133,840) | verb aspect |
| `lexinfo:clitic` | 423,585 (`yes` 423,503) | clitics |
| `lexinfo:partOfSpeech` | 400,180 | POS — **only on single-word entries** |
| `lexinfo:synonym` | 362,520 | synonyms |
| `skos:definition` (`@sl`, distinct senses) | **225,618** | definitions |
| `lexinfo:negative` | 121,229 (`yes` 6,630) | negation |
| `vartrans:*` | 77,570, all `directEquivalent`, all **`@hun`** | Hungarian only |
| `lexinfo:antonym` | **6,898** | antonyms |
| `lexinfo:hypernym`/`hyponym` | **5,415** each | effectively unusable (0.12 %) |

**Two findings that reshape the design:**

**(a) Usage examples are overwhelmingly attached to MWE senses.** Of 14,722,502 examples,
**14,650,773 (99.5 %) hang off MWE senses** and only **71,729 (0.5 %)** are reachable from
a single-word entry — spread over just **11,048 words**. Words that do have examples have
them directly on their own senses (*dedek* 25, *plesen* 27, *abiturient* 1 — matching the
single sentence the reference file gives for *abiturient*), so word-level usage-example
questions need no MWE traversal, but the eligible pool is small.

**(b) The KG has two populations.** Content richness differs by 3–4×:

| | entries | sl definition | collocations | synonyms | usage example | **any** |
|---|--:|--:|--:|--:|--:|--:|
| **core** (id < 1 M) | 100,801 | 40.6 % | 65.0 % | 35.4 % | 10.2 % | **72.0 %** |
| **tail** (id > 1 M) | 299,379 | 7.3 % | 12.2 % | 4.4 % | 0.3 % | **20.4 %** |

The core is a curated lexicon; the tail is a thin auto-generated extension. This is why
the reference file is full of *garsid* and *misailović* — it sampled across both.

### 3.1 The `sestavina` hub is directional

| direction | n | p50 | p90 | p99 | max |
|---|--:|--:|--:|--:|--:|
| **MWE → its constituent words** | 3,940,417 | 3 | 3 | 4 | **22** |
| **word → MWEs containing it** | 114,033 | 4 | 136 | 1,207 | **423,510** |

Downward traversal is bounded at 22 across the entire graph. Upward traversal is the only
structure that explodes, and its hubs are function words: *biti* (423,510), *in*, *kot*,
*se*, *ne*, *imeti*, *do*, *iz*, *med*, *na*, *o*, *ob*, *od*, *po*, *pred*, *leto*. Only
114,033 of 400,180 words are constituents of any MWE at all, which is why the median word
seed stays flat at 19 nodes across all three hops.

### 3.1b The `sense → kolokacija` hub — a second explosion, found 2026-08-21

> **Re-measured 2026-08-22 against `kg_graph_v7_gemma3`, with `kg_graph_v6_gemma3` as a
> control.** The original table was taken on v5, whose collocation layer held one node per
> member *set* rather than one per phrase and was therefore missing 20 % of the curated
> phrases, non-uniformly (`README.md` Finding 11). Both columns below come from one script,
> `analysis/measure_d5b.py`, run twice — so the difference between them is the dedup key and
> nothing else. **The v6 column reproduces the originally published figures exactly**, which
> is what licenses reading the v7 column as the effect of the fix rather than of a new
> method.

`sestavina` is **not** the only structure that explodes:

| sense's fan-out into `kolokacija:` nodes | v6 (= as published) | **v7** |
|---|--:|--:|
| collocation-bearing senses | 120,871 | 120,871 |
| p50 | 4 | **4** |
| p90 | 101 | **114** |
| p99 | 791 | **1,055** |
| max | 14,233 (`pomen 1: imeti`) | **64,842** (`pomen 1: biti`) |
| senses above K = 15 | 34,827 (28.8 %) | **35,446 (29.3 %)** |

Top hubs, v7: `biti` 64,842 · `imeti` 30,032 · `leto` 24,242 · `iti` 14,570 · `tako` 12,993
· `priti` 11,377 · `čas` 11,197 · `dan` 11,072 — the same function-word population as the
`sestavina` hubs, but with `biti` now clear of the field. That reordering is itself
informative: the copula participates in more *distinct curated phrases* than any other word,
and the member-set dedup had been collapsing precisely those.

**The number of collocation-bearing senses does not move.** 120,871 before and after. The fix
adds phrases to senses that already had one; it does not reach any new sense. So every
statement in this design about *which* anchors have collocations still holds — only *how
many* changed.

**Why the p50 of 4 is misleading here.** D9 defines the frequency proxy as
`collocation memberships + MWE memberships`, and D10 bands on it. The high bands are
therefore, *by construction*, the anchors with the largest collocation fan-out. A
uniform sample over lemmas says this is a rare tail; the sample this dataset actually
draws says it is the common case.

**Ball cost, top 400 anchors by proxy** (2 hops, D5 already applied, tokens per ball).
Re-measured on v7, v6 alongside:

| `sense → kolokacija` cap | v6 p50 | **v7 p50** | v7 p90 | v7 p99 | v7 max |
|---|--:|--:|--:|--:|--:|
| 0 (collocations excluded) | 2,142 | **2,151** | 3,228 | 6,812 | 9,936 |
| **K = 15** | 2,285 | **2,292** | 3,372 | 6,974 | 10,065 |
| K = 50 | 2,608 | **2,619** | 3,685 | 7,294 | 10,378 |
| K = 100 | 3,088 | **3,099** | 4,171 | 7,722 | 10,833 |
| uncapped | 23,009 | **30,936** | 58,527 | 111,292 | **588,537** |

**Read the two columns together — this is the result that matters.** Every *capped* row is
unchanged by the fix (2,285 → 2,292 at K = 15, +0.3 %), while the *uncapped* row rises 34 %
at the median and 4.3× at the maximum. The fix made the cap more necessary and no more
expensive. A 13× cut at the median, up from 10×.

The marginal cost of K is still nearly flat — 0 → 100 moves p50 by under 1 k tokens — so the
whole cliff sits between K = 100 and uncapped. **K is cheap; only the presence of a cap
matters.** That conclusion survives the rebuild unchanged, which is the point of restating it
here rather than assuming it.

*Method, stated because it differs slightly from the original run and explains a ~10 %
level shift in the low rows:* the top-400 set is taken by the D9 proxy computed over all
100,801 core anchors, and D5's upward `sestavina` cap is applied by lowest node id rather
than by its content-based rank. Which 15 MWEs are kept changes a ball's contents, not its
size, so the cost column is unaffected by that simplification — but the anchor set is not
identical to the one behind the original table, and the absolute numbers should not be
compared across the two runs. **The v6/v7 comparison above is within one run and is the
comparison to trust.**

### 3.1c Selecting *which* K collocations — sampling beats ranking

> **Re-measured 2026-08-22 on v7, with v6 as a control** (`analysis/measure_d5b.py`).

**The partner-proxy distribution is a power law.** Over 120,871 collocation-bearing senses
the log-log rank/value fit gives slope **−1.806, R² = 0.932** on v7 and **−1.748, R² = 0.926**
on v6 — the distribution steepened slightly, as it must when the fix adds most to the head.
Either way it is a power law, and that is what justifies log-damped weights rather than
linear ones.

*One honest discrepancy.* The originally published figure was **−1.418, R² = 0.933**, and my
v6 control does **not** reproduce it, although every other v6 figure in §3.1b does, exactly.
The R² matches to three decimals while the slope does not, which points at a different fitted
*range* or a different quantity (per-collocation partner proxy rather than per-sense fan-out)
rather than at a different graph. It is recorded as unresolved rather than quietly
overwritten. **Nothing downstream turns on the exponent's value** — the design uses the fact
that the tail is heavy, not its slope — so this is a provenance loose end, not a live risk.

**15 draws from `voda`'s collocations**, mean ± sd over 30 seeds. The pool itself grew
3,026 → **4,589** with the v7 fix:

| scheme | v6 partner proxy p50 | **v7 partner proxy p50** | v6 hub partners / 15 | **v7 hub partners / 15** |
|---|--:|--:|--:|--:|
| uniform | 109 | 180 | 0.5 ± 0.6 | 0.8 ± 0.7 |
| **w ∝ log(1+proxy)** | 245 | **315** | 1.2 ± 1.0 | **1.7 ± 1.0** |
| w ∝ proxy | 1,417 | 14,637 | 5.6 ± 1.8 | **10.2 ± 1.5** |

v7 samples: `barake brez vode`, `hladilnik vode`, `odvesti vodo` under `log1p`, against
`imeti med vodami`, `biti med vodami`, `podjetje nad vodo` under linear weighting.

**The choice of `log1p` is more clearly right after the fix, not less.** Linear weighting
degrades sharply — 10.2 of 15 picks are now hub partners, two thirds of the list — because
the head grew fastest. `log1p` moves from 1.2 to 1.7 hub partners per 15, which is the
intended behaviour: `iti po vodo` is a perfectly good Slovene collocation and should not be
banned, only damped.

For reference, the v7 pool: p10 = 11, p50 = 184, p90 = 1,741, max = 64,842, with 369 of 4,589
partners above 2,000 (v6: p10 = 7, p50 = 105, p90 = 955, max = 14,233, 115 of 3,026).

*(The octave-stratified scheme is not re-run: it was rejected on the grounds that it
deliberately samples the hapax tail, which is a property of the scheme and not of the store.)*

> **Measured on the realised dataset 2026-08-22 — the `voda` figure generalises** (check
> C19, `analysis/measure_ball_mix.py`, results in `analysis/results/ball_mix_v1.json`).
> Everything above is one anchor; this is all **632** T17 anchors the generated v1 actually
> drew, with each ball redrawn through the same `d5b.sample` call (a pure function of the
> node code, so it is the same ball, not a resample) and once more under linear weights with
> the same seed and the same sorted pool.
>
> Candidate pool before the cap: p50 **36**, p90 **526**, p99 **1,918**, max **10,559** — so
> **61.4 %** of anchors hit K = 15 and the weighting is doing real work on those. On the 388
> capped anchors, mean hub slots out of 15:
>
> | partner degree ≥ | `log1p` | linear |
> |---|--:|--:|
> | 1,055 *(p99 of the global fan-out, §3.1b)* | 7.49 | 11.91 |
> | 5,000 | **1.73** | **5.44** |
> | 14,233 *(`imeti`, v6's largest hub)* | 0.48 | 2.56 |
> | 64,842 *(`biti`, v7's largest)* | 0.19 | 1.45 |
>
> The `1.73` cell is the `voda` table's `1.7 ± 1.0` — the single-anchor measurement was not
> a lucky anchor. Linear weighting is worse at **every** threshold, by 3× at the top. Mean
> distinct partners per ball is **11.1 of 15**, so a ball is rarely one partner repeated.
>
> One caveat on reading the top row: 1,055 is the p99 of fan-out over *all* collocation-
> bearing senses, but ball partners are drawn *weighted toward* high degree, so their median
> (1,007) sits right at that threshold and "49 % of slots are hubs" is close to tautological.
> The threshold sweep is there so no single cut carries the claim.

### 3.2 The surface-form reverse index

Built from `canonicalForm`/`otherForm` + `writtenRep`: **8,960,506 forms with a resolvable
owner, 3,148,457 distinct surface strings.**

| a surface string maps to | share |
|---|--:|
| exactly 1 lexical unit | **88.9 %** |
| ≤ 2 | 97.2 % |
| ≤ 3 | 99.1 % |
| worst case in the KG | 18 (*goli*, *meti*, *ti*, *peti*) |

Resolving a known token to a node is therefore a dict lookup, not a model. The hard part is
deciding *which* token is the target — which D3 hands to an external extraction model, and
the residual 11.1 % of strings that resolve to more than one unit are handed to the GTLM
model as a union of balls.

---

## 4. Locked design decisions

**D1 — Generate, don't relabel.** Fresh questions and answers generated from the graph.
The reference files supply the type inventory and phrasing style only; no reference rows
are reused.

**D2 — Every item is anchored to a `lexical-unit-N` id.** Same id space as the newer
reference file (verified 1,140/1,140). The persisted store keeps `(type << 56) | id` in
`node_codes.npy`, so graph node ↔ lexical unit is recoverable in both directions.

**D3 — No delimiters. Entity linking is an external extractor plus a plain-text lookup, and
the union it returns is part of the model's input.** *(Locked 2026-08-19; supersedes the
earlier "delimiters required in v1, ~60/40 delimited/plain".)* We cannot require general-
public users to mark the word they are asking about, and the UI question is moot. The
production pipeline is:

1. an **external model extracts the relevant word(s)** from the user's question — a separate
   model, not the GTLM one;
2. each extracted string is **looked up verbatim** in the surface-form reverse index
   (Section 3.2);
3. the **hop-2 ball of every exact match is extracted**, and where there is more than one
   match — several extracted words, or one surface string owning several lexical units — the
   GTLM model receives their **union**.

Four consequences, all binding on generation:

- **Discriminating the intended neighbourhood from the accidental ones is a learned skill,
  and it is in v1 by construction.** This supersedes the earlier claim that entity
  resolution is a pipeline heuristic the model cannot help with: *resolution* is upstream,
  but *selection among what resolution returned* is the model's job and is supervised
  throughout the dataset. It is also the only mechanism that handles cross-POS homonymy
  (`kot` = angle / as), which delimiters could never have resolved anyway.
- **Dataset extraction runs the identical pipeline.** Neighbourhoods for training items are
  built by the same extract-look-up-union procedure, never by privileged knowledge of which
  lexical unit the generator started from. Multi-entity balls are therefore **not a quota'd
  slice** — their share is whatever the pipeline naturally yields on real question strings,
  which is what makes the training distribution representative of inference.
- **All question phrasings are plain prose.** No `»…«` or `"…"` variants are generated. A
  user who does type delimiters costs nothing: the extractor consumes them as ordinary
  text.
- **Cost stays affordable.** 88.9 % of surface strings resolve to exactly one lexical unit,
  97.2 % to ≤ 2, 99.1 % to ≤ 3 (worst case 18 — *goli, meti, ti, peti*). Two or three balls
  against single-seed sizes of p50 = 19 / p99 = 740 nodes stays well inside the budget in
  D4.

Whether users will paste a whole sentence and ask about one word in it is **unknown** (asked
2026-08-19, no answer). The extractor makes it a non-issue for linking either way, and
`primeri_uporabe/analiza_oblike_v_povedi` supplies sentence-bearing items regardless.

**D4 — One extraction policy for all questions: hop 2, uniform.** Production accepts
arbitrary questions, so extraction must not branch on question type. Hop 2 is the floor
(synonyms, antonyms and collocations are reified nodes one hop past the sense) and also
the ceiling (those nodes carry both lemmas in their own text). Measured cost on word
seeds: p50 = 19, p90 = 76, p99 = 740, max = 1,039 nodes. The policy applies **per matched
lexical unit**; when D3's lookup returns several, the union multiplies the node count but
not the policy.

**D5 — `sestavina` traversal is directional and capped.** Downward (MWE → word) always,
being bounded at 22. Upward (word → MWE) capped at **K = 10** *(was 15 until 2026-08-22)*,
deterministically ranked —
prefer MWEs whose senses carry a definition or example, then fewest constituents. A single
content-independent rule, so D4 still holds. This supersedes the earlier idea of capping
MWE *seeds* at hop 1, which is unnecessary once traversal is directional.

**D5b — `sense → kolokacija` is capped per anchor.** Added 2026-08-21, after §3.1b showed
`sestavina` is not the only hub. Same rank-then-truncate shape as D5:

- **Trigger** on the edge class only — `sense → kolokacija:` node. Everything else the
  sense touches (examples, synonyms, its own anchor) is untouched.
- **Pooled per ANCHOR, not per sense.** A polysemous anchor's collocations are *not*
  spread over its senses — 14,233 of `imeti`'s 14,249 sit on `pomen 1` (99.9 %), 3,003 of
  `voda`'s 3,026 (99.2 %). A per-sense cap would still admit 17 × K for `voda`, so the
  cap applies to the union across all of the anchor's senses.
- **Select** K = 15 by **seeded weighted sampling without replacement**, with
  `w(c) ∝ log(1 + proxy(partner))`, where the *partner* is the member sense that is not
  the one traversed from. Not a sort — a draw. §3.1c has the comparison.
- **K = 10**, matching D5. *(K = 15 until 2026-08-22; the tables in §3.1b and §3.1c were
  measured at 15 and are left as measured — the marginal cost of K is nearly flat, so the
  conclusions carry.)* §3.1b measures the cost as ~141 tokens at p50 over excluding
  collocations entirely, against **28,644 saved** (v7; it was 20,867 on v6 — the fix made
  the cap more valuable and no more expensive).

**K does not affect T17's gradability, and the reason is worth stating** — it was got wrong
once. T17 grades in `membership` mode against `all_items`, the anchor's *complete*
collocation set shipped inside the item, plus a count band (`qa/grade.py` `count_ok`). The
sampled list in `gold_items` is the F1 reference only, never the success criterion. The
strictest band asks for `n_asked ∈ [2, 10]` and `vague_large` is a floor of 5, not a demand
for 15 — so any K ≥ 10 satisfies every band, and the set of correct answers does not depend
on K at all. What K controls is *which* correct phrases the model can see, not *whether* it
can be right. `qa/check_balls.py` tests the graded predicate rather than the sampled list.

**But the SUPERVISION target did depend on K, and that was a real defect.** Grading and
training are two different questions and the answer differed. `gen_T17` verbalised `answer`
at generation time from the K = 15 sample, before any ball existed; the ball holds K = 10.
Measured on v2: **26.5 % of T17's target phrases (978 of 3,686) named a collocation absent
from the item's own ball, across 62.6 % of its positives (389 of 621)**, in all four bands.
That trains the model to emit a plausible-sounding collocation whether or not the evidence is
in front of it — precisely the behaviour the graph is there to prevent.

**Fix: membership targets are verbalised after the ball is known** (`qa/build_balls.py`
`reverbalise`). The ball record's `answer` and `gold_items` are re-drawn from
`ball ∩ all_items`, using the generator's own band-count logic against the ball's pool
instead of the sampler's. Verified: 0 target phrases outside the ball, and all 621
re-verbalised targets still pass their own grader. Nothing about grading changed — the
contract already accepted any acceptable subset of the right size; what changed is that we
now supervise on the acceptable answers the model can actually see.

This is why the generator's split of concerns matters: `answer` is a pure function of
`(gold_items, spec[type])` and touches the store not at all, so re-rendering it downstream is
cheap and safe. Any future membership-mode type gets the same treatment for free, keyed on
`grading.mode` rather than on T17 by name.

Two implementation constraints, both silent failure modes:

- **Sort the pool by node id before drawing.** Sampling picks by index, and CSR
  adjacency order is *not* stable across builds — the v5 build reordered 0.44 % of
  `indices` rows via `imap_unordered`. An unsorted pool means the same seed yields a
  different ball after any rebuild.
- **Derive the RNG seed from the anchor's node code**, e.g.
  `seed = blake2b(node_code) ^ DATASET_SEED` — never from a loop counter or extraction
  order, or the ball depends on how many questions were processed first.

Both preserve D4: the ball stays a pure function of `(seed, policy)`, identical for every
question about that anchor.

**Why sampling rather than a ranked truncation.** A rank was tried first, in three
variants, and all three are biased in a way that sampling is not:

| key | top of `voda`'s list | verdict |
|---|---|---|
| partner proxy **ASC** | `akratotermalna voda`, `demineralizirana voda` | obscure technical hapaxes |
| partner proxy **DESC** | `imeti nad vodo`, `biti si pod vodo`, `tako za vodo` | function-word partners |
| cut at 2,000, then DESC | `problem z vodo`, `priprava vode`, `povezava med vodo` | good — but see below |

The third looks best and is the most fragile: its picks have partner proxies of
2,000 / 1,998 / 1,973 / 1,972, i.e. **it returns whatever sits just under the cutoff**.
The cutoff *is* the ranking, and it was chosen by eye. Any deterministic key also
systematically re-selects the same slice of the distribution on every anchor, so the
dataset never sees the rest of it.

Sampling removes the free parameter and the bias in one move. It also removes a whole
class of bug: T17/T18 draw their gold **from the sampled 15**, so gold is in the ball by
construction and no shared comparator has to be maintained between the extractor and the
answer generator.

**Why `log`, not uniform or linear.** The partner-proxy distribution is a power law —
log-log slope **−1.418**, R² = **0.933** over 120,871 collocation-bearing senses (§3.1c),
so linear weighting is dominated by the tail and uniform ignores frequency entirely.
Use `log1p`, not `log`: `log(1)` is 0 and would give every hapax partner zero probability.

**On "function word junk".** An earlier draft of this decision excluded top-band partners
outright. That was too blunt: `imeti nad vodo` and `biti si pod vodo` are odd, but
`iti po vodo` is a perfectly good Slovene collocation. Log-damping keeps them at
1.2 ± 1.1 per 15 rather than banning them.

**Why not a frequency-ranked cap.** The KG carries no frequency on the collocation node,
so the selection cannot be "the K most typical" and T17 already forbids claiming it is.
Note the choice above was made by
inspecting output on real anchors, **not** by matching the reference files — those cannot
adjudicate it: their rows reproduce the export verbatim because they were drawn *from*
it, which is provenance, not curation.

**The key is shared with T17.** That type draws its gold **from this sampler's output**, so
the gold is in the ball by construction and no comparator has to be kept in sync. Grading
does not compare against the ball at all — it checks membership in `ALL(anchor)`, the full
set, plus a count rule (`QA_TASKS.md` §0.8.3), which is what makes an item survive a change
of K or a rebuild.

> **Implemented 2026-08-22 as `qa/d5b.py`, ahead of the ball builder.** Neighbourhood
> extraction is a separate piece of work, but T17's gold cannot be generated without knowing
> which 15 phrases the ball will hold. The selection is therefore written as a pure function
> of `(anchor node code, K, dataset seed)` — Efraimidis–Spirakis weighted sampling without
> replacement over a node-id-sorted pool — and **the extractor is expected to call it rather
> than reimplement it**. That is what decouples the two without letting them drift. The four
> C18 reproducibility properties are asserted in `qa/selftest.py`.

**D6 — Two-part answers.** A minimal, machine-gradeable line **first**, with a fixed
prefix on one line and no markdown (e.g. `ODGOVOR: tožilnik, ednina`), followed by a
user-facing formatted rendering for the UI. Minimal-part-first so the model commits before
elaborating and a truncated generation is still gradeable. Eval parses part 1 only; a
validation script checks part 2 does not contradict it. **No MSD tags** — morphology is
answered in words.

> **Amended 2026-08-21 — v1 emits the gradeable line and nothing else.** The owner
> postponed the UI half outright: *"I would not bother with the UI part of every answer.
> First, I want to train the model to actually solve the problem and we will work on
> aesthetics only later."* The line keeps its position and its shape, so restoring the prose
> half later is purely additive, and the eval parser already ignores everything after it.
> What lapses with it: the "part 2 does not contradict part 1" validator, the negative's
> "here is what *is* attached instead" sentence, and — the one piece of real supervision —
> the collocation under-supply slice's *say so explicitly*. See `QA_TASKS.md` §0.1.

**D7 — A "not recorded" slice is included by construction.** Four flavours: the entity does
not exist; it exists but lacks the relation; it has the relation but not for the sense asked
about; and — following from D3 — **the relation is present in the extracted ball but only on
a co-extracted distractor unit**, where answering from it would be wrong. That last flavour
is the negative counterpart of the discrimination skill D3 makes the model's job, and it is
the only thing that stops "the answer is somewhere in the input" from being a safe bet. The gradeable line gets a fixed sentinel (working form
`ODGOVOR: ni podatka v bazi`) so identification can be scored exactly. The UI part states
that no explicit information is recorded and then offers whatever *is* attached to the
entity — e.g. no definition, but here are its collocations. **The negatives must mostly be
ordinary lemmas**, not invented words; otherwise the model learns "weird-looking word →
say you don't know" and never learns to check the graph. The graph supplies these in bulk:
among core lemmas 59 % lack a definition, 65 % lack synonyms, 90 % lack a usage example.

**D8 — Seed pool: core entries passing a content filter.** Restrict to `id < 1 M` and
require at least one of {`@sl` definition, synonym, collocation, usage example}. Yields
**72,528 lemmas**. This is what keeps *garsid* out without needing frequency data.

> **One exclusion added 2026-08-22: entries whose lemma is not determined by the source.**
> 958 lexical-units (196 of them core) have a canonical form carrying two different words as
> `writtenRep`, and the RDF offers no signal for choosing between them:
>
> ```
> word-form-2773103  ontolex:writtenRep  """se"""@sl
> word-form-2773103  ontolex:writtenRep  """prikazati"""@sl
> ```
>
> That is the reflexive clitic *se* — 423,458 MWE memberships, the second-largest hub in the
> graph — which the builder's tie-break (least capitalised, then alphabetical) renders as
> `iztočnica: prikazati (zaimek, naslonska oblika)`. Also affected: *v*, *pri*, *s/z*, *za*,
> *na*, i.e. disproportionately the highest-frequency function words, which is exactly where
> a wrong lemma does the most damage. Some of the 958 are legitimate variants (*molsti* /
> *molzti*, the *s* / *z* allomorphs) rather than contamination, but a variant still leaves
> "which one is the lemma" undecided, so all are excluded. Cost: **0.27 %** of the pool.
> Found by `analysis/scan_multi_writtenrep.py`, whose output the generator reads directly.

**D9 — Frequency proxy is computed from the graph; no external data.** The KG has no
frequency predicate. `proxy(lemma) = collocation memberships + MWE memberships`. Both are
corpus-derived (collocations are extracted under frequency thresholds), both have real
dynamic range, and the ranking they produce is a frequency ranking — the top of it is
*biti, in, kot, se, ne, imeti* and the prepositions.

> **Superseded:** an earlier proposal to use *usage-example count* as the proxy was
> measured and **fails** — p50 = 4, p90 = 4, p99 = 4, max = 198, with 740,746 entries at
> zero. The data is capped at ~4 examples per sense, so there is no dynamic range.

Sloleks/Gigafida frequency data is deliberately **not** used: the in-graph proxy is
sufficient for banding, and the external route is not worth the integration work.

**D10 — Fixed log-spaced frequency bands, not quantiles.** See Section 5.

**D11 — Lemma-disjoint train/test split.** Partition *lemmas* first, then generate items
within each partition, then stratify within each split so both sides share the same band
profile. Test lemmas are never seen in training, so the score measures graph reading rather
than memorisation. Neighbourhood-level disjointness is **not** required — a test lemma's
ball may overlap a training lemma's ball.

**D12 — Three tiers of held-out generalisation.**
- **Tier A** — unseen *phrasings* of trained types (2–3 templates per type withheld).
- **Tier B** — unseen *question type* over seen relations (e.g. train "list the synonyms",
  test "which sense of X has a usage example?").
- **Tier C** — an unseen *relation*: **antonyms are held out of training entirely.** Only
  6,898 antonym pairs exist, so the training cost is negligible, and it tests whether a
  model that learned to read `sopomenka: …` nodes can read `protipomenka: …` nodes it was
  never supervised on. The strongest available probe of whether graph attention generalises.

The owner asked specifically for good out-of-distribution behaviour (2026-08-19), so these
three tiers are a **headline result** in v1, not a side experiment.

**D13 — v1 is retrieval-focused**, matching the reference set. Reasoning-heavier items
(aggregation, filtering, multi-hop) are deferred; Tier B/C will be the only place reasoning
is probed in v1.

**D14 — v1 question-type inventory: the reference 20, reworked, plus antonyms.**

| | types |
|---|---|
| **Kept as-is (16)** | `sklanjanje/{celotna_sklanjatev, sklanjatev_po_stevilu, posamezen_sklon, osnovna_oblika_leme}` · `spreganje/{celotno_spreganje, spreganje_v_casu, neosebne_oblike}` · `besedna_vrsta/{osnovne_lastnosti, spol_samostalnika, vrsta_in_vid_glagola}` · `pomen/{razlaga_pomena, nastevanje_pomenov, stevilka_pomenov}` · `sopomenke/navedi_sopomenke` · `stopnjevanje/vse_stopnje` · `primeri_uporabe/povedi_z_besedo` |
| **Reworked (2)** | `sklanjanje/analiza_oblike` and `primeri_uporabe/analiza_oblike_v_povedi` — see D15 |
| **Kept, rework withdrawn (1)** | `kolokacije/*` — D14 downgraded these to **lemma pairs only**; withdrawn 2026-08-20 because the inflected phrase turned out to be in the export after all (§6.2, `README.md` Finding 8). The gold answer is a real phrase and the questions keep their original *besedne zveze* wording. |
| **Added (1)** | **antonyms** — generated, but held out of training entirely as Tier C (D12) |

`stopnjevanje/vse_stopnje` is regenerated from `lexinfo:degree` rather than imitated: the
reference version is degenerate in 99/100 rows. `pomen/stevilka_pomenov` fixes the
Slovene number agreement that is wrong in 100/100 reference rows.

> **Amended 2026-08-21 — the inventory is 19 types, not 21.** Two merges, both because the
> pair produced the *same gold from the same source* and the split bought only a second
> chance to drift:
>
> - `pomen/nastevanje_pomenov` → **`pomen/razlaga_pomena`**: against this KG the sense list
>   *is* the definition list.
> - `kolokacije/stevilsko_opredeljene_kolokacije` → **`kolokacije/navedi_kolokacije`**: the
>   count is now a `quantity_band` field on the item (`none` / `vague_small` / `vague_large`
>   / `exact`), which also fixes a modelling error — a member of the public asks for *nekaj*
>   or *veliko* far more often than for *exactly five*.
>
> D16's per-type arithmetic follows: ~2 k test over 19 types is **~105 test items per type**.
> Full arguments in `QA_TASKS.md` T12 and T17.
>
> **`stopnjevanje/vse_stopnje` is thinner than this section assumes.** The 107,717
> comparative *forms* of Section 3 are ~108 forms per entry, so they come from roughly a
> **thousand entries** — and the KG stores degenerate gradation on some of those
> (`oblika: mikaven (…, primernik, določna oblika)`), which the generator must filter or it
> reproduces the reference file's own 99/100 defect. This is the inventory's thinnest type,
> thinner than the usage-example types; see the realised availability matrix.

Asked which types matter most in practice and which are generator filler, the owner said
**all of them matter** (2026-08-19). No type is dropped as filler, and the inventory is not
trimmed further for v1.

**D15 — Morphological ambiguity is split across the two form-analysis types.** The graph
lists every reading a surface form can carry but cannot pick between them; only the
sentence can (this is why the reference generator is wrong 5/5 on the accusative /
instrumental class). Rather than force one type to do both jobs:

- **`sklanjanje/analiza_oblike`** (no sentence) takes the **ambiguous** forms and answers
  with the disjunction — `ODGOVOR: tožilnik ali orodnik, ednina`. Well-posed and fully
  graph-derivable; ambiguity is the correct answer there.
- **`primeri_uporabe/analiza_oblike_v_povedi`** (with sentence) is restricted to
  **unambiguous** forms, so every label is correct by construction.

No tagger dependency either way. **To verify during implementation:** the second type needs
a lemma with a direct usage example *whose* form in that example is unambiguous, and only
11,048 lemmas have direct examples. If that intersection is too thin for the quota, fall
back to the disjunction for that type as well, and record the fallback here.

> **Amended 2026-08-22 — the ambiguity test is only as good as the paradigm.** Found by
> reading a generated item, not by a check: *odstotkov* in a real corpus sentence came out
> labelled **`rodilnik dvojine`**. The label is wrong, and it is wrong *because the test
> worked as specified* — the store's `odstotek` paradigm reads `rodilnik/množina → odstotek`
> and `imenovalnik/množina → odstotek` instead of `odstotkov`/`odstotki`, so the genitive
> plural occupied exactly one cell and passed as unambiguous. "Every label is correct by
> construction" rests on the ambiguity test; the ambiguity test rests on the paradigm being
> **complete**, which D15 assumed without saying so.
>
> This is the same defect class as T11's degenerate gradation (`oblika: mikaven (…,
> primernik)`), one word class over: the cell exists in the RDF, the surface was never
> inflected. It is a property of the source lexicon, not of the builder — v7 renders
> faithfully what the export contains.
>
> Measured over the **49,078** noun entries with ≥ 12 filled cells: **4,406 (8.98 %)** have
> nominative plural equal to nominative singular, and **3,335 (6.8 %)** carry the lemma
> surface in all eighteen cells. The population is mostly foreign proper nouns
> (*Baudelaire*, *Apollinaire*, *SMS*) plus ordinary words whose plural column was never
> filled (*pilot*). Some lemmas have **both** a healthy and a defective entry (*Moliere*,
> *Gilmore*), so the filter is decided per entry, never per lemma.
>
> The filter (`gen.healthy_grid`, check C23) requires the nominative plural to differ from
> the nominative singular *and* the filled cells to hold ≥ 6 distinct surfaces. It lives
> inside `nominal_grid` rather than at the five call sites, so no type can forget it — it
> also protects T1/T2/T3 (a table of one repeated surface) and T21 (an eighteen-way
> disjunction), where the damage is obvious rather than plausible. Cost: ~9 % of the noun
> pool for all six affected types.

**D16 — Scale: ~10 k train (≤1 k held out as dev) + ~2 k test.** Deliberately modest;
regenerating more later is cheap. One consequence: at 2 k test over ~21 types over 7 bands,
a per-(type × band) cell holds ~12 items — too thin to read. Results are therefore reported
**per band** (~285 each) and **per type** (~95 each), but not as a cross-tab. The
difficulty-versus-graph-size curve is an aggregate result in v1.

**D17 — Slice sizes.** "Not recorded" (D7) ~10 % of items, of which ~70 % are ordinary
lemmas genuinely missing the relation and ~30 % nonexistent entities. Held-out tiers (D12):
Tier A ~5 %, Tier B ~3 %, Tier C = 100 % of antonym items. Sentinel wording:
**`ODGOVOR: ni podatka v bazi`**.

> **Settled 2026-08-21 — the negative rate is 10 % PER TYPE, uniform.** The owner delegated
> the choice; it is recorded in full in `QA_TASKS.md` §0.2. The reasoning in one line: a rate
> that varies by type makes *which type this is* predict *whether this is a negative*, and
> the model learns that shortcut instead of learning to check the graph. Where a type's every
> valid seed has an answer (T1–T3, T5–T7, T9, T10), the ordinary-lemma half is sourced as a
> **category mismatch** — the conjugation of *gora*, the gender of a verb — which is a real
> lemma genuinely missing the relation and the confusion a real user actually produces. One
> documented deviation: **T4's negatives are 100 % nonexistent entities**, because every real
> form belongs to some lemma and there is no ordinary-lemma negative to draw.
>
> A frame that names a word class may not be used for a category-mismatch negative: *"kakšnega
> spola je samostalnik teči"* has already asserted the false premise (`QA_TASKS.md` C20).

**D18 — The builder renders `aspect`, `vform`, `clitic`, `person` and
`definiteness` into node text.** *(Done 2026-08-20 — the **v4** stores
`data/stores/kg_graph_v4_gemma3` and `data/stores/kg_graph_v4_gams2b`; see `README.md`
**Finding 6**.)*

Scoping this decision turned up that it was worse than "some features are not
rendered". Three separate faults kept **all** verb morphology out of the text:

- `FEATURE_PROPS` listed `person`, but `VALUE_SL` mapped
  `firstPerson`/`secondPerson`/`thirdPerson` while the KG emits
  `first`/`second`/`third` — and `feat_string()` drops unmapped values silently.
  Person vanished from **253,497** forms.
- `FEATURE_PROPS` listed `tense` and `mood`. **Neither predicate exists in this
  KG.** The distinction lives in `lexinfo:vform` (`present`, `imperative`,
  `participle`, `infinitive`, `supine`) on **452,782** forms — and `vform` was
  not listed at all. **Preteklik and prihodnjik are periphrastic and are not
  stored**, which is a hard constraint on `spreganje/*` (see `QA_TASKS.md` T5/T6).
- `aspect` and `clitic` hang off the **lexical-unit**, but the feature branch
  only accepted word-form subjects, so they were dropped even when listed.

So `spreganje/celotno_spreganje`, `spreganje/spreganje_v_casu`,
`spreganje/neosebne_oblike` and `besedna_vrsta/vrsta_in_vid_glagola` were all
ungeneratable, not merely impoverished. Measured effect of the fix: form nodes
byte-identical to a sibling fall **86,848 → 418 (−99.5 %)**. `definiteness` was
added in the same pass because without it an adjective's definite and indefinite
forms carry byte-identical labels, which blocks citation-form selection for
`stopnjevanje/vse_stopnje` and `sklanjanje/*`.

**Correction to the Section 3 census, and it supersedes part of Section 6.3.**
The corpus-wide counts (`aspect` 1,735,245 · `vform` 2,353,283 · `clitic`
423,585) are dominated by `lexical-unit-part` subjects — MWE components, which
the builder's `Component` collapse discards. Reachable on word entries: `vform`
**452,782**, `person` **253,497**, `definiteness` **160,524**, `aspect`
**18,157**, `clitic` **25**. All 18,157 single-word verbs carry an aspect, so
verb-aspect questions are fully supported — but **a clitic/negation question type
is dead at 25 units**, which supersedes Section 6.3's claim that it is "arguably
the first thing to add back after the MWE family".

**D19 — Two baselines, isolating two different things.** There is no external target
(Section 1.1), so every result is reported against both:

1. **No-retrieval** — the same model, the same questions, nothing retrieved. Measures how
   much of this is already in the weights.
2. **Serialised graph** — the same model given the *same* extracted subgraph flattened to
   plain text instead of presented as graph structure. Measures what the structure buys over
   merely having the facts in context.

The gap from (1) to (2) is the value of retrieval; the gap from (2) to GTLM is the value of
structure. Both run on the identical item set, so neither costs extra generation. The
colleagues' parallel method is a comparison point, not a threshold.

**D20 — Single-turn, no personas, no JSON.** Every item is one self-contained question and
one answer. Assistant-style multi-turn behaviour was called "not necessary" (2026-08-19);
follow-up handling ("*in v množini?*") and off-topic refusal were both left unspecified, so
neither is trained for or evaluated in v1. The older reference file's `has_role` (286 items)
and `is_json_format` (353 items) behaviours are **not reproduced** — nobody knows who added
them or why (asked 2026-08-19), and neither is part of the service. Output shape is fixed by
D6 for every item.

**D21 — Capability retention is phase two, deliberately.** Preserving the base model's
general abilities is "highly preferable", but it comes *after* establishing that the model
can answer these questions from a subgraph at all. v1 therefore carries **no
general-instruction replay slice**: the dataset is pure task data. Mitigating forgetting —
replay mixing, LoRA, adapters — is taken up once the task itself is demonstrated in
isolation. Recorded so that the absence of replay data reads as a decision rather than an
oversight.

---

## 5. Frequency bands

Bands are defined by **fixed log-spaced thresholds on the proxy value**, not by quantiles.

The two schemes nearly coincide here, because the proxy is close to log-uniform over the
pool: the quantile boundaries land at 4, 16, 55, 217, 534, 1,260, 3,602 — successive ratios
of ×4.0, ×3.4, ×3.9, ×2.5, ×2.4, ×2.9. So fixed powers-of-four thresholds already produce
roughly equal band sizes, and absolute boundaries cost nothing in balance.

| band | proxy value | lemmas | share of pool | **sampling quota** |
|---|---|--:|--:|--:|
| **B0** | 0 — no corpus signal | 6,264 | 8.6 % | **5 %** |
| **B1** | 1–4 | 9,700 | 13.4 % | **10 %** |
| **B2** | 5–16 | 13,855 | 19.1 % | **15 %** |
| **B3** | 17–64 | 15,385 | 21.2 % | **20 %** |
| **B4** | 65–256 | 14,357 | 19.8 % | **20 %** |
| **B5** | 257–1024 | 9,214 | 12.7 % | **20 %** |
| **B6** | 1025+ | 3,753 | 5.2 % | **10 %** |

The quotas sum to 100 and tilt upward: B5+B6 rise from their natural 17.9 % of the pool to
**30 %** of the dataset, so ordinary-to-common vocabulary is well represented instead of
being ~1 % of a Zipf-shaped sample. B6 is deliberately held at 10 % rather than more,
because the very top of the proxy is dominated by high-membership function-like words
(*med*, *raven*, and above them *biti*, *in*, *kot*) whose lexicographic questions are less
meaningful. The counter-consideration for the low bands is that dictionary users look words
up *because* they are unfamiliar, so rare lemmas are not unrealistic — hence a tilt rather
than a reversal.

What the bands actually contain (random draws, proxy value in parentheses):

```
B0/B1  sprašujoče (2), oživčiti (0), črnuhinja (2), prašanje (2), kromač (2)
B1/B2  prašnica (12), polkrepko (8), navrtan (9), opismenjevalen (8), blaznica (10)
B2/B3  portugalsko (17), brezoviški (21), zeleneti (37), nesocialen (16), šivan (47)
B3/B4  videc (59), bavarski (168), precenjen (178), vzvišeno (103), vihtenje (83)
B4/B5  tašča (356), gradnik (522), izpiranje (351), lojalnost (472), licitacija (362)
B5     ventil (808), označevanje (761), kovanec (863), uradno (935), pridržati (651)
B5/B6  format (1395), prekršek (1844), preveriti (2080), izvajalec (2092), viseti (1838)
B6     med (39830), raven (6982), prodaja (6061), telefon (4283), dejstvo (3816)
```

**Why fixed thresholds rather than quantiles:**

1. **Version stability.** Quantile boundaries are relative to the pool; change the content
   filter or add a newer snapshot and p90 moves, so per-band accuracy stops being
   comparable across dataset versions. `257–1024 memberships` means the same thing forever.
2. **Interpretability.** An absolute band is a claim about the language; a percentile band
   is a claim about our filtering.
3. **Per-type stratification requires it.** Sub-pools differ enormously — 65,480 lemmas
   have collocations, 35,691 have synonyms, only **11,048** have direct usage examples. With
   quantiles, each sub-pool gets its own boundaries, so "band 5" would mean a different
   frequency range per question type and cross-type comparison would be meaningless.

**Why stratify at all.** The pool is Zipf-shaped, so uniform sampling puts ~80 % of items
in the four lowest bands — exactly the failure mode of the reference file. Beyond realism,
the proxy *is* a degree count, so it predicts subgraph size and therefore difficulty
(*oživčiti*, proxy 0, has a ~19-node ball; *med*, proxy 39,830, saturates the MWE cap).
Stratifying by frequency therefore stratifies by graph size, which lets results be reported
as a function of how much graph the model had to read — the most informative axis available
for a model whose thesis is inference-time structure reading.

**Implementation note.** The band × question-type matrix will have thin cells (a
"B0 × usage example" cell is nearly empty, since only 11,048 lemmas have direct examples and
those skew high). Compute the availability matrix before generating and either shrink that
cell's quota or borrow from the adjacent band, rather than silently producing a skewed set.
Per-type quotas therefore cannot all be identical.

---

## 6. Deferred to v2 — do not lose these

Everything in Section 4 is decided. The items below were **considered, judged worth doing,
and consciously postponed** to keep v1 small. They are not rejected.

### 6.1 The MWE question family — the largest unused capability in the graph

`mwe_structure`, `mwe_component_search`, `idiom_completion`. **The single best-supported
area of the KG that nothing currently uses:** 10,265,230 `decomp:constituent` edges, every
one carrying a `lexinfo:category` (noun 4.63 M, verb 1.90 M, preposition 1.49 M, adjective
1.09 M …), plus `rdf:_1 … _22` giving constituent order, plus MWE headwords that *do* store
their real surface phrase (`form-lexical-unit-8148598 → "divji brin"`) unlike collocations.

Deferred purely on implementation risk: this is the only family that depends on the D5
upward cap behaving, and it was not worth carrying that risk into a first 12 k-item set.
The owner confirmed the deferral independently (2026-08-19): idiom and multiword questions
are **not urgent, "probably nice to have"**.
Nothing else about it is problematic — the downward traversal it needs is bounded at 22
across the entire graph.

**When picked up:** validate the D5 cap empirically first (measure real ball sizes at
K = 15 for a sample of high-band lemmas), then generate. `mwe_component_search` answers
should stay non-exhaustive ("*Primeri so …*"), matching both the reference phrasing and
what a capped traversal can honestly support.

### 6.2 Verbalised collocations — ✅ **RESOLVED 2026-08-20, no external source needed**

> **This section is closed.** The premise below — that the phrase is not in the export and
> must be fetched from the DDDS API — was **wrong**. The phrase is in the export. v5 of the
> builder writes it, `kolokacija: mineralna voda` is now the node text, and no API, no bulk
> fetch and no credentials are required. Full write-up: `README.md` **Finding 8**.
>
> **The correction in one line:** a `frac:Collocation` node really is textless, but its IRI
> `dependent-sense-D-lexical-unit-H` embeds `D`, the sense of the multi-word entry that
> spells the pairing out — and *that* entry carries `canonicalForm → writtenRep`. The link
> exists only as a naming convention, never as a triple, and nothing in the dump points at a
> collocation node (0 object-position triples), so no traversal could ever have found it.
>
> **Measured:** 4,717,090 / 4,717,090 collocation nodes resolve, 0 failures, 3,744,473
> distinct phrases, 75.3 % of them differing from their lemmas concatenated. **91.5 % of the
> reference file's 1,307 collocation phrases come back verbatim**, which also answers the
> provenance question — that file was derived from this export, not from the API.
>
> Containment is satisfied without qualification: the phrase is on a node inside the
> extracted subgraph, not behind a service.
>
> **Consequence:** Group F of `QA_TASKS.md` is unparked, and the D14 lemma-pair rework is
> withdrawn — T17/T18 can ask for *besedne zveze* and the gold answer is a real phrase.
>
> ---
>
> **Correction, 2026-08-22 — the 91.5 % was a property of the resolution PATH, not of the
> store, and for a year of store versions the two were not the same number.** The v5 and v6
> stores reproduced only **82.8 %** (1,082 / 1,307). The missing 8.7 points were not
> unresolvable phrases; they were phrases the builder resolved correctly and then discarded,
> because the collocation dedup still keyed on the member *set* while several curated phrases
> share one. `README.md` **Finding 11** has the full account. **v7 fixes the key and the store
> now reproduces 1,196 / 1,307 = 91.5 %** — i.e. exactly what the path can supply, with the
> gap closed.
>
> The lesson worth keeping: *"the export contains it"* and *"the store contains it"* are
> different claims, and this section asserted the first while being read as the second.
> Measure containment on the artefact the model actually reads.

<details>
<summary>Superseded reasoning, kept for the record</summary>

v1 emits collocations as **lemma pairs** (`mineralen + voda`) because the inflected phrase
is not in the RDF export — a `frac:Collocation` node carries only `rdfs:member` and
`frac:head`, and there is no surface string on it anywhere in 42 GB.

**Lead (from the data owner, 2026-08-18):** the phrases came from the DDDS **API**, not the
export — endpoint `api/retrieve/lexical-unit-collocations/[lexeme_id]`. The key is a
`lexeme_id`, i.e. **the same id space as our graph nodes**, so the join is free.

This cannot be a runtime dependency: if the phrase lives behind an API and not in the
graph, the answer is not in the subgraph the model reads, which breaks the containment
invariant the design rests on. It has to be a **build-time bulk fetch** baked into node
text, turning `kolokacija: mineralen + voda` into `kolokacija: mineralna voda`. Bounded
scale — 65,480 core lemmas have collocations.

**To ask Luka (he knows this database):** is the endpoint reachable from our network and
with what credentials; what exactly does it return per lexeme; is there a bulk export
rather than 65 k individual calls.

*Why the lead was plausible but wrong: the id space really is shared, and the API really
does serve these phrases — but it serves them from the same underlying database that
produced this export, so the export had them all along.*
</details>

### 6.3 Other

- **A no-LLM extraction fallback.** If the external extractor of D3 is ever unavailable at
  serving time: tokenize → drop metalanguage stopwords (a closed list, since we author the
  templates) → look up the survivors in the reverse index and union whatever resolves. Same
  downstream path as D3, so nothing else changes. A backstop, not a plan.
- ~~**Clitics and negation as a question type.**~~ **Effectively dead — see the
  reachability correction in D18.** The `clitic yes` 423,503 figure counts
  `lexical-unit-part` subjects, i.e. MWE components, which the builder collapses
  away. Only **25** single-word entries carry a `clitic` value in the whole KG,
  so there is no pool to sample from. Negation is thinner still on the `yes`
  side (6,630) and is not rendered at all. Anything here would have to come from
  the MWE family (Section 6.1), not from word entries.
- **Reasoning-tier items** (aggregation, filtering, multi-hop) — D13.
- **CLASSLA-disambiguated form analysis** — would let `analiza_oblike_v_povedi` use
  ambiguous forms with a single correct reading, at the cost of a tagger dependency and
  the tagger's errors entering the gold labels (D15).
- **Sloleks/Gigafida frequency data** — the in-graph proxy is sufficient (D9); revisit only
  if the sampled vocabulary looks wrong on inspection.
- **Neighbourhood-disjoint splitting** — D11 requires only lemma-disjointness.

---

## 7. Residual open items

| # | item |
|---|---|
| **R1** | ~~Token budgets need a re-run.~~ **Closed 2026-08-20.** Both halves shipped. *Tokenizer:* rebuilt on Gemma 3 (job 129258), worth **−0.80 %** over the whole graph (913,315,688 vs 920,680,698 tokens; 45.97 % of nodes agree node for node), so pre-existing token figures are within ~1 % and were not restated. Every Gemma 3 size shares this tokenizer with GaMS3-12B — identical ids, not just identical counts — so one store serves the whole iteration ladder. *Text:* D18's morphology rendering shipped as the **v4** stores (`kg_graph_v4_gemma3`, `kg_graph_v4_gams2b`); it adds labels to 452,782 form nodes and 18,182 anchors. Token effect is recorded in `README.md` Finding 6. |
| **R2** | ~~**Verify the D15 intersection**~~ **Closed 2026-08-22.** Measured by generation itself: T20 is generatable, so no fallback to the disjunction is needed. The intersection is thin but non-empty — see the realised availability matrix in `datasets/generated/v1/report.json`. |
| **R3** | ~~**Compute the band × type availability matrix**~~ **Closed 2026-08-22.** Computed by *running every generator over every pool entry* rather than predicting eligibility, and recorded in `report.json`. Thin cells are shrunk and the shortfall redistributed across bands with headroom, per Section 5. |
| **R4** | **The extraction model of D3 is unspecified** — which model, where it runs, what it costs. Its misses are end-to-end service errors: if it does not return the target word, the right ball is never extracted and no amount of GTLM training recovers it. Pick the model, then measure its recall against our own templates (we author them, so the gold target word is known for free) before generating at scale. |
| **R5** | **Measure the natural multi-entity rate.** D3 makes the union share whatever the pipeline produces rather than a quota, so it has to be measured on real question strings, not assumed from the 88.9 % single-match figure — that figure is over *all* surface strings, and our templates sample lemmas, not forms. If the rate comes out near zero, the discrimination skill is untrained in practice and we oversample ambiguous seeds to compensate. *Now measurable for free: the generated questions exist and every one records the surface string it names.* |
| **R6** | **Reconcile the §3.1c power-law exponent.** The published slope (−1.418) is not reproduced by a v6 control run (−1.748) even though every other v6 figure in §3.1b is reproduced exactly, which points at a difference of fitted range or fitted quantity rather than of graph. Nothing downstream uses the exponent's value, so this is a provenance loose end; recorded rather than overwritten. |
| **R7** | **Flavour (d) negatives are not generated.** §0.2's fourth flavour — the relation present in the ball but only on a co-extracted distractor — is the one that cannot be built without the extracted ball, which is out of scope for the generation pass. It is the negative counterpart of D3's discrimination skill, so it should land with the ball builder, not later. |

---

## 8. How these numbers were measured

All measurements are parallel scans of `kg_raw/OntoLex DSB/` with `xargs -P 32` over awk,
each worker writing to its own output file (a shared stdout interleaves and corrupts ~0.3 %
of lines). A full-corpus predicate census takes ~30 s; a value census ~60 s.

Reproducing them is a matter of re-running those scans; the specific ones behind this
document are:

- predicate census (all files, count by `$2`)
- value census for `lexinfo:type/category/aspect/vform/clitic/negative/degree/definiteness/animate/partOfSpeech`, `vartrans:category`, `rdf:type`
- language-tag census for `writtenRep`, `skos:definition`, `rdf:value`
- `lexical-unit → canonicalForm` and `→ partOfSpeech` maps (400,180 each)
- `decomp:constituent` out-degree by subject, `decomp:correspondsTo` in-degree by object
- `lexicog:usageExample` count by sense, joined through `ontolex:sense` to entries
- `rdfs:member` count by sense (collocation membership), `skos:definition`/`lexinfo:synonym` by sense
- surface index: `canonicalForm`/`otherForm` joined to `writtenRep`, lowercased
- reference-file cross-checks: id resolution, lemma agreement, MSD decode vs KG features

Use the shared venv `/shared/workspace/povejmo/graph_model/.venv/bin/python`, not the local
`.venv` symlink (see [`README.md`](README.md)).