# Lexicographical QA dataset — design decisions

> **Status: design fully locked 2026-08-18; generation not yet started.** This document records the design of the
> question–answer dataset that pairs with the v3 GTLM input graph. It supersedes any
> assumption that the two reference QA files in `data/` will be used as training data
> directly. Companion documents: [`README.md`](README.md) for how the graph is built,
> and the root [`../README.md`](../README.md) for the model target.

All figures below were measured over the **full** raw KG (all 2,594 `.nt` files in
`kg_raw/OntoLex DSB/`) unless stated otherwise. Where a number contradicts an earlier
belief, the earlier belief is called out explicitly.

### v1 at a glance

| | |
|---|---|
| **Size** | ~10 k train (≤1 k dev) + ~2 k test |
| **Types** | 20 — the reference inventory reworked, plus antonyms |
| **Seeds** | 72,528 core lemmas passing a content filter, sampled across 7 fixed frequency bands |
| **Extraction** | uniform hop 2; `sestavina` downward always, upward capped at K = 15 |
| **Answers** | gradeable line first (`ODGOVOR: …`), then UI prose; no MSD tags |
| **Split** | lemma-disjoint; three held-out generalisation tiers, antonyms entirely unseen |
| **Deferred** | the MWE question family, verbalised collocations, the no-delimiter path — **Section 6** |

---

## 1. Why we generate our own items

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

**Two of its subtypes cannot be backed by this KG:**

- `kolokacije/*` — 100 % of ids resolve, yet the answers are not reproducible. A
  `frac:Collocation` node carries only `rdfs:member` (two *senses*) and `frac:head`; there
  is no surface string on it anywhere in 42 GB. We can produce the lemma pair
  *mineralen + voda*; the inflected "*mineralna voda*" is not stored. Corpus fragments
  like "*Potrpežljivost je vrlina*" (175/1,307 collocation items start uppercase) came
  from a corpus-extraction source outside this KG.
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

### 3.2 The surface-form reverse index

Built from `canonicalForm`/`otherForm` + `writtenRep`: **8,960,506 forms with a resolvable
owner, 3,148,457 distinct surface strings.**

| a surface string maps to | share |
|---|--:|
| exactly 1 lexical unit | **88.9 %** |
| ≤ 2 | 97.2 % |
| ≤ 3 | 99.1 % |
| worst case in the KG | 18 (*goli*, *meti*, *ti*, *peti*) |

Resolving a known token to a node is therefore a dict lookup, not a model. The hard part
is deciding *which* token is the target.

---

## 4. Locked design decisions

**D1 — Generate, don't relabel.** Fresh questions and answers generated from the graph.
The reference files supply the type inventory and phrasing style only; no reference rows
are reused.

**D2 — Every item is anchored to a `lexical-unit-N` id.** Same id space as the newer
reference file (verified 1,140/1,140). The persisted store keeps `(type << 56) | id` in
`node_codes.npy`, so graph node ↔ lexical unit is recoverable in both directions.

**D3 — Entity linking: delimiters required in v1.** Production marks the target word (UI
tap-to-select, or typed `»…«`); this converts a ranking problem into a lookup. Note that
**entity resolution is a pipeline heuristic, not something the model can learn** —
extraction happens before the model runs, so the model cannot help choose its own input.

The dataset carries **both delimited and plain phrasings, ~60/40**. This is free — the
graph and the answer are identical, only the question string differs — and it buys
robustness to how users actually type.

The no-delimiter production path is **deferred** (see Section 6). Its design, for the
record: tokenize → drop metalanguage stopwords (a closed list, since we author the
templates) → look up survivors in the reverse index, where 88.9 % of strings resolve to
exactly one lexical unit → if 2–3 candidates survive, extract the **union** of their balls
(p50 = 19 / p99 = 740 nodes, so this is affordable) and let the model select. That last
step is the only learned part, and it needs a training slice of multi-entity balls — which
is why it is deferred together with the path that would consume it. Union extraction is
also the only thing that handles cross-POS homonymy delimiters cannot resolve
(`»kot«` = angle / as).

**D4 — One extraction policy for all questions: hop 2, uniform.** Production accepts
arbitrary questions, so extraction must not branch on question type. Hop 2 is the floor
(synonyms, antonyms and collocations are reified nodes one hop past the sense) and also
the ceiling (those nodes carry both lemmas in their own text). Measured cost on word
seeds: p50 = 19, p90 = 76, p99 = 740, max = 1,039 nodes.

**D5 — `sestavina` traversal is directional and capped.** Downward (MWE → word) always,
being bounded at 22. Upward (word → MWE) capped at **K = 15**, deterministically ranked —
prefer MWEs whose senses carry a definition or example, then fewest constituents. A single
content-independent rule, so D4 still holds. This supersedes the earlier idea of capping
MWE *seeds* at hop 1, which is unnecessary once traversal is directional.

**D6 — Two-part answers.** A minimal, machine-gradeable line **first**, with a fixed
prefix on one line and no markdown (e.g. `ODGOVOR: tožilnik, ednina`), followed by a
user-facing formatted rendering for the UI. Minimal-part-first so the model commits before
elaborating and a truncated generation is still gradeable. Eval parses part 1 only; a
validation script checks part 2 does not contradict it. **No MSD tags** — morphology is
answered in words.

**D7 — A "not recorded" slice is included by construction.** Three flavours: the entity
does not exist; it exists but lacks the relation; it has the relation but not for the sense
asked about. The gradeable line gets a fixed sentinel (working form
`ODGOVOR: ni podatka v bazi`) so identification can be scored exactly. The UI part states
that no explicit information is recorded and then offers whatever *is* attached to the
entity — e.g. no definition, but here are its collocations. **The negatives must mostly be
ordinary lemmas**, not invented words; otherwise the model learns "weird-looking word →
say you don't know" and never learns to check the graph. The graph supplies these in bulk:
among core lemmas 59 % lack a definition, 65 % lack synonyms, 90 % lack a usage example.

**D8 — Seed pool: core entries passing a content filter.** Restrict to `id < 1 M` and
require at least one of {`@sl` definition, synonym, collocation, usage example}. Yields
**72,528 lemmas**. This is what keeps *garsid* out without needing frequency data.

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

**D13 — v1 is retrieval-focused**, matching the reference set. Reasoning-heavier items
(aggregation, filtering, multi-hop) are deferred; Tier B/C will be the only place reasoning
is probed in v1.

**D14 — v1 question-type inventory: the reference 20, reworked, plus antonyms.**

| | types |
|---|---|
| **Kept as-is (16)** | `sklanjanje/{celotna_sklanjatev, sklanjatev_po_stevilu, posamezen_sklon, osnovna_oblika_leme}` · `spreganje/{celotno_spreganje, spreganje_v_casu, neosebne_oblike}` · `besedna_vrsta/{osnovne_lastnosti, spol_samostalnika, vrsta_in_vid_glagola}` · `pomen/{razlaga_pomena, nastevanje_pomenov, stevilka_pomenov}` · `sopomenke/navedi_sopomenke` · `stopnjevanje/vse_stopnje` · `primeri_uporabe/povedi_z_besedo` |
| **Reworked (3)** | `sklanjanje/analiza_oblike` and `primeri_uporabe/analiza_oblike_v_povedi` — see D15 · `kolokacije/*` — **lemma pairs only**, questions rephrased so the gold answer is a pair, never an inflected phrase |
| **Added (1)** | **antonyms** — generated, but held out of training entirely as Tier C (D12) |

`stopnjevanje/vse_stopnje` is regenerated from `lexinfo:degree` rather than imitated: the
reference version is degenerate in 99/100 rows. `pomen/stevilka_pomenov` fixes the
Slovene number agreement that is wrong in 100/100 reference rows.

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

**D16 — Scale: ~10 k train (≤1 k held out as dev) + ~2 k test.** Deliberately modest;
regenerating more later is cheap. One consequence: at 2 k test over ~21 types over 7 bands,
a per-(type × band) cell holds ~12 items — too thin to read. Results are therefore reported
**per band** (~285 each) and **per type** (~95 each), but not as a cross-tab. The
difficulty-versus-graph-size curve is an aggregate result in v1.

**D17 — Slice sizes.** "Not recorded" (D7) ~10 % of items, of which ~70 % are ordinary
lemmas genuinely missing the relation and ~30 % nonexistent entities. Held-out tiers (D12):
Tier A ~5 %, Tier B ~3 %, Tier C = 100 % of antonym items. Sentinel wording:
**`ODGOVOR: ni podatka v bazi`**.

**D18 — The builder renders `aspect`, `vform` and `clitic` into node text.**
`FEATURE_PROPS` in `build_gtlm_graph_v3.py` currently covers only
case/number/gender/person/tense/mood/degree, so `besedna_vrsta/vrsta_in_vid_glagola` and
`spreganje/neosebne_oblike` would be ungeneratable — the facts are in the RDF
(`aspect` 1,735,245 · `vform` 2,353,283 · `clitic` 423,585) but not in what the model reads.
~12 min rebuild plus a re-save of the 3.8 GB store, combined with the tokenizer
re-measurement in R1 so it is one run, not two.

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
Nothing else about it is problematic — the downward traversal it needs is bounded at 22
across the entire graph.

**When picked up:** validate the D5 cap empirically first (measure real ball sizes at
K = 15 for a sample of high-band lemmas), then generate. `mwe_component_search` answers
should stay non-exhaustive ("*Primeri so …*"), matching both the reference phrasing and
what a capped traversal can honestly support.

### 6.2 Verbalised collocations

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

### 6.3 The no-delimiter production path

The lexicon-match fallback plus union extraction described in D3, together with the
training slice of **multi-entity balls** that its final step would need. Deferred as a
unit — building the training slice before the path that consumes it is work with no
consumer.

### 6.4 Other

- **Clitics and negation as a question type.** Dropped by the newer reference file, though
  the KG supports it (`clitic yes` 423,503 · `negative yes` 6,630). D18 renders `clitic`
  into node text anyway, so this becomes cheap once v1 ships — arguably the first thing to
  add back after the MWE family.
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
| **R1** | **Token budgets need a re-run.** The store's `token_len` used the `cjvt/GaMS-2B` tokenizer, while the model target moved to Gemma 3 / GaMS3-12B, *and* the D-locked builder change adds `aspect`/`vform`/`clitic` text to form nodes. Both shift the numbers; one rebuild settles both. All token figures in [`README.md`](README.md) are stale until then. |
| **R2** | **Verify the D15 intersection** — lemmas with a direct usage example whose form in that example is unambiguous. If too thin, fall back to the disjunction for `analiza_oblike_v_povedi` and record it. |
| **R3** | **Compute the band × type availability matrix** before generating, and shrink or borrow for thin cells (Section 5). |

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