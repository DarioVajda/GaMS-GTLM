# QA task types — specification for the generation pipeline

> **Status: drafted 2026-08-20. Scope: the 21 question types locked by D14 of
> [`QA_DATASET_DESIGN.md`](QA_DATASET_DESIGN.md).** That document decides *what* the dataset
> is; this one decides *how each individual question type gets generated*. It is written to
> be read on its own — every fact it relies on (graph predicate, node text convention, answer
> format, seed pool) is restated here rather than cross-referenced away. Companions:
> [`QA_DATASET_DESIGN.md`](QA_DATASET_DESIGN.md) for the design decisions and the
> full-corpus census, [`README.md`](README.md) for how the graph store is built.

For every type this document gives:

1. **What it is** — the task in one paragraph, and what the gold answer looks like.
2. **How to implement it** — which graph predicates and node kinds it reads, how the seed
   pool is filtered, how the gold answer is assembled, what the gradeable line is, and what
   the "not recorded" case looks like.
3. **Existing formulations** — every distinct question frame found in the two reference
   datasets, verbatim, with the row count behind it.
4. **Suggested new templates** — additional phrasings, written to widen the phrasing
   distribution along axes the reference files do not cover.

---

## 0. Conventions that apply to every type

Restated from `QA_DATASET_DESIGN.md` so that this file stands alone.

### 0.1 Answer format (D6)

Every answer is **two parts**:

```
ODGOVOR: <minimal, machine-gradeable, one line, no markdown>

<user-facing prose or table for the UI>
```

The minimal part comes **first** so that a truncated generation is still gradeable and so the
model commits before elaborating. Eval parses part 1 only; a validation script checks that
part 2 does not contradict part 1. **No MSD tags anywhere** — morphology is answered in
words (`tožilnik, ednina`, never `Sozet`).

### 0.2 The "not recorded" sentinel (D7, D17)

~10 % of items have no answer in the graph. The gradeable line is then exactly:

```
ODGOVOR: ni podatka v bazi
```

Four flavours, all generated: (a) the entity does not exist; (b) it exists but lacks the
relation; (c) it has the relation but not for the sense asked about; (d) **the relation is
present in the extracted ball but only on a co-extracted distractor unit**, where answering
from it would be wrong. Flavour (d) is what stops "the answer is somewhere in the input"
from being a safe bet. ~70 % of negatives must be **ordinary lemmas genuinely missing the
relation**, only ~30 % nonexistent entities — otherwise the model learns "weird-looking word
→ say you don't know" instead of learning to read the graph.

The UI part of a negative states that nothing is recorded and then offers whatever *is*
attached to the entity ("no definition is recorded, but here are its collocations").

### 0.3 Question surface (D3)

**No delimiters.** No `»…«`, no `"…"`, no `'…'` around the target word — the reference
files' quoted style is listed below as *phrasing* inspiration only, and every quote must be
stripped when a frame is reused. The production pipeline is: an external model extracts the
word(s) from the user's question → each extracted string is looked up verbatim in the
surface-form reverse index → the hop-2 ball of every exact match is extracted → the model
receives their **union**. Dataset extraction runs the identical pipeline, never privileged
knowledge of which lexical unit the generator started from.

Consequence binding on every template below: **the target word must appear in the question
as a verbatim surface string that exists in the reverse index** — a lemma for most types, an
inflected form for `osnovna_oblika_leme` / `analiza_oblike` / `analiza_oblike_v_povedi` (the
index covers `canonicalForm` *and* `otherForm`, so both resolve).

### 0.4 Seeds, bands, split

- **Seed pool (D8):** core entries, `lexical-unit` id < 1 M, requiring at least one of
  {`@sl` definition, synonym, collocation, usage example} → **72,528 lemmas**.
- **Bands (D9, D10):** `proxy(lemma) = collocation memberships + MWE memberships`, bucketed
  by fixed log-spaced thresholds — B0 = 0, B1 = 1–4, B2 = 5–16, B3 = 17–64, B4 = 65–256,
  B5 = 257–1024, B6 = 1025+. Sampling quota 5/10/15/20/20/20/10 %.
- **Split (D11):** lemma-disjoint. Partition lemmas first, generate inside each partition,
  then stratify so both sides share the band profile.
- **Held-out tiers (D12, D17):** Tier A = unseen *phrasings*, **2–3 templates per type
  withheld from training** (~5 % of items) — this is why each section below needs more
  templates than it strictly uses. Tier B = unseen *question type* over seen relations
  (~3 %). Tier C = **antonyms, held out entirely**.
- **Scale (D16):** ~10 k train (≤1 k dev) + ~2 k test → ~95 test items per type. Report per
  band and per type, never as a band × type cross-tab.

### 0.5 What the model actually reads

The extracted ball is a set of nodes whose **text** is self-describing. The generator must
only ever produce answers derivable from these strings:

```
iztočnica: bežnica (samostalnik, imenovalnik, ednina)     ← anchor: lemma + POS + morphology
iztočnica: popraskati (glagol, dovršni, nedoločnik)       ← verb anchor: POS + aspect + vform
iztočnica: medse (zaimek, naslonska oblika, tožilnik)     ← clitic (only 25 of these exist)
oblika: Afričanu (dajalnik, ednina)                       ← inflected form leaf
oblika: popraskam (sedanjik, 1. oseba, ednina)            ← finite verb form
oblika: popraskaj (velelnik, 2. oseba, ednina)            ← imperative
oblika: popraskal (deležnik na -l, ednina, moški spol)    ← -l participle
oblika: popraskat (namenilnik)                            ← supine
oblika: navrženi (imenovalnik, ednina, moški spol, osnovnik, določna oblika)
pomen 2: pes                                              ← sense, with dictionary ordinal
pomen 1: pes (zgled: Sosedov pes je spet lajal ...)       ← sense with no definition
zgled: Vsak konjenik je dobil simbolno darilo ...         ← usage example
kolokacija: boj + kriminaliteta                           ← lemma pair, NOT an inflected phrase
sopomenka: biblioteka ~ knjižnica
protipomenka: boj ~ sodelovanje
prevod (madžarsko): rüh kezelése                          ← @hun only; unused in v1
```

The parenthetical order is fixed: POS, then lexical-unit properties (`aspect`, `clitic`),
then form features in the order `vform, person, case, number, gender, degree, definiteness`.
A generator may rely on that order.

Edges are **untyped**; the relation type lives in the node text. Extraction is uniform
**hop 2** from every matched lexical unit (D4), with `sestavina` traversal directional:
downward (MWE → word) always, upward (word → MWE) capped at K = 15 (D5).

### 0.6 Feature vocabulary actually present in the RDF

Verified against `kg_raw/OntoLex DSB/` on 2026-08-20, and confirmed by the v4 build. This is
the complete set a generator may key on. **"Reachable" is the count on single-word entries** —
the corpus-wide totals are far larger but dominated by `lexical-unit-part` subjects (MWE
components), which the builder's `Component` collapse discards, so those values never appear
in any extracted ball.

| predicate | attaches to | values | reachable | rendered? |
|---|---|---|--:|:--:|
| `lexinfo:case` | word-form | nominative, genitive, dative, accusative, locative, instrumental | — | ✅ |
| `lexinfo:number` | word-form | singular, dual, plural | — | ✅ |
| `lexinfo:gender` | word-form | masculine, feminine, neuter | — | ✅ |
| `lexinfo:person` | word-form | **first, second, third** | 253,497 | ✅ v4 |
| `lexinfo:vform` | word-form | **present, participle, imperative, supine, infinitive** | 452,782 | ✅ v4 |
| `lexinfo:degree` | word-form | positive, comparative, superlative | — | ✅ |
| `lexinfo:definiteness` | word-form | yes, no | 160,524 | ✅ v4 |
| `lexinfo:partOfSpeech` | **lexical-unit** | noun, verb, adjective, adverb, pronoun, numeral, … | 400,180 | ✅ |
| `lexinfo:aspect` | **lexical-unit** | perfective, progressive, biaspectual | 18,157 | ✅ v4 |
| `lexinfo:clitic` | **lexical-unit** | bound, yes | **25** | ✅ v4 |
| `lexinfo:animate` | word-form | yes, no | — | ❌ |
| `lexinfo:negative` | lexical-unit | yes (6,630), no | — | ❌ |
| `lexinfo:synonym` / `antonym` | sense | — | 362,520 / 6,898 | ✅ |
| `skos:definition` | sense | `@sl` literal | 225,618 | ✅ |
| `lexicog:usageExample` | sense | literal | 71,729 | ✅ |

**There is no `lexinfo:tense` and no `lexinfo:mood` predicate in this KG.** Verb tense is
carried by `vform`, and only the present is stored synthetically — see §0.7.

### 0.7 The morphology blockers — fixed in v4, and the one that is not fixable

Three faults found while writing this document kept **all** verb morphology out of the node
text, silently disabling four of the 21 types. Two were builder bugs and are now fixed; the
third is a property of the source data and constrains the design permanently.

**(M1) `person` never reached the node text — fixed.** `FEATURE_PROPS` listed `person`, but
`VALUE_SL` mapped `firstPerson`/`secondPerson`/`thirdPerson` while the KG emits
`first`/`second`/`third`, and `feat_string()` drops unmapped values silently. 253,497 forms
affected.

**(M3) `aspect` and `clitic` were dropped by the subject-type guard — fixed.** They attach
to the **lexical-unit**, but the feature branch only accepted `T_WORDFORM`/`T_FORMLU`
subjects. They are now collected separately (`UNIT_PROPS`) and rendered into the anchor
parenthetical after the POS.

Both shipped in the **v4** stores (`kg_graph_v4_gemma3`, `kg_graph_v4_gams2b`, built
2026-08-20), together with `vform` and `definiteness`. What changed:

```
BEFORE (v3.1)                          AFTER (v4)
iztočnica: popraskati (glagol)      →  iztočnica: popraskati (glagol, dovršni, nedoločnik)
oblika: popraskat                   →  oblika: popraskat (namenilnik)
oblika: popraskal (ednina, m. spol) →  oblika: popraskal (deležnik na -l, ednina, moški spol)
oblika: popraskam (ednina)          →  oblika: popraskam (sedanjik, 1. oseba, ednina)
oblika: popraskata (dvojina)        →  oblika: popraskata (sedanjik, 2. oseba, dvojina)
oblika: popraskata (dvojina)        →  oblika: popraskata (sedanjik, 3. oseba, dvojina)
oblika: popraskaj (ednina)          →  oblika: popraskaj (velelnik, 2. oseba, ednina)
iztočnica: medse (zaimek, tožilnik) →  iztočnica: medse (zaimek, naslonska oblika, tožilnik)
oblika: navrženi (…, osnovnik)      →  oblika: navrženi (…, osnovnik, določna oblika)
```

Note rows 5 and 6: those two nodes were **byte-identical** before. Form nodes identical to a
sibling fell **86,848 → 418 (−99.5 %)**. Nominal strings are byte-identical to v3.1 by
construction — the `FEATURE_PROPS` order was chosen so, and `check_v4_text.py` asserts it.

**(M2) There is no `tense` and no `mood` predicate in this KG — not fixable.** The builder
listed both and neither ever fired. Verb tense is carried by `vform`: `present` is the only
synthetically stored tense, `imperative` the imperative, `participle` the *-l* participle
(gender + number), plus `infinitive` and `supine`. **Preteklik and prihodnjik are
periphrastic and are not stored anywhere** — the reference CSV's past/future tables
(`sem pritrgal`, `bom pritrgal`) are *composed*, not looked up. This is a permanent
constraint on T5 and T6; see those sections for the decision it forces.

All four types are now generatable: **T5, T6, T7, T10** — subject to M2's scope limit on
T5/T6.

---

## 1. The 21 types at a glance

| # | type | primary graph source | caveat |
|---|---|---|:--:|
| T1 | `sklanjanje/celotna_sklanjatev` | `oblika:` leaves, case × number | |
| T2 | `sklanjanje/sklanjatev_po_stevilu` | `oblika:` leaves, one number | |
| T3 | `sklanjanje/posamezen_sklon` | `oblika:` leaves, one cell | |
| T4 | `sklanjanje/osnovna_oblika_leme` | form → `iztočnica:` | |
| T5 | `spreganje/celotno_spreganje` | `vform=present` + person + number | M2 scope |
| T6 | `spreganje/spreganje_v_casu` | `vform=present` (+ participle) | M2 scope |
| T7 | `spreganje/neosebne_oblike` | `vform` ∈ {inf, supine, imperative} | |
| T8 | `besedna_vrsta/osnovne_lastnosti` | `partOfSpeech` on the anchor | |
| T9 | `besedna_vrsta/spol_samostalnika` | `gender` on forms | |
| T10 | `besedna_vrsta/vrsta_in_vid_glagola` | `aspect` on the anchor | |
| T11 | `stopnjevanje/vse_stopnje` | `degree` on forms | |
| T12 | `pomen/razlaga_pomena` | `pomen:` nodes with a definition | |
| T13 | `pomen/nastevanje_pomenov` | all `pomen:` nodes of the anchor | |
| T14 | `pomen/stevilka_pomenov` | count of `pomen:` nodes | |
| T15 | `sopomenke/navedi_sopomenke` | `sopomenka:` nodes | |
| T16 | `protipomenke/navedi_protipomenke` | `protipomenka:` nodes | Tier C |
| T17 | `kolokacije/navedi_kolokacije` | `kolokacija:` nodes | |
| T18 | `kolokacije/stevilsko_opredeljene_kolokacije` | `kolokacija:` nodes, counted | |
| T19 | `primeri_uporabe/povedi_z_besedo` | `zgled:` nodes | |
| T20 | `primeri_uporabe/analiza_oblike_v_povedi` | `zgled:` + unambiguous form | |
| T21 | `sklanjanje/analiza_oblike` | ambiguous form → disjunction | |

---

# Group A — `sklanjanje` (nominal inflection)

All five types in this group read the same structure: an anchor (`iztočnica: …`) with
`oblika: …` leaves hanging off it, each leaf's text carrying its own case and number.
`form_mode=expand` is mandatory for the whole group.

Two facts constrain every one of them:

- **Duplicate form nodes are normal.** *gora* has `word-form-2052`, `word-form-5218225` and
  `word-form-7736982` all reading `oblika: Gora (rodilnik, ednina)` — different lexical units
  contributing the same string. Deduplicate on (surface, case, number) before rendering.
- **Capitalisation is a heuristic.** The builder prefers the least-capitalised `writtenRep`
  variant; a proper noun that also lists a lowercase variant comes out lowercased. Do not
  build a question whose gold answer turns on capitalisation.

---

## T1 — `sklanjanje/celotna_sklanjatev`

### What it is

Give the **complete declension table** of a nominal (noun, adjective, or another declinable):
6 cases × 3 numbers = 18 cells. The gold answer is the full paradigm; missing cells are
rendered `/`.

### How to implement

- **Source:** every `oblika:` leaf of the anchor, keyed by (case, number). Group into an
  18-cell grid.
- **Seed filter:** anchor POS ∈ {noun, adjective, numeral, pronoun}; require ≥ 12 of 18 cells
  populated, otherwise the item degenerates into a mostly-`/` table. Log the cell-fill
  distribution before generating — this filter interacts with the band quota (low bands have
  thinner paradigms).
- **Gradeable line:** the 18 forms in fixed order, comma-separated, `/` for a gap:
  `ODGOVOR: gora, gore, gori, goro, gori, goro; gori, gora, gorama, gori, gorah, gorama; gore, gora, goram, gore, gorah, gorami`
  (ednina; dvojina; množina). Fixed order makes exact-match grading trivial.
- **UI part:** the markdown table the reference file uses. Prepositions (`pri gorah`,
  `z gorami`) are a **rendering convention added by the generator**, not graph data — keep
  them out of the gradeable line.
- **Negative:** an entry with no `oblika:` leaves at all (indeclinable, or a thin tail entry
  that slipped the filter).

### Existing formulations

Reference CSV, `sklanjanje/celotna_sklanjatev`, n = 100, 10 distinct frames:

| n | frame |
|--:|---|
| 13 | `Navedi vse sklone besede <L>.` |
| 13 | `Prikaži pregled sklanjatvenih oblik besede <L>.` |
| 11 | `Izpiši vse sklone in števila za besedo <L>.` |
| 11 | `Sklanjaj besedo <L> v ednini, dvojini in množini.` |
| 11 | `Kako se sklanja beseda <L>?` |
| 10 | `Izpiši sklanjatveno tabelo samostalnika <L>.` |
| 10 | `Prikaži celotno sklanjatev za besedo <L>.` |
| 8 | `Kako glasi celotna sklanjatev besede <L>?` |
| 7 | `Sestavi sklanjatveno tabelo za besedo <L>.` |
| 6 | `Izpiši celotno sklanjatev za samostalnik <L>.` |

### Suggested new templates

Adds: elliptical, polite/indirect, first-person-need, and a "why am I asking" frame.

```
Sklanjaj besedo <L> po vseh sklonih in številih.
Zanima me celotna sklanjatev besede <L>.
Rabim vse sklanjatvene oblike besede <L>.
Kako gre sklanjatev besede <L>?
Sklanjatev besede <L>, prosim.
Ali mi lahko izpišeš vse sklone besede <L>?
Potrebujem sklanjatveni vzorec za <L>.
Pišem besedilo in ne vem, kako se sklanja <L>. Lahko pomagaš?
Prikaži, kako se beseda <L> spreminja po sklonih.
Katere sklanjatvene oblike ima <L>?
```

---

## T2 — `sklanjanje/sklanjatev_po_stevilu`

### What it is

The same table as T1 but **restricted to one number** — ednina, dvojina or množina. 6 cells.

### How to implement

- **Source:** identical to T1, filtered to one `number` value. The number is a template slot,
  sampled uniformly over {singular, dual, plural} so dvojina is not underrepresented (the
  reference file is roughly balanced here — keep that).
- **Seed filter:** require ≥ 5 of the 6 cells for the chosen number.
- **Gradeable line:** six forms in case order —
  `ODGOVOR: ednina: gora, gore, gori, goro, gori, goro`.
- **Negative:** a **pluralia tantum** or singular-only entry asked about the missing number
  is the highest-value negative in this group — it is an ordinary lemma, and the answer is
  genuinely absent rather than invented.

### Existing formulations

Reference CSV, n = 100, 28 rows because the number word varies inside each frame. The 9
frames, with `<Š>` = ednini / dvojini / množini:

| n | frame |
|--:|---|
| 13 | `Izpiši vse sklanjatvene oblike besede <L> v <Š>.` |
| 12 | `Sklanjaj besedo <L> samo v <Š>.` |
| 11 | `Katere sklone ima beseda <L> v <Š>?` |
| 11 | `Prikaži tabelo sklanjatve besede <L> v <Š>.` |
| 11 | `Kako se beseda <L> sklanja v <Š>?` |
| 11 | `Kako poteka sklanjatev besede <L> za <Š>?` |
| 10 | `Izpiši <Š> besede <L> po sklonih.` |
| 8 | `Prikaži sklanjatev besede <L> za <Š>.` |
| 8 | `Prikaži sklanjatvene oblike v <Š> za besedo <L>.` |
| 5 | `Navedi vse sklone za <Š> besede <L>.` |

### Suggested new templates

Adds: the dual asked about semantically rather than by name (a real user behaviour), and
telegraphic forms.

```
Potrebujem samo množinske oblike besede <L>.
Kako se <L> sklanja, kadar govorimo o dveh?
Zanima me sklanjatev besede <L> v ednini.
Sklanjatev besede <L>, samo dvojina.
Katere oblike ima <L> v množini?
Izpiši <L> po sklonih, v ednini.
Ali ima beseda <L> posebne oblike v dvojini?
Rabim množino besede <L> po vseh sklonih.
Prikaži samo ednino besede <L>.
Kako bi <L> sklanjal v dvojini?
```

---

## T3 — `sklanjanje/posamezen_sklon`

### What it is

**One cell** of the paradigm: a named case in a named number. The gold answer is a single
surface form.

### How to implement

- **Source:** one `oblika:` leaf, selected by (case, number). Both are template slots.
- **Sampling:** sample the (case, number) pair **uniformly over the 18 cells**, not from the
  reference file's distribution — the CSV over-samples nominative and singular. This is the
  cheapest robustness win in the group.
- **Gradeable line:** `ODGOVOR: gori` — the bare form, nothing else.
- **UI part:** `Mestnik ednine besede **gora** se glasi: **pri gori**.` — note the
  preposition appears only in the UI part, per T1.
- **Ambiguity note:** several cells can share a surface (`gori` = dative sg and locative sg).
  That is harmless *in this direction* — the question fixes the cell and the answer is the
  form. The reverse direction is T21.
- **Negative:** ask for a cell the entry does not have (see T2's pluralia tantum).

### Existing formulations

Reference CSV, n = 100. Frames, with `<S>` = sklon name, `<Š>` = number, `<N>` = the case's
ordinal number:

| n | frame |
|--:|---|
| ~14 | `Izpiši <N>. <S> sklon <Š> besede <L>.` |
| ~10 | `Navedi <N>. <S> sklon za <Š> besede <L>.` |
| ~9 | `V kakšni obliki je <S> <Š> pri besedi <L>?` |
| ~9 | `Kako se glasi <S> <Š> besede <L>?` |
| ~8 | `Katero obliko ima <S> <Š> samostalnika <L>?` |
| ~5 | `Kako se v slovenščini glasi <S> <Š> besede <L>?` |
| ~4 | `Kako zapišemo <S> <Š> besede <L>?` |
| ~4 | `Izpiši obliko: <S> <Š> za <L>.` |
| ~3 | `Kateri je <S> <Š> besede <L>?` |
| ~3 | `Katera je ustrezna oblika za <N>. <S> sklon <Š> besede <L>?` |

The `<N>. <S>` frames ("`4. Tožilnik sklon`") are a generator artefact — half-numeral,
half-name, and not how anyone writes. Keep one such template for continuity and drop the
rest.

### Suggested new templates

```
Kako se reče <L> v dajalniku ednine?
Rabim rodilnik množine od <L>.
Kaj je tožilnik dvojine besede <L>?
Zapiši besedo <L> v orodniku množine.
Kako bi besedo <L> postavil v mestnik ednine?
V rodilniku ednine — kako se glasi <L>?
Orodnik ednine besede <L>?
Kako se glasi beseda <L>, če jo dam v dajalnik množine?
Ali mi lahko poveš tožilnik ednine besede <L>?
Napiši <L> v rodilniku dvojine.
```

---

## T4 — `sklanjanje/osnovna_oblika_leme`

### What it is

**Lemmatisation.** Given an inflected surface form, name the dictionary headword it belongs
to. The gold answer is the lemma, ideally with its POS.

### How to implement

- **Source:** pick a form leaf, ask about its surface, answer with the anchor's lemma.
- **The target word in the question is the FORM, not the lemma.** This is the one place in
  the group where the reverse index is exercised on `otherForm` rather than `canonicalForm`.
- **Ambiguity is the design problem here.** 11.1 % of surface strings own more than one
  lexical unit, worst case 18 (*goli, meti, ti, peti*). Two policies, both wanted:
  - **majority items** — sample forms that resolve to exactly one unit; answer is one lemma;
  - **a deliberate ambiguous slice** — sample forms resolving to 2–3 units and make the gold
    answer name **all** of them (`ODGOVOR: goli, gol`). This is the cheapest source of the
    D3 discrimination supervision and of the D7 flavour-(d) negative, and it is a direct
    probe of R5 (the natural multi-entity rate). Target ~15–20 % of this type.
- **Gradeable line:** `ODGOVOR: gora` — or the comma-separated set for the ambiguous slice.
- **UI part:** lemma + POS + which cell the given form is, which links this type to T21.
- **Negative:** a surface string that is not in the index at all.

### Existing formulations

Reference CSV, n = 100, `<F>` = the inflected form:

| n | frame |
|--:|---|
| 15 | `Katera beseda v osnovni obliki predstavlja obliko <F>?` |
| 12 | `Navedi imenovalniško obliko besede <F>.` |
| 12 | `Kakšna je osnovna oblika besede <F>?` |
| 10 | `Navedi lemo za besedno obliko <F>.` |
| 10 | `V kateri lemi se nahaja oblika <F>?` |
| 9 | `Katera je lema besede <F>?` |
| 8 | `Kateri je imenovalnik ednine za obliko <F>?` |
| 8 | `Iz katere osnovne oblike izvira oblika <F>?` |
| 8 | `Zapiši lemo (osnovno obliko) za besedo <F>.` |
| 7 | `Poišči osnovno imeno obliko za besedo <F>.` (sic — typo in the source) |

Older JSON file, `oblikoslovna/lemma_lookup`, n = 104 — noticeably more natural phrasing:

```
Pod katero slovarsko iztočnico spada oblika drugi?
Katera je osnovna slovarska oblika zaimka vaša?
Kaj je lema oblike dolgemu?
Povej mi osnovno obliko samostalnika konji.
Ali lahko določiš lemo za cerkvama?
Če iščem obliko novem, katera je njena lema?
Iz katere leme je tvorjena oblika razumeli?
Na katero osnovno obliko se reducira oblika vsemi?
Pod katero slovarsko obliko najdem besedo svežega?
Kateri glagol stoji za obliko začel?
```

### Suggested new templates

```
Od katere besede je oblika <F>?
Naletel sem na obliko <F>. Katera je njena osnovna oblika?
Pod katerim geslom najdem besedo <F>?
Kaj je slovarska oblika besede <F>?
Iščem lemo oblike <F>.
Katero geslo naj iščem, če imam obliko <F>?
Beseda <F> — katera je njena osnovna oblika?
Kako se glasi <F> v imenovalniku ednine?
Ali je <F> oblika kakšne druge besede? Katere?
Ne najdem besede <F> v slovarju. Kaj je njena osnovna oblika?
```

---

## T21 — `sklanjanje/analiza_oblike` *(reworked — D15)*

Listed here rather than at the end because it belongs to Group A.

### What it is

The **inverse of T3**: given a surface form (and its lemma, so the question is well-posed),
name the case and number it expresses. **Reworked by D15 to take specifically the
*ambiguous* forms and answer with the disjunction** — `ODGOVOR: tožilnik ali orodnik,
ednina`. The graph lists every reading a form can carry and cannot pick between them; with
no sentence there is nothing to pick with, so the disjunction *is* the correct answer.

This is the fix for the reference file's worst measured failure: on the cleanest ambiguity
class (*haljico, medicino, uniformo, marihuano, teorijo*) it guesses instrumental where
context forces accusative, **5/5 wrong**.

### How to implement

- **Source:** all `oblika:` leaves of the anchor sharing the same surface string. Collect the
  set of (case, number) pairs; that set is the answer.
- **Seed filter:** *require* |set| ≥ 2. Unambiguous forms belong to T20 (with a sentence);
  keep a minority (~25 %) of unambiguous ones here too so the model does not learn "always
  answer with a disjunction".
- **Gradeable line:** the readings, canonically ordered (case order, then number order),
  joined with `ali`: `ODGOVOR: tožilnik ali orodnik, ednina`. When the readings differ in
  number too, list full pairs: `ODGOVOR: rodilnik ednine ali imenovalnik množine`.
- **Grading:** exact set match after canonical ordering — the ordering rule must be fixed in
  the generator, not left to the renderer.
- **Negative:** a form string that does not belong to the named lemma at all.

### Existing formulations

Reference CSV, n = 100, `<F>` = form, `<L>` = lemma:

| n | frame |
|--:|---|
| 17 | `Razloži, v katerem sklonu in številu je oblika <F> besede <L>.` |
| 17 | `Katere slovnične lastnosti nosi oblika <F> besede <L>?` |
| 9 | `Določi sklon in število za obliko <F> (lema <L>).` |
| 9 | `Navedi sklon in število, ki ga izraža <F> pri besedi <L>.` |
| 9 | `Kateri sklon in število izraža oblika <F> pri besedi <L>?` |
| 9 | `Katero sklanjatveno obliko predstavlja beseda <F> pri lemi <L>?` |
| 9 | `Slovnično analiziraj obliko <F> besede <L>.` |
| 8 | `Opiši oblikoslovno vlogo besede <F> pri samostalniku <L>.` |
| 6 | `Katero slovnično funkcijo ima <F> v sklanjatvenem vzorcu <L>?` |
| 2 | `Kaj pomeni oblika <F> pri besedi <L>?` |

Older JSON file, `oblikoslovna/paradigm_identification`, n = 100 — and note that this file
handles ambiguity *better*, in at least one case naming it outright
("*orodnik ednine moškega ali srednjega spola*"):

```
Kakšna oblika je drugim?
Pri obliki avtomobili me zanima, ali gre za ednino ali množino in kateri sklon.
Določi sklon, število in spol pri obliki vsakemu.
Katera oseba in število sta izražena v obliki upaš?
Kaj označuje oblika sosedom z vidika sklona in števila?
Prepoznaj slovnične lastnosti oblike hotela.
Kateri sklon skriva oblika njegovemu v zvezi njegovemu bratu?
Kaj lahko poveš o obliki peti: je to števnik, pridevnik ali glagol?
```

### Suggested new templates

```
Kaj mi oblika <F> pove o sklonu in številu? Lema je <L>.
Oblika <F> pri besedi <L> — kateri sklon in katero število?
V katerem sklonu je <F>, če je osnovna oblika <L>?
Analiziraj obliko <F> pri lemi <L>.
Ali je <F> pri besedi <L> lahko več kot en sklon?
Katere sklone lahko izraža oblika <F> besede <L>?
Sem naletel na <F>. Pri lemi <L>, kaj je to po sklonu in številu?
Določi slovnične lastnosti oblike <F> (<L>).
Kaj vse je lahko <F> pri besedi <L>?
Je <F> pri lemi <L> ednina ali množina, in kateri sklon?
```

The last two are the ones worth withholding as **Tier A** for this type: they ask the same
question in a shape the training templates never take.

---

# Group B — `spreganje` (verb conjugation)

**Generatable from the v4 stores, and not before them** (§0.7). Until v4, `person` was
silently dropped and `vform` was never rendered, so a verb's whole paradigm collapsed to a
handful of strings. Read §0.7 before implementing any of these three — in particular **M2,
which no rebuild can fix**: preteklik and prihodnjik are periphrastic and are not stored.

What the KG actually stores for a verb:

| what | how it is stored |
|---|---|
| sedanjik | `vform=present` + `person` + `number` — 9 cells (3 persons × 3 numbers) |
| velelnik | `vform=imperative` + `person` + `number` |
| nedoločnik | `vform=infinitive` |
| namenilnik | `vform=supine` |
| deležnik na *-l* | `vform=participle` + `gender` + `number` |
| preteklik, prihodnjik, pogojnik | **not stored** — periphrastic |
| vid | `aspect` on the lexical-unit (perfective / progressive / biaspectual) |

---

## T5 — `spreganje/celotno_spreganje`

### What it is

The **full conjugation** of a verb: all persons × numbers, across the tenses the reference
answers show as a stacked set of markdown tables (sedanjik, preteklik, prihodnjik).

### How to implement

- **Source:** the 9 present-tense cells (`vform=present` × person × number) are looked up.
- **M2 decision required.** Past and future are not in the graph. Three options, in order of
  preference:
  1. **Restrict the gold answer to what is stored** — sedanjik, velelnik, nedoločnik,
     namenilnik, deležnik. Honest, fully graph-derivable, and consistent with the
     containment invariant the whole design rests on ("the answer is in the subgraph").
     **Recommended.**
  2. Compose preteklik/prihodnjik from the *-l* participle plus the auxiliary *biti* by a
     deterministic generator rule. Correct Slovene, but the auxiliary is generator knowledge,
     not graph content — the model would be supervised to produce something it cannot read.
  3. Reproduce the reference file's three-tense table. Rejected for the same reason as (2),
     with no added benefit.
  Whichever is chosen must be recorded here, since it changes what the type measures.
- **Seed filter:** POS = verb; require all 9 present cells present.
- **Gradeable line:** 9 forms in fixed order (1sg, 2sg, 3sg, 1du, 2du, 3du, 1pl, 2pl, 3pl):
  `ODGOVOR: delam, delaš, dela, delava, delata, delata, delamo, delate, delajo`.
- **Negative:** a defective verb, or a non-verb seed asked to be conjugated (an excellent
  ordinary-lemma negative — *gora* has no conjugation, and saying so is the correct answer).

### Existing formulations

Reference CSV, n = 100:

| n | frame |
|--:|---|
| 15 | `Prikaži pregled spreganja za glagol <L>.` |
| 14 | `Navedi vse osebe in čase za glagol <L>.` |
| 13 | `Kako glasi spreganje glagola <L> v sedanjiku, pretekliku in prihodniku?` |
| 12 | `Izpiši vse glagolske oblike za <L>.` |
| 11 | `Spregaj glagol <L>.` |
| 11 | `Kako se sprega glagol <L>?` |
| 8 | `Sestavi spreganje glagola <L> po časih in osebah.` |
| 7 | `Prikaži celotno spreganje glagola <L>.` |
| 6 | `Prikaži spreganje po osebah za glagol <L>.` |
| 3 | `Izpiši celotno spreganje glagola <L>.` |

Note frames 2 and 3 explicitly promise tenses the graph does not hold. **Under option (1)
they must be reworded** — a question that asks for preteklik and gets an answer without it
is a training signal to ignore the question.

### Suggested new templates

Written to be answerable under option (1):

```
Spregaj glagol <L> po vseh osebah.
Zanima me, kako se sprega <L>.
Rabim vse osebne oblike glagola <L>.
Kako gre spreganje glagola <L>?
Izpiši tabelo spreganja za <L>.
Prikaži, kako se <L> spreminja po osebah in številu.
Ali mi lahko spregaš glagol <L>?
Katere oblike ima glagol <L> po osebah?
Spreganje glagola <L>, prosim.
Potrebujem osebne oblike glagola <L> za vsa tri števila.
```

---

## T6 — `spreganje/spreganje_v_casu`

### What it is

Conjugation **restricted to one tense**, by person and number. 9 cells.

### How to implement

- **Source:** `vform=present` × person × number.
- **The reference file asks about sedanjik in 100/100 rows.** That is a robustness hole, but
  under M2 it is also the only tense the graph holds. Under option (1) of T5, this type is
  effectively "sedanjik only" and should be **renamed in the item metadata to say so**, or
  extended to cover the *-l* participle by gender ("kako se glasi deležnik na -l v ženskem
  spolu ednine?") which *is* stored and gives the type a second, genuinely graph-backed
  dimension. The older JSON file already asks exactly that.
- **Gradeable line:** as T5's, prefixed with the tense: `ODGOVOR: sedanjik: delam, delaš, …`.
- **Negative:** ask for preteklik under option (1) → `ODGOVOR: ni podatka v bazi`, with the
  UI part explaining that the past is formed periphrastically and offering the participle.
  This is a **good** negative: truthful, and it teaches the boundary of the resource.

### Existing formulations

Reference CSV, n = 100 — every row instantiates `<Č>` = Sedanjik:

| n | frame |
|--:|---|
| 17 | `Kako se po osebah sprega glagol <L> v <č>?` |
| 15 | `Izpiši <Č> glagola <L> po osebah.` |
| 11 | `Kako se glasi <Č> glagola <L>?` |
| 11 | `Navedi spreganje glagola <L> v <č>.` |
| 11 | `Spregaj glagol <L> v <č>.` |
| 10 | `Prikaži oblike za <Č> glagola <L>.` |
| 7 | `Sestavi tabelo za <Č> glagola <L>.` |
| 7 | `Izpiši oblike po osebah in številu za <Č> glagola <L>.` |
| 7 | `Katere so vse osebe in oblike za <Č> glagola <L>?` |
| 4 | `Prikaži spreganje v <č> za glagol <L>.` |

Older JSON file, `oblikoslovna/paradigm_generation`, n = 104 — single-cell rather than
whole-tense, and covers the participle:

```
Kakšna je oblika glagola ustvariti v 2. osebi množine sedanjika?
Kaj je srednjespolska dvojinska oblika deležnika na -l od glagola oditi?
Kako se glasi 3. oseba ednine sedanjika glagola oditi?
Kako se glasi velelnik v 2. osebi ednine za glagol motiti?
Kako bi zapisali 2. osebo ednine sedanjika glagola dati?
Izpiši namenilnik glagola prejeti.
```

### Suggested new templates

```
Kako se glagol <L> sprega v sedanjiku?
Rabim sedanjik glagola <L> po osebah.
Izpiši sedanjiške oblike glagola <L>.
Kako rečem <L> v 1. osebi množine?
Kaj je 3. oseba ednine sedanjika glagola <L>?
Kako se glasi deležnik na -l glagola <L> v ženskem spolu ednine?
Katere oblike ima <L> v sedanjiku?
Spregaj <L> v sedanjiku, vsa tri števila.
Kako bi rekel <L> v 2. osebi dvojine?
Sedanjik glagola <L>, prosim.
```

---

## T7 — `spreganje/neosebne_oblike`

### What it is

The **non-finite and special forms** of a verb: nedoločnik (infinitive), namenilnik (supine),
velelnik (imperative), and — worth adding — the *-l* participle.

### How to implement

- **Source:** `vform` ∈ {infinitive, supine, imperative}, plus `participle` if included.
  Imperative forms carry person + number, so the imperative is itself a small paradigm
  (`popraskajva` = 1st person dual imperative).
- **Coverage caution:** the corpus census gives `vform` 2,353,283 total with infinitive
  1,580,662 and supine only **20,878**. A question that asks for the namenilnik will very
  often have no answer — either filter for its presence or let it feed the negative slice
  deliberately. The reference answers show exactly this shape (nedoločnik + namenilnik, two
  bullet points, nothing else).
- **Gradeable line:** `ODGOVOR: nedoločnik: zaripniti; namenilnik: zaripnit; velelnik: zaripni`
  — labelled slots, so partial credit is possible and a missing slot is visible.
- **Negative:** a verb with no supine recorded.

### Existing formulations

Reference CSV, n = 100:

| n | frame |
|--:|---|
| 13 | `Kako se glasi velelnik glagola <L>?` |
| 13 | `Kakšna sta nedoločnik in namenilnik glagola <L>?` |
| 11 | `Izpiši neosebne oblike glagola <L>.` |
| 11 | `Kako se glasijo neosebne oblike pri glagolu <L>?` |
| 11 | `Navedi nedoločnik in velelniške oblike za glagol <L>.` |
| 10 | `Katere so neosebne ter posebne oblike glagola <L>?` |
| 10 | `Izpiši velelnik ter nedoločnik glagola <L>.` |
| 9 | `Prikaži nedoločniški in namenilniški obliki glagola <L>.` |
| 7 | `Opiši posebne glagolske oblike za <L>.` |
| 5 | `Navedi nedoločnik, namenilnik in velelnik za <L>.` |

### Suggested new templates

```
Kateri sta nedoločnik in namenilnik glagola <L>?
Kako se glasi namenilnik glagola <L>?
Rabim velelnik glagola <L> v 2. osebi ednine.
Katere neosebne oblike ima <L>?
Izpiši nedoločnik in deležnik na -l glagola <L>.
Kako komu rečem, naj <L>? Zanima me velelnik.
Ali ima <L> namenilnik? Kako se glasi?
Zapiši velelniške oblike glagola <L>.
Nedoločnik glagola <L>?
Katere so posebne oblike glagola <L>, poleg osebnih?
```

---

# Group C — `besedna_vrsta` (word class and grammatical properties)

---

## T8 — `besedna_vrsta/osnovne_lastnosti`

### What it is

**Word class plus its basic grammatical properties.** For a noun: POS + gender (+ proper-noun
flag). For a verb: POS + aspect. For an adjective: POS (+ definiteness where it matters).

### How to implement

- **Source:** `lexinfo:partOfSpeech` on the **anchor** (400,180 entries — single-word entries
  only; MWEs carry no POS at all), plus whatever the anchor's morphology parenthetical
  already renders. The anchor text is `iztočnica: bežnica (samostalnik, imenovalnik, ednina)`
  — but note only **9.2 %** of all anchors carry that parenthetical, because 91 % of anchors
  are MWEs. Restricted to word anchors it is **99.9 %**.
- **Seed filter:** word anchors only (this is automatic under D8's core-entry pool).
- **Gradeable line:** `ODGOVOR: samostalnik, moški spol` — POS first, then the properties
  that exist for that POS, in a fixed per-POS order.
- **`properNoun` is a separate POS value in this KG**, and the reference answers surface it
  as a second sentence ("Beseda je lastno ime in je moškega spola"). Decide once whether
  `lastno ime` is the POS or a flag on `samostalnik`, and apply it uniformly.
- **Negative:** an MWE seed (no POS recorded anywhere) — again a truthful negative that
  teaches a real boundary of the resource.

### Existing formulations

Reference CSV, n = 100:

| n | frame |
|--:|---|
| 15 | `Kako slovnično opredelimo besedo <L>?` |
| 12 | `V katero vrsto besed uvrščamo besedo <L>?` |
| 12 | `Slovnično opredeli besedo <L>.` |
| 11 | `Katero besedno vrsto predstavlja <L>?` |
| 11 | `Kaj po besedni vrsti pomeni <L>?` |
| 8 | `Kakšne so slovnične lastnosti besede <L>?` |
| 8 | `Opiši osnovne gramatične lastnosti besede <L>.` |
| 8 | `Navedi slovnične značilnosti za lemo <L>.` |
| 8 | `Kateri besedni vrsti pripada beseda <L>?` |
| 7 | `Ali je beseda <L> samostalnik ali druga besedna vrsta?` |

Older JSON file, `oblikoslovna/part_of_speech`, n = 104 — includes **sentence-embedded**
targets, which the newer file dropped entirely and which D3 makes relevant again:

```
Kakšna besedna vrsta je bogat?
Kam po besednih vrstah spada nekaj?
Besedo prijatelj bi rad besednovrstno razvrstil. Kaj je?
V povedi Voda hitro izhlapeva: katera besedna vrsta je beseda voda?
V stavku Moj oče čaka doma: katera besedna vrsta je oče?
Je beseda grd v stavku Pes je grd pridevnik?
Beseda visoko me zanima z vidika besedne vrste — kaj je to?
Rad bi samo besednovrstno oznako za jezik.
Kateri vrsti besed pripada jesenski v primerjavi z jesen?
```

### Suggested new templates

The sentence-embedded frames are worth reviving here: they exercise the D3 extractor on a
question that is not a bare template, and R5 says the multi-entity rate has to be measured on
real strings.

```
Kaj je <L> po besedni vrsti?
Katera besedna vrsta je <L>?
Zanima me, v katero besedno vrsto spada <L>.
Rabim samo besednovrstno oznako za <L>.
Ali je <L> samostalnik?
V povedi <S> me zanima besedna vrsta besede <L>.
Kako bi besedo <L> slovnično opredelil?
Povej mi osnovne slovnične lastnosti besede <L>.
Je <L> pridevnik ali prislov?
Kaj vse mi lahko poveš o besedi <L> z vidika slovnice?
```

---

## T9 — `besedna_vrsta/spol_samostalnika`

### What it is

**Grammatical gender of a noun** — moški, ženski or srednji spol. One word of answer.

### How to implement

- **Source:** `lexinfo:gender` on the anchor's forms. Take the gender of the nominative
  singular form; verify it is constant across the paradigm and skip (or flag) the entry if
  not.
- **Seed filter:** POS = noun.
- **Gradeable line:** `ODGOVOR: moški spol`.
- **Ambiguity:** some nouns are recorded with two genders. Answer with both
  (`ODGOVOR: moški ali ženski spol`) rather than picking — same principle as T21.
- **Negative:** ask the gender of a verb or adverb. This is the single easiest ordinary-lemma
  negative in the whole inventory and should carry a good share of this type's negative
  quota.

### Existing formulations

Reference CSV, n = 100:

| n | frame |
|--:|---|
| 16 | `Ali je samostalnik <L> moškega, ženskega ali srednjega spola?` |
| 13 | `Navedi spol za samostalnik <L>.` |
| 12 | `Kateri spol ima beseda <L>?` |
| 12 | `Kakšnega spola je samostalnik <L>?` |
| 11 | `Določi spol samostalnika <L>.` |
| 11 | `Povej, kakšnega spola je beseda <L>.` |
| 10 | `Ali spada <L> med samostalnike moškega, ženskega ali srednjega spola?` |
| 7 | `Kateri slovnični spol ima beseda <L>?` |
| 5 | `V kateri spol razvrstimo samostalnik <L>?` |
| 3 | `Kateri slovnični spol pripada besedi <L>?` |

### Suggested new templates

```
Kakšnega spola je <L>?
Je <L> moškega ali ženskega spola?
Zanima me spol besede <L>.
Rečemo ta <L> ali ta <L>a? Zanima me spol. [only where the pair is safe to render]
Spol samostalnika <L>?
Ali je <L> srednjega spola?
Katerega spola je beseda <L>, ko jo uporabim v stavku?
Povej spol za besedo <L>.
Pri besedi <L> — moški, ženski ali srednji spol?
Ne vem, ali je <L> moškega ali ženskega spola. Kaj je pravilno?
```

*(The 4th template needs a generator-side safety check; drop it if the surface pair cannot be
built without inventing a form.)*

---

## T10 — `besedna_vrsta/vrsta_in_vid_glagola`

### What it is

**Verb aspect** — dovršni, nedovršni or dvovidski — and the verb's type. The reference
answers are a single sentence ("Beseda **pregibati** je dvovidski glagol").

### How to implement

- **Source:** `aspect` on the anchor ∈ {perfective → dovršni, progressive → nedovršni,
  biaspectual → dvovidski}, rendered since v4 as `iztočnica: zgrinjati (glagol, nedovršni,
  nedoločnik)`. **18,157 reachable verb units, and every single-word verb in the KG has an
  aspect** — so coverage for this type is 100 % of its pool, the best of any type in the
  inventory. (The 1,735,245 corpus-wide figure counts MWE components, which are collapsed
  away; see §0.6.)
- **Seed filter:** POS = verb. No aspect filter is needed — coverage is total.
- **Gradeable line:** `ODGOVOR: dvovidski`.
- **"Vrsta glagola" is not in the graph.** Transitivity, reflexivity and valency are absent
  (the census finds no valency frames at all). Either restrict the questions to aspect and
  reword the type's phrasings accordingly, or answer the "vrsta" half from the reflexive *se*
  visible in MWE headwords — the first option is cleaner and is what the reference answers do
  in practice anyway.
- **Negative:** a verb with no aspect recorded; or a non-verb.

### Existing formulations

Reference CSV, n = 100:

| n | frame |
|--:|---|
| 18 | `Ali je glagol <L> dovršni ali nedovršni?` |
| 13 | `Opiši vid in vrsto glagola <L>.` |
| 13 | `Kateri vid izraža glagol <L>?` |
| 11 | `Kakšnega vida je glagol <L>?` |
| 11 | `Določi glagolski vid za besedo <L>.` |
| 10 | `Kateri vid in katero vrsto glagola predstavlja <L>?` |
| 10 | `Ali je <L> dovršni, nedovršni ali dvovidski glagol?` |
| 6 | `Opredeli glagolski vid besede <L>.` |
| 5 | `Ali spada glagol <L> med dovršne ali nedovršne glagole?` |
| 3 | `Kakšne vrste in vida je glagol <L>?` |

Older JSON file, `oblikoslovna/aspect_and_vform`, n = 101 — mixes aspect with infinitive
recovery, and asks about aspectual pairs, which the graph cannot answer:

```
Kateri glagolski vid ima glagol leteti?
Ali je glagol umreti dovršni ali nedovršni?
Je glagol delati dovršen?
Ali lahko glagol telefonirati štejemo za dvovidskega?
Kaj je nedoločnik glagolske oblike hotel?
Kateri dovršni glagol navadno stoji ob nedovršnem hoditi?   ← NOT answerable from this KG
```

### Suggested new templates

```
Kakšen vid ima glagol <L>?
Je <L> dovršni glagol?
Zanima me glagolski vid besede <L>.
Ali je <L> nedovršen?
Vid glagola <L>?
Kako je z vidom pri glagolu <L> — dovršni ali nedovršni?
Povej, ali je <L> dovršni, nedovršni ali dvovidski.
Ali glagol <L> izraža dovršeno dejanje?
Določi vid glagola <L>.
Rabim podatek o vidu za <L>.
```

---

## T11 — `stopnjevanje/vse_stopnje`

### What it is

**Gradation** of an adjective or adverb: osnovnik, primernik, presežnik.

### How to implement

**This type is regenerated from scratch, not imitated.** The reference version is degenerate
in **99/100 rows** (comparative = superlative = lemma) and in 76/100 rows the "positive" cell
is an unsorted dump of every inflected form:

```
| 1. Osnovnik  | brunastega / brunasti / brunastih / brunaste / brunastimi / ... |
| 2. Primernik | brunast |     ← wrong: this is the lemma
| 3. Presežnik | brunast |     ← wrong: this is the lemma
```

The bug is obvious once the RDF is read: `degree` sits on **word-forms**, and each graded
form also carries case, number and gender. The reference generator neither selected a cell
nor checked that a comparative existed.

- **Source:** `lexinfo:degree` on forms — 4,542,306 instances, but only **107,717
  comparative** and **72,856 superlative** against 4,361,733 positive. The graded forms are
  the scarce resource and they define the seed pool.
- **Cell selection:** pick the **nominative singular masculine** form of each degree.
  Where `definiteness` is present, prefer the indefinite form for the positive (`brunast`)
  and the definite for the superlative (`najbrunastejši`) — matching how Slovene dictionaries
  cite them. Do **not** dump the whole paradigm.
- **Seed filter:** POS ∈ {adjective, adverb} **and at least one comparative form exists**.
  Without that filter the type reproduces the reference defect.
- **Gradeable line:** `ODGOVOR: lep, lepši, najlepši`.
- **Negative — and this one is the point of the type.** Most adjectives in the graph have no
  graded forms (107,717 comparatives over ~4.4 M positive forms). Non-gradable adjectives
  (*lesen*, *kovinski*, *mrtev*) asked for their comparative are the ideal ordinary-lemma
  negative: `ODGOVOR: ni podatka v bazi`, with the UI part saying that no graded forms are
  recorded. The older JSON file asks exactly this ("Ali se glagol odnesti stopnjuje?",
  "Ali beseda noben tvori stopnjevane oblike?") and it should be a substantial slice here.

### Existing formulations

Reference CSV, n = 100:

| n | frame |
|--:|---|
| 15 | `Izpiši stopnjevalni vzorec za besedo <L>.` |
| 13 | `Navedi primernik in presežnik za pridevnik/prislov <L>.` |
| 13 | `Kako poteka stopnjevanje besede <L>?` |
| 11 | `Prikaži stopnjevanje pridevnika <L>.` |
| 10 | `Navedi vse stopnje za <L>.` |
| 10 | `Stopnjuj besedo <L>.` |
| 8 | `Izpiši osnovnik, primernik in presežnik za <L>.` |
| 7 | `Prikaži vse tri stopnje besede <L>.` |
| 7 | `Sestavi tabelo stopnjevanja za besedo <L>.` |
| 6 | `Kako se stopnjuje beseda <L>?` |

Older JSON file, `oblikoslovna/gradation`, n = 107:

```
Povej stopnjevanje za pridevnik zanimiv.
Napiši stopnjevanje prislova zgodaj.
Ali ima prislov visoko primerjalnik in presežnik?
Ali ima pridevnik dober nepravilno stopnjevanje? Če da, navedi oblike.
Ali se glagol odnesti stopnjuje?
Ali beseda noben tvori stopnjevane oblike?
Kako stopnjujemo prislov blizu?
Izpiši tri stopnje prislova lepo.
```

### Suggested new templates

```
Kako se stopnjuje <L>?
Katere so tri stopnje besede <L>?
Rabim primernik in presežnik za <L>.
Kako rečem, da je nekaj bolj <L>?
Ali se beseda <L> sploh stopnjuje?
Stopnjuj pridevnik <L>.
Kaj je presežnik besede <L>?
Zanima me primernik besede <L>.
Ima <L> primernik in presežnik?
Napiši osnovnik, primernik in presežnik za <L>.
```

---

# Group D — `pomen` (senses and definitions)

Shared constraint for all three: **definition coverage is thin.** Only **225,618** `@sl`
`skos:definition` literals exist over 8.47 M senses, so most `pomen:` nodes fall back to the
entry's lemma and carry no new information. v3.1 makes sibling senses distinguishable by
adding the dictionary ordinal (`pomen 2: pes`) and, for definition-less senses, a bounded
usage-example snippet (`pomen 1: pes (zgled: Sosedov pes je spet lajal ...)`) — 43 % of
senses got a snippet. **The generator must distinguish a real definition from a
lemma-fallback**, or it will emit `pomen/razlaga_pomena` items whose gold answer is the
headword repeated.

Also: the reference file's COBUILD-style full-sentence glosses ("*Kadar sodišče dostavi
pravno odredbo nekomu, mu jo pošlje*") came partly from **WordNet and a bridge dictionary**,
not from this KG. Do not imitate that answer style — it is not reproducible here.

---

## T12 — `pomen/razlaga_pomena`

### What it is

**Define the word.** Give the recorded definition; for a polysemous entry, give the
definitions of all senses that have one.

### How to implement

- **Source:** `pomen:` nodes carrying a genuine `skos:definition`.
- **Seed filter:** at least one sense with a real definition (not a lemma fallback). Among
  core lemmas, **40.6 %** have an `@sl` definition; among tail lemmas only 7.3 % — D8's pool
  already biases toward the core.
- **Gradeable line:** the definition text of the first sense, verbatim from the graph:
  `ODGOVOR: Plesen je mehka bela gobasta tvorba, ki se razvije na vlažnih in toplih krajih.`
  Grading is then string/embedding similarity against a string that is guaranteed present in
  the input, which is exactly the property that makes this type a clean retrieval probe.
- **UI part:** all defined senses as a bulleted list, matching the reference style.
- **Negative:** **59 % of core lemmas lack a definition** — the single largest natural
  negative pool in the dataset. The UI part should offer what *is* attached (examples,
  collocations, synonyms).

### Existing formulations

Reference CSV, n = 100:

| n | frame |
|--:|---|
| 14 | `Podaj razlago pomena za besedo <L>.` |
| 13 | `Opiši pomen besede <L>.` |
| 11 | `Kakšen je pomen besede <L>?` |
| 10 | `Napiši obrazložitev pomena besede <L>.` |
| 10 | `Kaj pomeni beseda <L>?` |
| 9 | `Navedi definicijo besede <L>.` |
| 9 | `Kakšna je leksikalna definicija besede <L>?` |
| 9 | `Kaj v slovenščini pomeni izraz <L>?` |
| 9 | `Razloži, kaj označuje beseda <L>.` |
| 6 | `Razloži pomen besede <L>.` |

Older JSON file, `semanticna/definitions`, n = 107 — markedly more conversational:

```
Kako se v slovarju pojasni beseda obraz?
Kako razložiti besedo okus?
Mi lahko podaš kratko definicijo besede knjiga?
Kaj pomeni beseda naglas?
Kako bi pojasnili pomen besede učenec?
Prosim za kratko slovarsko razlago besede prijatelj.
Kaj v slovenščini pomeni prislov notri?
Kako slovar definira besedo svetloba?
Kako se razume izraz skrb?
Kaj označuje glagol misliti?
```

### Suggested new templates

```
Kaj pomeni <L>?
Kaj je <L>?
Razloži mi besedo <L>.
Ne poznam besede <L>. Kaj pomeni?
Kako bi razložil pomen besede <L>?
Prosim za definicijo besede <L>.
Kaj pomeni izraz <L> v slovenščini?
Zanima me, kaj pomeni <L>.
Naletel sem na besedo <L> in je ne razumem.
Pomen besede <L>?
```

`Kaj je <L>?` and `Ne poznam besede <L>. Kaj pomeni?` are the most realistic public-user
phrasings in this whole document, and neither appears in either reference file. Both are
good **Tier A** candidates.

---

## T13 — `pomen/nastevanje_pomenov`

### What it is

**Enumerate the senses** of a polysemous word, as a numbered or bulleted list.

### How to implement

- **Source:** every `pomen:` node of the anchor, in dictionary-ordinal order — v3.1 numbers
  them, so the ordering is graph-derived rather than invented.
- **Seed filter:** ≥ 2 senses. **91.3 % of anchors have ≥ 2 senses**, so the pool is large,
  but most sibling senses are lemma fallbacks. Require that ≥ 2 senses are *distinguishable*
  — either both defined, or both carrying an example snippet. Without that check the type
  degenerates into "pomen 1: jajčast, pomen 2: jajčast".
- **Watch for junk sense labels.** The reference file has 12 rows whose senses read
  "*raba obliki*", "*raba živali*" — artefacts. Filter definitions that begin with `raba`
  or that are shorter than ~3 tokens.
- **Gradeable line:** the sense heads, semicolon-separated, in ordinal order:
  `ODGOVOR: 1. mehka bela gobasta tvorba …; 2. mehka siva snov na stari hrani …`
- **Negative:** a monosemous entry — answer with the one sense and say so, rather than the
  sentinel; reserve the sentinel for entries whose senses are all lemma fallbacks.

### Existing formulations

Reference CSV, n = 100:

| n | frame |
|--:|---|
| 17 | `Razvrsti in naštej pomene besede <L>.` |
| 13 | `Kateri so pomeni besede <L>?` |
| 12 | `Navedi različne pomene besede <L>.` |
| 11 | `Naštej leksikalne pomene za besedo <L>.` |
| 10 | `Navedi vse pomene besede <L>.` |
| 9 | `Izpiši vse registrirane pomene besede <L>.` |
| 8 | `Katere pomene ima beseda <L>?` |
| 8 | `Naštej pomene besede <L>.` |
| 6 | `Prikaži seznam pomenov za besedo <L>.` |
| 6 | `Poišči in naštej pomene leme <L>.` |

Older JSON file, `semanticna/polysemy`, n = 107 — largely **not** reproducible here, because
it asks about figurative vs literal readings, domain-specific senses and sense extension,
none of which the KG marks:

```
Kaj pomeni beseda ključ poleg predmeta za odklepanje?
Ima pridevnik rdeč poleg barvnega še kak drug pomen?
Ali beseda cvet pomeni samo del rastline?
Je člen primer besede z več pomeni?
Ali je pri besedi polje mogoče govoriti o večpomenskosti?
Kako ločimo pomena besede most v gradbeništvu in zobozdravstvu?   ← not answerable
Kaj je preneseni pomen pri glagolu plesati?                       ← not answerable
```

Only the first four shapes survive. The rest are listed so nobody re-derives them and then
discovers the KG has no register, domain or figurative-sense marking.

### Suggested new templates

```
Katere pomene ima <L>?
Naštej pomene besede <L>.
Koliko različnih stvari lahko pomeni <L>?
Ali ima <L> več pomenov? Naštej jih.
Zanimajo me vsi pomeni besede <L>.
Beseda <L> — kateri pomeni so zabeleženi?
Izpiši pomene besede <L> po vrsti.
Kaj vse lahko pomeni <L>?
Prikaži vse registrirane pomene za <L>.
Rabim seznam pomenov besede <L>.
```

---

## T14 — `pomen/stevilka_pomenov`

### What it is

**Count the senses.** A number, optionally followed by short sense heads.

### How to implement

- **Source:** count of `pomen:` nodes on the anchor.
- **Fix the agreement bug.** The reference file gets Slovene number agreement wrong in
  **100/100 rows** ("*ima registrirana 1 pomen*"). Correct forms:
  - 1 → `ima 1 pomen`
  - 2 → `ima 2 pomena`
  - 3–4 → `ima 3 pomene`
  - 5+ → `ima 5 pomenov`
  Implement this as a table lookup on `n mod 100`, not a heuristic, and unit-test it.
- **Gradeable line:** the bare integer — `ODGOVOR: 3`. This is the only type in the inventory
  whose gold answer is exact-match gradeable with zero ambiguity, which makes it the natural
  smoke test for the whole pipeline.
- **Counting policy must be fixed and stated:** does a lemma-fallback sense count? Recommend
  **yes** (the count is a structural fact about the entry, independent of definition
  coverage), but state it in the UI part so the answer is not silently surprising.
- **Negative:** an entry with no senses at all.

### Existing formulations

Reference CSV, n = 100:

| n | frame |
|--:|---|
| 14 | `Ali ima beseda <L> več pomenov?` |
| 12 | `Navedi število in kratek opis pomenov besede <L>.` |
| 12 | `Izpiši število pomenov besede <L>.` |
| 11 | `Navedi število pomenov za besedo <L>.` |
| 10 | `Preveri število registriranih pomenov besede <L>.` |
| 10 | `Koliko pomenov ima beseda <L>?` |
| 9 | `Ali je beseda <L> enopomenska ali večpomenska?` |
| 8 | `Koliko ločenih pomenov ima lema <L>?` |
| 8 | `Preveri večpomenskost besede <L>.` |
| 6 | `Koliko leksikalnih pomenov vsebuje beseda <L>?` |

### Suggested new templates

```
Koliko pomenov ima <L>?
Je <L> večpomenska beseda?
Koliko različnih pomenov je zabeleženih za <L>?
Ali ima beseda <L> samo en pomen?
Število pomenov besede <L>?
Povej, koliko pomenov ima <L>.
Je <L> enopomenska?
Koliko gesel oziroma pomenov ima <L>?
Zanima me, ali je <L> večpomenska.
Ima <L> več kot en pomen?
```

---

# Group E — sense relations

## T15 — `sopomenke/navedi_sopomenke`

### What it is

**List the synonyms** of a word.

### How to implement

- **Source:** reified `sopomenka:` nodes — `sopomenka: biblioteka ~ knjižnica`. They hang one
  hop past the sense, which is precisely why D4 sets the extraction floor at **hop 2**. The
  node text carries **both** lemmas, so the answer is readable even if the partner's anchor
  is pruned.
- **Coverage:** `lexinfo:synonym` 362,520 instances; **35.4 % of core lemmas** have at least
  one synonym, 4.4 % of tail lemmas.
- **Gradeable line:** the partner lemmas, comma-separated, in a canonical order (alphabetical
  — the graph's order is not meaningful): `ODGOVOR: pasiven, resigniran, spokojen, vdan`.
- **Sense scoping:** synonyms attach to *senses*, not lemmas. For a polysemous word, either
  merge across senses (simple, matches the reference) or ask about one sense
  ("*sopomenke besede <L> v pomenu …*"). The second is a natural **Tier B** item — an unseen
  question type over a seen relation.
- **Negative:** **65 % of core lemmas lack synonyms** — a large, natural, ordinary-lemma
  negative pool.

### Existing formulations

Reference CSV, n = 100:

| n | frame |
|--:|---|
| 16 | `Katere so sopomenke besede <L>?` |
| 16 | `Poišči sopomenke za <L>.` |
| 14 | `Katere besede imajo podoben pomen kot <L>?` |
| 9 | `Katere sopomenke obstajajo za izraz <L>?` |
| 9 | `S katerimi besedami lahko nadomestimo besedo <L>?` |
| 8 | `Izpiši soznačilne besede za <L>.` |
| 8 | `Navedi soznačice za besedo <L>.` |
| 7 | `Navedi sinonime za besedo <L>.` |
| 7 | `Prikaži seznam sopomenk za besedo <L>.` |
| 5 | `S katerimi sinonimi opisujemo besedo <L>?` |

Older JSON file, `semanticna/synonyms`, n = 103 — mostly **sense- or register-scoped**, and
register is not in the KG:

```
Kateri sinonimi pridejo v poštev za glagol začeti?
Katere besede lahko uporabim namesto glagola izboljšati?
Kako bi še izrazil hoteti, ne da bi uporabil isto besedo?
Iščem sinonime za hrana, kadar mislim na to, kar jemo.          ← sense-scoped: possible
Katera sopomenka za umazan je najprimernejša za formalno besedilo?  ← register: NOT possible
Če v stavku dvakrat uporabim spet, s čim ga lahko zamenjam?
```

### Suggested new templates

```
Katere so sopomenke besede <L>?
S čim lahko zamenjam besedo <L>?
Rabim drugo besedo za <L>.
Kako drugače rečem <L>?
Zanimajo me sinonimi za <L>.
Ali obstaja sopomenka za <L>?
Napiši nekaj besed, ki pomenijo isto kot <L>.
V besedilu sem že uporabil <L>. Katero besedo lahko uporabim namesto nje?
Sopomenke za <L>?
Katere besede so po pomenu blizu besedi <L>?
```

---

## T16 — `protipomenke/navedi_protipomenke` *(added — Tier C, held out of training)*

### What it is

**List the antonyms** of a word. New in v1's inventory: the newer reference file dropped
antonyms, though the KG supports them.

### How to implement

- **Source:** reified `protipomenka:` nodes — `protipomenka: boj ~ sodelovanje`. Structurally
  **identical to `sopomenka:`**, differing only in the tag word. That identity is the entire
  point of the experiment.
- **Coverage:** `lexinfo:antonym` = **6,898** instances — two orders of magnitude below
  synonyms. This is the smallest relation in the inventory, which is why holding it out costs
  almost nothing.
- **D12 Tier C:** **100 % of antonym items are test items.** Nothing about antonyms appears
  in training — not the relation, not the templates, not the tag word. The probe is whether a
  model trained to read `sopomenka: A ~ B` nodes can read `protipomenka: A ~ B` nodes it was
  never supervised on. `QA_DATASET_DESIGN.md` calls this the strongest available test of
  whether graph attention generalises, and the owner asked specifically for good OOD
  behaviour — so this is a **headline result**, not a side experiment.
- **Generation discipline:** because the type is unseen, its templates must not leak. Do not
  reuse a synonym template with the word swapped; write them independently, and **verify that
  no training item anywhere in the dataset contains the string `protipomenka`, `antonim` or
  `nasprotje`**.
- **Gradeable line:** `ODGOVOR: sodelovanje` (or the comma-separated set).
- **Negative:** ample — the overwhelming majority of lemmas have no antonym. Keep the
  negative share here at the dataset default; a Tier C type that is mostly sentinels measures
  nothing.

### Existing formulations

Not present in the newer CSV at all. Older JSON file, `semanticna/antonyms`, n = 106:

```
Kakšno nasprotje ima samostalnik prihod?
Ali je nasprotje besede glasen kar tih?
Kateri antonim ustreza prislovu daleč?
Kateri antonim ima beseda desno?
Kaj je antonim prislova blizu?
Za izraz uspeh me zanima protipomenka.
Kako se glasi antonim za dan?
Katero besedo bi uporabili kot antonim za mlad?
Za pridevnik mehak potrebujem protipomenko.
Ali obstaja antonim za besedo enak?
Kateri glagol je antonim glagolu kupiti?
Kateri antonim ustreza besedi mir?
```

### Suggested new templates

```
Kaj je nasprotje besede <L>?
Katera beseda pomeni nasprotno od <L>?
Rabim protipomenko za <L>.
Ali ima <L> nasprotni pomen?
Kako rečem nasprotno od <L>?
Protipomenka besede <L>?
Katera je nasprotna beseda od <L>?
Zanima me antonim besede <L>.
Če <L> ni pravi izraz, ker mislim ravno obratno — katero besedo naj uporabim?
Napiši nasprotje za <L>.
```

---

# Group F — `kolokacije` *(reworked — lemma pairs only)*

**Shared, and non-negotiable.** A `frac:Collocation` node carries only `rdfs:member` (two
*senses*) and `frac:head`. **There is no surface string on it anywhere in 42 GB.** The graph
gives `kolokacija: mineralen + voda`; the inflected *mineralna voda* is not stored. The
reference file's inflected phrases came from a separate DDDS **API** endpoint
(`api/retrieve/lexical-unit-collocations/[lexeme_id]`), and its corpus fragments
("*Potrpežljivost je vrlina*" — 175 of 1,307 collocation items start with a capital) came from
a corpus extraction outside this KG entirely.

**Therefore both collocation types are rephrased so that the gold answer is a lemma pair,
never an inflected phrase.** This is the whole content of the D14 rework, and it changes the
question wording as much as the answer: a question that asks for *besedne zveze* invites an
inflected phrase, so it must be replaced by one that asks which words combine with the target.

Verbalised collocations are deferred to v2 (Section 6.2 of `QA_DATASET_DESIGN.md`) — as a
**build-time bulk fetch** baked into node text, never a runtime dependency, because a fact
that is not in the extracted subgraph breaks the containment invariant the design rests on.

Coverage: **65.0 % of core lemmas** have collocations (12.2 % of tail lemmas) — 65,480 core
lemmas in total, the largest sub-pool of any relation type. 2,981,731 distinct pairings,
99.8 % binary.

---

## T17 — `kolokacije/navedi_kolokacije`

### What it is

**Which words does this word combine with?** The answer is a list of partner lemmas (or
lemma pairs), unnumbered and non-exhaustive.

### How to implement

- **Source:** `kolokacija:` nodes attached to the anchor's senses. Each has degree exactly 2
  and is a leaf-like connector, not a hub — which is why wiring collocations in costs ~0 % at
  the median ball size.
- **Ranking:** the graph carries no frequency on the collocation node. Rank deterministically
  (e.g. by the partner's own proxy value, then alphabetically) and say in the UI part that
  the list is a selection, not a ranking by corpus frequency. **Do not claim "najpogostejše"**
  — the reference file does, and it cannot back the claim either.
- **Gradeable line:** partner lemmas in canonical order:
  `ODGOVOR: voda, gnojilo, snov, surovina, olje`
  — or the full pairs (`mineralen + voda; mineralen + gnojilo; …`) if the pair form is judged
  clearer for grading. Pick one and keep it fixed across both collocation types.
- **Cap:** 5–10 partners. High-band lemmas have hundreds; an uncapped list makes grading
  meaningless and the answer unreadable.
- **Negative:** 35 % of core lemmas have no collocations.

### Existing formulations

Reference CSV, n = 100 — **most of these frames ask for *besedne zveze* and must be reworded**
under the rework; they are recorded here as the phrasing baseline, marked:

| n | frame | reusable? |
|--:|---|---|
| 15 | `Navedi primerne kolokacije za besedo <L>.` | yes |
| 14 | `Katere so najbolj znane kolokacije besede <L>?` | reword — "znane" implies frequency |
| 12 | `S katerimi besedami se najpogosteje povezuje beseda <L>?` | **yes — this is the model frame for the rework** |
| 9 | `Izpiši tipične besedne zveze z besedo <L>.` | no — invites an inflected phrase |
| 9 | `Katere besedne zveze so pogoste z besedo <L>?` | no |
| 9 | `S katerimi izrazi se kolocira beseda <L>?` | yes |
| 9 | `Poišči pogoste kolokacije za besedo <L>.` | yes |
| 8 | `Navedi tipične kolokacije z besedo <L>.` | yes |
| 8 | `Prikaži seznam pogostih kolokacij z besedo <L>.` | yes |
| 7 | `Navedi kolokativne zveze za lemo <L>.` | yes |

Older JSON file, `raba/collocations`, n = 104 — asks by POS of the partner, which the KG
*can* support (`lexinfo:category` is on every constituent, and POS is on every word anchor):

```
S katerimi samostalniki se navadno povezuje pridevnik tradicionalen?
S katerimi pridevniki se pogosto pojavlja samostalnik otrok?
Katere glagole navadno uporabljamo z besedo odločitev?
Katere pridevniške kolokacije so običajne z besedo kruh?
```

These POS-filtered frames are the best idea in either reference file for this type and are
**not** in the newer one. They are a natural **Tier B** family: an unseen question shape over
a seen relation.

### Suggested new templates

```
S katerimi besedami se povezuje <L>?
Katere besede se pogosto pojavljajo skupaj z <L>?
S čim se navadno kombinira beseda <L>?
Naštej besede, ki se povezujejo z <L>.
Katere sopojavitve so zabeležene za <L>?
S katerimi samostalniki se povezuje <L>?
Kateri glagoli se pojavljajo ob besedi <L>?
Kateri pridevniki se pojavljajo ob besedi <L>?
Rabim seznam besed, ki gredo skupaj z <L>.
Katere besedne povezave so zabeležene za <L>?
```

---

## T18 — `kolokacije/stevilsko_opredeljene_kolokacije`

### What it is

The same as T17 but with an **explicit count** in the question: "give me exactly 3". The
answer must have exactly that many items.

### How to implement

- Identical to T17, with `<N>` as a template slot.
- **Sample `<N>` from 2–10** and — critically — **only from seeds that actually have ≥ N
  collocations**. Otherwise the type teaches the model to pad, which is the worst possible
  habit to train into a lexicographic service.
- **Deliberately include an under-supplied slice** (~10 %): ask for 5 when only 3 exist, and
  make the gold answer give the 3 and say so explicitly. This is the type's most valuable
  supervision — it is the only place in the inventory that trains *refusal to invent* under
  direct numeric pressure.
- **Gradeable line:** exactly N partner lemmas, canonical order, or the honest short list.
  The grader checks both the count and the membership.

### Existing formulations

Reference CSV, n = 100:

| n | frame |
|--:|---|
| 13 | `Izpiši <N> kolokacij besede <L>.` |
| 13 | `Sestavi seznam <N> kolokacij z besedo <L>.` |
| 12 | `Prikaži številčni seznam <N> kolokacij besede <L>.` |
| 12 | `Katere so <N> tipične kolokacije z besedo <L>?` |
| 11 | `Poišči <N> tipičnih besednih zvez za besedo <L>.` (reword) |
| 11 | `Prikaži <N> pogostih kolokacij z besedo <L>.` |
| 9 | `Izpiši točno <N> kolokacij besede <L>.` |
| 7 | `Navedi <N> najbolj pogostih kolokacij za <L>.` (reword — frequency claim) |
| 7 | `Zapiši <N> kolokativnih primerov za besedo <L>.` |
| 5 | `Navedi <N> kolokativnih zvez za besedo <L>.` |

### Suggested new templates

```
Naštej <N> besed, ki se povezujejo z <L>.
Daj mi <N> kolokacij besede <L>.
Rabim <N> primerov sopojavitev za <L>.
S katerimi <N> besedami se najpogosteje pojavlja <L>?
Izpiši natanko <N> kolokacij za <L>.
Zanima me <N> besed, ki gredo skupaj z <L>.
Napiši <N> kolokacij besede <L>, oštevilčeno.
Ali mi lahko daš <N> kolokacij za <L>?
Samo <N> primerov, prosim: s čim se povezuje <L>?
Potrebujem <N> zabeleženih sopojavitev besede <L>.
```

---

# Group G — `primeri_uporabe` (usage examples)

**Shared constraint, and it is severe.** Of 14,722,502 `lexicog:usageExample` values,
**14,650,773 (99.5 %) hang off MWE senses**; only **71,729 (0.5 %)** are reachable from a
single-word entry, spread over just **11,048 words**. Words that do have examples carry them
directly on their own senses, so no MWE traversal is needed — but the eligible pool is
**11,048 lemmas**, versus 65,480 for collocations and 35,691 for synonyms.

Consequences:

- Only **10.2 % of core lemmas** (0.3 % of tail lemmas) have a usage example.
- The band × type availability matrix for both types in this group will have **nearly empty
  low-band cells** — B0 lemmas almost never have examples. This is exactly the case
  Section 5 of `QA_DATASET_DESIGN.md` warns about: compute the matrix first (**R3**), then
  shrink the quota or borrow from the adjacent band. Do not silently produce a skewed set.
- Examples are capped at ~4 per sense (p50 = p90 = p99 = 4, max 198), so "give me five
  sentences" is usually unanswerable.

---

## T19 — `primeri_uporabe/povedi_z_besedo`

### What it is

**Show the word in use** — return recorded corpus sentences containing it.

### How to implement

- **Source:** `zgled:` nodes on the anchor's senses, verbatim. v3.1 unescaped the literals,
  so `\"` no longer leaks into example text (it affected 3.53 % of `zgled:` nodes in v3).
- **Seed filter:** ≥ 1 direct example. Pool = 11,048 lemmas; expect this type to be the
  binding constraint on the low bands.
- **Gradeable line:** the first example verbatim —
  `ODGOVOR: Je oče petih otrok in dedek štirih vnukov.`
  Verbatim reproduction from the input is the cleanest possible retrieval signal, and it is
  the one type where an exact string match is both fair and meaningful.
- **UI part:** up to 4 examples, italicised, as the reference does.
- **Do not ask for a specified number** of sentences beyond 4.
- **Negative:** **90 % of core lemmas have no usage example** — the largest natural negative
  pool of all. Keep this type's negative share at the dataset default anyway; a type that is
  90 % sentinel teaches "say you don't know".

### Existing formulations

Reference CSV, n = 100:

| n | frame |
|--:|---|
| 16 | `Kako se beseda <L> pojavlja v besedilnem kontekstu?` |
| 13 | `Kako se beseda <L> uporablja v stavkih?` |
| 13 | `Prikaži primere rabe besede <L>.` |
| 13 | `Izpiši primere iz korpusa za besedo <L>.` |
| 11 | `Prikaži uporabo besede <L> v dejanskih povedih.` |
| 9 | `Kako v praksi uporabimo besedo <L>?` |
| 7 | `Navedi stavčne primere za besedo <L>.` |
| 7 | `Zapiši primere stavkov z besedo <L>.` |
| 6 | `Izpiši nekaj povedi, ki vsebujejo besedo <L>.` |
| 5 | `Navedi primere uporabe besede <L> v povedih.` |

Older JSON file, `raba/usage_example`, n = 106 — almost all **sense-** or **context-scoped**,
which this KG mostly cannot honour:

```
Pokaži primer, kjer beseda pes nastopa v dobesednem pomenu živali.     ← sense-scoped
Kako bi v stavku uporabil luč, ko govorimo o prometu?                  ← domain: NOT possible
Bi lahko dal primer rabe za spati v zanikanem stavku?                  ← NOT possible
Prosim za primer, kjer se v istem stavku pojavita čaj in zajtrk.       ← NOT possible
Ali lahko navedeš primer za besedo koren v botaničnem pomenu?          ← NOT possible
```

Only sense-scoped variants survive, and only where the sense actually has its own examples.
Worth one **Tier B** family: "*Kateri pomen besede <L> ima zabeležen zgled?*" — an unseen
question type over a seen relation, and exactly the example `QA_DATASET_DESIGN.md` gives for
Tier B.

### Suggested new templates

```
Pokaži mi primer rabe besede <L>.
Kako se <L> uporablja v povedi?
Daj primer stavka z besedo <L>.
Rabim zgled uporabe za <L>.
V kakšnem stavku se pojavi <L>?
Ali imaš kakšen primer rabe besede <L>?
Napiši povedi, v katerih nastopa <L>.
Kako bi <L> uporabil v stavku?
Primeri rabe za <L>?
Zanima me, kako se <L> pojavlja v besedilih.
```

---

## T20 — `primeri_uporabe/analiza_oblike_v_povedi` *(reworked — D15)*

### What it is

**Form analysis with a sentence.** Given a real sentence and one word in it, say what
grammatical form that word takes. **Reworked by D15 to be restricted to *unambiguous* forms**,
so every label is correct by construction — and **answered in words, never as an MSD tag**.

The reference version is defective in **100/100 rows**: it answers with a raw JOS tag and
nothing else, names the case in words in exactly 1 row, and is vacuous ("form X of lemma X")
in 36. The tags themselves are well-formed — category letter agrees with the KG POS 91/91,
decoded number+case is a legal KG reading 58/58 — but **50 of 58 noun forms are
morphologically ambiguous**, and on the cleanest ambiguity class the disambiguation is wrong
**5/5**. D6 removes MSD entirely; D15 removes the ambiguity.

### How to implement

- **Source:** a `zgled:` node (the sentence) plus the `oblika:` leaf whose surface appears in
  it. The sentence comes from the graph, so the item is fully contained in the extracted
  ball.
- **The unambiguity requirement is the hard part, and it is an open item (R2).** The type
  needs a lemma that has a **direct** usage example *and* whose form in that example is
  unambiguous. Only 11,048 lemmas have direct examples to begin with, and roughly 86 % of the
  reference file's noun forms were ambiguous. **Measure this intersection before generating.**
  If it is too thin for the quota, D15's recorded fallback is to answer with the disjunction
  here too (as T21 does) — and that fallback must then be **recorded in
  `QA_DATASET_DESIGN.md`**, not left implicit.
- **Matching the form to the sentence:** string-match the form's `writtenRep` against the
  example text, case-insensitively, on word boundaries. Reject if the surface occurs more
  than once in the sentence with different possible analyses, and reject if it matches more
  than one form leaf of the entry (that is the ambiguity test, done cheaply).
- **No tagger.** CLASSLA disambiguation is deferred (Section 6.3) precisely so the tagger's
  errors do not enter the gold labels.
- **Gradeable line:** `ODGOVOR: tožilnik, ednina` — words, never `Sozet`.
- **Negative:** a sentence in which the named word does not appear at all.

### Existing formulations

Reference CSV, n = 100, `<F>` = the form as it appears, `<S>` = the sentence. Nine frames:

| frame |
|---|
| `Analiziraj rabo besede <F> v povedi: '<S>'.` |
| `V katerem sklonu/obliki je beseda <F> v povedi '<S>'?` |
| `Katero slovnično obliko izraža beseda <F> v stavku: '<S>'?` |
| `Razloži oblikoslovno vlogo besede <F> v stavku '<S>'.` |
| `Kakšna je slovnična oblika besede <F> v povedi: '<S>'?` |
| `V kateri obliki nastopa beseda <F> v povedi: '<S>'?` |
| `Določi slovnične lastnosti besede <F> v stavku '<S>'.` |
| `Določi sklon in število besede <F> v povedi '<S>'.` |
| `Slovnično analiziraj besedo <F> v povedi '<S>'.` |

Note the reference sentences are **truncated mid-word** in several rows ("*To območje je
običajno gospodarsko, polit*"). Do not inherit that: `zgled:` nodes hold complete sentences,
so there is no reason to truncate.

### Suggested new templates

Per D3 the sentence should *not* be wrapped in quotes in every template — a user pasting a
sentence will not quote it, and the extractor consumes quotes as ordinary text either way.
Half the templates below leave the sentence unquoted.

```
V povedi <S> — v kateri obliki je beseda <F>?
Kateri sklon in število ima <F> v tem stavku: <S>
Napisal sem: <S> Kaj je <F> po sklonu in številu?
Določi obliko besede <F> v povedi <S>.
Kaj je <F> v tem stavku — kateri sklon? Stavek: <S>
V stavku <S> me zanima oblika besede <F>.
Analiziraj besedo <F> v naslednji povedi: <S>
Poglej stavek <S> in povej, v kateri obliki nastopa <F>.
Katero slovnično obliko ima <F> tukaj: <S>
Zanima me, kaj je <F> v povedi <S> — sklon in število.
```

---

## 2. Cross-cutting checks before generation starts

| # | check | why |
|---|---|---|
| C1 | ~~Rebuild the store.~~ **Done 2026-08-20** — the v4 stores render `vform`/`person`/`definiteness` on forms and `aspect`/`clitic` on anchors (§0.7). Generate against `kg_graph_v4_gemma3`. | T5, T6, T7 and T10 were ungeneratable before this and are generatable now. |
| C2 | **Compute the band × type availability matrix** (R3) and record it. | T19/T20 have ~11 k eligible lemmas against 72,528 in the pool; low-band cells will be near-empty. |
| C3 | **Measure the T20 intersection** (R2): direct example × unambiguous form. | If thin, fall back to the disjunction and record the fallback. |
| C4 | **Measure the natural multi-entity rate** (R5) on real generated question strings, not on the 88.9 % figure. | D3 makes the union share emergent; if it comes out near zero, oversample ambiguous seeds via T4/T21. |
| C5 | **Pick and evaluate the extraction model** (R4). Its recall against our own templates is measurable for free — we author them, so the gold target word is known. | An extractor miss is an end-to-end service error no amount of GTLM training recovers. |
| C6 | **Verify Tier C leakage**: no training item anywhere contains `protipomenka` / `antonim` / `nasprotje`. | T16's whole value is that the relation is unseen. |
| C7 | **Unit-test the Slovene number agreement table** (T14) and the canonical orderings (T1, T5, T15, T17, T21). | Grading is exact-match on these strings; an ordering bug reads as a model failure. |
| C8 | **Withhold 2–3 templates per type** for Tier A before generation, not after. | Retrofitting a held-out split from generated items risks the same phrasing appearing on both sides. |

---

## 3. Appendix — types deliberately NOT in v1

Recorded so nobody re-derives them. All are from Section 6 of `QA_DATASET_DESIGN.md`, judged
worth doing and consciously postponed.

| family | status | note |
|---|---|---|
| `mwe_structure`, `mwe_component_search`, `idiom_completion` | **deferred to v2** | The single best-supported unused capability: 10,265,230 `decomp:constituent` edges, all carrying `lexinfo:category`, plus `rdf:_1 … _22` for constituent order, plus MWE headwords that *do* store their surface phrase. Deferred purely on implementation risk — it is the only family that depends on the D5 upward cap behaving. Validate the cap empirically first. |
| clitics and negation | **effectively dead for word entries** | The `clitic` 423,585 and `negative` 121,229 figures count `lexical-unit-part` subjects — MWE components, which the builder collapses away. Only **25** single-word entries carry a `clitic` value in the whole KG, so there is no pool to sample from. Anything here would have to come from the MWE family, not from word entries. Supersedes the earlier "cheapest type to add back". |
| verbalised collocations | **deferred to v2** | Build-time bulk fetch from the DDDS API, never a runtime dependency. See Section 6.2 of the design doc, and the open questions for Luka. |
| reasoning-tier items (aggregation, filtering, multi-hop) | **deferred (D13)** | v1 is retrieval-focused; Tier B/C are the only places reasoning is probed. |
| hypernyms/hyponyms, meronyms/holonyms | **cannot be supported** | `hypernym`/`hyponym` = 5,415 each (0.12 %); meronym/holonym absent entirely. |
| translations | **cannot be supported** | `vartrans:*` = 77,570, all `directEquivalent`, all **`@hun`**. |
| register, style, dialect, domain, frequency, typo metadata | **cannot be supported** | Absent from the KG entirely. This is what makes ~half the older reference file's types unanswerable. |
| valency, word formation | **cannot be supported** | No valency frames, no derivational morphology. |
| role-play personas, JSON output | **not reproduced (D20)** | The older file's `has_role` (286 items) and `is_json_format` (353) behaviours: nobody knows who added them or why, and neither is part of the service. Output shape is fixed by D6 for every item. |
| multi-turn / follow-ups | **out of scope (D20)** | Every item is one self-contained question and one answer. |
| general-instruction replay | **out of scope (D21)** | Capability retention is phase two, deliberately. |
