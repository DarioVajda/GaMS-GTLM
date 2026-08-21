# QA task types — specification for the generation pipeline

> **Status: drafted 2026-08-20; revised 2026-08-20 (T8 schema, T12/T13 merge, §0.8 grading);
> revised 2026-08-21 (Group D `sense_class` — T12 and T14 now read one shared sense filter,
> and T14's "90.1 %" alarm is retired as a measurement artefact).
> Scope: the question types locked by D14 of
> [`QA_DATASET_DESIGN.md`](QA_DATASET_DESIGN.md).** That document decides *what* the dataset
> is; this one decides *how each individual question type gets generated*. It is written to
> be read on its own — every fact it relies on (graph predicate, node text convention, answer
> format, seed pool) is restated here rather than cross-referenced away. Companions:
> [`QA_DATASET_DESIGN.md`](QA_DATASET_DESIGN.md) for the design decisions and the
> full-corpus census, [`README.md`](README.md) for how the graph store is built.
>
> **D14 drift.** D14 locked **21** types. This document now specifies **20**: T13
> (`pomen/nastevanje_pomenov`) is retired into T12, which produces the same gold answer
> against this KG — see T12 for the argument. D14 and D16's per-type arithmetic in
> `QA_DATASET_DESIGN.md` have **not** yet been updated to match.

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
- **Scale (D16):** ~10 k train (≤1 k dev) + ~2 k test → ~100 test items per type at 20 types
  (D16 says ~95, computed at 21 — see the D14-drift note above). Report per
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
kolokacija: boj proti kriminaliteti                       ← the inflected phrase (v5; was a lemma pair through v4)
sopomenka: biblioteka ~ knjižnica
protipomenka: boj ~ sodelovanje
prevod (madžarsko): rüh kezelése                          ← @hun only; unused in v1
```

The parenthetical order is fixed: POS, then lexical-unit properties (`aspect`, `clitic`),
then form features in the order `vform, person, case, number, gender, degree, definiteness`.
A generator may rely on that order.

Edges are **untyped**; the relation type lives in the node text. Extraction is uniform
**hop 2** from every matched lexical unit (D4), with two hub caps:

- `sestavina` traversal is directional: downward (MWE → word) always, upward
  (word → MWE) capped at K = 15 (D5).
- `sense → kolokacija` is capped at K = 15 **pooled across all of the anchor's senses**,
  selected by seeded weighted sampling with `w ∝ log(1 + proxy(partner))` over a pool
  sorted by node id, seeded from the anchor's node code (D5b). Uncapped, a high-band
  anchor costs a p50 of 22,704 tokens; capped, 1,986.

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
| `lexinfo:gender` | word-form **+ lexical-unit** | masculine, feminine, neuter | 310,362 units | ⚠️ forms only — M4 |
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

Four faults found while writing this document kept morphology out of the node text, silently
disabling six of the types. Two were builder bugs, fixed in v4. The third is a property of
the source data and constrains the design permanently. The fourth was found on 2026-08-20
while specifying T8 and is **still open**.

**(M1) `person` never reached the node text — fixed.** `FEATURE_PROPS` listed `person`, but
`VALUE_SL` mapped `firstPerson`/`secondPerson`/`thirdPerson` while the KG emits
`first`/`second`/`third`, and `feat_string()` drops unmapped values silently. 253,497 forms
affected.

**(M3) `aspect` and `clitic` were dropped by the subject-type guard — fixed.** They attach
to the **lexical-unit**, but the feature branch only accepted `T_WORDFORM`/`T_FORMLU`
subjects. They are now collected separately (`UNIT_PROPS`) and rendered into the anchor
parenthetical after the POS.

Both shipped in the **v4** stores (`kg_graph_v4_gemma3`, `kg_graph_v4_gams2b`, built
2026-08-20), together with `vform` and `definiteness`, and carry forward unchanged into the
v5 stores that supersede them (v5 touches collocation text only — see Group F). What changed:

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

**(M4) Noun `gender` sits on the lexical-unit and never reaches the node text — open.**
`UNIT_PROPS` is `("aspect", "clitic")`; `gender` is missing from it. Noun gender is an
**entry** property in this KG, not a form property: `lexical-unit-69883` (*miza*) carries
`lexinfo:gender lexinfo:feminine`, while every one of its forms renders as
`oblika: miza (imenovalnik, množina)` — no gender anywhere. There are **310,362**
`lexinfo:gender` triples on `lexical-unit` subjects, *exactly* the noun count, so every noun
entry has one and none of them is in the store. (The 3,242,639 `word-form` gender triples are
adjective and participle forms, where gender genuinely is a form feature and does render —
`oblika: popraskal (deležnik na -l, ednina, moški spol)`.)

Consequence: **T9 is not generatable from the v4 store at all, and T8's gender slot is
unfillable.** The fix is one line —

```python
UNIT_PROPS = ("aspect", "gender", "clitic")
```

— and `VALUE_SL` already maps `masculine/feminine/neuter` → `moški/ženski/srednji spol`, so
noun anchors become `iztočnica: miza (samostalnik, ženski spol, imenovalnik, ednina)`. It costs
a ~35-minute rebuild and it changes every noun **anchor** string, so `check_v4_text.py`'s
"nominal strings are byte-identical to v3.1" assertion has to be narrowed to nominal *form*
nodes. Nothing else in the inventory depends on it. **Decision pending:** until it lands, T8
answers POS only and T9 cannot run.

Four types are generatable again after v4 — **T5, T6, T7, T10** — subject to M2's scope limit
on T5/T6. **T8 (partially) and T9 (entirely) remain blocked on M4.**

### 0.8 Grading — exact match after a fixed normalization

Every gradeable line in this document is scored by **exact string match after a fixed
normalization**. No embedding model, no LLM judge, no fuzzy similarity. The entire grader is a
script that runs in microseconds:

```python
import re, unicodedata

def norm(s):
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s).strip().rstrip(".")
    return s.casefold()

def parse(answer, sep="|"):
    for line in answer.splitlines():
        if line.startswith("ODGOVOR:"):
            return [norm(x) for x in line[len("ODGOVOR:"):].split(sep)]
    return None                     # unparseable -> scored wrong, no partial credit

def correct(pred, gold, *, ordered):
    p, g = parse(pred), parse(gold)
    if p is None:
        return False
    return p == g if ordered else sorted(p) == sorted(g)
```

**`ordered` is a per-type constant**, taken from the type's Output template. It is **not** a
grader heuristic and never inferred at run time.

Two numbers per item, both deterministic:

- **`exact`** — normalized item lists identical, order included. **The headline metric.**
- **`f1`** — per-item precision/recall over the multiset. Diagnostic only: it separates "missed
  a sense" from "invented one" from "right content, wrong order". Never reported alone.

**Why exact match is defensible here, and not merely convenient.** Every gradeable string is
present **verbatim in the model's input** — these types are retrieval probes over an extracted
subgraph, not open generation. A model that paraphrases has not done the task.

### 0.8.1 Order: always generated, sometimes graded

Two separate concerns, and they must not be collapsed:

- **Generation is always canonical.** Every gold answer is emitted in one fixed order, with no
  exceptions and no dependence on graph-walk order. Without this the same fact yields different
  gold across runs, the training target for an item is unstable, and dataset diffs are noise.
- **Grading is order-sensitive only where order carries meaning.** In `ODGOVOR: gora, gore,
  gori, …` position 3 *is* the dative — that is addressing, not ordering, and it must be
  graded. In `ODGOVOR: pasiven, resigniran, spokojen` the order is an artefact of our own sort;
  penalising a reordering would score a formatting preference as a retrieval miss.

So set-valued types are compared as **multisets** (`ordered: no`) while slot-valued and
data-ordered types are compared as **sequences** (`ordered: yes`). Each type's Output template
states which.

Verified: nothing in the KG orders these sets. `lexinfo:synonym` and `lexinfo:antonym` carry no
score, rank or frequency anywhere in the dump. The graph's one explicit ordering —
`rdf:_1 … rdf:_n`, 118,260 triples — runs `lexical-unit → lexical-unit-part`, i.e. **word order
inside a multiword expression**, and touches none of these types.

### 0.8.2 Canonical order for set-valued types

Alphabetical, by the **Slovene** alphabet. This needs an explicit table: Python's default sort
is wrong here and the `sl_SI` locale is **not installed on this cluster**, so a
`locale.strxfrm` implementation would fail or fall back silently.

```
sorted():          cesta, riba, sok, zima, čas, šola, žito     <- č/š/ž dumped after z
correct Slovene:   cesta, čas, riba, sok, šola, zima, žito
```

Use a fixed key, no dependency:

```python
_SL = "abcčdefghijklmnoprsštuvzž"
_RANK = {ch: i for i, ch in enumerate(_SL)}

def sl_key(s):
    """Slovene collation key. Unknown chars sort after the alphabet, by codepoint."""
    return [_RANK.get(ch, len(_SL) + ord(ch)) for ch in s.casefold()]
```

Because order is not graded for these types, a collation bug can never cost a point — but the
training data must still be strictly canonical, so this is generated once and **checked** (C15),
not assumed.

**Separator.** `|` for any answer whose items are free text. Measured over all 230,606 `@sl`
definition literals: `|` occurs **0** times and a newline **0** times, while `;` occurs in
**8.5 %** and `,` in **33.0 %**. Comma and semicolon are therefore unusable wherever an item can
be definition text. Types whose items are single word forms (T1, T5, T7, T11, T15) keep their
comma-separated lines — a word form cannot contain a comma — but any new list type defaults
to `|`.

**The normalization is deliberately shallow.** It folds case, collapses whitespace and drops one
trailing period. Nothing else: no diacritic stripping (that would merge real Slovene
distinctions), no stemming, no reordering. Over the definition corpus its targets are real but
small — 0.1 % have stray leading/trailing whitespace, 0.0 % double spaces, 8.4 % end with a
period, 6.0 % begin with a capital — so it removes formatting noise the model cannot be
expected to guess and changes nothing else.

### 0.9 Output templates — the notation used in every section

Each type below carries an **Output template** block. It is the implementation contract: the
generator emits exactly this shape, and the §0.8 grader parses exactly this shape. Fields:

| field | meaning |
|---|---|
| `arity` | how many items. **Fixed** = always this many, positionally graded, gaps marked. **Open** = 1 or more, membership graded. |
| `sep` | the separator, or `—` for single-item answers |
| `order` | the canonical order used when generating, and whether the grader enforces it — **graded** (sequence comparison) or **not graded** (multiset comparison). See §0.8.1. Generation is canonical either way. |
| `gap` | how a missing slot in a fixed-arity answer is written |
| `regex` | shape check for the generator's own self-test (C9), not the grader |

**Arity is part of the contract, not a detail.** The grader must know it per type, because a
single-item answer is *never* split on the separator. Measured: 4 of 51,172 usage examples
contain a `|`, and 65 % contain a comma — so a T19 answer split on either would be shredded.
Single-item types (`arity: 1`) are compared whole, after `norm()` and nothing else.

**Gaps.** In a fixed-arity answer a missing cell is `/`, never an omitted item — otherwise a
missing 3rd form silently turns into a wrong 4th form and every later position is misgraded.
`/` never occurs inside a word form (verified: 0 occurrences across every form surface sampled).

---

## 1. The 20 types at a glance

| # | type | primary graph source | caveat |
|---|---|---|:--:|
| T1 | `sklanjanje/celotna_sklanjatev` | `oblika:` leaves, case × number | |
| T2 | `sklanjanje/sklanjatev_po_stevilu` | `oblika:` leaves, one number | |
| T3 | `sklanjanje/posamezen_sklon` | `oblika:` leaves, one cell | |
| T4 | `sklanjanje/osnovna_oblika_leme` | form → `iztočnica:` | |
| T5 | `spreganje/celotno_spreganje` | `vform=present` + person + number | M2 scope |
| T6 | `spreganje/spreganje_v_casu` | `vform=present` (+ participle) | M2 scope |
| T7 | `spreganje/neosebne_oblike` | `vform` ∈ {inf, supine, imperative} | |
| T8 | `besedna_vrsta/osnovne_lastnosti` | `partOfSpeech` on the anchor | M4 (gender slot) |
| T9 | `besedna_vrsta/spol_samostalnika` | `gender` on the **anchor** | **M4 — blocked** |
| T10 | `besedna_vrsta/vrsta_in_vid_glagola` | `aspect` on the anchor | |
| T11 | `stopnjevanje/vse_stopnje` | `degree` on forms | |
| T12 | `pomen/razlaga_pomena` | `pomen:` nodes classed `defined` (Group D) | absorbs T13 |
| ~~T13~~ | ~~`pomen/nastevanje_pomenov`~~ | — | **retired → T12** |
| T14 | `pomen/stevilka_pomenov` | count of the senses T12 lists | shares T12's filter |
| T15 | `sopomenke/navedi_sopomenke` | `sopomenka:` nodes | |
| T16 | `protipomenke/navedi_protipomenke` | `protipomenka:` nodes | Tier C |
| T17 | `kolokacije/navedi_kolokacije` | `kolokacija:` nodes (v5: the phrase itself) | **unparked** |
| T18 | `kolokacije/stevilsko_opredeljene_kolokacije` | `kolokacija:` nodes, counted | **unparked** |
| T19 | `primeri_uporabe/povedi_z_besedo` | `zgled:` nodes | |
| T20 | `primeri_uporabe/analiza_oblike_v_povedi` | `zgled:` + unambiguous form | |
| T21 | `sklanjanje/analiza_oblike` | ambiguous form → disjunction | |

## 1.1 Verifiability audit

Every type was checked against one question: **can a 10-line script decide whether an answer is
right, with no model in the loop and no judgement call?** Measured against the v4 store on
2026-08-20 (samples of 4,000–20,000 anchors; scripts under the session scratchpad).

T17/T18 were skipped in that pass because they were parked. They are unparked (Group F) and
their rows below are filled in from the raw-dump measurement; **re-run the audit for both
against a v5 store** before generation — the open item is C14, not a known defect.

| type | arity | verdict | defect found |
|---|---|---|---|
| T1 | 18 fixed | ✅ | in-cell doublets — 0.2 % of nouns, all `tožilnik, ednina`; **filter them** |
| T2 | 6 fixed | ✅ | same filter |
| T3 | 1 | ✅ | same filter |
| T4 | open | ⚠️→✅ | spec said "`gora` **or** the comma-separated set" — two shapes; **fixed to always-a-set** |
| T5 | 9 fixed | ✅ | none (0 % of verbs have a doublet slot) |
| T6 | 9 fixed | ✅ | none |
| T7 | 3 fixed | ⚠️→✅ | omitting an absent slot made arity variable; **fixed to `/`, slot never dropped** |
| T8 | 1–2 fixed | ✅ | closed vocabulary, fully enumerable in the regex |
| T9 | 1 | ✅ | blocked on M4, but trivially gradeable once rendered |
| T10 | 1 | ✅ | best in the inventory — 3 values, 100 % coverage |
| T11 | 3 fixed | ⚠️→✅ | no gap marker for "comparative but no superlative"; **fixed to `/`** |
| T12 | open | ✅ | `\|` safe: 0 of 230,606 definitions contain one |
| T14 | 1 int | ⚠️→✅ | the "90.1 % answer 2" alarm was measured over the **wrong population**; **fixed** by counting what T12 lists — see below |
| T15 | open | ✅ | comma safe: 0 of 523 sampled synonym partners contain one (max 11 words) |
| T16 | open | ✅ | same; 75 % of anchors have exactly 1 antonym |
| T17 | open | ✅ | **unparked** — the KG has the verbalisations after all; gold is the inflected phrase, graded case-insensitively (Group F) |
| T18 | 1 int + open | ✅ | same |
| T19 | 1 | ⚠️→✅ | examples reach **1,487 words**, and 4 of 51,172 contain `\|`; **fixed by a 60-word seed cap + arity 1 (never split)** |
| T20 | 2 fixed | ⚠️→✅ | had its own shape; **unified with T21** |
| T21 | open | ⚠️→✅ | spec had **two** shapes (`tožilnik ali orodnik, ednina` vs `rodilnik ednine ali imenovalnik množine`); **unified to full pairs always** |

**The T14 alarm was a measurement artefact — resolved 2026-08-21.** The earlier figure
(1 sense 8.6 %, **2 senses 90.1 %**, 3+ 1.3 %) was measured over *all* anchors. But **90.7 % of
the store's 4.34 M anchors are multi-word-expression entries**, and 98.1 % of those have exactly
two senses — where, in 93.5 % of cases, the lower-id one is a completely bare shell carrying
nothing at all. The 90.1 % was a fact about how MWEs are exported, not about words.

Measured on the D8 seed pool T14 will actually be generated from (72,561 lemmas), counting
every `pomen:` node as the old spec said, the distribution is **1: 37.8 %, 2: 17.7 %, 3: 15.3 %,
4: 8.7 %, 5: 5.4 %, >5: 15.0 %** — majority-class baseline **37.8 %**, not 90.1 %.

That number is defensible but the answer still is not, because it counts empty shells and the
entry-reified placeholder. T14 now counts **the senses T12 lists** (Group D, `sense_class`),
seeded from the 40,926 pool lemmas that have at least one: **1: 37.1 %, 2: 25.1 %, 3: 12.8 %,
4: 7.4 %, 5: 4.8 %, >5: 12.9 %**, baseline **37.1 %**. Same difficulty, true gold, and the count
is the length of T12's list by construction. C12 is closed; option (2) solved a problem that
did not exist.

**What made the rest easy.** Two measurements did most of the work. No word-form surface
anywhere contains `,`, `;`, `|` or `/` — so positional comma-separated paradigm lines are safe
and `/` is free as a gap marker. And every gradeable string is already present verbatim in the
model's input (§0.8), so no type needs semantic comparison.

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

### Output template

```
ODGOVOR: {ed.im}, {ed.rod}, {ed.daj}, {ed.tož}, {ed.mest}, {ed.or}; {dv.×6}; {mn.×6}
```

| | |
|---|---|
| arity | **18, fixed** — six cases × three numbers, positionally graded |
| sep | `, ` within a number, `; ` between numbers |
| order | ednina; dvojina; množina — inside each: im, rod, daj, tož, mest, or — **graded** (position is the case) |
| gap | `/` |
| regex | `^ODGOVOR: (?:[^,;]+(?:, [^,;]+){5}; ){2}[^,;]+(?:, [^,;]+){5}$` |
| example | `ODGOVOR: gora, gore, gori, goro, gori, goro; gori, gora, gorama, gori, gorah, gorama; gore, gora, goram, gore, gorah, gorami` |
- **Seed filter (added by the audit):** skip entries where any (case, number) slot holds more
  than one surface. Measured at **0.2 % of nouns, 0 % of verbs and adjectives**, and every
  instance is `tožilnik, ednina` on a masculine noun — the animacy alternation
  (*vidim Mercator* / *vidim Mercatorja*). `lexinfo:animate` exists in the RDF but is **not
  rendered** (§0.6), so the model cannot see which reading is meant; asking would be unfair as
  well as ungradeable. Filtering costs 0.2 % of the pool and keeps every line strictly
  positional.
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

### Output template

```
ODGOVOR: {število}: {im}, {rod}, {daj}, {tož}, {mest}, {or}
```

| | |
|---|---|
| arity | **6, fixed**, after the number label |
| sep | `, ` |
| order | im, rod, daj, tož, mest, or — **graded** (position is the case) |
| gap | `/` |
| regex | `^ODGOVOR: (ednina\|dvojina\|množina): [^,]+(?:, [^,]+){5}$` |
| example | `ODGOVOR: ednina: gora, gore, gori, goro, gori, goro` |

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

### Output template

```
ODGOVOR: {oblika}
```

| | |
|---|---|
| arity | **1** — compared whole, never split |
| sep | — |
| order | — (arity 1) |
| gap | the sentinel (a missing cell is a negative, not a `/`) |
| regex | `^ODGOVOR: \S.*$` |
| example | `ODGOVOR: gori` |

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
- **Gradeable line:** **always a set**, comma-separated and alphabetically ordered, even when it
  has exactly one member. *(The earlier wording — "`ODGOVOR: gora`, or the comma-separated set
  for the ambiguous slice" — gave the type two shapes and left the grader guessing which was
  intended for a one-member answer.)*
- **UI part:** lemma + POS + which cell the given form is, which links this type to T21.
- **Negative:** a surface string that is not in the index at all.

### Output template

```
ODGOVOR: {lema}[, {lema}]…
```

| | |
|---|---|
| arity | **open, ≥ 1** — membership graded; one member is not a special case |
| sep | `, ` (safe: no lemma surface contains a comma) |
| order | `sl_key` alphabetical — **not graded** (§0.8.1); the candidate lemmas of a form are a set |
| gap | — |
| regex | `^ODGOVOR: [^,]+(?:, [^,]+)*$` |
| example | `ODGOVOR: gora` · `ODGOVOR: gol, gola` |

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
- **Gradeable line:** **always full case+number pairs**, joined with ` ali `, canonically
  ordered. *(This replaces the earlier two-shape rule — `tožilnik ali orodnik, ednina` when the
  number was shared, `rodilnik ednine ali imenovalnik množine` when it was not. Two shapes for
  one fact meant the grader had to decide which the model "meant"; one shape removes the
  question. It costs a repeated `ednine` and buys a template that is machine-checkable with a
  single regex.)*
- **Grading:** exact match after canonical ordering — the ordering rule lives in the generator,
  never in the renderer.
- **Shared with T20.** T20 is this analysis with a sentence around it; both emit this exact
  template, for the same reason T8/T9/T10 share a normalizer.
- **Negative:** a form string that does not belong to the named lemma at all.

### Output template

```
ODGOVOR: {sklon} {števila}[ ali {sklon} {števila}]…
```

| | |
|---|---|
| arity | **open, ≥ 1** — one reading is not a special case |
| sep | ` ali ` |
| order | case order (im, rod, daj, tož, mest, or), then number order (ed, dv, mn) — **not graded** (§0.8.1): the readings of an ambiguous form are a set, and the case sequence is our convention, not the data's |
| gap | — |
| regex | `^ODGOVOR: \w+ (ednine\|dvojine\|množine)(?: ali \w+ (?:ednine\|dvojine\|množine))*$` |
| example | `ODGOVOR: tožilnik ednine ali orodnik ednine` · `ODGOVOR: rodilnik ednine ali imenovalnik množine` |

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

### Output template

```
ODGOVOR: {1ed}, {2ed}, {3ed}, {1dv}, {2dv}, {3dv}, {1mn}, {2mn}, {3mn}
```

| | |
|---|---|
| arity | **9, fixed**, positionally graded |
| sep | `, ` |
| order | person-major within number: 1ed, 2ed, 3ed, 1dv, 2dv, 3dv, 1mn, 2mn, 3mn — **graded** (position is the person/number cell) |
| gap | `/` |
| regex | `^ODGOVOR: [^,]+(?:, [^,]+){8}$` |
| example | `ODGOVOR: delam, delaš, dela, delava, delata, delata, delamo, delate, delajo` |

No verb in the sample had a slot with competing surfaces (0 of 188), so unlike T1 this type
needs no doublet filter.

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

### Output template

```
ODGOVOR: {čas}: {1ed}, {2ed}, {3ed}, {1dv}, {2dv}, {3dv}, {1mn}, {2mn}, {3mn}
```

| | |
|---|---|
| arity | **9, fixed**, after the tense label |
| sep | `, ` |
| order | as T5 — **graded** |
| gap | `/` |
| regex | `^ODGOVOR: sedanjik: [^,]+(?:, [^,]+){8}$` |
| example | `ODGOVOR: sedanjik: delam, delaš, dela, delava, delata, delata, delamo, delate, delajo` |

Under M2 option (1) the label vocabulary is `sedanjik` only; a request for preteklik or
prihodnjik takes the sentinel, not a fabricated periphrastic row.

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
- **Gradeable line:** three labelled slots, **always all three, in this order**. An absent form
  is `/`; the slot is never dropped. *(The earlier wording allowed a missing slot to be omitted,
  which made arity variable — and a variable-arity positional answer cannot be graded
  positionally at all.)*
- **Negative:** a verb with no supine recorded → `/` in that slot. Reserve the sentinel for a
  verb with none of the three.

### Output template

```
ODGOVOR: nedoločnik: {inf}; namenilnik: {sup}; velelnik: {imp}
```

| | |
|---|---|
| arity | **3, fixed** — labelled and positional |
| sep | `; ` between slots, `: ` after each label |
| order | nedoločnik, namenilnik, velelnik — **graded** (slots are labelled *and* positional) |
| gap | `/` |
| regex | `^ODGOVOR: nedoločnik: [^;]+; namenilnik: [^;]+; velelnik: [^;]+$` |
| example | `ODGOVOR: nedoločnik: zaripniti; namenilnik: zaripnit; velelnik: zaripni` |

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

**Word class, plus the one lexeme-level property that word class carries.** A single fixed
schema, keyed on POS and on nothing else.

The phrasings run from `V katero vrsto besed uvrščamo besedo <L>?` to `Kakšne so slovnične
lastnosti besede <L>?`, which reads as though the second asks for more than the first. It does
not. The reference CSV settles this empirically: 100 rows, 11 distinct question frames, and
only **six** distinct answer shapes — one per POS, with no correlation between frame and
shape. There is **one answer contract**, and that is what makes the type gradeable: a Tier A
phrasing withheld from training has to land on a contract the model already knows, which a
per-phrasing contract could never guarantee.

### The schema

Every slot is lexeme-level. **Case, number, person, vform and degree never appear** — they
describe the citation *form*, not the word, and are T1–T7's material. The anchor
`iztočnica: bežnica (samostalnik, imenovalnik, ednina)` says the headword is *rendered* in the
nominative singular; it does not say that *bežnica* "is nominative singular", and putting that
in the gold answer would teach a falsehood.

| `partOfSpeech` | entries | `ODGOVOR:` |
|---|--:|---|
| `noun` | 310,362 | `samostalnik, <spol>` — gender slot blocked by **M4**, see below |
| `adjective` | 52,634 | `pridevnik` |
| `verb` | 18,159 | `glagol, <vid>` |
| `adverb` | 15,228 | `prislov` |
| `numeral` | 2,434 | `števnik` |
| `interjection` | 507 | `medmet` |
| `pronoun` | 284 | `zaimek` |
| `residual` | 219 | *excluded* |
| `abbreviation` | 129 | `okrajšava` |
| `preposition` | 97 | `predlog` |
| `particle` | 69 | `členek` |
| `conjunction` | 58 | `veznik` |

That is the **complete** `partOfSpeech` object vocabulary and it sums to exactly the 400,180
entries of §0.6 — verified against `kg_raw/OntoLex DSB/` on 2026-08-20.
`<vid>` ∈ {`dovršni`, `nedovršni`, `dvovidski`}, `<spol>` ∈ {`moški`, `ženski`, `srednji`} `spol`.

Three consequences of that vocabulary, each of which overrides an earlier draft of this
section:

**There is no `properNoun` and no `commonNoun` in this KG** — zero triples, zero files. The
reference CSV puts `lastno ime` / `občno ime` in **70 of its 100 answers**; not one of them is
derivable here. The builder's `POS_SL` still carries a dead `"properNoun": "lastno ime"` key
that can never fire. Drop the distinction entirely, and do not reconstruct it from
capitalisation — the CSV appears to have tried something of the sort and labelled
`physiological` a Slovene proper noun.

**`residual` is excluded from the seed pool.** It is the tag set's garbage class (foreign
words, symbols, typos) and has no name in the Slovene grammatical tradition. It is also absent
from `POS_SL`, so `feat_string()` drops it silently: those 219 anchors render with case and
number but **no POS at all**. From the model's side the word class is simply not there, so
there is nothing to answer. They would be a legitimate flavour-(b) negative if more were ever
needed; 219 entries do not justify the special case.

**`abbreviation` is included, rendered `okrajšava`.** It is not a besedna vrsta in the school
tradition, but it is what the anchor the model reads actually says
(`POS_SL["abbreviation"] = "okrajšava"`). Gold that contradicts the input is worse than gold
that is taxonomically loose.

### How to implement

- **Source:** `lexinfo:partOfSpeech` on the **anchor** — single-word entries only; MWEs carry
  no POS at all, which is why only **9.2 %** of all anchors show a POS parenthetical (91 % of
  anchors are MWEs). Restricted to word anchors it is **99.9 %**. Aspect comes from
  `UNIT_PROPS` on the same node and has been rendered since v4.
- **Seed filter:** word anchors carrying a POS from the table above, `residual` excluded.
  Automatic under D8's core-entry pool.
- **Gradeable line:** POS first, then the property, in that fixed order —
  `ODGOVOR: samostalnik, ženski spol`, `ODGOVOR: glagol, dovršni`, `ODGOVOR: prislov`.
- **Shared normalizer with T9 and T10, mandatory.** T8's noun answer *is* T9's answer with the
  POS prepended, and its verb answer *is* T10's. If the three types spell the same fact
  differently — `moški spol` in one and `moški` in another — the supervision contradicts
  itself on the one fact all three assert. One function, called by all three.
- **Property missing but POS present:** fall back to POS alone. The sentinel is for a missing
  *word class*, not a missing gender — the word class genuinely is recorded.
- **Negative:** an MWE seed (no POS recorded anywhere) → `ODGOVOR: ni podatka v bazi`. A
  truthful negative that teaches a real boundary of the resource.
- **Blocked slot (M4).** Noun gender lives on the lexical-unit and `UNIT_PROPS` omits it, so it
  is not in the v4 store at all. Until the one-line builder fix and the rebuild land, the noun
  row answers `ODGOVOR: samostalnik` and nothing more — never invent the gender. This is the
  one decision T8 is waiting on; see §0.7 (M4).

### Output template

```
ODGOVOR: {besedna_vrsta}[, {lastnost}]
```

| | |
|---|---|
| arity | **1 or 2, fixed by POS** — 2 for nouns (after M4) and verbs, 1 for everything else |
| sep | `, ` |
| order | word class first, then the property — **graded** (two semantic slots) |
| gap | — (a missing property drops the slot; it never becomes `/`) |
| regex | `^ODGOVOR: (samostalnik\|pridevnik\|glagol\|prislov\|števnik\|medmet\|zaimek\|okrajšava\|predlog\|členek\|veznik)(?:, (moški spol\|ženski spol\|srednji spol\|dovršni\|nedovršni\|dvovidski))?$` |
| example | `ODGOVOR: samostalnik, ženski spol` · `ODGOVOR: glagol, dovršni` · `ODGOVOR: prislov` |

Both vocabularies are **closed and complete** — 11 word classes (the 12th, `residual`, is
excluded) and 6 property values — so the regex is not a shape heuristic here but an exhaustive
validator. This is the most tightly checkable type in the inventory after T10.

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

…and the **six** answer shapes those 11 frames produce across all 100 rows. This is the
evidence that the type has one contract, not one per phrasing:

| n | shape |
|--:|---|
| 59 | `Beseda **<L>** je samostalnik. Beseda je lastno ime in je <spol> spola.` |
| 22 | `Beseda **<L>** je pridevnik.` |
| 11 | `Beseda **<L>** je samostalnik. Beseda je občno ime in je <spol> spola.` |
| 5 | `Beseda **<L>** je prislov.` |
| 2 | `Beseda **<L>** je glagol. Je nedovršni glagol.` |
| 1 | `Beseda **<L>** je števnik.` |

Note what those frames were applied *to*: `jacinto, physiological, boniciolli, corbacho,
bimhuis, sörenstam, macbeth, 6500, mmmmmm, sčg, klu, smoothtouch, practices` — near-uniformly
foreign surnames and OCR noise, with 59 rows labelled `lastno ime` while only 2 of the 100
lemmas are even capitalised. Independent confirmation that D8's core-entry pool was the right
call. Take the frames; take nothing else.

Older JSON file, `oblikoslovna/part_of_speech`, n = 104 — a **different task** wearing the same
label, and the source of this section's earlier over-breadth:

| n / 104 | property | reusable? |
|--:|---|:--:|
| 48 | target embedded in a sentence or phrase | ✅ as phrasing |
| 25 | answer names a **subtype**: `svojilni zaimek`, `vprašalni zaimek`, `priredni veznik` | ❌ not in the KG |
| 15 | question carries a format instruction (`Odgovori samo z oznako brez ločil`) | ❌ D6 fixes the format |
| 14 | yes/no answer (`Da, »ali« je veznik.`) | ✅ as phrasing |
| 10 | demands JSON output against a supplied schema | ❌ |
| 7 | role-play preamble (`Si bibliotekarka, ki …`) | ❌ |
| 2 | hedged (`praviloma prislov; v nekaterih rabah tudi …`) | ❌ |

Its POS distribution is function-word heavy — `zaimek` 26, `samostalnik` 23, `pridevnik` 19,
`glagol` 18, `prislov` 7, `veznik` 5, `predlog` 5, `medmet` 3 — which our pool could not supply
even if we wanted it to: the entire KG holds 284 pronoun, 97 preposition, 69 particle and 58
conjunction *entries*. **Reuse the sentence-embedded frames as phrasing; reuse none of the
answer styles.**

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

The sentence-embedded frames are worth reviving: they exercise the D3 extractor on a question
that is not a bare template, and R5 says the multi-entity rate has to be measured on real
strings.

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
Slovnično opredeli <L>.
```

**Dropped from the earlier draft of this list:** `Kaj vse mi lahko poveš o besedi <L> z vidika
slovnice?` — unbounded, and no fixed schema can make it gradeable; that is summarisation, not
this type. `Kako bi besedo <L> slovnično opredelil?` is kept but should stay **out of the Tier
A withheld set**: it is the loosest of the frames, and an unseen phrasing is exactly where a
loose frame does the most damage.

---

## T9 — `besedna_vrsta/spol_samostalnika`

### What it is

**Grammatical gender of a noun** — moški, ženski or srednji spol. One word of answer.

### How to implement

- **Source:** `lexinfo:gender` on the **anchor's lexical-unit**, not on its forms — 310,362
  triples, one per noun entry, so coverage is 100 % of the pool. *(Corrected 2026-08-20: this
  section previously said "the gender of the nominative singular form" and told the generator
  to check the paradigm for consistency. Noun word-forms carry no gender at all in this KG;
  there is one value per entry and nothing to reconcile.)*
- **Blocked on M4.** That unit-level gender is not rendered into the node text, so it is absent
  from the v4 store and this type **cannot be generated today**. One-line builder fix plus a
  ~35-minute rebuild; see §0.7 (M4). Everything below assumes it has landed.
- **Seed filter:** POS = noun.
- **Gradeable line:** `ODGOVOR: moški spol`.
- **No gender ambiguity exists in this KG.** 310,362 gender triples over 310,362 *distinct*
  lexical-unit subjects — exactly one value each, and that count matches the noun total
  exactly. The earlier note here ("some nouns are recorded with two genders, answer with
  both") described a case the data does not contain; there is nothing to reconcile and no
  `moški ali ženski spol` answer to generate.

- **Negative:** ask the gender of a verb or adverb. This is the single easiest ordinary-lemma
  negative in the whole inventory and should carry a good share of this type's negative
  quota.

### Output template

```
ODGOVOR: {spol}
```

| | |
|---|---|
| arity | **1** |
| sep | — |
| order | — (arity 1) |
| gap | — |
| regex | `^ODGOVOR: (moški\|ženski\|srednji) spol$` |
| example | `ODGOVOR: moški spol` |

The three-value vocabulary is shared verbatim with T8's second slot — same string, same
normalizer, no `moški` / `moški spol` drift.

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

### Output template

```
ODGOVOR: {vid}
```

| | |
|---|---|
| arity | **1** |
| sep | — |
| order | — (arity 1) |
| gap | — |
| regex | `^ODGOVOR: (dovršni\|nedovršni\|dvovidski)$` |
| example | `ODGOVOR: dvovidski` |

Three values, 100 % coverage of the verb pool, one closed regex: **the most cleanly verifiable
type in the inventory.** Use it as the pipeline smoke test alongside T14.

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
- **Partial paradigms need a gap marker.** There are 107,717 comparatives but only 72,856
  superlatives, so ~32 % of comparative-bearing entries have no superlative. Those emit `/` in
  the third slot rather than a two-item answer — arity stays fixed at 3.

### Output template

```
ODGOVOR: {osnovnik}, {primernik}, {presežnik}
```

| | |
|---|---|
| arity | **3, fixed**, positionally graded |
| sep | `, ` |
| order | osnovnik, primernik, presežnik — **graded** (position is the degree) |
| gap | `/` |
| regex | `^ODGOVOR: [^,]+, [^,]+, [^,]+$` |
| example | `ODGOVOR: lep, lepši, najlepši` · `ODGOVOR: brunast, brunastejši, /` |

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

Shared constraint for both types in this group: **definition coverage is thin.** Only **225,618** `@sl`
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

### What counts as a `pomen` — the shared `sense_class` contract

Measured against `kg_graph_v5_gemma3`, 2026-08-21. **A `pomen:` node is not a meaning.** It is
one `ontolex:LexicalSense` IRI, and the export mints those for structural reasons that have
nothing to do with polysemy. Both types in this group read the same classification, and
**T14's number is the length of T12's list, by construction.**

The store's senses come in two layers:

| layer | how to recognise it | what it carries |
|---|---|---|
| **native** | `sense id == entry id`, one per core entry | the entry's collocations, forms and synonyms — `otrok`'s has **degree 3,824** — and **never a definition** |
| **imported** | any other sense id | the definitions; **77.6 %** of them attach to nothing but the entry |

Verified over the whole store: 100,801 entries carry a native sense — that is **every** core
(`id < 1 M`) word entry and only those; **100 %** render as `pomen 1:` and **0 %** carry a
definition. In the D8 pool the split is 100,801 native (all fallback) against 173,168 imported
(127,082 defined, 46,086 fallback), and **49.0 %** of the pool's sense nodes are degree-1
shells with nothing attached at all. Raw-dump confirmation: `sense-97358` on
`lexical-unit-97358` has exactly two triples in all 42 GB — `rdf:type LexicalSense` and
`isSenseOf`. It is a shell. Its sibling `sense-16281180` carries `"del sistema dihal"@sl`.

One function, called by both generators — never two implementations:

```
sense_class(sense, anchor) ->
    placeholder   sense id == the entry's own id.  The entry reified: `pomen 1: <lemma>`,
                  never defined, often the busiest node in the neighbourhood.
    fallback      text is the headword repeated, optionally + ` (zgled: …)`.
    defined       anything else — the sense carries a real skos:definition.
```

**Only `defined` counts, for either type.** That rule survives a test the alternatives fail:
the model sees node **text, not ids**, and `pomen 1: otrok` (placeholder) is identical in shape
to `pomen 2: otrok` (an ordinary fallback). Any policy that keeps one and drops the other is
not derivable from the model's input. *"Count a sense iff its text is not the headword
repeated"* is. Four policies were scored on the pool before this was settled; the alternatives
— drop the placeholder only, or keep fallbacks that have material attached — disagree with
this one on **31.2 %** of anchors and push 37.8 % of the pool to an answer of `0`.

**Reliability.** Checked against the RDF itself (every sense id carrying a `skos:definition`,
grepped from the dump and joined back to the store): **154** of the pool's senses carry a
definition whose text *is* the headword — `pomen 8: imeti`, `pomen: Shakespearejev` — and are
classed `fallback`, which is the wanted behaviour, since such an item would be a gold answer
that repeats the question. A further **363** are classed `defined` without a matching
`skos:definition` in my extraction — but the extraction itself lost ~528 sense ids to
interleaved parallel `grep` output, which accounts for them; re-derive from the builder's `dfn`
map rather than from a grep if the residual matters. The rule is otherwise exact, because
**every sense belongs to exactly one entry** — verified over all 8,468,227, zero shared — so
"text equals the anchor's lemma" is never ambiguous.

**What no filter can fix.** The imported layer records *several glosses per meaning*:
`fotograf` gets both *kdor se ukvarja s fotografiranjem* and *oseba, ki fotografira poklicno*;
`dekan` has 3, `dekanica` 4, `dekanja` 4 for the same content; `dvigniti` has 36. The KG never
says which records describe the same sense, so the true sense count is not recoverable.
**Neither type may ask how many meanings a word *has*** — only how many are recorded in this
database. T14's wording follows from that, not from style.

---

## T12 — `pomen/razlaga_pomena` (absorbs the former T13, `nastevanje_pomenov`)

### What it is

**Give the recorded meaning(s) of the word** — every sense that carries a real definition, in
dictionary-ordinal order. Monosemous and polysemous entries take the **same** form: a one-item
answer is not a special case, which is precisely what makes the merge clean.

**Why this was two types and is now one.** The reference dataset separates `razlaga_pomena`
("what does X mean?") from `nastevanje_pomenov` ("list the senses of X"). Against this KG the
two produce the *same gold answer* — the sense list **is** the definition list — so the split
bought nothing but a second inventory slot, a second template pool, and a standing risk of
contradictory gold for one lemma under two labels. Both template pools are merged below and
both question styles map to the one contract. **T13 is retired; T14–T21 keep their numbers so
cross-references elsewhere stay valid.** The inventory is 20 types.

### What the definitions actually look like

Measured over all 230,606 `@sl` `skos:definition` literals, 2026-08-20:

| property | value |
|---|---|
| length | median **5 words / 35 chars**; p75 9; p95 17; max 84 |
| ends with `.` | 8.4 % |
| starts uppercase | 6.0 % |
| contains `,` | 33.0 % |
| contains `;` | 8.5 % |
| contains `\|` or a newline | **0** |
| 1–2 words | 21.7 % (49,987) |
| empty | 26 |

These are **terse glosses, not sentences**: `pomen 1: zgradba za bivanje`, not *"Hiša je
zgradba, v kateri ljudje bivajo."* The earlier draft of this section illustrated the type with
a full-sentence example ending in a period — that style came from the reference file's
WordNet/bridge-dictionary imports and does not exist in this KG. The shortness is what makes
exact-match grading practical rather than brittle (§0.8).

### How to implement

- **Read the gold from the store, not from the RDF.** The builder collapses a sense's
  definitions to one string (`dfn` is a `sense -> str` map, last write wins; 4,980 senses carry
  more than one in the RDF), and it is the *store* text the model is shown. Taking gold from the
  node text guarantees the answer string is present verbatim in the input — the property the
  whole grading contract rests on.
- **Which senses count:** exactly those `sense_class` calls `defined` (Group D). `body = d or
  lemma_of_lu(lu)`, so a sense with no definition renders as the entry's own lemma, optionally
  plus ` (zgled: …)`. Strip that suffix; if what remains equals the anchor's lemma, the sense is
  `fallback` (or the `placeholder`) and is excluded. Without this check the type emits items
  whose gold answer is the headword repeated. **Call the shared function — T14 counts the list
  this produces, and a second implementation is how the two silently drift apart.**
- **Seed filter:** at least one `defined` sense. **56.4 %** of the 72,561-lemma D8 pool qualify
  (40,926 lemmas); the often-quoted **40.6 %** is the same count over the *unfiltered* 100,801
  core word entries, not over the pool — don't mix the denominators. Among tail lemmas 7.3 %.
- **Ordering:** dictionary ordinal ascending, as rendered in `pomen N:`. Senses with no ordinal
  sort last, stable by sense id.
- **Gradeable line:** the definition texts only, `|`-separated, in ordinal order, **no
  numbering**:
  `ODGOVOR: zgradba za bivanje | rodbina, družina`
  Ordinals are omitted deliberately — the order already carries them, and requiring the model to
  also emit `1.`/`2.` adds a failure mode that measures counting rather than retrieval. Note the
  comma inside the second item: this is exactly why `|` and not `,` (§0.8).
- **UI part:** one sense per line, numbered with the **graph's** ordinal so that a gap (sense 2
  undefined) is visible. This is where the user-facing "one definition per line" rendering
  lives; the gradeable line stays single-line so the eval parser is one rule for all 20 types.
- **Deduplicate identical definitions within an entry.** 292 pool anchors (**0.40 %**) carry the
  same definition on two senses — `naslikati` → *ustvariti sliko* twice. Emit it once. **T14 must
  dedup identically**, or the count stops matching the list.
- **Junk items.** Exclude the 26 empty definitions. For the 21.7 % that are 1–2 words, recommend
  **keeping** them — they are genuine lexicographic data, and dropping a fifth of the pool to
  make the metric look harder is the wrong trade. Instead report `exact` broken down by gold
  length, so a score dominated by one-word glosses is visible rather than hidden.
- **Sense *labels* stored as definitions — keep them, but know they are there.** The former T13's
  warning was dismissed on too narrow a test (2 of 230,606 definitions begin with `raba`). The
  class does exist in this KG, it just uses other forms: **15,210 (12.0 %)** of the pool's
  definitions begin `o …` — `leten` → *o starosti*, `vnaprej` → *o prihodnosti*, `otrok` → *v
  razmerju do staršev* — which is the "used of X" label, not a paraphrase. Also `zveza: hočem
  reči` (200), `zanikano` (15), `nerazvrščeno` (3), `npr. streho` (50), and **32 unanswered
  editor's notes** shipped verbatim: `obsegati` → *vsebovati?*, `dvigniti` → *dati del telesa ali
  nečesa (pokrov) v drug položaj - ali dati sem samo telo?*. **Recommendation: keep them.**
  Filtering would change both the list and T14's count for a reason **invisible in the node text
  the model reads**, which is the same property that decided `sense_class` in the first place.
  Report a breakdown instead. If they are ever filtered, both types must filter identically.
- **Negative:** **43.6 % of the D8 pool has no defined sense** (31,635 lemmas) — the largest
  natural negative pool in the dataset, and **shared with T14**: the same lemma answers
  `ODGOVOR: ni podatka v bazi` under both types, never `0` under one and a list under the other.
  The UI part offers what *is* attached (examples, collocations, synonyms). A monosemous entry is
  **not** a negative; it takes an ordinary one-item answer. *(The "59 % of core lemmas" figure
  elsewhere is over the unfiltered 100,801 core entries — see the seed-filter bullet.)*
- **Relation to T14.** They no longer disagree: **T14's gold is the length of this list**, from
  the same `sense_class` call, the same dedup and the same seed pool. The earlier design let them
  differ and papered over it in the UI text; that taught the training data to say *ima 4 pomene*
  next to a two-item list for one lemma.
- **Expected arity.** Among the 40,926 anchors with at least one defined sense: **1: 37.1 %,
  2: 25.1 %, 3: 12.8 %, 4: 7.4 %, 5: 4.8 %, >5: 12.9 %.** One item is the plurality but not the
  majority, which is another reason the merged type must not treat it as a special shape.

### Output template

```
ODGOVOR: {razlaga}[ | {razlaga}]…
```

| | |
|---|---|
| arity | **open, ≥ 1** — membership graded, order significant |
| sep | ` \| ` (mandatory: 33.0 % of definitions contain a comma, 8.5 % a semicolon, **0 %** a pipe) |
| order | dictionary ordinal ascending; ordinal-less senses last, stable by sense id — **graded** (the ordinal is data, not our convention) |
| gap | — (an undefined sense is omitted, not `/`; the gap is shown in the UI part) |
| regex | `^ODGOVOR: .+(?: \| .+)*$` |
| example | `ODGOVOR: zgradba za bivanje \| rodbina, družina` |

### Existing formulations

Both merged pools follow. First, the CSV's `razlaga_pomena`, n = 100:

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

And the CSV's `nastevanje_pomenov`, n = 100 — the "list them" half of the merged pool:

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

One merged pool. The two halves are kept visibly apart because they must stay **balanced** in
sampling: if the "define it" phrasings dominate, the model learns to answer with one definition
and silently drop the later senses of polysemous entries — the exact failure the merge is
supposed to make measurable.

*"Define it" phrasings — natural for a one-item answer, must still produce all items:*

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

*"List them" phrasings — natural for many items, must still work when there is only one:*

```
Katere pomene ima <L>?
Naštej pomene besede <L>.
Ali ima <L> več pomenov? Naštej jih.
Zanimajo me vsi pomeni besede <L>.
Beseda <L> — kateri pomeni so zabeleženi?
Izpiši pomene besede <L> po vrsti.
Kaj vse lahko pomeni <L>?
Prikaži vse registrirane pomene za <L>.
Rabim seznam pomenov besede <L>.
```

`Kaj je <L>?` and `Ne poznam besede <L>. Kaj pomeni?` are the most realistic public-user
phrasings in this whole document, and neither appears in either reference file — both are good
**Tier A** candidates.

**Dropped in the merge:** `Koliko različnih stvari lahko pomeni <L>?` — it asks for a count, so
it belongs to T14, and leaving it here would give one surface two different gold answers. It is
not usable in T14 either, as written: it asks what the *word* can mean rather than what the base
records, which T14's ⚠️ bullet rules out. Reword to `Koliko pomenov je za <L> zabeleženih?` or
drop it.

---

## T14 — `pomen/stevilka_pomenov`

### What it is

**Count the recorded senses.** A number, optionally followed by short sense heads.

### How to implement

- **Source: the senses T12 lists.** `len(T12_items(anchor))` — the same `sense_class` call
  (Group D), the same dedup, the same seed pool, obtained by **calling the same function**, not
  by reimplementing the filter. If the two ever disagree on a lemma, the generator has a bug —
  C17 asserts they never do.
- **Seed only anchors with at least one `defined` sense** — 40,926 of the 72,561-lemma pool. The
  remaining 31,635 are the **shared negative**: `ODGOVOR: ni podatka v bazi`, the same string T12
  emits, so the type never has to answer `0` and no lemma gets a number from one type and a
  refusal from the other.
- **Fix the agreement bug.** The reference file gets Slovene number agreement wrong in
  **100/100 rows** ("*ima registrirana 1 pomen*"). Correct forms, in the database framing the
  bullet below requires:
  - 1 → `v bazi je zabeležen 1 pomen`
  - 2 → `v bazi sta zabeležena 2 pomena`
  - 3–4 → `v bazi so zabeleženi 3 pomeni`
  - 5+ → `v bazi je zabeleženih 5 pomenov`
  Note that the verb and the participle agree too, not just the noun — a table lookup on
  `n mod 100` returning the whole phrase, not a heuristic, and unit-tested (C7). The case is a
  property of the frame, so a UI sentence that keeps a `ima` frame needs the accusative table
  instead (`ima 1 pomen` / `2 pomena` / `3 pomene` / `5 pomenov`). Pick one frame per type and
  keep it.
- **Gradeable line:** the bare integer — `ODGOVOR: 3`. Exact-match gradeable with zero
  ambiguity, which makes it a natural smoke test for the pipeline.
- **The answer distribution, on the pool as now seeded:** **1: 37.1 %, 2: 25.1 %, 3: 12.8 %,
  4: 7.4 %, 5: 4.8 %, >5: 12.9 %** — majority-class baseline **37.1 %**, reported beside every
  score (C13). *(The old "90.1 % answer 2" figure was measured over all 4.34 M anchors, 90.7 % of
  which are MWE entries whose second sense is an empty shell — see §1.1. It was never a fact
  about words.)*
- **⚠️ Ask about the database, not about Slovene.** The KG records several glosses per meaning
  and never says which describe the same sense (`fotograf` 2, `dekan` 3 vs `dekanica` 4,
  `dvigniti` 36 — Group D), so *"koliko pomenov **ima** beseda X"* has no true answer here.
  Question and UI text must both say **zabeleženih / razloženih v bazi**. This is a correctness
  constraint, not a stylistic one: it is what makes the gold true.
- **Counting policy — settled.** Only `defined` senses count. The entry-reified `placeholder`
  does not (it is the entry, not a meaning), and neither does a `fallback` sense whose text is
  the headword repeated. Rationale and the three rejected alternatives are in Group D; the short
  version is that no other policy is derivable from the text the model is shown.
- **What this type actually measures.** With the shared filter, T14 is `len(T12)` — it is a
  **format smoke test and a count-vs-list consistency probe** ("does the model's number match
  the list it would give?"), not an independent capability. That is a real thing to measure and
  worth keeping, but it must not be read as sense knowledge. Recorded here because the doc's
  earlier option 3 (drop T14) becomes reasonable if the probe is not wanted.
- **Negative:** shared with T12 — a lemma with no defined sense. 31,635 of the pool.

### Output template

```
ODGOVOR: {n}
```

| | |
|---|---|
| arity | **1**, a decimal integer, no words, no agreement |
| sep | — |
| order | — (arity 1) |
| gap | — |
| regex | `^ODGOVOR: [1-9][0-9]*$` — **`0` is deliberately not allowed**; a lemma with no defined sense is a §0.2 negative, not a count of zero, and a `0` reaching the gold means the seed filter leaked |
| example | `ODGOVOR: 3` |

The Slovene agreement table (see the implementation bullet) belongs to the **UI part only**.
Keeping the number bare in the gradeable line means the agreement bug the reference file has in
100/100 rows cannot affect the score.

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

**Six of these ten are unusable as written.** `Ali ima beseda <L> več pomenov?`, `Koliko pomenov
ima beseda <L>?`, `Ali je beseda <L> enopomenska ali večpomenska?`, `Koliko ločenih pomenov ima
lema <L>?`, `Preveri večpomenskost besede <L>.` and `Koliko leksikalnih pomenov vsebuje beseda
<L>?` all ask what the **word** has; our gold can only say what the **database records**, and the
two differ wherever the imported layer duplicates a gloss. The four `število … registriranih /
zabeleženih` frames are already database-framed and are the model to follow.

### Suggested new templates

Every frame names the database. This is a correctness requirement (see the ⚠️ bullet above), so
new frames must keep it — a phrasing that asks about Slovene rather than about the entry makes
the gold false no matter how the count is computed.

```
Koliko pomenov je za <L> zabeleženih v bazi?
Koliko različnih pomenov je zabeleženih za <L>?
Koliko pomenov besede <L> je razloženih v slovarju?
Število zabeleženih pomenov za <L>?
Koliko pomenskih razlag ima geslo <L>?
Povej, koliko pomenov je pri <L> opisanih.
Koliko razlag je v bazi pri besedi <L>?
Za koliko pomenov besede <L> obstaja razlaga?
Koliko ločenih pomenov je pri geslu <L> popisanih?
Zanima me, koliko pomenov je za <L> registriranih.
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
- **Cap the list at 10.** Measured over 523 sampled synonym items: 41 % of anchors have exactly
  one partner, but **7 % have 11 or more**. An uncapped answer makes both grading and reading
  meaningless, so take the **first 10 in canonical order** and say in the UI part that the list
  was truncated. The cap is part of the gold, not a rendering choice.

### Output template

```
ODGOVOR: {lema}[, {lema}]…
```

| | |
|---|---|
| arity | **open, 1–10** — membership graded |
| sep | `, ` (safe: 0 of 523 sampled partners contain a comma, though they run up to 11 words) |
| order | `sl_key` alphabetical — **not graded** (§0.8.1); synonyms carry no rank in the KG |
| gap | — |
| regex | `^ODGOVOR: [^,]+(?:, [^,]+){0,9}$` |
| example | `ODGOVOR: pasiven, resigniran, spokojen, vdan v usodo` |

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
- **Gradeable line:** **always a set**, exactly as T15 — comma-separated, alphabetical, even
  when it has one member (75 % of anchors do). *(The earlier "`ODGOVOR: sodelovanje` (or the
  comma-separated set)" left the one-member case with two possible shapes.)*
- **Negative:** ample — the overwhelming majority of lemmas have no antonym. Keep the
  negative share here at the dataset default; a Tier C type that is mostly sentinels measures
  nothing.

### Output template

Identical to T15 — same shape, same separator, same ordering, same grader path.

```
ODGOVOR: {lema}[, {lema}]…
```

| | |
|---|---|
| arity | **open, 1–10** (75 % of anchors have exactly 1, 12 % have 2, 12 % have 4) |
| sep | `, ` |
| order | `sl_key` alphabetical — **not graded** (§0.8.1) |
| gap | — |
| regex | `^ODGOVOR: [^,]+(?:, [^,]+){0,9}$` |
| example | `ODGOVOR: sodelovanje` |

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

# Group F — `kolokacije` — ▶ **UNPARKED, 2026-08-20**

> **Parked and unparked the same day.** The park notice said: *"Reopen when a source for the
> actual verbalisations is settled — the DDDS API endpoint is the known candidate."* No
> external source is needed. **The verbalisations are in the export.** T17/T18 are live
> again and both keep their original *besedne zveze* wording.
>
> **What was wrong.** A `frac:Collocation` node genuinely carries only `rdfs:member`,
> `frac:head` and `rdf:type` — verified across all 42 GB — and nothing in the dump ever
> points *at* one (0 object-position triples), so no traversal reaches a surface string from
> either side. But the node's IRI `dependent-sense-D-lexical-unit-H` embeds `D`, the sense of
> the **multi-word entry** that spells the pairing out, and that entry carries
> `canonicalForm → writtenRep`. The link is a naming convention, never a triple. (The trap:
> the *other* id, `H`, **is** restated as a real triple, so the convention looks redundant.)
>
> **Measured over the raw dump:** 4,717,090 / 4,717,090 collocation nodes resolve, 0
> failures, 3,744,473 distinct phrases, **75.3 % differing from their lemmas concatenated**.
> The builder writes them as of v5 (`--colloc-text phrase`), so the phrase is a node inside
> the extracted subgraph and containment holds without qualification.
>
> **The D14 lemma-pair rework is withdrawn.** Its objection — "nobody asks about collocations
> as lemma pairs" — was correct, and is now moot rather than accommodated. Both types are
> back in the verifiability audit (§1.1) and in the per-type count used for D16.
>
> Full write-up: `README.md` **Finding 8**, and §6.2 of `QA_DATASET_DESIGN.md`.

**Shared, and non-negotiable.** The collocation node's text is the **inflected phrase**:
`kolokacija: mineralna voda`, not `kolokacija: mineralen + voda`. It falls back to the pair
form only when the phrase does not resolve, which measured 0 times in 4.7 M.

The phrases are lexicographer-curated, and the pair form was losing three things at once:

| | |
|---|---|
| agreement | *cvileč + guma* → **cvileče gume** |
| function words | *odnašati + na + glava* → **odnašati na glavah** (the preposition is a constituent) |
| word order and the copula | *soba + biti + enoposteljen* → **sobe so enoposteljne** |

The uppercase items in the reference file ("*Potrpežljivost je vrlina*" — 175 of 1,307) are
**not** foreign corpus fragments as previously recorded; they are stored that way in the KG,
and they come back verbatim.

**Provenance of the reference file, settled.** Its `id` column is the KG `lexical-unit` id,
and **1,196 of its 1,307 phrases (91.5 %) are reproducible verbatim** from the export alone
(155/200 rows in full). It was not built from the DDDS API. The 8.5 % residual is 2.8 % the
same collocation in another number/case, 2.4 % anchors with no collocations at all, and
3.4 % genuinely absent — concentrated in possessive adjectives of proper names (*Bergerjev*,
*Žalgirisov*, *Charlottin*).

Coverage: **65.0 % of core lemmas** have collocations (12.2 % of tail lemmas) — 65,480 core
lemmas in total, the largest sub-pool of any relation type. 2,981,731 distinct pairings,
99.8 % binary. Median anchor offers 13 phrases; 101,861 lexical units have at least one.

---

## T17 — `kolokacije/navedi_kolokacije`

### What it is

**Which phrases does this word appear in?** The answer is a list of inflected collocations,
unnumbered and non-exhaustive.

### How to implement

- **Source:** `kolokacija:` nodes attached to the anchor's senses. Since v5 the node text
  **is** the phrase (`kolokacija: mineralna voda`), so the gold answer is read straight off
  the node with no post-processing.
- **The anchor side is a hub** *(corrected 2026-08-21)*. An earlier version of this spec
  said collocation nodes are "a leaf-like connector, not a hub — which is why wiring
  collocations in costs ~0 % at the median ball size". The degree-2 part is true of the
  collocation node, but the *sense* on the other end fans out without bound: p99 = 791,
  max = 14,233 (`pomen 1: imeti`). Extraction applies the **D5b** cap — K = 15 pooled
  across the anchor's senses.
- **Selection:** the graph carries no frequency on the collocation node, so there is no
  "top N" to report. **Draw the gold from the D5b sample** — the 15 collocations the
  extractor already put in the ball — so gold is in-ball by construction and no comparator
  has to be kept in sync. Order the drawn phrases canonically (`sl_key`) for the gradeable
  line; that ordering is presentation, not ranking.
- **Say it is a selection**, not a ranking by corpus frequency. **Do not claim
  "najpogostejše"** — the reference file does, and it cannot back the claim either.
  Deterministic ranking was tried and rejected in favour of sampling: every key biases
  toward one slice of the partner-frequency distribution, and the best-looking one merely
  returned whatever sat just under a hand-picked cutoff (D5b).
- **Gradeable line:** the phrases in canonical order:
  `ODGOVOR: mineralna voda, mineralno gnojilo, mineralna snov, mineralna surovina, mineralno olje`
- **Grading note:** match case-insensitively. 175 of the reference file's 1,307 items are
  stored capitalised (*Cvileče gume*, *Potrpežljivost je vrlina*) because that is how the
  lexicographer entered them, not because a sentence boundary leaked in.
- **Cap:** 5–10 phrases. High-band lemmas have hundreds (max 5,570); an uncapped list makes
  grading meaningless and the answer unreadable. Median anchor offers 13.
- **Negative:** 35 % of core lemmas have no collocations.

### Existing formulations

Reference CSV, n = 100. **All ten frames are reusable as written** — the D14 rework that
required rewording the *besedne zveze* frames is withdrawn, since the gold answer is once
again an inflected phrase:

| n | frame | reusable? |
|--:|---|---|
| 15 | `Navedi primerne kolokacije za besedo <L>.` | yes |
| 14 | `Katere so najbolj znane kolokacije besede <L>?` | reword — "znane" implies frequency |
| 12 | `S katerimi besedami se najpogosteje povezuje beseda <L>?` | yes |
| 9 | `Izpiši tipične besedne zveze z besedo <L>.` | **yes — restored** |
| 9 | `Katere besedne zveze so pogoste z besedo <L>?` | **yes — restored** |
| 9 | `S katerimi izrazi se kolocira beseda <L>?` | yes |
| 9 | `Poišči pogoste kolokacije za besedo <L>.` | yes |
| 8 | `Navedi tipične kolokacije z besedo <L>.` | yes |
| 8 | `Prikaži seznam pogostih kolokacij z besedo <L>.` | yes |
| 7 | `Navedi kolokativne zveze za lemo <L>.` | yes |

`Katere so najbolj znane …` is still the one to reword, for the unrelated reason that the KG
carries no frequency.

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
- **Gradeable line:** exactly N phrases, canonical order, or the honest short list. The
  grader checks both the count and the membership, case-insensitively.

### Existing formulations

Reference CSV, n = 100:

| n | frame |
|--:|---|
| 13 | `Izpiši <N> kolokacij besede <L>.` |
| 13 | `Sestavi seznam <N> kolokacij z besedo <L>.` |
| 12 | `Prikaži številčni seznam <N> kolokacij besede <L>.` |
| 12 | `Katere so <N> tipične kolokacije z besedo <L>?` |
| 11 | `Poišči <N> tipičnih besednih zvez za besedo <L>.` (**restored** — no longer needs rewording) |
| 11 | `Prikaži <N> pogostih kolokacij z besedo <L>.` |
| 9 | `Izpiši točno <N> kolokacij besede <L>.` |
| 7 | `Navedi <N> najbolj pogostih kolokacij za <L>.` (reword — frequency claim) |
| 7 | `Zapiši <N> kolokativnih primerov za besedo <L>.` |
| 5 | `Navedi <N> kolokativnih zvez za besedo <L>.` |

### Suggested new templates

```
Naštej <N> besednih zvez z besedo <L>.
Daj mi <N> kolokacij besede <L>.
Rabim <N> primerov sopojavitev za <L>.
V katerih <N> zvezah se najpogosteje pojavlja <L>?
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
- **UI part:** up to 4 examples, italicised, as the reference does. **98 % of anchors that have
  examples have exactly 4**, so "up to 4" is in practice "all of them".
- **Do not ask for a specified number** of sentences beyond 4.
- **Cap the length in the seed filter.** Examples reach **1,487 words**; median is 16, p95 is
  39, p99 is 60. A 1,487-word exact-match line is not a retrieval probe, it is a transcription
  test. **Cap at 60 words, which retains 99.0 %** of examples.
- **"First" must be deterministic:** the example child with the lowest node id. The graph's
  edge order is not a ranking and must not be treated as one.

### Output template

```
ODGOVOR: {poved}
```

| | |
|---|---|
| arity | **1** — compared whole. **Never split**: 65 % of examples contain a comma, 1.2 % a semicolon, and 4 of 51,172 contain a `\|` |
| sep | — |
| order | — (arity 1) |
| gap | — |
| regex | `^ODGOVOR: \S.*$` |
| example | `ODGOVOR: Je oče petih otrok in dedek štirih vnukov.` |

This is the one place where §0.9's rule that arity is part of the contract does real work: the
same string that is a valid single answer here would be shredded into three items by a
`|`-splitting parser.
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
- **Gradeable line:** **T21's template exactly** — full case+number pairs joined by ` ali `,
  words never MSD tags. T20 is T21 with a sentence around it, and giving the same analysis two
  shapes across two types would make one of them wrong by construction. *(The earlier
  `ODGOVOR: tožilnik, ednina` was a third shape, distinct from both of T21's two.)*
- **Negative:** a sentence in which the named word does not appear at all.

### Output template

```
ODGOVOR: {sklon} {števila}[ ali {sklon} {števila}]…
```

| | |
|---|---|
| arity | **open, ≥ 1** — in practice 1, since the seed filter selects unambiguous forms (R2) |
| sep | ` ali ` |
| order | case order, then number order — identical to T21, and **not graded** (§0.8.1) |
| gap | — |
| regex | `^ODGOVOR: \w+ (ednine\|dvojine\|množine)(?: ali \w+ (?:ednine\|dvojine\|množine))*$` |
| example | `ODGOVOR: tožilnik ednine` |

Shares T21's generator function and grader path outright. If R2's intersection turns out thin
and the fallback disjunction is adopted, the template does not change — only how often the
answer has more than one item.

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
| C1 | ~~Rebuild the store.~~ **Done 2026-08-20** — the v4 stores render `vform`/`person`/`definiteness` on forms and `aspect`/`clitic` on anchors (§0.7), and **v5** verbalises the collocation nodes (Group F). Generate against `kg_graph_v5_gemma3`. | T5, T6, T7 and T10 were ungeneratable before v4; T17 and T18 were parked before v5. |
| C2 | **Compute the band × type availability matrix** (R3) and record it. | T19/T20 have ~11 k eligible lemmas against 72,528 in the pool; low-band cells will be near-empty. |
| C3 | **Measure the T20 intersection** (R2): direct example × unambiguous form. | If thin, fall back to the disjunction and record the fallback. |
| C4 | **Measure the natural multi-entity rate** (R5) on real generated question strings, not on the 88.9 % figure. | D3 makes the union share emergent; if it comes out near zero, oversample ambiguous seeds via T4/T21. |
| C5 | **Pick and evaluate the extraction model** (R4). Its recall against our own templates is measurable for free — we author them, so the gold target word is known. | An extractor miss is an end-to-end service error no amount of GTLM training recovers. |
| C6 | **Verify Tier C leakage**: no training item anywhere contains `protipomenka` / `antonim` / `nasprotje`. | T16's whole value is that the relation is unseen. |
| C7 | **Unit-test the Slovene number agreement table** (T14), the canonical orderings (T1, T5, T15, T17, T21), and **`sense_class`** (Group D) against a fixture holding one placeholder, one fallback with an example snippet, one definition equal to the headword, and one ordinary definition. | Grading is exact-match on these strings; an ordering bug reads as a model failure. `sense_class` decides both the T12 list and the T14 count, so a bug there is a systematic wrong-gold, not noise. |
| C8 | **Withhold 2–3 templates per type** for Tier A before generation, not after. | Retrofitting a held-out split from generated items risks the same phrasing appearing on both sides. |
| C9 | **Run the §0.8 grader over the gold answers themselves** — every item must score `exact = 1.0` against its own gold. | Catches separator collisions, stray whitespace and normalization bugs before they are misread as model failures. Free, and it is the one test that validates the grading contract end to end. |
| C10 | **Assert every gold answer matches its type's `regex`**, and that no *multi-item* gold contains its own separator inside an item. | Measured safe today for definitions (0/230,606 contain `\|`) and word forms (0 contain `,;\|/`), but nothing enforces it. Single-item types are exempt by design — 4 of 51,172 examples contain a `\|` and must not be split (§0.9). |
| C11 | **Decide M4** (render noun `gender` into the anchor, one line + a ~35 min rebuild). | T9 is ungeneratable and T8's noun row is POS-only until it lands (§0.7). |
| C12 | ~~**Decide T14** — keep with the 90.1 % majority-class baseline reported, restrict the pool, or drop.~~ **Closed 2026-08-21.** The 90.1 % was measured over all anchors, 90.7 % of which are MWE entries whose second sense is an empty shell; on the seed pool the baseline is **37.1 %**. T14 now counts the senses T12 lists (Group D). Remaining work is C17, not a decision. | The premise was a measurement artefact. Kept, with its role restated: a count-vs-list consistency probe, not independent sense knowledge (§T14). |
| C13 | **Report a majority-class / constant-answer baseline for every type**, not just T14. | T14 is the extreme case, but T9 (3 values), T10 (3 values) and T16 (75 % single-antonym) all admit cheap constant strategies. A score without its baseline is unreadable. |
| C14 | ~~**Reopen T17/T18** only with a verbalisation source (Group F).~~ **Done 2026-08-20 — the source is the export itself.** Remaining work: regenerate both types against a v5 store and confirm the gold lines come off the node text unmodified. | The verbalisation is now node text, so the check is that nothing downstream still splits on `+`. |
| C15 | **Assert every generated gold is in its type's canonical order** — including the set-valued types whose order is *not* graded. | Grading tolerance is not a licence for sloppy training data: the model must see exactly one ordering for a given set, or it is being taught noise on a surface it is forced to emit. Cheap to check (re-sort and compare), and it is the only thing standing between "order is not graded" and "order is arbitrary". |
| C16 | **Unit-test `sl_key`** against a fixture including `č`, `š`, `ž` and a non-Slovene character. | `sl_SI` is not installed on this cluster and Python's default sort is wrong for Slovene (§0.8.2); a silent fallback would make the training data non-canonical without failing anything. |
| C17 | **Assert `T14 gold == len(T12 gold)`** for every lemma appearing in both, and that a lemma is a negative in both types or in neither. | The two share `sense_class`, the dedup and the seed pool *by design*; nothing enforces it. Drift here teaches the training data to say *ima 4 pomene* beside a two-item list — the exact contradiction the shared filter exists to remove. |
| C18 | **Implement D5b's sampler and pin its reproducibility.** Assert (a) the candidate pool is sorted by node id before drawing, (b) the RNG seed derives from the anchor's node code and nothing else, (c) rebuilding the store leaves every ball byte-identical, and (d) every gold phrase is in its own ball. | Found 2026-08-21: `sestavina` was never the only hub. Uncapped, a top-band anchor costs a p50 of 22,704 tokens (max 135,515) against 1,986 capped. (a) and (b) are silent failures — CSR adjacency order is not stable across builds (0.44 % of `indices` rows moved in v5), so an unsorted pool or an order-dependent seed makes the dataset unreproducible without failing anything. |
| C19 | **Record the realised collocation mix** once generation runs: partner-proxy percentiles and function-word share over all sampled balls, against the §3.1c single-anchor figures. | §3.1c measures the sampler on `voda` alone. The weights are justified by a power law fitted globally (slope −1.418, R² = 0.933), but the mix the dataset actually gets is unmeasured until items exist. |

---

## 3. Appendix — types deliberately NOT in v1

Recorded so nobody re-derives them. All are from Section 6 of `QA_DATASET_DESIGN.md`, judged
worth doing and consciously postponed.

| family | status | note |
|---|---|---|
| `mwe_structure`, `mwe_component_search`, `idiom_completion` | **deferred to v2** | The single best-supported unused capability: 10,265,230 `decomp:constituent` edges, all carrying `lexinfo:category`, plus `rdf:_1 … _22` for constituent order, plus MWE headwords that *do* store their surface phrase. Deferred purely on implementation risk — it is the only family that depends on the D5 upward cap behaving. Validate the cap empirically first. |
| clitics and negation | **effectively dead for word entries** | The `clitic` 423,585 and `negative` 121,229 figures count `lexical-unit-part` subjects — MWE components, which the builder collapses away. Only **25** single-word entries carry a `clitic` value in the whole KG, so there is no pool to sample from. Anything here would have to come from the MWE family, not from word entries. Supersedes the earlier "cheapest type to add back". |
| ~~verbalised collocations~~ | ✅ **shipped in v1** | Was "build-time bulk fetch from the DDDS API". No fetch needed — the phrase is in the export, one dereference from the collocation IRI, and the builder writes it as of v5. See §6.2 of the design doc and `README.md` Finding 8. |
| reasoning-tier items (aggregation, filtering, multi-hop) | **deferred (D13)** | v1 is retrieval-focused; Tier B/C are the only places reasoning is probed. |
| hypernyms/hyponyms, meronyms/holonyms | **cannot be supported** | `hypernym`/`hyponym` = 5,415 each (0.12 %); meronym/holonym absent entirely. |
| translations | **cannot be supported** | `vartrans:*` = 77,570, all `directEquivalent`, all **`@hun`**. |
| register, style, dialect, domain, frequency, typo metadata | **cannot be supported** | Absent from the KG entirely. This is what makes ~half the older reference file's types unanswerable. |
| valency, word formation | **cannot be supported** | No valency frames, no derivational morphology. |
| role-play personas, JSON output | **not reproduced (D20)** | The older file's `has_role` (286 items) and `is_json_format` (353) behaviours: nobody knows who added them or why, and neither is part of the service. Output shape is fixed by D6 for every item. |
| multi-turn / follow-ups | **out of scope (D20)** | Every item is one self-contained question and one answer. |
| general-instruction replay | **out of scope (D21)** | Capability retention is phase two, deliberately. |
