# The 19 question types

What each type asks, where its answer comes from, and the exact shape it is graded
in. `qa/spec.py` is the executable half of this document — the generator emits the
shape it declares and the grader parses that same shape, so the two cannot be edited
apart. Dataset-level design (seeds, bands, splits, the extraction policy) is in
[`QA_DATASET_DESIGN.md`](QA_DATASET_DESIGN.md).

## 0. Conventions that apply to every type

### 0.1 Answer format

Every answer is exactly one line:

```
ODGOVOR: <minimal, machine-gradeable, one line, no markdown>
```

**No MSD tags anywhere** — morphology is answered in words (`tožilnik ednine`, never
`Sozet`). A user-facing prose rendering is deliberately postponed: the line keeps its
position and shape, so adding prose after it later is purely additive, and the eval
parser reads the first `ODGOVOR:` line and ignores the rest.

### 0.2 The "not recorded" sentinel

~10 % of items have no answer in the graph. The line is then exactly:

```
ODGOVOR: ni podatka v bazi
```

Three flavours are generated: the entity does not exist (`nonexistent`), it exists
but lacks the relation (`absent`), or the question does not apply to it at all
(`mismatch` — the gender of a verb, the conjugation of a noun).

**The rate is 10 % per type, uniform** — not proportional to how many natural
negatives each type happens to have. A rate that varies by type makes *which type
this is* predict *whether this is a negative*, and the model learns that shortcut
instead of checking the graph.

**~70 % of negatives are ordinary lemmas**, only ~30 % nonexistent entities;
otherwise the model learns "weird-looking word → say you don't know". The
nonexistent half is built by seeded single-character perturbation of a real lemma,
verified absent from the reverse index — so a perturbed *gora* is *gara*, not
*qxzzy*, and orthography gives nothing away. Negatives are drawn from the same band
distribution as positives, at most one per (lemma, type), and T12/T14 share their
negative set exactly.

### 0.3 Question surface

**No delimiters** — no `»…«`, no quotes around the target word. The production
pipeline is: an external model extracts the word(s) from the user's question → each
string is looked up verbatim in the surface-form reverse index → the hop-2 ball of
every exact match is extracted → the model receives their **union**. Dataset
extraction runs the identical pipeline, never privileged knowledge of which lexical
unit the generator started from.

Binding on every frame: **the target word must appear in the question as a verbatim
surface string that exists in the reverse index** — a lemma for most types, an
inflected form for T4, T20 and T21. The index covers `canonicalForm` and
`otherForm`, so both resolve.

### 0.4 Seeds, bands, split

Core entries passing a content filter, banded by a graph-internal frequency proxy,
split lemma-disjointly with three held-out tiers. Numbers and rationale:
`QA_DATASET_DESIGN.md` §§4–5. What matters per type below is the **seed filter** —
which entries can produce an item at all.

### 0.5 What the model actually reads

The ball is a set of nodes whose text is self-describing. A generator may only
produce answers derivable from these strings:

```
iztočnica: bežnica (samostalnik, ženski spol, imenovalnik, ednina)   ← anchor
iztočnica: popraskati (glagol, dovršni, nedoločnik)                  ← verb anchor
oblika: Afričanu (dajalnik, ednina)                                  ← form leaf
oblika: popraskam (sedanjik, 1. oseba, ednina)                       ← finite verb form
oblika: popraskal (deležnik na -l, ednina, moški spol)               ← -l participle
oblika: navrženi (imenovalnik, ednina, moški spol, osnovnik, določna oblika)
pomen 2: pes                                                          ← sense, with ordinal
pomen 1: pes (zgled: Sosedov pes je spet lajal ...)                   ← undefined sense
zgled: Vsak konjenik je dobil simbolno darilo ...                     ← usage example
kolokacija: boj proti kriminaliteti                                   ← the inflected phrase
sopomenka: biblioteka ~ knjižnica
protipomenka: boj ~ sodelovanje
```

The parenthetical order is fixed — POS, then lexical-unit properties (`aspect`,
`gender`, `clitic`), then form features in the order `vform, person, case, number,
gender, degree, definiteness` — and a generator may rely on it.

Edges are untyped; the relation lives in the node text. Extraction is uniform hop 2
with two hub caps (`QA_DATASET_DESIGN.md` §3.1, §3.1b).

### 0.6 Feature vocabulary present in the RDF

The complete set a generator may key on. **"Reachable" counts single-word entries** —
corpus-wide totals are far larger but dominated by MWE components, which the builder
collapses away, so those values never appear in a ball.

| predicate | attaches to | values | reachable |
|---|---|---|--:|
| `lexinfo:case` | word-form | nominative, genitive, dative, accusative, locative, instrumental | — |
| `lexinfo:number` | word-form | singular, dual, plural | — |
| `lexinfo:gender` | word-form **and** lexical-unit | masculine, feminine, neuter | 310,362 units |
| `lexinfo:person` | word-form | first, second, third | 253,497 |
| `lexinfo:vform` | word-form | present, participle, imperative, supine, infinitive, future, conditional | 452,782 |
| `lexinfo:degree` | word-form | positive, comparative, superlative | — |
| `lexinfo:definiteness` | word-form | yes, no | 160,524 |
| `lexinfo:partOfSpeech` | lexical-unit | noun, verb, adjective, adverb, … | 400,180 |
| `lexinfo:aspect` | lexical-unit | perfective, progressive, biaspectual | 18,157 |
| `lexinfo:clitic` | lexical-unit | bound, yes | **25** |
| `lexinfo:synonym` / `antonym` | sense | — | 362,520 / 6,898 |
| `skos:definition` | sense | `@sl` literal | 225,618 |
| `lexicog:usageExample` | sense | literal | 71,729 |

`lexinfo:animate` and `lexinfo:negative` exist but are **not rendered** into node
text, so nothing may key on them. There is **no `tense` and no `mood` predicate**:
verb tense is carried by `vform`.

`gender` is the one property this KG stores at two levels, and they mean different
things — an agreement feature on adjective and participle forms, the inherent gender
of a noun on the entry. Both are rendered; the noun's goes on the anchor and its form
leaves inherit it.

### 0.7 What the morphology supports

**Single-word entries carry full morphology.** Every noun has a gender, every
single-word verb has an aspect, and forms carry `vform`, `person`, `case`, `number`,
`gender`, `degree` and `definiteness`.

**MWEs carry none.** POS and morphology sit on single-word entries only, and MWEs are
91 % of all anchors; the component collapse also discards each constituent's case and
number. No type in this document can ask a grammatical question about a multi-word
expression, and an MWE seed is a truthful `mismatch` negative for the ones that try.

**Preteklik and prihodnjik are composed, and the composition is graph content (M2).**
No verb entry stores a past or future form of its own — they are periphrastic. But
the *-l* participle is stored for every verb, and the auxiliary *biti* is stored as an
ordinary verb entry with all 17 of its present and future forms. So the rule is
`aux[person, number] + participle[gender, number]`, and both halves are node text
somewhere in the graph. Measured over 2,086 sampled core verbs, **100 % are
composable**; the participle table is all-or-nothing, so a verb either supports the
whole thing or is already excluded by the "require all 9 present cells" filter.

**The auxiliary is not injected into the ball.** It would break train/inference
parity: no extractor returns *biti* from *"Kako se glagol pisati spreže v
prihodnjiku?"*, so every conjugation ball at inference would be missing 17 nodes the
model trained on, and closing that gap means a hard-coded POS-conditional special case
in the production retriever. The cost is stated plainly: T5 and T6 gold contains
tokens absent from the input (8.4 % of the corpus), so an exact-string grade cannot
separate *failed to retrieve the participle* from *misconjugated biti*. The prediction
dumps carry the full table, so splitting the two is a grep over the failures rather
than a second metric maintained forever.

### 0.8 Grading — exact match after a fixed normalization

No embedding model, no LLM judge, no fuzzy similarity. The whole grader is a script:

```python
def norm(s):
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s).strip().rstrip(".")
    return s.casefold()

def correct(pred, gold, *, mode, allow=None, band=None, n_all=None):
    p = parse(pred)                        # first ODGOVOR: line, split on the type's sep
    if p is None:      return False        # unparseable -> wrong, no partial credit
    if mode == "sequence":   return p == parse(gold)
    if mode == "multiset":   return sorted(p) == sorted(parse(gold))
    if mode == "membership":               # §0.8.3
        d = list(dict.fromkeys(p))
        return len(d) == len(p) and set(d) <= allow and count_ok(len(d), band, n_all)
```

`mode` is a **per-type constant** from `qa/spec.py`, never inferred at run time. One
headline number: **`success`**, the share of items whose answer satisfies their type's
condition. `f1` is computed per item as a diagnostic — it separates "missed a sense"
from "invented one" from "right content, wrong order" — and is never reported as a
score.

**Why exact match is defensible here.** Every gradeable string is present verbatim in
the model's input: these types are retrieval probes over an extracted subgraph, not
open generation. A model that paraphrases has not done the task.

The normalization is deliberately shallow — it folds case, collapses whitespace and
drops one trailing period. No diacritic stripping (that would merge real Slovene
distinctions), no stemming, no reordering.

#### 0.8.1 Order: always generated, sometimes graded

Two concerns that must not be collapsed. **Generation is always canonical**: every
gold answer is emitted in one fixed order, with no dependence on graph-walk order, or
the training target for an item is unstable across runs. **Grading is order-sensitive
only where order carries meaning** — in `gora, gore, gori, …` position 3 *is* the
dative, which is addressing rather than ordering; in `pasiven, resigniran, spokojen`
the order is an artefact of our own sort, and penalising a reordering would score a
formatting preference as a retrieval miss.

Nothing in the KG orders these sets: synonyms and antonyms carry no score, rank or
frequency anywhere, and the graph's one explicit ordering (`rdf:_1 … rdf:_n`) is word
order inside a multi-word expression, which touches none of these types.

#### 0.8.2 Canonical order for set-valued types

Alphabetical, by the **Slovene** alphabet. Python's default sort is wrong here and the
`sl_SI` locale is not installed on this cluster, so a `locale.strxfrm` implementation
would fail or fall back silently:

```
sorted():          cesta, riba, sok, zima, čas, šola, žito     <- č/š/ž dumped after z
correct Slovene:   cesta, čas, riba, sok, šola, zima, žito
```

`qa/sl.py:sl_key` is a fixed collation key with no dependency, unit-tested against
`č`/`š`/`ž` and a non-Slovene character (C16).

**Separator.** ` | ` for any answer whose items are free text: over all 230,606 `@sl`
definitions, `|` occurs **0** times while `;` occurs in 8.5 % and `,` in 33.0 %.
Types whose items are single word forms (T1, T5, T7, T11, T15, T16) keep
comma-separated lines — a word form cannot contain a comma — but any new list type
defaults to `|`.

#### 0.8.3 The third mode: membership + count

`sequence` and `multiset` both compare against one gold list. That breaks for
collocations, because the ball holds a **seeded sample** of the anchor's
collocations, so two extractions of the same anchor legitimately support different
answers and a grader holding one list would mark a correct answer wrong whenever the
sample moved.

```
GOLD(item)  ⊆  BALL(anchor)  ⊆  ALL(anchor)
the gold line   what the model can see   what the grader accepts
```

The gold is drawn from the ball, so it is answerable from the input; grading is
membership in `ALL` plus a count rule, so any phrase the model could legitimately
have read is accepted. An answer is correct iff it repeats nothing, every item is in
`ALL`, and the count matches what was asked. `ALL` is materialised per anchor at
generation time and shipped inside the item, so the grader opens no store.

### 0.9 Output templates — the notation used below

The contract table in §1 gives each type's `arity`, `sep` and `mode`; `qa/spec.py`
adds the shape `regex` its self-test uses (C10).

**Arity is part of the contract, not a detail.** A single-item answer is *never* split
on a separator: 4 of 51,172 usage examples contain a `|` and 65 % contain a comma, so
a T19 answer split on either would be shredded. Single-item types are compared whole
after `norm()` and nothing else.

**Gaps.** In a fixed-arity answer a missing cell is `/`, never an omitted item —
otherwise a missing third form silently becomes a wrong fourth form and every later
position is misgraded. `/` occurs in no word-form surface anywhere.

---

## 1. The 19 types at a glance

| # | type | source | arity | sep | mode |
|---|---|---|---|---|---|
| T1 | `sklanjanje/celotna_sklanjatev` | `oblika:` leaves, case × number | 18 fixed | `, ` / `; ` | sequence |
| T2 | `sklanjanje/sklanjatev_po_stevilu` | one number of T1's grid | 6 fixed | `, ` | sequence |
| T3 | `sklanjanje/posamezen_sklon` | one cell of T1's grid | 1 | — | sequence |
| T4 | `sklanjanje/osnovna_oblika_leme` | form → `iztočnica:` | open | `, ` | multiset |
| T21 | `sklanjanje/analiza_oblike` | ambiguous form → its readings | open | ` ali ` | multiset |
| T5 | `spreganje/celotno_spreganje` | present cells + composed tenses | 27 fixed | `, ` / `; ` | sequence |
| T6 | `spreganje/spreganje_v_casu` | one tense of T5's table | 9 fixed | `, ` | sequence |
| T7 | `spreganje/neosebne_oblike` | `vform` ∈ {inf, supine, imperative} | 3 fixed | `; ` | sequence |
| T8 | `besedna_vrsta/osnovne_lastnosti` | `partOfSpeech` + one property | 1–2 | `, ` | sequence |
| T9 | `besedna_vrsta/spol_samostalnika` | `gender` on the anchor | 1 | — | sequence |
| T10 | `besedna_vrsta/vrsta_in_vid_glagola` | `aspect` on the anchor | 1 | — | sequence |
| T11 | `stopnjevanje/vse_stopnje` | `degree` on forms | 3 fixed | `, ` | sequence |
| T12 | `pomen/razlaga_pomena` | `pomen:` nodes classed `defined` | open | ` \| ` | sequence |
| T14 | `pomen/stevilka_pomenov` | the length of T12's list | 1 | — | sequence |
| T15 | `sopomenke/navedi_sopomenke` | `sopomenka:` nodes | open, ≤10 | `, ` | multiset |
| T16 | `protipomenke/navedi_protipomenke` | `protipomenka:` nodes | open, ≤10 | `, ` | multiset |
| T17 | `kolokacije/navedi_kolokacije` | `kolokacija:` nodes | open | ` \| ` | **membership** |
| T19 | `primeri_uporabe/povedi_z_besedo` | `zgled:` nodes | 1 | — | membership |
| T20 | `primeri_uporabe/analiza_oblike_v_povedi` | `zgled:` + unambiguous form | open | ` ali ` | multiset |

T13 and T18 are retired — the first into T12, the second into T17's `exact` band — and
the remaining types keep their numbers, so the inventory is 19 types numbered to 21.

Four filters are shared across types rather than repeated in each, because a type that
forgets one produces gold that is plausible and wrong:

| filter | what it rejects | why |
|---|---|---|
| `gen.healthy_grid` (C23) | nominal paradigms whose nominative plural equals the nominative singular, or whose filled cells hold fewer than 6 distinct surfaces | The KG stores lemma-filled paradigms — `odstotek` has `rodilnik/množina → odstotek`. 8.98 % of noun entries with ≥12 filled cells fail. It makes T1/T2/T3 a table of one repeated surface, and it makes T20's ambiguity test pass on a cell that only *looks* unambiguous. Decided per entry, never per lemma. |
| degenerate gradation (C22) | a comparative equal to the positive or the lemma | The KG stores `oblika: mikaven (…, primernik, …)` — the cell exists, the surface was never inflected. 29 % of comparative-bearing entries; they become truthful negatives. |
| premise truth (C20) | a frame naming a word class used on an entry of another class | *"…samostalnika izbrisen?"* where *izbrisen* is an adjective makes the question false before the model reads anything. A filter on the frame pool, not a hand-kept list. |
| frame scope (C24) | a frame asking for fewer slots than the answer contract fills | T7's contract is arity 3, so *"Kako se glasi velelnik glagola krasti?"* was answered with all three forms — teaching the model to ignore which form the question named. |
| metalanguage agreement (C21) | ungrammatical Slovene in our own question and answer text | Case and number names decline (*v ednini*, *tožilnik ednine*) and a counted noun agrees with its numeral (*2 kolokaciji*, *3 kolokacije*, *5 kolokacij*). Both are tables in `qa/sl.py`, unit-tested (C7), never heuristics — and the defect would sit in the **question**, where the model would learn to imitate it. |

---

# Group A — `sklanjanje` (nominal inflection)

All five types read one structure: an anchor with `oblika:` leaves, each leaf's text
carrying its own case and number. Two facts constrain every one of them. **Duplicate
form nodes are normal** — different lexical units contribute the same string, so
deduplicate on (surface, case, number) before rendering. And **capitalisation is a
heuristic**: the builder prefers the least-capitalised variant, so never build a
question whose gold turns on it.

### T1 — `sklanjanje/celotna_sklanjatev`

The complete declension table: 6 cases × 3 numbers = 18 cells, ednina then dvojina
then množina, and inside each im, rod, daj, tož, mest, or. Seed filter: POS ∈ {noun,
adjective, numeral, pronoun}, ≥ 12 of 18 cells populated, `healthy_grid`, and no cell
holding two surfaces — the doublets are 0.2 % of nouns and every one is the animacy
alternation in `tožilnik, ednina` (*vidim Mercator* / *vidim Mercatorja*), which the
model cannot see because `animate` is not rendered.

```
ODGOVOR: gora, gore, gori, goro, gori, goro; gori, gora, gorama, gori, gorah, gorama; gore, gora, goram, gore, gorah, gorami
```

### T2 — `sklanjanje/sklanjatev_po_stevilu`

T1 restricted to one number, which is a template slot sampled uniformly so that
dvojina is not underrepresented. Seed filter: ≥ 5 of the 6 cells. The best negative in
the group is a **pluralia tantum** asked about the missing number — an ordinary lemma
whose answer is genuinely absent rather than invented.

```
ODGOVOR: ednina: gora, gore, gori, goro, gori, goro
```

### T3 — `sklanjanje/posamezen_sklon`

One cell, answered with the bare form. The (case, number) pair is sampled **uniformly
over the 18 cells** rather than from the reference file's nominative-heavy
distribution. Several cells can share a surface (`gori` is dative and locative
singular); harmless in this direction, because the question fixes the cell. The
reverse direction is T21.

```
ODGOVOR: gori
```

### T4 — `sklanjanje/osnovna_oblika_leme`

Lemmatisation: given an inflected surface, name the headword. This is the one type in
the group whose question word is a **form**, so it exercises the reverse index on
`otherForm`.

Ambiguity is the design problem: 11.1 % of surface strings own more than one lexical
unit (worst case 18 — *goli, meti, ti, peti*). Both policies are wanted — majority
items resolve to exactly one unit, while a deliberate ~15–20 % slice resolves to 2–3
and the gold names **all** of them. That slice is the cheapest source of supervision
for choosing among what entity linking returned. The answer is **always a set**, even
with one member, so the grader never has to guess which shape was meant.

```
ODGOVOR: gora          ODGOVOR: gol, gola
```

### T21 — `sklanjanje/analiza_oblike`

The inverse of T3: given a surface form and its lemma, name the case and number it
expresses. Restricted to **ambiguous** forms, answered with the disjunction — the
graph lists every reading a form carries and cannot pick between them, and with no
sentence there is nothing to pick with, so the disjunction *is* the correct answer.
~25 % unambiguous forms are kept so the model does not learn "always answer with a
disjunction".

Always **full case+number pairs**, even when the number is shared: one shape for one
fact costs a repeated *ednine* and buys a template checkable with a single regex.
T20 is this type with a sentence around it and shares its generator outright.

```
ODGOVOR: tožilnik ednine ali orodnik ednine
ODGOVOR: rodilnik ednine ali imenovalnik množine
```

---

# Group B — `spreganje` (verb conjugation)

Both tense-bearing types rest on M2 (§0.7): the present cells are stored, preteklik
and prihodnjik are composed from the *-l* participle and the auxiliary. The exceptions
are enumerated and none of them blocks the design — the composed tenses are rendered
in the **masculine** (the citation default), *biti* is excluded from the seed pool
because its future is suppletive, the 3 verbs in 10,242 with a doubled present cell
take the **non-negated** surface, and word order is auxiliary-first (*sem delal*).

### T5 — `spreganje/celotno_spreganje`

The full conjugation: 3 tenses × 9 person/number cells = 27 forms, tense-major. Seed
filter: POS = verb, all 9 present cells and all 3 masculine participle cells — which
costs ~0.1 % of the verb pool, because the tables are all-or-nothing. The negative is
a non-verb asked to be conjugated; the seed filter leaves no verb without an answer,
so that is the whole ordinary-lemma negative pool for this type.

```
ODGOVOR: sedanjik: delam, delaš, dela, delava, delata, delata, delamo, delate, delajo; preteklik: sem delal, …; prihodnjik: bom delal, …
```

### T6 — `spreganje/spreganje_v_casu`

One tense of T5's table, obtained by **calling T5's generator and taking one block** —
never a second implementation. The tense is a three-value slot sampled uniformly. The
*-l* participle carries gender, so the two composed tenses can also be asked in the
feminine or neuter (~20 % of those items); the default is masculine, which is the only
legal value for sedanjik.

**The gender goes in parentheses, not after a comma**, because `, ` is this type's
item separator — `preteklik, ženski spol: …` would split the label into two items and
make the arity depend on whether the slot is filled.

```
ODGOVOR: sedanjik: delam, delaš, dela, delava, delata, delata, delamo, delate, delajo
ODGOVOR: preteklik (ženski spol): sem delala, si delala, je delala, …
```

### T7 — `spreganje/neosebne_oblike`

The non-finite forms — nedoločnik, namenilnik, velelnik — **always all three, in this
order**, with `/` for an absent one. The slot is never dropped: a variable-arity
positional answer cannot be graded positionally at all. Supine coverage is thin
(20,878 against 1,580,662 infinitives), so a missing namenilnik is common and is a
gap, not a sentinel; the sentinel is for a verb with none of the three.

```
ODGOVOR: nedoločnik: zaripniti; namenilnik: zaripnit; velelnik: zaripni
```

---

# Group C — `besedna_vrsta` (word class and grammatical properties)

T8, T9 and T10 assert the same facts about the same anchor, so they **share one
normalizer**. If they spelled a gender `moški spol` in one and `moški` in another, the
supervision would contradict itself on the one fact all three state.

### T8 — `besedna_vrsta/osnovne_lastnosti`

Word class, plus the one lexeme-level property that word class carries. One contract,
whatever the phrasing: the reference file's 11 question frames produce only six answer
shapes, one per POS, with no correlation between frame and shape — which is what makes
a Tier A phrasing withheld from training land on a contract the model already knows.

Every slot is lexeme-level. **Case, number, person, vform and degree never appear** —
they describe the citation *form*, not the word, and are T1–T7's material.

| POS | entries | answer |
|---|--:|---|
| noun | 310,362 | `samostalnik, <spol>` |
| adjective | 52,634 | `pridevnik` |
| verb | 18,159 | `glagol, <vid>` |
| adverb | 15,228 | `prislov` |
| numeral · interjection · pronoun | 2,434 · 507 · 284 | `števnik` · `medmet` · `zaimek` |
| abbreviation · preposition · particle · conjunction | 129 · 97 · 69 · 58 | `okrajšava` · `predlog` · `členek` · `veznik` |

That is the complete `partOfSpeech` vocabulary, summing to the 400,180 entries of
§0.6. Two consequences: **there is no `properNoun`/`commonNoun` distinction in this
KG** — zero triples — so `lastno ime` / `občno ime` is not derivable and must not be
reconstructed from capitalisation; and **`residual`** (219 entries, the tag set's
garbage class) is excluded from the pool, since those anchors render with no POS at
all and there is nothing to answer. `abbreviation` is included and rendered
`okrajšava`: gold that contradicts the input is worse than gold that is taxonomically
loose.

An MWE seed carries no POS anywhere and is the type's truthful negative.

```
ODGOVOR: samostalnik, ženski spol      ODGOVOR: glagol, dovršni      ODGOVOR: prislov
```

### T9 — `besedna_vrsta/spol_samostalnika`

Grammatical gender of a noun, from `lexinfo:gender` on the **anchor's lexical-unit**,
not on its forms — noun word-forms carry no gender in this KG. Coverage is 310,362
triples over 310,362 distinct noun entries: every noun has exactly one, none has two,
and no other POS has any, so there is nothing to reconcile and no `moški ali ženski
spol` answer to generate. Asking the gender of a verb or adverb is the easiest
ordinary-lemma negative in the inventory.

```
ODGOVOR: moški spol
```

### T10 — `besedna_vrsta/vrsta_in_vid_glagola`

Verb aspect, from `aspect` on the anchor. **Every single-word verb in the KG has
one** — 18,157 units, 100 % coverage of the pool, the best of any type. Three values,
one closed regex: the cleanest verifiable type in the inventory, and the natural
pipeline smoke test alongside T14.

*Vrsta glagola* in the wider sense is **not** in the graph — transitivity, reflexivity
and valency are absent, with no valency frames at all — so the questions ask about
aspect and are worded accordingly.

```
ODGOVOR: dvovidski
```

### T11 — `stopnjevanje/vse_stopnje`

Gradation of an adjective or adverb: osnovnik, primernik, presežnik. `degree` sits on
**word-forms**, and each graded form also carries case, number and gender, so the
generator picks the **nominative singular masculine** of each degree rather than
dumping the paradigm; where `definiteness` is present it prefers the indefinite for the
positive and the definite for the superlative, as Slovene dictionaries cite them.

The graded forms are the scarce resource and define the pool: 107,717 comparatives and
72,856 superlatives against 4,361,733 positives. Seed filter: at least one comparative
that survives the degenerate-gradation check (C22). ~32 % of comparative-bearing
entries have no superlative and emit `/` in the third slot — arity stays 3.

Non-gradable adjectives asked for a comparative are the ideal ordinary-lemma negative
and are a substantial slice of this type.

```
ODGOVOR: lep, lepši, najlepši          ODGOVOR: brunast, brunastejši, /
```

---

# Group D — `pomen` (senses and definitions)

### What counts as a `pomen` — the shared `sense_class` contract

**A `pomen:` node is not a meaning.** It is one `ontolex:LexicalSense` IRI, and the
export mints those for structural reasons that have nothing to do with polysemy. One
function classifies them, called by both types — never two implementations:

```
sense_class(sense, anchor) ->
    placeholder   sense id == the entry's own id.  The entry reified: `pomen 1: <lemma>`,
                  never defined, often the busiest node in the neighbourhood.
    fallback      text is the headword repeated, optionally + ` (zgled: …)`.
    defined       anything else — the sense carries a real skos:definition.
```

**Only `defined` counts, for either type.** That rule survives a test the alternatives
fail: the model sees node **text, not ids**, and `pomen 1: otrok` (a placeholder) is
identical in shape to `pomen 2: otrok` (an ordinary fallback). Any policy that keeps
one and drops the other is not derivable from the model's input; *"count a sense iff
its text is not the headword repeated"* is. The alternatives disagree with it on
31.2 % of anchors.

**What no filter can fix.** The imported layer records several glosses per meaning —
`fotograf` gets two, `dekan` three, `dvigniti` 36 — and the KG never says which
describe the same sense, so the true sense count is not recoverable. **Neither type may
ask how many meanings a word *has***, only how many are recorded in this database.
T14's wording follows from that, not from style.

### T12 — `pomen/razlaga_pomena`

Every sense that carries a real definition, in dictionary-ordinal order. Monosemous
and polysemous entries take the same shape — a one-item answer is not a special case,
which is what makes the merge with the former T13 clean.

Definitions are **terse glosses, not sentences**: median 5 words, p95 17, and 21.7 %
are one or two words. That shortness is what makes exact match practical rather than
brittle. Gold is read from the **store**, not the RDF, so the answer string is
guaranteed present verbatim in the model's input.

Ordinals are omitted from the line — the order already carries them, and requiring
`1.`/`2.` adds a failure mode that measures counting rather than retrieval. Identical
definitions within an entry are deduplicated (0.40 % of anchors), and **T14 must dedup
identically** or its count stops matching the list.

Seed filter: at least one `defined` sense — 40,926 of the 72,561-lemma pool. The
remaining 31,635 are the largest natural negative pool in the dataset, **shared with
T14**, so no lemma gets a list from one type and a refusal from the other.

Sense *labels* stored as definitions are kept, not filtered: 12.0 % of the pool's
definitions begin `o …` (*leten* → *o starosti*), and there are editor's notes shipped
verbatim. Filtering them would change both the list and T14's count for a reason
**invisible in the node text the model reads** — the same property that decided
`sense_class`.

```
ODGOVOR: zgradba za bivanje | rodbina, družina
```

### T14 — `pomen/stevilka_pomenov`

The count of the senses T12 lists — `len(T12_items(anchor))`, from the same
`sense_class` call, the same dedup and the same pool, obtained by calling the same
function. C17 asserts they never disagree, and that a lemma is a negative in both
types or in neither.

The line is the bare integer. `0` is deliberately not allowed by the regex: a lemma
with no defined sense is a sentinel, not a count of zero, so a `0` in the gold means
the seed filter leaked. Answer distribution on the pool as seeded — 1: 37.1 %,
2: 25.1 %, 3: 12.8 %, 4: 7.4 %, 5: 4.8 %, >5: 12.9 % — so the majority-class baseline
is **37.1 %** and is reported beside every score (C13).

**Ask about the database, not about Slovene** (see the group header): question text
says *zabeleženih / razloženih v bazi*. This is a correctness constraint, not a
stylistic one — it is what makes the gold true.

The surrounding Slovene must agree with the count — *je zabeležen 1 pomen*, *sta
zabeležena 2 pomena*, *so zabeleženi 3 pomeni*, *je zabeleženih 5 pomenov* — and the
**verb and the participle agree too, not only the noun**. It is a table lookup on
`n mod 100` returning the whole phrase, never a heuristic (C21, unit-tested by C7), and
the case is a property of the frame, so a frame built on *ima* needs the accusative
table instead. The gradeable line stays a bare integer, so an agreement bug can never
affect the score.

What this type measures is a **format and consistency probe** — does the model's
number match the list it would give? — not independent sense knowledge.

```
ODGOVOR: 3
```

---

# Group E — sense relations

### T15 — `sopomenke/navedi_sopomenke`

The synonyms of a word, from reified `sopomenka:` nodes. They hang one hop past the
sense, which is why extraction's floor is hop 2; the node text carries **both** lemmas,
so the answer is readable even when the partner's anchor is pruned.

Coverage: 35.4 % of core lemmas have at least one, so **65 % lack them** — a large,
natural, ordinary-lemma negative pool. The list is **capped at 10** in canonical order
(7 % of anchors have 11 or more); the cap is part of the gold, not a rendering choice.

Synonyms attach to senses rather than lemmas, so a sense-scoped variant
(*"sopomenke besede X v pomenu …"*) is a natural Tier B item — an unseen question type
over a seen relation.

```
ODGOVOR: pasiven, resigniran, spokojen, vdan v usodo
```

### T16 — `protipomenke/navedi_protipomenke`

The antonyms of a word, from `protipomenka:` nodes — **structurally identical to
`sopomenka:`, differing only in the tag word. That identity is the entire point.**

**Tier C: 100 % of antonym items are test items.** Nothing about antonyms appears in
training — not the relation, not the templates, not the tag word. The probe is whether
a model trained to read `sopomenka: A ~ B` can read `protipomenka: A ~ B` it was never
supervised on, which is the strongest available test of whether graph attention
generalises. At 6,898 antonym instances — two orders of magnitude below synonyms —
holding it out costs almost nothing.

Templates must be written independently rather than by swapping the tag word into a
synonym frame. C6 asserts the leak boundary in two halves, because only one of them is
ours to control: **tag words** (`protipomenka`, `antonim`) name the relation and are
banned from training everywhere; **ordinary Slovene for "opposite"** (`nasprotje`,
`nasproten`) is banned from training *questions*, which we author, but allowed in
*answers*, which are the graph's own data — the KG defines `obratno` as *na nasproten
način*, and banning that would delete real lexicographic data to protect against
nothing.

Same shape as T15, and 75 % of anchors have exactly one antonym — which is also why
the answer is always a set.

```
ODGOVOR: sodelovanje
```

---

# Group F — `kolokacije`

### T17 — `kolokacije/navedi_kolokacije`

Which phrases does this word appear in? A list of inflected collocations, unnumbered
and non-exhaustive. **How many** is a slot on the item rather than a second type: a
member of the public asks for *nekaj* or *veliko* far more often than for exactly five,
and four subtypes would quarter the per-cell counts of a type that has ~100 test items.

`k` = phrases emitted, `N` = the number asked for, `|ALL|` = every phrase the store
holds for the anchor.

| band | question says | `count_ok(k, ·)` | share |
|---|---|---|--:|
| `none` | nothing about quantity | `2 ≤ k ≤ min(15, \|ALL\|)` | 40 % |
| `vague_small` | *nekaj*, *par* | `2 ≤ k ≤ min(6, \|ALL\|)` | 20 % |
| `vague_large` | *veliko*, *čim več* | `k ≥ min(5, \|ALL\|)` | 20 % |
| `exact` | *natanko N*, N ∈ 2–10 | `k == min(N, \|ALL\|)` | 20 % |

Three properties of that table are deliberate. **The bands are loose**: a user who asks
an open question has no count in mind, so scoring one would score our own arbitrary
choice. **`vague_large` has no upper bound** — membership in `ALL` is the real
constraint and it is a hard one, so a model that pads is wrong on the membership test
rather than on the count. And **`min(·, |ALL|)` everywhere**, so under-supply is never
an error: if the anchor has 3 phrases and the question asks for 5, giving 3 is correct.
That is the *refusal to invent* slice, expressed in the count rule instead of prose.

Since the collocation node's text **is** the phrase, gold is read straight off the node
with no post-processing. The gold is drawn from the ball's own sample, so it is
in-ball by construction. The graph carries no frequency on a collocation node, so
there is **no "top N"** and the questions must not claim *najpogostejše*; ordering is
`sl_key`, which is presentation, not ranking. Matching is case-insensitive: some
phrases are stored capitalised because that is how the lexicographer entered them.

Gold length comes from the band — `none` → 5, `vague_small` → 4, `vague_large` → the
whole sample, `exact` → `min(N, |ALL|)` — every one capped by what the ball holds.
35 % of core lemmas have no collocations at all.

```
ODGOVOR: mineralna voda | mineralno gnojilo | mineralna snov | mineralna surovina | mineralno olje
```

---

# Group G — `primeri_uporabe` (usage examples)

### T19 — `primeri_uporabe/povedi_z_besedo`

Show the word in use: a recorded corpus sentence containing it, verbatim. The cleanest
possible retrieval signal.

Graded `membership` with a count of exactly one, **not `sequence`**: a lemma commonly
has several recorded examples and every one of them answers the question. The KG
stores them as a set, so grading against whichever has the lowest node id would reward
an architecture for preserving an accident of file layout and penalise one for
discarding it — and discarding it is permutation equivariance, which a graph model is
supposed to have.

Seed filter: ≥ 1 direct example (11,048 lemmas, so this type is the binding constraint
on the low bands) and a **60-word cap**, which retains 99.0 % of examples — an example
reaches 1,487 words, and a line that long is a transcription test rather than a
retrieval probe. Arity 1 and never split: 65 % of examples contain a comma and 4 of
51,172 contain a `|`.

90 % of core lemmas have no usage example — the largest natural negative pool of all —
but this type's negative share stays at the dataset default, since a type that is 90 %
sentinel teaches "say you don't know".

```
ODGOVOR: Je oče petih otrok in dedek štirih vnukov.
```

### T20 — `primeri_uporabe/analiza_oblike_v_povedi`

Form analysis with a sentence: given a real sentence and one word in it, say what
grammatical form that word takes — in words, never as an MSD tag. Restricted to
**unambiguous** forms, so every label is correct by construction.

The sentence comes from a `zgled:` node, so the item is fully contained in the ball.
The form is matched to the sentence by case-insensitive word-boundary string match,
rejecting a surface that occurs twice with different analyses or that matches more than
one form leaf of the entry — that is the ambiguity test, done cheaply. **No tagger**,
so no tagger's errors enter the gold.

The ambiguity test is only as good as the paradigm, which is what `healthy_grid` (C23)
protects: a lemma-filled paradigm makes a genitive plural look unambiguous because the
cell it should share was never inflected, and the result is a *plausible* wrong label.

Emits **T21's template exactly** and shares its generator — T20 is T21 with a sentence
around it, and giving one analysis two shapes across two types would make one of them
wrong by construction. Read this type against its **majority-class baseline of 53.9 %**
(C13): a dictionary example cites its headword in the base form more often than not,
and forcing that distribution flat would mean discarding true items to manufacture a
balance the language does not have.

```
ODGOVOR: tožilnik ednine
```

---

## 2. The checks

`qa/selftest.py` runs these against a generated dataset; the four generation filters
are in §1.

| # | check |
|---|---|
| C6 | **Tier C leakage** — no training item contains a tag word, and no training *question* contains a soft word. |
| C7 | **Unit tests** for the Slovene number-agreement table, the canonical orderings, and `sense_class` against a fixture holding one placeholder, one fallback with an example snippet, one definition equal to the headword, and one ordinary definition. |
| C9 | **The grader over the gold itself** — every item must score correct against its own answer. Catches separator collisions, stray whitespace and normalization bugs before they are misread as model failures, and it is the one test that validates the grading contract end to end. |
| C10 | **Every gold matches its type's `regex`**, and no multi-item gold contains its own separator inside an item. Single-item types are exempt by design (§0.9). |
| C11 | **The split is lemma-disjoint.** |
| C13 | **A majority-class baseline for every type**, reported beside the score. T9 (3 values), T10 (3 values), T14 (37.1 %), T16 (75 % single-antonym) and T20 (53.9 %) all admit cheap constant strategies, and a score without its baseline is unreadable. |
| C15 | **Every gold is in its type's canonical order** — including the set-valued types whose order is not graded. Grading tolerance is not a licence for non-canonical training data: the model must see exactly one ordering for a given set, or it is being taught noise on a surface it is forced to emit. |
| C16 | **`sl_key`** against a fixture including `č`, `š`, `ž` and a non-Slovene character. |
| C17 | **`T14 gold == len(T12 gold)`** for every lemma in both, and a lemma is a negative in both or in neither. T12 records its choices and T14 replays them, which is exact rather than probable — drawing negatives from two random streams made their agreement a coincidence that held in one generation and broke in the next. |
| C18 | **Sampler reproducibility and gold-in-ball**: the candidate pool is sorted by node id before drawing, the RNG seed derives from the anchor's node code and nothing else, and every gold item is inside its own ball. The first two are silent failures — CSR adjacency order is not stable across builds, so an unsorted pool or an order-dependent seed makes the dataset unreproducible without failing anything. T5/T6's composed auxiliary is the one enumerated exemption (§0.7). |

---

## 3. The item record

One JSON object per line in `train.jsonl` / `dev.jsonl` / `test.jsonl`, produced by
`qa/build_dataset.py` and consumed by the grader, the training pipeline and every
analysis.

```json
{
  "id": "T12-000123",
  "type": "T12",
  "type_name": "pomen/razlaga_pomena",
  "lemma": "cistitis",
  "lu_id": 23901,
  "node_code": 72057594037951293,
  "band": "B3",
  "proxy": 42,
  "split": "train",
  "tier": "core",
  "template_id": "T12/03",
  "question": "Podaj razlago pomena za besedo cistitis.",
  "answer": "ODGOVOR: vnetje sečnega mehurja in sečevodov | vnetje sečnega mehurja",
  "gold_items": ["vnetje sečnega mehurja in sečevodov", "vnetje sečnega mehurja"],
  "slots": {"L": "cistitis"},
  "negative": false,
  "negative_flavour": null,
  "grading": {}
}
```

| field | why it is there |
|---|---|
| `lu_id` / `node_code` | every item is anchored in the store's own id space, so item ↔ graph node is recoverable both ways. `node_code` is what the collocation sampler seeds from. |
| `band` / `proxy` | results are reported per band, and the raw proxy is kept so a boundary can move without regenerating. |
| `tier` | `core`, `A` (unseen phrasing) or `C` (unseen relation) — a headline result, so it is a field rather than a filename convention. |
| `template_id` | which frame produced the question, so a per-frame breakdown is free and a badly worded frame can be found after the fact. |
| `gold_items` | the answer already split, so analysis never re-implements the parser. |
| `grading` | the item-level half of the contract — `{}` for most types, since the type-level constants live in `qa/spec.py`. T17 and T19 carry `all_items`, `n_all`, `quantity_band` and `n_asked`. **The grader opens no store.** |
| `negative_flavour` | `absent`, `mismatch` or `nonexistent`, so the §0.2 mix is measurable rather than assumed. |

Two invariants hold by shape: the answer is a single line, so adding a prose half later
is purely additive; and the grader's input is the item, never the graph, so an eval run
cannot silently depend on which store version is mounted.

---

## 4. Types deliberately not included

| family | why |
|---|---|
| `mwe_structure`, `mwe_component_search`, `idiom_completion` | The best-supported unused capability — 10.3 M constituent edges with categories, constituent order, and MWE headwords that store their surface phrase. Deferred on implementation risk: it is the one family that depends on the upward `sestavina` cap behaving. |
| clitics and negation | Only **25** single-word entries carry a clitic value in the whole KG. The large corpus-wide counts are MWE components, which the builder collapses away. No pool to sample from. |
| reasoning-tier items (aggregation, filtering, multi-hop) | The dataset is retrieval-focused; the held-out tiers are the only place reasoning is probed. |
| hypernyms/hyponyms, meronyms/holonyms | 5,415 each (0.12 %); meronym and holonym absent entirely. |
| translations | 77,570 instances, all `directEquivalent`, all **`@hun`**. |
| register, style, dialect, domain, frequency, typo metadata | Absent from the KG entirely — which is what makes about half of the older reference file's types unanswerable. |
| valency, word formation | No valency frames, no derivational morphology. |
| role-play personas, JSON output, multi-turn | Not part of the service. Output shape is fixed for every item, and every item is one self-contained question and one answer. |
