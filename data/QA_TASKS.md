# The 34 question types

What each type asks, where its answer comes from, and the exact shape it is graded
in. `qa/spec.py` is the executable half of this document — the generator emits the
shape it declares and the grader parses that same shape, so the two cannot be edited
apart. Dataset-level design (seeds, bands, splits, the extraction policy) is in
[`QA_DATASET_DESIGN.md`](QA_DATASET_DESIGN.md).

> **Status — the output format below is specified, not yet implemented.** §0.1 states
> the labelled-pair rule that replaces the 19 per-type templates, and every type's
> block gives its output under that rule. `qa/spec.py`, `qa/gen.py` and the grader
> still implement the old per-type contract, and `datasets/generated/` still holds
> data in it, so *for now this document and its executable half disagree on purpose*.
> The measurements quoted in §0.1 and C25 were taken against the existing corpus and
> hold regardless. Closing the gap is one regeneration and one retrain
> ([`../TODO.md`](../TODO.md) §2).

## 0. Conventions that apply to every type

### 0.1 Answer format — the labelled-pair rule

Every answer is exactly one line, and every line has the same shape:

```
ODGOVOR: <oznaka>: <vrednost> | <oznaka>: <vrednost> | …
```

> **The answer labels its own content, using the same vocabulary the graph uses to
> label it.** `oznaka` is read off the ball node that carries the value. It is never
> a per-type constant chosen by the generator.

**What this replaces, and why.** Until now `arity`, `sep`, `mode` and `regex` were
keyed on the type id in `qa/spec.py`, so the *shape* of an answer was a pure function
of which type an item belonged to. The type, in turn, is a pure function of one
keyword in the question — *sklanjaj / spregaj / sopomenke / kolokacije / pomen / spol
/ vid / stopnjuje / zgled* separate the 19 types nearly perfectly. So the whole
question → answer-shape path bypassed the graph and only the slot fillers came from
it, and the cheapest policy fitting that data is *classify into 1 of 19 → emit that
template → fill from the ball*. That is what got learned: shown a relation it was
never trained on, the model has no class to route to and abstains. Tier C scores
**0.165**, below the **0.200** of the no-retrieval control, with a false-sentinel rate
of 0.356.

Under the labelled-pair rule an unseen relation no longer requires an unseen output
contract:

```
ODGOVOR: protipomenka: sodelovanje
```

is constructible by a model supervised only on `sopomenka:`, because the label is
copied off `protipomenka: boj ~ sodelovanje` in the ball.

**Four clauses make the rule exact.**

1. **The label vocabulary is the graph's own**, down to the number in `pomen N` (see
   T12). `iztočnica`, `oblika`, `pomen N`,
   `sopomenka`, `protipomenka`, `kolokacija` and `zgled` are the literal tags of §0.5.
   Feature bundles (`mestnik ednine`, `sedanjik 1. osebe ednine`, `presežnik`,
   `spol`) are rendered from the same parentheticals by `qa/sl.py`, which already
   declines them for T20/T21.
2. **Nothing is positional.** The line parses to a **multiset of pairs**: split on
   ` | `, then split each field at its **first** `: ` — a label never contains a
   colon, a value may. `sequence` and `multiset` collapse into that one comparison;
   only `membership` survives as a separate mode, for T17 and T19, whose gold is a
   subset draw (§0.8.3).
3. **Arity belongs to the question, not to the type.** *Give the locative* yields one
   pair, *give the whole declension* yields eighteen of **the same** pairs, and a
   missing cell is an absent key rather than a `/`. T2 and T3 are T1 with a smaller
   key set; T6 is T5 with a smaller key set; T9 and T10 are T8 with a smaller key set.
4. **No MSD tags anywhere** — morphology is answered in words (`tožilnik ednine`,
   never `Sozet`), as before.

#### The eight labels that are not in the ball

Measured by `qa/check_labels.py` over all **11,179 positive items** of
`datasets/balls/{train,dev,test}.jsonl`, matching whole words against node text only —
this is check **C25**. A label is only as readable as its worst component, so
`preteklik 1. osebe ednine` counts as a constant even though two of its three
components are derived:

| label | used by | present in the ball | status |
|---|---|--:|---|
| every case, number, person, vform, degree and `spol` label | T1–T3, T5–T7, T9, T11, T20, T21 | 100 % | **derived** |
| `iztočnica` · `pomen N` · `sopomenka` · `protipomenka` · `kolokacija` · `zgled` | T4, T12, T15–T17, T19 | 100 % | **derived** |
| `preteklik` · `prihodnjik` | T5, T6 | **0 / 1,233** | constant |
| `besedna vrsta` | T8 | **1 / 617** | constant |
| `vid` | T8, T10 | **2 / 1,233** — both incidental (*kolokacija: okvariti vid*) | constant |
| `število pomenov` | T14, T24 | **11 / 621**, all incidental (*oksidacijsko število …*) | constant |
| `več` · `manj` · `enako` | T27 only | not lexical content at all | constant |

The derived rows are 100 % **wherever the pair is emitted**, and the ten apparent
misses prove that rather than weakening it. `check_labels.py` names them, and every
one is an item with no such pair to label: the 2 T7 items without `velelnik` and the 5
T11 items without `presežnik` are exactly the answers carrying `/` in that slot
(`ODGOVOR: nedoločnik: zebsti; namenilnik: zebst; velelnik: /`), and the 3 T8 items
without `spol` are the three non-nouns — `števnik`, `prislov`, `prislov`. The label is
absent precisely when the value is.

`preteklik` and `prihodnjik` follow from **M2** (§0.7): the KG stores no past or
future form, so for those 18 of T5's 27 cells the generator composes both the value
*and* the label. Labelling them `deležnik na -l …` instead would be fully derived —
that string is in **1,233 / 1,233** conjugation balls — but it would misdescribe the
value, which is a periphrastic form and not a participle. They stay constants,
declared rather than hidden.

That list is a **budget, not an exemption.** Its length is the number of labels the
model must still memorise instead of read — how much of the old classification
behaviour survives the reformat. C25 fails on any label outside it, so a new type
cannot quietly add a ninth, and the diff that adds one also grows a list somebody
reviews.

The budget is the reason Group H is cheap: it adds 15 types and **3** constants, because
fourteen of the fifteen answer with a label an existing type already uses (see that
group's header).

C25 is also the **fairness proof for Tier C**: `protipomenka` is present in **104 /
104** antonym balls, so those items are *unseen* rather than unanswerable, and the
0.165 is a statement about the model rather than about an impossible task.

**The prose half.** A user-facing rendering is still deliberately postponed: the line
keeps its position and shape, so adding prose after it later is purely additive, and
the eval parser reads the first `ODGOVOR:` line and ignores the rest.

### 0.2 The "not recorded" sentinel

~10 % of items have no answer in the graph. The line is then exactly:

```
ODGOVOR: ni podatka v bazi
```

Three flavours are generated: the entity does not exist (`nonexistent`), it exists
but lacks the relation (`absent`), or the question does not apply to it at all
(`mismatch` — the gender of a verb, the conjugation of a noun).

**The sentinel is not a negative answer.** T23 and T25 ask a yes/no question, and their
`ne` means *the database holds this word and it has no such relation* — a fact read off
the ball. The sentinel means *the database has nothing to say about this word at all*.
Collapsing the two would reward exactly the behaviour Tier C exposes, where the model
abstains on 35.6 % of items rather than commit to a reading, so an `absent`-flavoured
sentinel is **never** generated for those two types: for them, absence is the answer.

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
    s = re.sub(r"\s+", " ", s).strip().rstrip(".").strip()   # idempotent
    return s.casefold()

def parse(line):                           # first ODGOVOR: line -> multiset of pairs
    body = line.split("ODGOVOR: ", 1)[1]
    out = []
    for field in body.split(" | "):
        oznaka, sep, vrednost = field.partition(": ")   # FIRST ": " only
        if not sep: return None            # unlabelled field -> unparseable
        out.append((norm(oznaka), norm(vrednost)))
    return sorted(out)

def correct(pred, gold, *, mode, allow=None, band=None, n_all=None):
    p = parse(pred)
    if p is None:      return False        # unparseable -> wrong, no partial credit
    if mode == "pairs":        return p == parse(gold)
    if mode == "membership":               # §0.8.3
        d = list(dict.fromkeys(p))
        return len(d) == len(p) and set(d) <= allow and count_ok(len(d), band, n_all)
```

**One comparison for 30 of the 34 types.** `parse` returns a sorted multiset, so
`pairs` is simultaneously the old `sequence` (a paradigm cell is addressed by its
label, not by its position) and the old `multiset` (synonyms are order-blind because
their labels are identical). The type-keyed `sep` and `arity` are gone; `mode` is now
one of two values and `membership` is carried on the item, not on the type.

One headline number: **`success`**, the share of items whose answer satisfies its
condition. `f1` is computed per item as a diagnostic — over pairs, so it now separates
*right value under the wrong label* from *wrong value*, which the positional version
could not — and is never reported as a score.

**Why exact match is defensible here.** Every gradeable string is present verbatim in
the model's input: these types are retrieval probes over an extracted subgraph, not
open generation. A model that paraphrases has not done the task.

The normalization is deliberately shallow — it folds case, collapses whitespace and
drops trailing periods. No diacritic stripping (that would merge real Slovene
distinctions), no stemming, no reordering.

**The trailing `strip()` is load-bearing, not tidiness.** `rstrip(".")` can uncover
whitespace the periods were hiding — a corpus sentence ending `… prestopka ...`
normalises to `… prestopka ` — so without it `norm` moves on its second application. A
membership item's allow-list is normalised when it is written and again when it is
compared, so a normalizer that is not **idempotent** marks a correct answer wrong;
observed on T19-000364, whose gold is its own only legal member. C7 holds the fixture.

#### 0.8.1 Order: always generated, never graded

Two concerns that must not be collapsed. **Generation is always canonical**: every
gold answer is emitted in one fixed order, with no dependence on graph-walk order, or
the training target for an item is unstable across runs. **Grading is never
order-sensitive**, because under §0.1 nothing needs it to be: what position used to
carry, the label now carries. In `gora, gore, gori, …` position 3 *was* the dative —
addressing dressed up as ordering — and it is now spelled `dajalnik ednine: gori`; in
`pasiven, resigniran, spokojen` the order was always an artefact of our own sort, and
penalising a reordering would score a formatting preference as a retrieval miss.

This is a strict gain in robustness, and it deletes the whole class of separator
collisions the positional grader had to be defended against.

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

**Separator.** ` | ` for every type, with no exceptions and no per-type choice. It is
the only safe candidate: over all 230,606 `@sl` definitions `|` occurs **0** times,
while `;` occurs in 8.5 % and `,` in 33.0 %, and 65 % of usage examples contain a
comma. The old arrangement gave the word-form types (T1, T5, T7, T11, T15, T16)
comma-separated lines on the grounds that a word form cannot contain a comma; that
was sound and is now moot, since one separator for all 34 types is exactly the
property §0.1 is buying.

The second delimiter is `: `, and it is split on **first occurrence only**, which is
what makes a definition containing a colon safe as a value.

#### 0.8.3 The second mode: membership + count

`pairs` compares against one gold multiset. That breaks for collocations, because the
ball holds a **seeded sample** of the anchor's
collocations, so two extractions of the same anchor legitimately support different
answers and a grader holding one list would mark a correct answer wrong whenever the
sample moved.

```
GOLD(item)  ⊆  BALL(anchor)  ⊆  ALL(anchor)
the gold line   what the model can see   what the grader accepts
```

The gold is drawn from the ball, so it is answerable from the input; grading is
membership in `ALL` plus a count rule, so any phrase the model could legitimately
have read is accepted. An answer is correct iff it repeats nothing, every pair is in
`ALL`, and the count matches what was asked. `ALL` is materialised per anchor at
generation time and shipped inside the item, so the grader opens no store. Under §0.1
its members are pairs (`kolokacija: mineralna voda`), so the label is checked along
with the phrase and the mode is a restriction of `pairs` rather than a different kind
of comparison.

The one place `|` inside a value would still be fatal is T19 — 4 of 51,172 usage
examples contain one. They are excluded by the same 60-word cap that already trims
the type (§T19).

### 0.9 Output templates — the notation used below

There is no per-type output template any more. §1 gives each type's **key set** — the
labels its answer uses — and its mode; the shape is §0.1's for all of them, and
`qa/spec.py`'s 19 regexes collapse to one, which is what C10 now checks:

```
^ODGOVOR: [^:|]+: [^|]+(?: \| [^:|]+: [^|]+)*$          or the §0.2 sentinel line
```

**The key set is a property of the item, not of the type.** T1's frame asks for
18 pairs and T3's for 1, and they draw from the same key set; which keys an item
actually carries is fixed by its question and recorded in `gold_items` (§3). This is
what retires the **C24** frame-scope filter: a frame asking for fewer forms than the
old contract filled used to produce gold that answered more than the question, and
now it cannot, because the question determines the keys.

**Gaps.** A missing cell is an **absent key**. The old `/` marker existed only to keep
positional alignment — a missing third form silently becoming a wrong fourth one — and
positions are gone, so the marker goes with them. An answer that legitimately has
nothing at all is the §0.2 sentinel, unchanged.

---

## 1. The 34 types at a glance

Every type emits §0.1's one shape, so the columns that used to distinguish them
(`arity`, `sep`, `regex`) are gone. What is left is the **key set** — which labels the
type's pairs may carry — and how many of them a given item asks for.

| # | type | source | key set | pairs / item |
|---|---|---|---|---|
| T1 | `sklanjanje/celotna_sklanjatev` | `oblika:` leaves, case × number | 6 cases × 3 numbers | 18 |
| T2 | `sklanjanje/sklanjatev_po_stevilu` | one number of T1's grid | ⊂ T1 | 6 |
| T3 | `sklanjanje/posamezen_sklon` | one cell of T1's grid | ⊂ T1 | 1 |
| T4 | `sklanjanje/osnovna_oblika_leme` | form → `iztočnica:` | `iztočnica` | 1–3 |
| T21 | `sklanjanje/analiza_oblike` | ambiguous form → its readings | ⊂ T1 | 1–4 |
| T5 | `spreganje/celotno_spreganje` | present cells + composed tenses | 3 tenses × 9 person/number | 27 |
| T6 | `spreganje/spreganje_v_casu` | one tense of T5's table | ⊂ T5 | 9 |
| T7 | `spreganje/neosebne_oblike` | `vform` ∈ {inf, supine, imperative} | `nedoločnik`, `namenilnik`, `velelnik` | 1–3 |
| T8 | `besedna_vrsta/osnovne_lastnosti` | `partOfSpeech` + one property | `besedna vrsta`, `spol`, `vid` | 1–2 |
| T9 | `besedna_vrsta/spol_samostalnika` | `gender` on the anchor | ⊂ T8 | 1 |
| T10 | `besedna_vrsta/vrsta_in_vid_glagola` | `aspect` on the anchor | ⊂ T8 | 1 |
| T11 | `stopnjevanje/vse_stopnje` | `degree` on forms | `osnovnik`, `primernik`, `presežnik` | 2–3 |
| T12 | `pomen/razlaga_pomena` | `pomen:` nodes classed `defined` | `pomen N` | open |
| T14 | `pomen/stevilka_pomenov` | the length of T12's list | `število pomenov` | 1 |
| T15 | `sopomenke/navedi_sopomenke` | `sopomenka:` nodes | `sopomenka` | open, ≤10 |
| T16 | `protipomenke/navedi_protipomenke` | `protipomenka:` nodes | `protipomenka` | open, ≤10 |
| T17 | `kolokacije/navedi_kolokacije` | `kolokacija:` nodes | `kolokacija` | open · **membership** |
| T19 | `primeri_uporabe/povedi_z_besedo` | `zgled:` nodes | `zgled` | 1 · **membership** |
| T20 | `primeri_uporabe/analiza_oblike_v_povedi` | `zgled:` + unambiguous form | ⊂ T1 | 1 |
| | **Group H** — composition, comparison, phrases | | | |
| T22 | `izbor/izbrane_oblike` | any subset of T1's grid | ⊂ T1 | 1–18 |
| T23 | `preveri/ali_obstaja` | presence of any relation | the relation's tag | 1 |
| T24 | `stevilo/koliko` | size of any *complete* relation | `število …` | 1 |
| T25 | `primerjava/ali_sta_v_relaciji` | a reified `A ~ B` node, or its absence | the relation's tag | 1 |
| T26 | `primerjava/skupne_lastnosti` | two anchors' parentheticals, intersected | ⊂ T8 | 0–2 |
| T27 | `primerjava/vec_ali_manj` | two anchors' relation counts | `več`/`manj`/`enako …` | 1 |
| T28 | `pomen/relacija_pomena` | one sense's own children | ⊂ T15/T17/T19 | open |
| T29 | `pomen/kateri_pomen` | child → its sense | `pomen N` | 1 |
| T30 | `zveze/sestava_zveze` | MWE → constituent headwords | `iztočnica` | 2–4 |
| T31 | `zveze/zveze_z_besedo` | word → MWEs containing it | `iztočnica` | open · **membership** |
| T32 | `zveze/pomen_zveze` | MWE's `defined` senses | `pomen N` | open |
| T33 | `zveze/zgled_za_zvezo` | MWE sense → `zgled:` | `zgled` | 1 · **membership** |
| T34 | `zveze/dopolni_zvezo` | fragment → the whole MWE | `iztočnica` | 1–3 |
| T35 | `kolokacije/beseda_kolokacije` | `kolokacija:` → the anchor it hangs off | `iztočnica` | 1 |
| T36 | `prevodi/madzarski_prevod` | `prevod (madžarsko):` nodes | `prevod (madžarsko)` | open |

T13 and T18 are retired — the first into T12, the second into T17's `exact` band — so
the inventory is 34 types numbered to 36. **T16 and T36 are Tier C**, held out of
training entirely.

**Twenty-one of them share six key sets.** T1/T2/T3/T20/T21/T22 are one key set at six
sizes, T5/T6 one at two, T8/T9/T10/T26 one at three, T12/T29/T32 one, T15/T17/T19/T28
one, and T4/T30/T31/T34/T35 one. Those distinctions stop being learnable as separate
output modes and become a question about *which keys were asked for* — which is the
property the reformat exists to buy. T3 and T21 are indistinguishable in output space
(`mestnik ednine: gori` either way), and so are the five `iztočnica` types, so a model
that routes by type gains nothing on any of them.

Four filters are shared across types rather than repeated in each, because a type
that forgets one produces gold that is plausible and wrong. A fourth, **frame scope
(C24)**, is retired by §0.9: it rejected a frame asking for fewer slots than the
answer contract filled, and there is no longer a contract to overfill.

| filter | what it rejects | why |
|---|---|---|
| `gen.healthy_grid` (C23) | nominal paradigms whose nominative plural equals the nominative singular, or whose filled cells hold fewer than 6 distinct surfaces | The KG stores lemma-filled paradigms — `odstotek` has `rodilnik/množina → odstotek`. 8.98 % of noun entries with ≥12 filled cells fail. It makes T1/T2/T3 a table of one repeated surface, and it makes T20's ambiguity test pass on a cell that only *looks* unambiguous. Decided per entry, never per lemma. |
| degenerate gradation (C22) | a comparative equal to the positive or the lemma | The KG stores `oblika: mikaven (…, primernik, …)` — the cell exists, the surface was never inflected. 29 % of comparative-bearing entries; they become truthful negatives. |
| premise truth (C20) | a frame naming a word class used on an entry of another class | *"…samostalnika izbrisen?"* where *izbrisen* is an adjective makes the question false before the model reads anything. A filter on the frame pool, not a hand-kept list. |
| constituent validity (C26) | an MWE whose `decomp:constituent` list names a word that is not in the phrase | The KG resolves function-word components to junk entries: `za → silo`, `v → celoti`, `se → prikazati`, plus typo lemmas like `abramba`. Keeping only MWEs whose every constituent's lemma or one of its forms occurs in the phrase passes **74.2 %** of them (clean pool ≈ 2.93 M); what it rejects is `predlog` (463 of 853) and `zaimek` (309). Without it, T30/T34 gold is confidently wrong on a quarter of the pool. |
| metalanguage agreement (C21) | ungrammatical Slovene in our own question and answer text | Case and number names decline (*v ednini*, *tožilnik ednine*) and a counted noun agrees with its numeral (*2 kolokaciji*, *3 kolokacije*, *5 kolokacij*). Both are tables in `qa/sl.py`, unit-tested (C7), never heuristics. Under §0.1 this table is load-bearing in a second place: it renders the **labels**, so a bug in it now corrupts the answer as well as the question. |

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

The 18 pairs are emitted in that canonical order, and graded without it (§0.8.1).

```
ODGOVOR: imenovalnik ednine: gora | rodilnik ednine: gore | dajalnik ednine: gori | tožilnik ednine: goro | mestnik ednine: gori | orodnik ednine: goro | imenovalnik dvojine: gori | … | orodnik množine: gorami
```

### T2 — `sklanjanje/sklanjatev_po_stevilu`

T1 restricted to one number, which is a template slot sampled uniformly so that
dvojina is not underrepresented. Seed filter: ≥ 5 of the 6 cells. The best negative in
the group is a **pluralia tantum** asked about the missing number — an ordinary lemma
whose answer is genuinely absent rather than invented.

The number is no longer a group leader — it is part of every label, so a T2 answer is
literally a 6-key subset of T1's.

```
ODGOVOR: imenovalnik ednine: gora | rodilnik ednine: gore | dajalnik ednine: gori | tožilnik ednine: goro | mestnik ednine: gori | orodnik ednine: goro
```

### T3 — `sklanjanje/posamezen_sklon`

One cell, answered with the labelled form. The (case, number) pair is sampled
**uniformly over the 18 cells** rather than from the reference file's nominative-heavy
distribution. Several cells can share a surface (`gori` is dative and locative
singular); harmless in this direction, because the question fixes the cell. The
reverse direction is T21 — and under §0.1 the two emit **the same string**, which is
the point: nothing in the output tells the model which type it was given.

```
ODGOVOR: mestnik ednine: gori
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
with one member — which under §0.1 costs nothing, since a repeated label is how every
set-valued type is written, and a one-member answer stops being a shape of its own.

```
ODGOVOR: iztočnica: gora          ODGOVOR: iztočnica: gol | iztočnica: gola
```

### T21 — `sklanjanje/analiza_oblike`

The inverse of T3: given a surface form and its lemma, name the case and number it
expresses. Restricted to **ambiguous** forms, answered with the disjunction — the
graph lists every reading a form carries and cannot pick between them, and with no
sentence there is nothing to pick with, so the disjunction *is* the correct answer.
~25 % unambiguous forms are kept so the model does not learn "always answer with a
disjunction".

Always **full case+number labels**, even when the number is shared: one shape for one
fact costs a repeated *ednine* and buys a key set identical to T1's. T20 is this type
with a sentence around it and shares its generator outright.

The disjunction is no longer spelled — the ` ali ` separator is gone, and two readings
are simply two pairs sharing a value. That is what makes T21 and T3 the same contract
read in two directions.

```
ODGOVOR: tožilnik ednine: goro | orodnik ednine: goro
ODGOVOR: rodilnik ednine: gore | imenovalnik množine: gore
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

Tense-major ordering survives as the canonical emission order; it is no longer a
grouping in the syntax, because the tense is part of each label.

**This is the type that pays for §0.1's constants.** `preteklik` and `prihodnjik` are
in **0 of 1,233** conjugation balls (C25), so 18 of these 27 labels are supplied by
the generator rather than read — matching M2, under which the values are composed too.

```
ODGOVOR: sedanjik 1. osebe ednine: delam | sedanjik 2. osebe ednine: delaš | … | sedanjik 3. osebe množine: delajo | preteklik 1. osebe ednine: sem delal | … | prihodnjik 3. osebe množine: bodo delali
```

### T6 — `spreganje/spreganje_v_casu`

One tense of T5's table, obtained by **calling T5's generator and taking one block** —
never a second implementation. The tense is a three-value slot sampled uniformly. The
*-l* participle carries gender, so the two composed tenses can also be asked in the
feminine or neuter (~20 % of those items); the default is masculine, which is the only
legal value for sedanjik.

**The gender is an ordinary part of the label.** It used to go in parentheses, because
`, ` was this type's item separator and `preteklik, ženski spol: …` would have split
the label into two items and made the arity depend on whether the slot was filled.
Neither hazard exists now: the separator is ` | `, the label is delimited by its first
`: `, and there is no arity. It is written like every other feature bundle, in
`qa/sl.py`'s order.

```
ODGOVOR: sedanjik 1. osebe ednine: delam | … | sedanjik 3. osebe množine: delajo
ODGOVOR: preteklik 1. osebe ednine ženskega spola: sem delala | … | preteklik 3. osebe množine ženskega spola: so delale
```

### T7 — `spreganje/neosebne_oblike`

The non-finite forms — nedoločnik, namenilnik, velelnik. **This type was already
written in the labelled shape**, which is the best evidence that §0.1 is native to
this data rather than imposed on it; only the separator changes, `; ` → ` | `.

An absent form is now an **absent key** rather than a `/`. The old marker existed to
hold a position, and there are no positions; a variable number of pairs is graded
exactly as a fixed one. Supine coverage is thin (20,878 against 1,580,662
infinitives), so a missing namenilnik is common and is a gap, not a sentinel; the
sentinel is for a verb with none of the three. The 2 balls in 611 that carry no
`velelnik` label are exactly the 2 items whose answer has the gap (C25).

```
ODGOVOR: nedoločnik: zaripniti | namenilnik: zaripnit | velelnik: zaripni
```

---

# Group C — `besedna_vrsta` (word class and grammatical properties)

T8, T9 and T10 assert the same facts about the same anchor, so they **share one
normalizer**. If they spelled a gender `moški spol` in one and `moški` in another, the
supervision would contradict itself on the one fact all three state. Under §0.1 they
also share a key set, and T9 and T10 are one-key subsets of T8.

**Gender is the one value whose graph string embeds its own label.** Node text says
`ženski spol`, so the pair is written `spol: ženski` — the ` spol` suffix moves from
the value to the label rather than being repeated. That is a one-line normalisation in
the shared normalizer, and it is why `spol` counts as derived in C25 (present in 100 %
of the balls of every item that emits it) while `besedna vrsta` and `vid` do not: the
graph never names those two properties, only their values.

### T8 — `besedna_vrsta/osnovne_lastnosti`

Word class, plus the one lexeme-level property that word class carries. One contract,
whatever the phrasing: the reference file's 11 question frames produce only six answer
shapes, one per POS, with no correlation between frame and shape — which is what makes
a Tier A phrasing withheld from training land on a contract the model already knows.

Every slot is lexeme-level. **Case, number, person, vform and degree never appear** —
they describe the citation *form*, not the word, and are T1–T7's material.

| POS | entries | answer |
|---|--:|---|
| noun | 310,362 | `besedna vrsta: samostalnik \| spol: <spol>` |
| adjective | 52,634 | `besedna vrsta: pridevnik` |
| verb | 18,159 | `besedna vrsta: glagol \| vid: <vid>` |
| adverb | 15,228 | `besedna vrsta: prislov` |
| numeral · interjection · pronoun | 2,434 · 507 · 284 | `besedna vrsta: števnik` · `… medmet` · `… zaimek` |
| abbreviation · preposition · particle · conjunction | 129 · 97 · 69 · 58 | `besedna vrsta: okrajšava` · `… predlog` · `… členek` · `… veznik` |

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
ODGOVOR: besedna vrsta: samostalnik | spol: ženski
ODGOVOR: besedna vrsta: glagol | vid: dovršni
ODGOVOR: besedna vrsta: prislov
```

### T9 — `besedna_vrsta/spol_samostalnika`

Grammatical gender of a noun, from `lexinfo:gender` on the **anchor's lexical-unit**,
not on its forms — noun word-forms carry no gender in this KG. Coverage is 310,362
triples over 310,362 distinct noun entries: every noun has exactly one, none has two,
and no other POS has any, so there is nothing to reconcile and no `moški ali ženski
spol` answer to generate. Asking the gender of a verb or adverb is the easiest
ordinary-lemma negative in the inventory.

```
ODGOVOR: spol: moški
```

### T10 — `besedna_vrsta/vrsta_in_vid_glagola`

Verb aspect, from `aspect` on the anchor. **Every single-word verb in the KG has
one** — 18,157 units, 100 % coverage of the pool, the best of any type. Three values,
one closed value set: the cleanest verifiable type in the inventory, and the natural
pipeline smoke test alongside T14.

*Vrsta glagola* in the wider sense is **not** in the graph — transitivity, reflexivity
and valency are absent, with no valency frames at all — so the questions ask about
aspect and are worded accordingly.

```
ODGOVOR: vid: dvovidski
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
entries have no superlative and simply omit that key — the third slot and its `/` are
both gone. The 5 balls in 546 that carry no `presežnik` label are exactly the 5 items
that used to emit the gap marker (C25).

Non-gradable adjectives asked for a comparative are the ideal ordinary-lemma negative
and are a substantial slice of this type.

```
ODGOVOR: osnovnik: lep | primernik: lepši | presežnik: najlepši
ODGOVOR: osnovnik: brunast | primernik: brunastejši
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

**Ordinals are now explicit**, because the ordinal *is* the label the graph uses:
`TAG_SENSE_N` renders `pomen 2: …`, so `pomen 2` is copied off the node exactly like
`sopomenka` is. This reverses the previous rule, which omitted them on the grounds
that position already carried them and that requiring `1.`/`2.` would measure counting
rather than retrieval. Neither objection survives §0.1: there is no position left to
carry the ordinal, and the ordinal is read rather than counted — it is printed on the
node the definition comes from. Identical definitions within an entry are deduplicated
(0.40 % of anchors), and **T14 must dedup identically** or its count stops matching
the list. Deduplication can therefore leave the ordinals non-contiguous; that is
correct, since they name senses rather than positions.

Seed filter: at least one `defined` sense — 40,926 of the 72,561-lemma pool. The
remaining 31,635 are the largest natural negative pool in the dataset, **shared with
T14**, so no lemma gets a list from one type and a refusal from the other.

Sense *labels* stored as definitions are kept, not filtered: 12.0 % of the pool's
definitions begin `o …` (*leten* → *o starosti*), and there are editor's notes shipped
verbatim. Filtering them would change both the list and T14's count for a reason
**invisible in the node text the model reads** — the same property that decided
`sense_class`.

```
ODGOVOR: pomen 1: zgradba za bivanje | pomen 2: rodbina, družina
```

**N is the graph's ordinal, never a 1…n renumbering of the kept senses.** The filters
above drop the placeholder and fallback senses from the *answer*; the **ball** renders
all of the anchor's senses under the store's own numbers, and the dropped ones keep
theirs there. So `nakladnica`'s one real sense is `pomen 3` in the ball and would have
been `pomen 1` in a renumbered answer. Under §0.1 the ordinal is part of the **label**,
so renumbering makes the label name a node that is not in front of the model — in
**623 of 623** positives, every one of which begins with at least one undefined sense.
The generator reads the ordinal off `store.senses`, which already returns it and used
to discard it; a sense the KG gives no number is labelled `pomen`, exactly as its node
text spells it.

C25 does not catch this — it treats the ordinal as positional noise and asks only
whether `pomen` is in the ball. **C18 (d)** does, because containment is over every atom
of the pair including the digit, and a wrong ordinal names a node that is not there.

### T14 — `pomen/stevilka_pomenov`

The count of the senses T12 lists — `len(T12_items(anchor))`, from the same
`sense_class` call, the same dedup and the same pool, obtained by calling the same
function. C17 asserts they never disagree, and that a lemma is a negative in both
types or in neither.

One pair, `število pomenov: N`. `0` is deliberately not a legal value: a lemma with no
defined sense is a sentinel, not a count of zero, so a `0` in the gold means the seed
filter leaked. This is one of the five §0.1 constants — `pomen` is in 100 % of the
balls, `število` in 11 of 621 and every one of those incidental (*oksidacijsko število
bromidnih ionov*) — and it is the type where a graph-derived label is least available,
since a count is a fact about the ball rather than a node in it. Answer distribution on the pool as seeded — 1: 37.1 %,
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
table instead. The gradeable value stays a bare integer and the label is fixed, so an
agreement bug in the question can never affect the score.

What this type measures is a **format and consistency probe** — does the model's
number match the list it would give? — not independent sense knowledge.

```
ODGOVOR: število pomenov: 3
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
ODGOVOR: sopomenka: pasiven | sopomenka: resigniran | sopomenka: spokojen | sopomenka: vdan v usodo
```

### T16 — `protipomenke/navedi_protipomenke`

The antonyms of a word, from `protipomenka:` nodes — **structurally identical to
`sopomenka:`, differing only in the tag word. That identity is the entire point.**

**Tier C: 100 % of antonym items are test items**, as for T36 (Hungarian translation),
the second held-out relation. Nothing about antonyms appears in
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

**This is the type §0.1 exists for.** Under the old contract the model had to invent an
output shape it had never been supervised on; under the labelled-pair rule the only
novel token is the label, and the label is sitting in the ball. C25 confirms the probe
is fair rather than impossible: `protipomenka` is present in **104 / 104** antonym
balls, so every Tier C item is answerable by copying, and whatever the model scores is
a statement about the model.

**Read the Tier C score against C13's `morph` column, not against zero.** 28.7 % of
T16's test items have their gold antonym sitting inside the question already — not as a
word, but as the stem of it: *neangažiranost → angažiranost*, *nepriročnost →
priročnost*, *nesramežljiv → sramežljiv*.
Slovene forms a large share of its antonyms by `ne-` prefixation, so a model that has
learned nothing about the `protipomenka` relation can still clear roughly a third of
this type by deleting two letters. That is a fact about Slovene rather than a defect in
the sample — filtering the prefixed pairs out would leave an antonym set that is not
what the dictionary holds — but it means **a Tier C score near 28.7 % is not evidence
of generalisation at all**, and the interesting quantity is the margin above it. The
un-prefixed remainder is the part that actually requires reading the graph.

```
ODGOVOR: protipomenka: sodelovanje
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
ODGOVOR: kolokacija: mineralna voda | kolokacija: mineralno gnojilo | kolokacija: mineralna snov | kolokacija: mineralna surovina | kolokacija: mineralno olje
```

---

# Group G — `primeri_uporabe` (usage examples)

### T19 — `primeri_uporabe/povedi_z_besedo`

Show the word in use: a recorded corpus sentence containing it, verbatim. The cleanest
possible retrieval signal.

Graded `membership` with a count of exactly one, **not `pairs`**: a lemma commonly
has several recorded examples and every one of them answers the question. The KG
stores them as a set, so grading against whichever has the lowest node id would reward
an architecture for preserving an accident of file layout and penalise one for
discarding it — and discarding it is permutation equivariance, which a graph model is
supposed to have.

Seed filter: ≥ 1 direct example (11,048 lemmas, so this type is the binding constraint
on the low bands) and a **60-word cap**, which retains 99.0 % of examples — an example
reaches 1,487 words, and a line that long is a transcription test rather than a
retrieval probe. **A `|` in the value is the one hazard §0.1 introduces here**: 65 % of
examples contain a comma, which is now harmless, but 4 of 51,172 contain a `|`, and one
of those as a value would split a single pair into two fields. The seed filter must
reject them outright — a four-item exclusion, checked by C10 over the gold rather than
assumed from the word cap.

90 % of core lemmas have no usage example — the largest natural negative pool of all —
but this type's negative share stays at the dataset default, since a type that is 90 %
sentinel teaches "say you don't know".

```
ODGOVOR: zgled: Je oče petih otrok in dedek štirih vnukov.
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

Emits **T21's shape exactly** and shares its generator — T20 is T21 with a sentence
around it, and giving one analysis two shapes across two types would make one of them
wrong by construction. Unambiguity means one pair rather than several, which is now a
count and not a different template. Read this type against its **label baseline of
57.4 %, all of it `quote`-reachable** (C13): a dictionary example cites its headword
in the base form more often than not, and forcing that distribution flat would mean
discarding true items to manufacture a balance the language does not have. Note what
the two C13 columns say together — the *value* is never work here, because the
question quotes the very form it asks about, so T20 is a question about the label and
nothing else. Under the pre-§0.1 format this read as a 53.9 % constant-*answer*
baseline; the constant string is gone but the type is exactly as cheap as it was, and
only the split measure shows it.

```
ODGOVOR: tožilnik ednine: hišo
```

---

# Group H — composition, comparison and phrases

**Why this group exists.** T1–T21 are one shape: *anchor → relation → dump*. Adding a
22nd instance of that shape buys a 22nd class. Every type below is instead a **product**
— `{list, filter, count, decide, compare, decompose} × {every relation}` — and a product
is too large to memorise as N slots. That structural property, not the individual
types, is what forces reading over classifying.

**The group adds almost no new output vocabulary, on purpose.** Fourteen of these
fifteen types answer with a label some existing type already uses:

| new type | its label | already used by |
|---|---|---|
| T22 filtered paradigm | case + number | T1, T2, T3 |
| T30 phrase decomposition · T31 phrase search · T34 completion | `iztočnica` | T4 |
| T32 phrase meaning · T29 which sense | `pomen N` | T12 |
| T33 phrase example | `zgled` | T19 |
| T28 sense-scoped relation | `sopomenka` / `kolokacija` / `zgled` | T15, T17, T19 |
| T35 reverse collocation | `iztočnica` | T4 |
| T26 shared properties | `besedna vrsta` / `spol` / `vid` | T8 |
| T24 counting | `število …` | T14 |
| T23 · T25 polarity | the relation's own tag | T15–T17, T19 |

So T4, T30, T31, T34 and T35 are **indistinguishable in output space**: five different
questions, one contract, and no classifier shortcut between them. Only T27 needs new
label words (`več`, `manj`, `enako`), taking §0.1's constant budget from 5 to **8**.

### H.0 What the group rests on

**Three pipeline changes, all measured, none touching the hop-2 policy (D4).**

| | change | measured cost |
|---|---|---|
| **D3b** | the surface reverse index also keys MWE canonical forms | **Landed.** 911,404 word keys + **3,871,485 phrase keys** = 4,782,889. A phrase can never shadow a word, and that is now *enforced* rather than sampled: only keys containing a space are added and never over a key the core pool owns. The sampled claim was wrong in one half — over the full 3,940,417 MWE anchors, **711 have a single-word lemma** (`klorat`, `n-kotnik`), which a 20,000 sample read as 100 % multi-word. Measured collisions under the space rule: **0**. |
| **D3c** | when a multi-word span does not resolve **whole**, `qa/relabel.py` falls back to the union of its constituents' anchors | **Landed.** Of the 55 multi-word spans the current extraction run produced, D3b resolves **43 (78.2 %)** whole and D3c the remaining **12** — analytic comparatives (`bolj nebistven`) and extraction noise (`mm popisanih`), which are not dictionary entries and never will be. Together: **0 of 55 unresolved**, against 55 of 55 before either. On the corpus this flips **39 items from `extract_miss` to `keep`** and rescues **37 T8 items** from the single-node ball — exactly the cohort C28 reports. |
| **D5c** | D5's `word → MWE` cap applies at **every** expansion, not only at the root | **Landed, and it is not optional once D3b/D3c are in.** Making the phrase items resolve is what exposed the hub: the first build with D3c and without D5c put three balls of **423,606 nodes / 5,028,776 tokens** into `train` — `zgoditi se`, `obnašati se`, `zaljubiti se`, whose clitic constituent sits at hop 1 and belongs to ~423 k phrases. With D5c they are **109, 75 and 79 nodes**; train's mean falls **268.5 → 88.0** and its token max **5,028,776 → 14,055**, while p50 holds at **1,317**. Diffed over the whole corpus: **12,450 of 12,490 balls byte-identical, the 40 that changed are exactly the MWE-anchored ones, and all 40 shrank.** |

D5c is a correction, not a tuning knob. `qa/build_balls.py` said everything but the
root's two hubs "is traversed unrestricted, because nothing else in the graph
explodes." That is false for an MWE root: its constituents sit at hop 1, and *biti* is
a constituent of 423,510 phrases, so hop 2 pulls all of them. The cap was always meant
to bound this hub; it was only ever applied in the one place a word-seeded ball could
reach it.

**Order these two correctly: D3b/D3c without D5c is worse than neither.** While the
phrase spans resolved to nothing, the hub was unreachable and the omission cost
nothing — every phrase item shipped a harmless single-node ball. Making them resolve is
what walks the builder into the hub, and the first build with D3c and without D5c put
three balls of 423,606 nodes / 5,028,776 tokens into `train`. The explosion is not a
pre-existing bug that D3c happens to reveal; it is *created* by fixing the lookup, and
the two changes belong in the same landing.

**A note on reading a ball row.** `anchors` on a ball row are positions inside that
row's own `nodes` list — they index `edges`, not the store. The store anchors are the
dataset row's `targets`, which are node **codes** and need `searchsorted` against
`store.codes`. The two are both small integers and neither errors when mistaken for the
other; it just silently compares a different entry. This cost a wrong D5c measurement
before the corpus-wide diff caught it.

D3c is the one that makes the phrase types exist at all, and it is not the change the
first draft of this group named. **The extractor is already correct**: its prompt
carries `Za besedno zvezo: ["črni pes"]`, it returned 55 multi-word spans over the
12,490-item run, and it returned more than one span on 1.6 % of them. The break was one
line downstream — `qa/relabel.py` resolved a span by `idx.get(x)`, a whole-string
lookup into an index that held **0 keys containing a space out of 911,404**. So
every phrase the extractor found resolved to nothing, and the failure was invisible
because an unresolved item still ships a well-formed `ni v bazi` ball. D3b fixes this
for MWE entries; D3c fixes it for everything else, including the collocation phrases of
T35, which are *not* MWE entries 70.4 % of the time.

Both now live behind `qa/relabel.py:resolve()`, one span at a time: whole first, and the
constituent union only for a span that contains a space. A single word that misses is a
genuine miss — "rescuing" it by splitting it would be rescuing it by doing nothing.

**Two-anchor items and the split.** C11 says the split is lemma-disjoint, and
`qa/seeds.py` assigns it by hashing **one** lemma. An item naming two lemmas therefore
puts both into whatever split it lands in, and the second one agrees only by chance —
measured over the corpus, a synonym partner that is itself a seed shares its anchor's
split **64.9 %** of the time (T15, n = 467) and **24.1 %** (T16, n = 29); the
chance rate is Σ share² ≈ 0.60. T25, T26 and T27 therefore admit a pair only when
**both lemmas already sit in the same split**, which costs ~40 % of the pair pool and
is the only rule that keeps C11 true. Nothing about the pair decides the split; the
split decides which pairs exist.

**The two held-out relations are not free slot values.** T23, T24 and T27 take the
relation as a slot, and two of the relations in the coverage table below belong to
Tier C: `protipomenka` (T16) and `prevod (madžarsko)` (T36). C6 bans their tag words
from *every* training item, so those two slot values are **test-only** — a training
question asking whether a word has an antonym is exactly the leak C6 exists to stop,
whatever its answer. Every other relation is available in both. This shrinks the
training slot vocabulary of those three types from nine relations to seven and leaves
their test vocabulary whole, which is what makes an unseen-relation slot value a
second, cheaper probe of the same generalisation T16 and T36 measure directly.

**What is complete in the ball, and therefore countable.** Measured by comparing each
ball's node counts against the store, over 1,200 single-anchor items:

| relation | ball == store | |
|---|--:|---|
| `oblika:` · `sopomenka:` · `protipomenka:` · `zgled:` · `prevod …:` | **100 %** | countable, listable exhaustively |
| `kolokacija:` | 29.2 % | D5b samples it — **membership only**, never a count |
| `pomen` | 4.2 % | complete for the anchor's *own* senses; the ball also carries the senses of every MWE it holds, so counting means reading the edges, not the tags |

That last row is not a defect, it is the group's best single item: `pomen 1: beli
kumulusi` and `pomen 1: kumulus` are the same shape and differ only in what they attach
to. A model that counts senses by counting `pomen` tags gets it wrong; one that reads
the graph gets it right.

**Anchor coverage**, over 4,000 sampled core anchors — the seed pool of every polarity,
counting and comparison type below:

| relation | ≥ 1 | ≥ 2 | median | p90 | max |
|---|--:|--:|--:|--:|--:|
| `oblika` | 91.5 % | 91.5 % | 17 | 56 | 166 |
| `pomen` (own) | 100 % | 45.5 % | 1 | 6 | 88 |
| `pomen` defined | 40.7 % | 25.9 % | 0 | 4 | 70 |
| `sopomenka` | 35.4 % | 26.1 % | 0 | 6 | 62 |
| `protipomenka` | 3.5 % | 1.1 % | 0 | 0 | 8 |
| `kolokacija` | 63.7 % | 56.6 % | 3 | 137 | 5,279 |
| `zgled` | 10.4 % | 9.1 % | 0 | 1 | 137 |
| `prevod` | 9.6 % | 4.8 % | 0 | 0 | 17 |
| MWE membership | 67.8 % | 60.5 % | 4 | 145 | 162,004 |

Every one of those rates is a **natural negative pool**, and the group is designed so
that the same relation is sometimes listed, sometimes counted, sometimes only asked
about — which is what stops *relation* from predicting *answer shape*.

---

### H.1 Selection over the paradigm

### T22 — `izbor/izbrane_oblike`

An arbitrary **subset** of T1's grid: *"katere oblike besede gora so v dvojini"*,
*"rodilnik in dajalnik množine"*, *"vse oblike v mestniku"*. Same key set as T1, same
source, same seed filter (`healthy_grid`, C23) — only the requested subset moves, and it
is drawn uniformly over the non-empty subsets of one axis rather than over all 2¹⁸.

This is the type that makes T1, T2 and T3 stop being three things. They are the subset
sizes 18, 6 and 1 of one contract; T22 is every other size, so no per-type arity can be
memorised and the model has to read which cells the question named. 91.5 % of core
anchors carry forms, median 17, so the pool is the whole nominal lexicon.

The negative is a number or case the entry does not fill — a pluralia tantum asked for
dvojina — which is an ordinary lemma with a genuinely absent answer, not an invented one.

```
ODGOVOR: imenovalnik dvojine: gori | rodilnik dvojine: gora | dajalnik dvojine: gorama | tožilnik dvojine: gori | mestnik dvojine: gorah | orodnik dvojine: gorama
```

---

### H.2 Polarity and quantity over any relation

Both types below take **the relation as a slot**. That is the whole point: the same
`sopomenka:` node is listed by T15, decided by T23 and counted by T24, so knowing which
relation the question names tells the model nothing about what shape to emit.

### T23 — `preveri/ali_obstaja`

*"Ali ima beseda X sopomenko / zgled / kolokacijo / več pomenov?"* One pair: the
relation's own tag, and `da` or `ne`. In **test** items the slot also takes the two
held-out relations, *protipomenka* and *prevod (madžarsko)*; in training it cannot —
see H.0.

**`ne` is an answer, not the sentinel.** §0.2's sentinel says *the database has nothing
to say about this word*; `ne` says *the database has this word and it has no such
relation*. Keeping them apart is the point of the type — the model that abstains
whenever it is unsure scores 0 here, and Tier C's 0.356 false-sentinel rate says that
is exactly what the current model does.

Every relation of the coverage table is in the slot — seven of them in training, all
nine in test — and their rates (3.5 % to 100 %) are what make both answers common
somewhere. The rate is balanced **per relation**, not globally, or the relation
predicts the answer and the graph goes unread.

Seed filter: exclude the handful of entries whose relation value is literally *da* or
*ne* — 2 of 13,015 synonym partners and 1 of 16,494 senses — since `sopomenka: da` is
then ambiguous between the two readings. Same class of exclusion as T1's animacy
doublets, and about as large.

```
ODGOVOR: sopomenka: da
ODGOVOR: zgled: ne
```

The second example is deliberately not `protipomenka: ne`. That line is a legal **test**
answer and an illegal training one, and writing the illegal case as the type's
illustration is how a held-out relation quietly becomes a seen one.

### T24 — `stevilo/koliko`

T14 generalised off senses and onto every countable relation: *"koliko sopomenk /
oblik / zgledov je zabeleženih za X"*. Label `število <relation, genitive plural>`,
declined by `qa/sl.py` (C21), value a bare integer.

**Only the six complete relations of H.0 are eligible.** `kolokacija` is sampled and
MWE membership is capped, so a count over either would grade the model on D5b's seed;
they are absent from this type by construction, and that asymmetry — listable but not
countable — is itself a thing the model has to read rather than assume. Two of the six,
`protipomenka` and `prevod`, are held out, so in training the eligible set is **four**
and the full six appear only in test (H.0).

Shares T14's rule that the question asks about **the database**, not about Slovene, and
shares its negative pool per relation.

```
ODGOVOR: število sopomenk: 4
```

---

### H.3 Two anchors

D3 unions the ball of every matched unit and calls selection among them "the model's
job". 90.7 % of current training balls have exactly one anchor, so that job is
essentially untrained (**R5**). These three types are where it gets trained.

**Affordable, measured.** Real 2-anchor balls in the current corpus: median **2,511**
tokens, p90 4,333, p99 7,716. Real 3-anchor: median 2,802. Against 1,275 median for a
single anchor, a pair costs about double and nothing overflows.

**Same-split pairs only.** These are the first types whose item names two lemmas, and
C11 is a statement about lemmas, not about items: a test item naming a training lemma
breaks it just as surely as a duplicated seed does. `qa/seeds.py` decides the split by
hashing a single lemma, so agreement between a pair's two halves is chance. The pair is
therefore drawn from **within one split**, and the distractor for a negative is drawn
from the same split as well.

**Capacity after the filter, measured over the whole 72,334-entry pool** (not over the
corpus sample the first draft used):

| relation | partner mentions | partner is itself a seed | …and in the same split |
|---|---|---|---|
| `sopomenka` | 206,476 | 160,843 (77.9 %) | **96,456** (60.0 %) |
| `protipomenka` | 5,654 | 5,119 (90.5 %) | **3,044** (59.5 %) |

Two corrections to what this section used to claim. First, the same-split rate is
**60.0 % and 59.5 %** — both sitting on the Σ share² ≈ 0.5998 chance baseline, exactly
as the argument predicts. The earlier figures of 64.9 % and 24.1 % were corpus samples
of n = 467 and **n = 29**; the antonym number in particular was noise, and nothing
should be read into antonyms behaving differently from synonyms. Second, the claim that
thin relations lose proportionally more does **not** hold here: `protipomenka` is
40× thinner and loses *less* (46.2 % of mentions against 53.3 %), because its partners
are likelier to be seeds. Recomputing per relation is still right — but as a rule, not
because thinness predicts the loss.

96,456 same-split synonym pairs is ample for T25–T27. 3,044 antonym pairs is thin in
absolute terms but sufficient, since `protipomenka` is a **test-only** slot value under
H.0 and needs a test slice rather than a training population.

### T25 — `primerjava/ali_sta_v_relaciji`

*"Ali sta besedi X in Y sopomenki?"* — one pair, the relation tag and `da`/`ne`.

The positive is answered by one node, `sopomenka: X ~ Y`, which sits in **both** balls:
of synonym partner lemmas that resolve in the surface index (78.6 %), **98.1 % show the
relation from the partner's side too**. So the union ball contains the evidence
whichever word the extractor found first.

The negative is the type's real content, and it is not a sentinel: both words are in the
database, both balls are present, and no node links them. Answering it requires
checking *the absence of an edge between two named things* — the one skill no
single-anchor type in this document exercises. Distractor pairs are drawn from the same
band, the same POS and the same split, so *plausible* is not *related*.

The relation slot obeys H.0 exactly as T23's does: `sopomenka` in both splits,
`protipomenka` in test only.

```
ODGOVOR: sopomenka: da
ODGOVOR: sopomenka: ne
```

### T26 — `primerjava/skupne_lastnosti`

*"Kaj imata besedi X in Y skupnega?"* — T8's key set, restricted to the properties the
two anchors agree on. Both anchors carry their properties in their own parenthetical, so
the answer is two reads and an intersection, and the empty intersection is the sentinel.

One shape, two anchors, and a key set that shrinks with the data rather than with the
type — which is §0.9's rule doing real work rather than restating T8.

```
ODGOVOR: besedna vrsta: glagol | vid: dovršni
```

### T27 — `primerjava/vec_ali_manj`

*"Katera beseda ima več pomenov, X ali Y?"* Over the six countable relations only, for
T24's reason, and over the four of them that are not held out in training, for H.0's.

**Ties are common and are not filtered away**: two random core anchors tie on sense count
33.4 % of the time, on form count 29.3 %, on synonym count 42.1 %. A seed filter
demanding a strict difference would teach *there is always a winner*, so a measured slice
of items are ties and answer `enako`. This is the one type in the group that needs new
label words — `več`, `manj`, `enako` — and they are declared constants under §0.1,
taking the budget from 5 to 8.

```
ODGOVOR: več pomenov: hiša
ODGOVOR: enako pomenov: hiša | enako pomenov: gora
```

The tie is two pairs sharing a label, not one pair holding a list — §0.1 has no
comma-separated values anywhere, and a lemma may contain a comma-free space but a
value list would be indistinguishable from a definition.

---

### H.4 Sense-scoped relations

`QA_DATASET_DESIGN.md` D12 specifies **Tier B** — an unseen question type over a seen
relation — at ~3 %, and it was never built. These two types are its natural first
instances, and `QA_TASKS.md` §T15 already names the first of them.

The scoping is real rather than decorative. Over 4,000 core anchors, 45.5 % have ≥ 2
senses, and where a relation is present on a multi-sense entry it sits on **some but not
all** senses almost always:

| relation | sense-scoped on a multi-sense entry | of those, discriminating | attaches to exactly one sense |
|---|--:|--:|--:|
| `sopomenka` | 33.4 % of anchors | 95.3 % | 100 % |
| `kolokacija` | 38.8 % | 96.3 % | 100 % |
| `zgled` | 11.4 % | 100 % | 99.9 % |

### T28 — `pomen/relacija_pomena`

*"Katere sopomenke ima beseda X v pomenu 'Y'?"* The question names a sense by its
definition or its ordinal; the answer is that sense's own children, not the entry's.

Grading and labels are T15/T17/T19's exactly — the same contract, asked of a smaller
scope. The distractor is the entry's *other* senses' children, sitting in the same ball,
so an answer that ignores the scope is wrong on content rather than on shape. That is
what makes this a Tier B probe instead of a rephrasing of T15.

```
ODGOVOR: sopomenka: biblioteka | sopomenka: knjigarna
```

### T29 — `pomen/kateri_pomen`

The inverse: *"Kateremu pomenu besede X pripada zgled/kolokacija Z?"* Well-defined
because attachment is unique — 100 % for collocations and synonyms, 99.9 % for examples,
measured above; the 0.1 % are excluded by seed filter.

Answered with the sense itself, so the label is `pomen N` and the contract is T12's. The
question gives what T12 asks for and asks for what T12 gives, which is the same
reversal T3/T21 perform on the paradigm — and, as there, the two are indistinguishable
in output space.

```
ODGOVOR: pomen 2: rodbina, družina
```

---

### H.5 Phrases

§4 called the MWE family "the best-supported unused capability" and deferred it on the
risk that the upward `sestavina` cap would misbehave. H.0 measures that risk directly:
it is real, it is D5c, and D5c is a no-op on every ball the corpus already has.

The pool is **3,940,417 MWE anchors** against 100,801 single-word core anchors, and it
carries the examples: **93.4 % of MWEs have ≥ 1 usage example** (median 4), which is the
14.6 M-example half of §3's finding that the dataset has never been able to reach.

**A warning that shapes three of these types.** The MWE pool is corpus-extracted
pairings, not a phrase dictionary — *poskus tolmačenja*, *zdrsniti vozniku*, *Osebje
tiskarne*. Two consequences, both measured:

* **Only 1.4 % carry a defined sense** (pool ≈ 54,000). So *what does this phrase mean*
  is a narrow type, and every other phrase type must not need a definition.
* **Function-word constituents resolve to junk lemmas.** `za → silo`, `v → celoti`,
  `se → prikazati`, and a typo entry `abramba`. A constituent filter — keep the MWE only
  if every constituent's lemma or one of its forms actually occurs in the phrase —
  passes **74.2 %** of MWEs, and what it rejects is overwhelmingly `predlog` (463 of 853)
  and `zaimek` (309). Clean pool ≈ **2.93 M**.

**Two things the seed pool cannot do as written** (measured on a 20,000 MWE sample,
before writing any of step 6):

* **`entry_for`'s content rule does not filter phrases.** It keeps an anchor with any of
  {definition, synonym, collocation, example}, and **95.4 %** of MWEs have an example —
  so the rule keeps essentially everything and projects a pool of **≈ 3.76 M entries
  against today's 72,334**, a 52× pool that takes 14.3 min to walk before a single item
  is generated. Tightening the rule is the wrong fix: the example-bearing half *is* the
  capability H.5 exists to reach, and requiring a definition instead would cut the pool
  to ≈ 62,849 by throwing that half away. **The pool must be bounded by sampling, not by
  content** — draw the phrase seeds first and build `Entry` objects for the draw, rather
  than materialising 3.76 M of them and sampling after.
* **D9's band is meaningless for a phrase.** `proxy` is collocations + MWE memberships,
  and a phrase has neither: of the phrase seeds sampled, **316 of 319 land in B0** and
  3 in B1, against a core pool that spreads 6,260 / 11,058 / 13,883 / 15,455 / 14,177 /
  8,426 / 3,075 across B0–B6. Banding phrases on `proxy` would put the whole family in
  one band and flatten the stratification the sampler depends on. A phrase needs its own
  frequency proxy — the natural candidate is an aggregate over its constituents' proxies,
  since a phrase containing *biti* is common and one containing a technical term is not —
  and whichever is chosen has to be stated in D9 rather than inherited by accident.

### T30 — `zveze/sestava_zveze`

*"Iz katerih besed je sestavljena zveza *beli kumulusi*?"* → the constituent **headwords**,
`bel` and `kumulus`.

This is T4 with a phrase instead of a word, and it is not string splitting: the phrase is
inflected and the constituents are lemmas, so the answer differs from the surface words
in **93.1 %** of cases. Seed filter is the constituent filter above; constituent counts
are 2 (44 %), 3 (50 %) and 4 (6 %), and `MWE → constituent` is uncapped in both the
old policy and D5c, so the list is always complete.

```
ODGOVOR: iztočnica: bel | iztočnica: kumulus
```

### T31 — `zveze/zveze_z_besedo`

The reverse: *"V katerih zvezah nastopa beseda kumulus?"* 67.8 % of core anchors are in
at least one phrase, median 4, so the pool is most of the lexicon.

Graded **`membership`**, not `pairs`, and for a different reason than T17's: the ball
holds D5's top-10 by a deterministic content-independent rank rather than a seeded
sample, so the set is reproducible but still a *selection* — the anchor's true membership
reaches 162,004. The question therefore asks for *some* phrases and never for all, with
T17's count bands and `ALL` = the ball's own set, shipped on the item.

```
ODGOVOR: iztočnica: beli kumulusi | iztočnica: podjetje Kumulus
```

### T32 — `zveze/pomen_zveze`

*"Kaj pomeni zveza X?"* — T12's contract with an MWE anchor. Narrow by the 1.4 %
measured above, ≈ 54,000 seeds, and worth having anyway because it is the only type that
asks a *phrase* for a definition, and because D5's rank prefers contentful senses, so the
phrases that have one are the phrases most likely to be reachable at all.

```
ODGOVOR: pomen 1: kopast oblak lepega vremena
```

### T33 — `zveze/zgled_za_zvezo`

*"Navedi poved z zvezo X."* T19's contract, and **the largest single unlock in this
document**: 93.4 % of 3.94 M MWEs carry an example, against the 71,729 examples reachable
from single-word entries that T19 lives on. §3's *99.5 % of the KG's 14.7 M usage
examples hang off MWE senses* stops being a limitation the moment the MWE is an anchor,
because its senses are then hop 1 and their examples hop 2.

Inherits T19's 60-word cap and its `|`-exclusion, and is graded `membership` with a
count of one for T19's reason: median 4 examples per phrase, every one of them an answer.

```
ODGOVOR: zgled: Beli valoviti kopasti kumulusi na jasnem nebu.
```

### T34 — `zveze/dopolni_zvezo`

*"Dopolni zvezo: beli ___."* The extractor returns the fragment, the surviving word
resolves, and the completed phrase is one of the ≤ 10 MWEs in its ball.

The seed filter is that reachability: an MWE qualifies only if it survives D5's rank for
at least one of its constituents. Without D3b that route is the **only** one and admits
~20 % of MWEs; with D3b the phrase is an anchor in its own right and the filter exists
only to keep the item answerable from the fragment the question actually gives.

Ambiguity is the design problem, as in T4: a fragment can have several legal
completions, and the gold names **all** of them that the ball holds.

```
ODGOVOR: iztočnica: beli kumulusi
```

### T35 — `kolokacije/beseda_kolokacije`

*"Kolokacija katere besede je *beli kumulusi*?"* → `kumulus`.

Answerable from the graph as it stands, and **only** once D3c is in place. A collocation
node hangs off a sense of the word it belongs to, and **99.7 %** of collocation phrases
contain a word that resolves back to that anchor through the existing index — so if the
ball is built from the phrase's *words*, the `kolokacija:` node is inside it, attached
to the answer, and no new index and no new edge is required.

The catch is that the extractor returns the phrase as one span, `["beli kumulusi"]`,
and `qa/relabel.py` looks that string up whole. Today that resolves to nothing —
0 of 911,404 index keys contain a space, and 0 of the 55 multi-word spans in the current
extraction run resolved. D3b does not rescue this type either, because a collocation
phrase is a different node from the MWE entry of the same string and the two coincide
only 29.6 % of the time. D3c does: the constituent fallback reaches the anchor by the
same route the 99.7 % figure measures. This is the type that most repays reading the
pipeline rather than the graph — the data was always there and the lookup could not
reach it.

The type matters because it is the direction nothing else in this document asks. It is
also the one that cannot be answered by emitting the anchor's most obvious relation —
the anchor **is** the answer, and the phrase is the question.

Note that the collocation node and the MWE entry of the same phrase are **different
nodes that are not linked to each other**; the collocation hangs off the sense, the MWE
off the anchor. They coincide textually in 29.6 % of cases and the type must not assume
it.

```
ODGOVOR: iztočnica: kumulus
```

---

### H.6 A relation held out by construction

### T36 — `prevodi/madzarski_prevod`

*"Kako se beseda X reče madžarsko?"* — from `prevod (madžarsko):` nodes, complete in the
ball (100 %), on 9.6 % of core anchors, 77,570 instances, all `directEquivalent` and all
`@hun`.

§4 listed translations as *not included*. That was right when the question was whether to
train on them and is wrong now: 9.6 % is far too thin to train a relation on and exactly
the right size for a **second Tier C**. Tier C today is one relation at n = 115, where a
single item moves the score by 0.009; `TODO.md` §1 asks for more held-out relations for
precisely that reason, and this one needs no new machinery — the label is the node's own
tag, so §0.1 makes it constructible unseen in the same way `protipomenka:` is.

Held out entirely, like T16, with the same two-part leak boundary (C6): the tag words
(*prevod*, *madžarsko*, *madžarski*) banned from training everywhere, and ordinary
Slovene for *translate* banned from training questions only.

```
ODGOVOR: prevod (madžarsko): kényes
```

---

## 2. The checks

`qa/selftest.py` runs these against a generated dataset; the four generation filters
are in §1.

| # | check |
|---|---|
| C6 | **Tier C leakage** — no training item contains a tag word, and no training *question* contains a soft word. Two held-out relations now: T16's (`protipomenka`, `antonim`) and T36's (`prevod`, `madžarsko`, `madžarski`). T36's tag words need the same two-part treatment, since *prevod* is ordinary Slovene that can appear inside a definition. **Group H widens what this check has to cover.** T23, T24, T25 and T27 take the relation as a *slot*, so a held-out relation can now enter training as a question's subject rather than as its answer — and `Ali ima beseda X protipomenko?` leaks T16 whether the gold says `da` or `ne`. Those two slot values are test-only (H.0), and C6 is what enforces it: the check is over the item, not over the answer, so it already catches this the moment a generator forgets. |
| C7 | **Unit tests** for the Slovene number-agreement table, the canonical orderings, and `sense_class` against a fixture holding one placeholder, one fallback with an example snippet, one definition equal to the headword, and one ordinary definition. |
| C9 | **The grader over the gold itself** — every item must score correct against its own answer. Catches separator collisions, stray whitespace and normalization bugs before they are misread as model failures, and it is the one test that validates the grading contract end to end. |
| C10 | **Every gold matches the one shape regex** of §0.9, no value contains ` \| `, and no *label* contains a colon. There is no per-type regex left and no single-item exemption — one shape, one check, all 34 types. The value rule is what the T19 and T33 seed filters must satisfy. |
| C11 | **The split is lemma-disjoint.** Group H makes this a check over *every lemma an item names*, not over its seed: T25, T26 and T27 name two, and a test item whose second lemma is a training seed breaks lemma-disjointness exactly as a duplicated seed would. The pair filter (H.3) is the generation-side rule; this is the assertion that it held. |
| C13 | **The no-knowledge baselines for every type**, reported beside the score. A score without its baseline is unreadable, and since §0.1 a *constant answer string* is no longer the cheap strategy: the label came apart from the value, and the two are cheap for different reasons. C13 reports four numbers — `answer` (the old constant string), `labels` (the most common label multiset, i.e. how often the answer's shape is free), `quote` (that label set *and* every value standing in the question as a whole word) and `morph` (that label set and every value inside the question but not as a word — the answer is a derivation of something quoted). Measured on the current test split: **T20 is 57.4 % `quote`** — its question supplies the very form it asks about, so the type is entirely about the label; **T4 is 47.8 % `morph`** (strip the inflection) and **T7 30.4 %**; and **T16 is 28.7 % `morph`**, because its antonyms are dominated by `ne-` prefixation, so almost a third of the Tier C type is reachable by deleting two letters and never consulting the graph. `quote` and `morph` are upper bounds, not scores — they say the information suffices, not that a model finds it — but they are what a retrieval arm must beat before its gap over the no-retrieval control means anything. Group H raises the stakes further: **T23, T25 and T27 are two- and three-way decisions**, so always-`da` and always-`več` are strong constant strategies and their baselines are not optional. The rates are balanced per relation for exactly this reason. |
| C15 | **Every gold is in its type's canonical order** — now including the paradigm types, whose order stopped being graded when §0.1 removed positions. Grading tolerance is not a licence for non-canonical training data: the model must see exactly one ordering for a given set, or it is being taught noise on a surface it is forced to emit. |
| C16 | **`sl_key`** against a fixture including `č`, `š`, `ž` and a non-Slovene character. |
| C17 | **`T14 gold == len(T12 gold)`** for every lemma in both, and a lemma is a negative in both or in neither. T12 records its choices and T14 replays them, which is exact rather than probable — drawing negatives from two random streams made their agreement a coincidence that held in one generation and broke in the next. |
| C18 | **Sampler reproducibility and gold-in-ball**: the candidate pool is sorted by node id before drawing, the RNG seed derives from the anchor's node code and nothing else, and every gold item is inside its own ball. The first two are silent failures — CSR adjacency order is not stable across builds, so an unsorted pool or an order-dependent seed makes the dataset unreproducible without failing anything. **§0.1 made (d) much stronger**: containment is over every atom of the `oznaka: vrednost` pair, so it now asserts that a paradigm cell holds *that* cell's surface and that T12's sense ordinal names a node the ball really has — neither of which the positional line could express. Three enumerated exemptions, all declared rather than inferred: the eight labels of §0.1's budget, which no ball carries by definition; T5/T6's composed auxiliary *and person*, which sit on the `biti` form and not on the participle (§0.7); and T14, whose **value** is composed too — a count is not a node, and the old check scored it 100 % only because a one-token gold of `3` matched any node containing a 3. |
| C25 | **Label derivability.** Every `oznaka` in every gold is present in that item's own ball, or is one of the eight declared constants of §0.1 — `preteklik`, `prihodnjik`, `besedna vrsta`, `vid`, `število pomenov`, `več`, `manj`, `enako`. Matching is whole-word against node text only. Without it a generator can take a label from a per-type table instead of from the graph and emit **byte-identical** output, so the defect §0.1 exists to remove comes back invisibly, one type at a time; the two implementations differ by one line and look equally reasonable in review. The constant list is a **budget, not an exemption**: its length is how many labels the model must still memorise rather than read, and a ninth cannot be added without a diff that shows it. Run over Tier C the same check is the **fairness proof** that those items are unseen rather than unanswerable. |
| C26 | **Constituent validity** (§1) asserted over T30/T34 gold: every constituent lemma the answer names has a surface — its own lemma or one of its `oblika:` forms — occurring in the phrase. It is a generation filter and a check, because the failure is silent: a wrong constituent is a well-formed lemma in the right shape. |
| C27 | **D5c is a no-op on word balls.** *Live in `qa/selftest.py`.* A fixed sample of single-word anchors is built twice in one process — `ball_nodes(..., d5c=True)` and `d5c=False`, the second being the pre-D5c expansion exactly — and the node sets must be equal. **300 / 300 today.** D5c exists to make MWE balls finite; the day it changes a word ball, it has changed the corpus every published number was measured on. The `d5c=False` flag exists for this check and for nothing else: there is no reason to build a real ball without the cap. |
| C28 | **Phrase items resolve.** No item of a phrase type may ship the single-node `iztočnica: … (ni v bazi)` ball. This is the check the group would most have benefited from having earlier: before D3b/D3c the corpus resolved **0 of 55** multi-word spans, and nothing failed, because an unresolved item still produces a well-formed ball, a well-formed question and a gold answer nothing in it supports. Since there are no phrase types yet, the live assertion is the wider one — **no positive item of any type** ships that ball (11,179 checked) — with the multi-word cohort reported beside it. That cohort is what moved: **42 items, 37 unresolved before D3b/D3c and 0 after.** |

C25 lives in **`qa/check_labels.py`** rather than in `selftest.py`, for the same reason
C18's containment half lives in `check_balls.py`: it needs the written balls, not the
generated items. Its first run, against the pre-reformat corpus, produced the §0.1
table — 119 labels derived, five constants, 104 / 104 on the Tier C label, and ten
misses all explained, in `--legacy` mode, which takes the label set from a per-type
table. That set is a **superset** of what any one item emits, so the mode reports and
does not fail; it is now the flag rather than the default.

Against the reformatted corpus it reads real pairs out of `gold_items` and is
pass/fail: **179 labels derived from the ball, 76 hitting a declared constant, 0
partial.** The ten misses are gone rather than explained — a gap is an absent key now,
so a T7 item with no imperative emits no `velelnik` pair to fail on.

The eight declared labels live in `qa/spec.py`, not here, because **C18 (d) has to
excuse exactly the same eight**: one asks whether a label is in the ball and the other
whether the pair is, and two copies of the budget would drift in the direction of a
label being unreadable in one check and declared in the other.

---

## 3. The item record

One JSON object per line in `train.jsonl` / `dev.jsonl` / `test.jsonl`, produced by
`qa/build_dataset.py` and consumed by the grader, the training pipeline and every
analysis.

```json
{
  "id": "T12-000002",
  "type": "T12",
  "type_name": "pomen/razlaga_pomena",
  "lemma": "onomatopoija",
  "lu_id": 75354,
  "node_code": 72057594038003290,
  "band": "B0",
  "proxy": 0,
  "split": "train",
  "tier": "core",
  "template_id": "T12/11",
  "question": "Naštej pomene besede onomatopoija.",
  "answer": "ODGOVOR: pomen 2: uporaba besed, ki posnemajo zvok, ki ga označujejo | pomen 3: pesniško sredstvo",
  "gold_items": [["pomen 2", "uporaba besed, ki posnemajo zvok, ki ga označujejo"],
                 ["pomen 3", "pesniško sredstvo"]],
  "slots": {"L": "onomatopoija"},
  "negative": false,
  "negative_flavour": null,
  "grading": {}
}
```

| field | why it is there |
|---|---|
| `lu_id` / `node_code` | every item is anchored in the store's own id space, so item ↔ graph node is recoverable both ways. `node_code` is what the collocation sampler seeds from. |
| `band` / `proxy` | results are reported per band, and the raw proxy is kept so a boundary can move without regenerating. |
| `tier` | `core`, `A` (unseen phrasing), `B` (unseen question type over a seen relation — T28 and T29 are its first instances) or `C` (unseen relation — T16 and T36) — a headline result, so it is a field rather than a filename convention. |
| `template_id` | which frame produced the question, so a per-frame breakdown is free and a badly worded frame can be found after the fact. |
| `gold_items` | the answer already split into `[oznaka, vrednost]` pairs, so analysis never re-implements the parser. It is also the item's **key set** (§0.9) — which labels this question asked for — so a per-label breakdown is free and C25 has something to check against. The record above is a real one, and its `pomen 2` / `pomen 3` is the point: the ordinals are the graph's, not a 1…n renumbering of the two senses that survived filtering (§T12). |
| `grading` | the item-level half of the contract — `{}` for most types, since the type-level constants live in `qa/spec.py`. T17, T19, T31 and T33 carry `all_items` (as pairs), `n_all`, `quantity_band` and `n_asked`. **The grader opens no store.** |
| `negative_flavour` | `absent`, `mismatch` or `nonexistent`, so the §0.2 mix is measurable rather than assumed. |

Two invariants hold by shape: the answer is a single line, so adding a prose half later
is purely additive; and the grader's input is the item, never the graph, so an eval run
cannot silently depend on which store version is mounted.

---

## 4. Types deliberately not included

| family | why |
|---|---|
| **reverse definition** — *"katera beseda pomeni zgradba za bivanje?"* | **Not answerable by this pipeline, measured.** Entity linking resolves surface strings, so a question that gives only a definition builds its ball from the definition's *own* words. Over 481 defined senses, the target anchor was reachable from them — directly or inside their hop-2 balls — **5.2 %** of the time. `TODO.md` §3.3 proposed it; this is the measurement that retires it. The other half of that proposal, *"katera beseda ima sopomenko knjižnica?"*, **is** answerable and is T25, because the reified node names both endpoints. |
| clitics and negation | Only **25** single-word entries carry a clitic value in the whole KG. The large corpus-wide counts are MWE components, which the builder collapses away. No pool to sample from. |
| reasoning-tier items (aggregation, multi-hop) | The dataset is retrieval-focused; Group H's composition types are selection and comparison over what the ball states, never inference beyond it. |
| hypernyms/hyponyms, meronyms/holonyms | 5,415 each (0.12 %); meronym and holonym absent entirely. Too thin to train, and — unlike translations — too thin even as a held-out probe. |
| register, style, dialect, domain, frequency, typo metadata | Absent from the KG entirely — which is what makes about half of the older reference file's types unanswerable. |
| valency, word formation | No valency frames, no derivational morphology. |
| role-play personas, JSON output, multi-turn | Not part of the service. Output shape is fixed for every item, and every item is one self-contained question and one answer. |
