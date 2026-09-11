#!/usr/bin/env python3
"""Question frames, per type, with the Tier A withheld set marked.

Every frame is plain prose with **no delimiters** around the target word (D3):
no `»...«`, no `"..."`, no `'...'`.  The reference files' quoted style is
phrasing inspiration only and every quote is stripped.  The consequence, binding
on every frame here, is that the target word appears as a VERBATIM surface string
that exists in the reverse index -- a lemma for most types, an inflected form for
T4/T20/T21.

Slots.  The metalanguage words decline: a frame that says "v ednini" needs the
locative and one that says "za ednino" the accusative, so the case is part of the
slot name.  Getting it wrong writes ungrammatical Slovene into the training
distribution ("sklanjatev za dvojini"), which is a defect and not a cosmetic one.

    {L}         the lemma              {F}   an inflected form
    {S}         a sentence             {N}   a count
    {ORD}       the case's ordinal
    {STEV_NOM|_GEN|_LOC|_ACC}   ednina / ednine / ednini / ednino
    {SKLON}, {SKLON_LOC}        mestnik / mestniku
    {CAS_NOM|_LOC|_ACC}         sedanjik / sedanjiku / sedanjik
    {SPOL_NOM|_LOC}             moški / moškem

Tier A (D12): frames marked `A` are **withheld from training** and appear only in
test items, so the score on them measures generalisation to unseen phrasing
rather than memorisation of a frame.  They are chosen before generation (C8), not
retrofitted, and each type withholds 3.

Volume: ~13 frames per type, 3 of them withheld.  No paraphrase expansion -- at
D16 scale that is ~70 items per frame with the slot values varying every time,
which is enough for retrieval training.  Widening the pool later is cheap and
invalidates nothing already generated.
"""

# type -> [(frame, tier)], tier in {"train", "A"}
TEMPLATES = {

    # ---- Group A: sklanjanje -------------------------------------------
    "T1": [
        ("Navedi vse sklone besede {L}.", "train"),
        ("Prikaži pregled sklanjatvenih oblik besede {L}.", "train"),
        ("Izpiši vse sklone in števila za besedo {L}.", "train"),
        ("Sklanjaj besedo {L} v ednini, dvojini in množini.", "train"),
        ("Kako se sklanja beseda {L}?", "train"),
        ("Prikaži celotno sklanjatev za besedo {L}.", "train"),
        ("Sestavi sklanjatveno tabelo za besedo {L}.", "train"),
        ("Zanima me celotna sklanjatev besede {L}.", "train"),
        ("Rabim vse sklanjatvene oblike besede {L}.", "train"),
        ("Katere sklanjatvene oblike ima {L}?", "train"),
        ("Sklanjatev besede {L}, prosim.", "A"),
        ("Pišem besedilo in ne vem, kako se sklanja {L}. Lahko pomagaš?", "A"),
        ("Prikaži, kako se beseda {L} spreminja po sklonih.", "A"),
        # --- the paradigm has a GENDER axis (adjectives; gendered numerals and
        # pronouns).  Naming it is not decoration: without it the question is
        # under-specified against a 54-cell paradigm and its answer would not be
        # determined by it (0.1 clause 3).  Most are category-neutral so that a
        # gendered numeral or pronoun can draw them too.
        ("Navedi vse sklone besede {L} v {SPOL_LOC} spolu.", "train"),
        ("Sklanjaj besedo {L} v {SPOL_LOC} spolu.", "train"),
        ("Prikaži pregled sklanjatvenih oblik besede {L} v {SPOL_LOC} spolu.", "train"),
        ("Izpiši vse sklone in števila za besedo {L} v {SPOL_LOC} spolu.", "train"),
        ("Sklanjaj besedo {L} v {SPOL_LOC} spolu, po vseh treh številih.", "train"),
        ("Kako se beseda {L} sklanja v {SPOL_LOC} spolu?", "train"),
        ("Prikaži celotno sklanjatev besede {L} za {SPOL_NOM} spol.", "train"),
        ("Sestavi sklanjatveno tabelo za besedo {L} v {SPOL_LOC} spolu.", "train"),
        ("Zanima me celotna sklanjatev pridevnika {L} v {SPOL_LOC} spolu.", "train"),
        ("Katere sklanjatvene oblike ima {L} v {SPOL_LOC} spolu?", "train"),
        ("Sklanjatev besede {L} v {SPOL_LOC} spolu, prosim.", "A"),
        ("Pišem besedilo in ne vem, kako se {L} sklanja v {SPOL_LOC} spolu. "
         "Lahko pomagaš?", "A"),
        ("Prikaži, kako se {L} v {SPOL_LOC} spolu spreminja po sklonih.", "A"),
        # --- and a DEFINITENESS axis as well: the masculine nominative and
        # accusative singular are the only cells that carry the split, and it is
        # what separates the doublet instead of discarding the entry.
        ("Navedi vse sklone besede {L} v {SPOL_LOC} spolu, v {DOLOCNOST_LOC}.", "train"),
        ("Sklanjaj besedo {L} v {SPOL_LOC} spolu, {DOLOCNOST}.", "train"),
        ("Izpiši sklanjatev besede {L} v {SPOL_LOC} spolu v {DOLOCNOST_LOC}.", "train"),
        ("Kako se beseda {L} sklanja v {SPOL_LOC} spolu, v {DOLOCNOST_LOC}?", "train"),
        ("Prikaži celotno sklanjatev besede {L} za {SPOL_NOM} spol v {DOLOCNOST_LOC}.", "train"),
        ("Sestavi sklanjatveno tabelo za {L} v {SPOL_LOC} spolu, {DOLOCNOST}.", "train"),
        ("Zanima me sklanjatev pridevnika {L} v {SPOL_LOC} spolu v {DOLOCNOST_LOC}.", "train"),
        ("Katere oblike ima {L} v {SPOL_LOC} spolu, v {DOLOCNOST_LOC}?", "train"),
        ("Sklanjatev besede {L}, {SPOL_NOM} spol, {DOLOCNOST}.", "A"),
        ("Kako bi {L} sklanjal v {SPOL_LOC} spolu, če rabim {DOLOCNOST}?", "A"),
    ],
    # {STEV_LOC} "v ednini" · {STEV_ACC} "za ednino" · {STEV_GEN} "sklone ednine"
    "T2": [
        ("Izpiši vse sklanjatvene oblike besede {L} v {STEV_LOC}.", "train"),
        ("Sklanjaj besedo {L} samo v {STEV_LOC}.", "train"),
        ("Katere sklone ima beseda {L} v {STEV_LOC}?", "train"),
        ("Prikaži tabelo sklanjatve besede {L} v {STEV_LOC}.", "train"),
        ("Kako se beseda {L} sklanja v {STEV_LOC}?", "train"),
        ("Kako poteka sklanjatev besede {L} za {STEV_ACC}?", "train"),
        ("Izpiši {STEV_ACC} besede {L} po sklonih.", "train"),
        ("Prikaži sklanjatev besede {L} za {STEV_ACC}.", "train"),
        ("Zanima me sklanjatev besede {L} v {STEV_LOC}.", "train"),
        ("Rabim oblike besede {L} v {STEV_LOC}, po vseh sklonih.", "train"),
        ("Sklanjatev besede {L}, samo {STEV_NOM}.", "A"),
        ("Kako bi {L} sklanjal v {STEV_LOC}?", "A"),
        ("Ali ima beseda {L} posebne oblike v {STEV_LOC}?", "A"),
        # --- gender axis
        ("Izpiši vse sklanjatvene oblike besede {L} v {STEV_LOC}, "
         "v {SPOL_LOC} spolu.", "train"),
        ("Sklanjaj besedo {L} v {STEV_LOC}, {SPOL_NOM} spol.", "train"),
        ("Katere sklone ima beseda {L} v {STEV_LOC} v {SPOL_LOC} spolu?", "train"),
        ("Prikaži tabelo sklanjatve besede {L} v {STEV_LOC}, "
         "v {SPOL_LOC} spolu.", "train"),
        ("Kako se beseda {L} v {SPOL_LOC} spolu sklanja v {STEV_LOC}?", "train"),
        ("Izpiši {STEV_ACC} besede {L} po sklonih, v {SPOL_LOC} spolu.", "train"),
        ("Prikaži sklanjatev pridevnika {L} v {STEV_LOC} za {SPOL_NOM} spol.", "train"),
        ("Rabim oblike besede {L} v {STEV_LOC} v {SPOL_LOC} spolu, "
         "po vseh sklonih.", "train"),
        ("Sklanjatev besede {L}, {STEV_NOM}, {SPOL_NOM} spol.", "A"),
        ("Kako bi {L} sklanjal v {STEV_LOC}, v {SPOL_LOC} spolu?", "A"),
        # --- gender and definiteness
        ("Izpiši sklanjatvene oblike besede {L} v {STEV_LOC}, v {SPOL_LOC} "
         "spolu, v {DOLOCNOST_LOC}.", "train"),
        ("Sklanjaj besedo {L} v {STEV_LOC}, {SPOL_NOM} spol, {DOLOCNOST}.", "train"),
        ("Katere sklone ima {L} v {STEV_LOC} v {SPOL_LOC} spolu, "
         "v {DOLOCNOST_LOC}?", "train"),
        ("Prikaži tabelo sklanjatve besede {L} v {STEV_LOC} za {SPOL_NOM} spol "
         "v {DOLOCNOST_LOC}.", "train"),
        ("Rabim oblike besede {L} v {STEV_LOC}, {SPOL_NOM} spol, "
         "{DOLOCNOST}.", "train"),
        ("Sklanjatev besede {L}, {STEV_NOM}, {SPOL_NOM} spol, {DOLOCNOST}.", "A"),
    ],
    # {SKLON} nominative · {SKLON_LOC} "v mestniku" · {STEV_GEN} "ednine"
    "T3": [
        ("Izpiši {ORD}. sklon ({SKLON}) {STEV_GEN} besede {L}.", "train"),
        ("Kako se glasi {SKLON} {STEV_GEN} besede {L}?", "train"),
        ("Katero obliko ima {SKLON} {STEV_GEN} samostalnika {L}?", "train"),
        ("V kakšni obliki je {SKLON} {STEV_GEN} pri besedi {L}?", "train"),
        ("Kako zapišemo {SKLON} {STEV_GEN} besede {L}?", "train"),
        ("Izpiši obliko: {SKLON} {STEV_GEN} za {L}.", "train"),
        ("Kateri je {SKLON} {STEV_GEN} besede {L}?", "train"),
        ("Zapiši besedo {L} v {SKLON_LOC} {STEV_GEN}.", "train"),
        ("Rabim {SKLON} {STEV_GEN} od {L}.", "train"),
        ("Kako bi besedo {L} postavil v {SKLON} {STEV_GEN}?", "train"),
        ("{SKLON} {STEV_GEN} besede {L}?", "A"),
        ("Ali mi lahko poveš {SKLON} {STEV_GEN} besede {L}?", "A"),
        ("Kako se glasi beseda {L}, če jo dam v {SKLON} {STEV_GEN}?", "A"),
        # --- gender axis
        ("Izpiši {ORD}. sklon ({SKLON}) {STEV_GEN} besede {L} "
         "v {SPOL_LOC} spolu.", "train"),
        ("Kako se glasi {SKLON} {STEV_GEN} besede {L} v {SPOL_LOC} spolu?", "train"),
        ("Katero obliko ima {SKLON} {STEV_GEN} pridevnika {L} "
         "v {SPOL_LOC} spolu?", "train"),
        ("V kakšni obliki je {SKLON} {STEV_GEN} pri besedi {L} "
         "v {SPOL_LOC} spolu?", "train"),
        ("Kako zapišemo {SKLON} {STEV_GEN} besede {L} v {SPOL_LOC} spolu?", "train"),
        ("Izpiši obliko: {SKLON} {STEV_GEN}, {SPOL_NOM} spol, za {L}.", "train"),
        ("Kateri je {SKLON} {STEV_GEN} besede {L} v {SPOL_LOC} spolu?", "train"),
        ("Zapiši besedo {L} v {SKLON_LOC} {STEV_GEN}, v {SPOL_LOC} spolu.", "train"),
        ("Rabim {SKLON} {STEV_GEN} od {L} v {SPOL_LOC} spolu.", "train"),
        ("Kako bi besedo {L} postavil v {SKLON} {STEV_GEN} "
         "{SPOL_NOM} spola?", "train"),
        ("{SKLON} {STEV_GEN} besede {L}, {SPOL_NOM} spol?", "A"),
        ("Ali mi lahko poveš {SKLON} {STEV_GEN} besede {L} "
         "v {SPOL_LOC} spolu?", "A"),
        ("Kako se glasi {L}, če jo dam v {SKLON} {STEV_GEN} "
         "{SPOL_NOM} spola?", "A"),
        # --- gender and definiteness
        ("Izpiši {ORD}. sklon ({SKLON}) {STEV_GEN} besede {L} v {SPOL_LOC} "
         "spolu, v {DOLOCNOST_LOC}.", "train"),
        ("Kako se glasi {SKLON} {STEV_GEN} besede {L} v {SPOL_LOC} spolu "
         "v {DOLOCNOST_LOC}?", "train"),
        ("Katero obliko ima {SKLON} {STEV_GEN} pridevnika {L} v {SPOL_LOC} "
         "spolu, {DOLOCNOST}?", "train"),
        ("Kako zapišemo {SKLON} {STEV_GEN} besede {L} za {SPOL_NOM} spol "
         "v {DOLOCNOST_LOC}?", "train"),
        ("Izpiši obliko: {SKLON} {STEV_GEN}, {SPOL_NOM} spol, {DOLOCNOST}, "
         "za {L}.", "train"),
        ("Rabim {SKLON} {STEV_GEN} od {L} v {SPOL_LOC} spolu, "
         "v {DOLOCNOST_LOC}.", "train"),
        ("{SKLON} {STEV_GEN} besede {L}, {SPOL_NOM} spol, {DOLOCNOST}?", "A"),
        ("Ali mi lahko poveš {SKLON} {STEV_GEN} besede {L} v {SPOL_LOC} spolu, "
         "v {DOLOCNOST_LOC}?", "A"),
    ],
    "T4": [
        ("Katera beseda v osnovni obliki predstavlja obliko {F}?", "train"),
        ("Navedi imenovalniško obliko besede {F}.", "train"),
        ("Kakšna je osnovna oblika besede {F}?", "train"),
        ("Navedi lemo za besedno obliko {F}.", "train"),
        ("V kateri lemi se nahaja oblika {F}?", "train"),
        ("Katera je lema besede {F}?", "train"),
        ("Iz katere osnovne oblike izvira oblika {F}?", "train"),
        ("Zapiši lemo (osnovno obliko) za besedo {F}.", "train"),
        ("Od katere besede je oblika {F}?", "train"),
        ("Pod katerim geslom najdem besedo {F}?", "train"),
        ("Naletel sem na obliko {F}. Katera je njena osnovna oblika?", "A"),
        ("Ne najdem besede {F} v slovarju. Kaj je njena osnovna oblika?", "A"),
        ("Ali je {F} oblika kakšne druge besede? Katere?", "A"),
    ],
    "T21": [
        ("Razloži, v katerem sklonu in številu je oblika {F} besede {L}.", "train"),
        ("Katere slovnične lastnosti nosi oblika {F} besede {L}?", "train"),
        ("Določi sklon in število za obliko {F} (lema {L}).", "train"),
        ("Navedi sklon in število, ki ga izraža {F} pri besedi {L}.", "train"),
        ("Kateri sklon in število izraža oblika {F} pri besedi {L}?", "train"),
        ("Katero sklanjatveno obliko predstavlja beseda {F} pri lemi {L}?", "train"),
        ("Slovnično analiziraj obliko {F} besede {L}.", "train"),
        ("Oblika {F} pri besedi {L} — kateri sklon in katero število?", "train"),
        ("V katerem sklonu in številu je {F}, če je osnovna oblika {L}?", "train"),
        ("Katere sklone lahko izraža oblika {F} besede {L}?", "train"),
        ("Kaj vse je lahko {F} pri besedi {L}?", "A"),
        ("Je {F} pri lemi {L} ednina ali množina, in kateri sklon?", "A"),
        ("Sem naletel na {F}. Pri lemi {L}, kaj je to po sklonu in številu?", "A"),
    ],

    # ---- Group B: spreganje --------------------------------------------
    "T5": [
        ("Prikaži pregled spreganja za glagol {L}.", "train"),
        ("Navedi vse osebe in čase za glagol {L}.", "train"),
        ("Kako glasi spreganje glagola {L} v sedanjiku, pretekliku in prihodnjiku?", "train"),
        ("Izpiši vse glagolske oblike za {L}.", "train"),
        ("Spregaj glagol {L}.", "train"),
        ("Kako se sprega glagol {L}?", "train"),
        ("Sestavi spreganje glagola {L} po časih in osebah.", "train"),
        ("Prikaži celotno spreganje glagola {L}.", "train"),
        ("Spregaj {L} po vseh osebah in časih.", "train"),
        ("Kako se sprega {L}?", "train"),
        ("Prikaži spreganje besede {L}.", "train"),
        ("Spreganje glagola {L}, prosim.", "A"),
        ("Ali mi lahko spregaš glagol {L}?", "A"),
        ("Prikaži, kako se {L} spreminja po osebah, številu in času.", "A"),
    ],
    # {CAS_NOM} "sedanjik" · {CAS_LOC} "v sedanjiku" · {CAS_ACC} "za sedanjik"
    "T6": [
        ("Kako se po osebah sprega glagol {L} v {CAS_LOC}?", "train"),
        ("Izpiši oblike glagola {L} v {CAS_LOC}, po osebah.", "train"),
        ("Kako se glasi {CAS_NOM} glagola {L}?", "train"),
        ("Navedi spreganje glagola {L} v {CAS_LOC}.", "train"),
        ("Spregaj glagol {L} v {CAS_LOC}.", "train"),
        ("Prikaži oblike za {CAS_ACC} glagola {L}.", "train"),
        ("Sestavi tabelo za {CAS_ACC} glagola {L}.", "train"),
        ("Katere so vse osebe in oblike za {CAS_ACC} glagola {L}?", "train"),
        ("Rabim {CAS_ACC} glagola {L} po osebah.", "train"),
        ("Katere oblike ima {L} v {CAS_LOC}?", "train"),
        ("{CAS_NOM} glagola {L}, prosim.", "A"),
        # C24: arity 9 — every person in all three numbers, so the frame must
        # ask for the whole table rather than name a single cell.
        ("Kako bi rekel {L} v {CAS_LOC}? Izpiši vse osebe in števila.", "A"),
        ("Spregaj {L} v {CAS_LOC}, vsa tri števila.", "A"),
    ],
    # {SPOL_NOM} "moški" · {SPOL_LOC} "moškem"
    "T6G": [
        ("Kako se glasi {CAS_NOM} glagola {L} v {SPOL_LOC} spolu?", "train"),
        ("Izpiši {CAS_ACC} glagola {L} za {SPOL_NOM} spol.", "train"),
        ("Spregaj glagol {L} v {CAS_LOC}, {SPOL_NOM} spol.", "train"),
        ("Navedi oblike glagola {L} v {CAS_LOC} za {SPOL_NOM} spol.", "train"),
        ("Kako se {L} sprega v {CAS_LOC}, kadar govorimo o {SPOL_LOC} spolu?", "train"),
        ("Katere oblike ima {L} v {CAS_LOC}, v {SPOL_LOC} spolu?", "train"),
        ("Izpiši {L} v {CAS_LOC} za {SPOL_NOM} spol.", "train"),
        ("Rabim {SPOL_NOM} oblike glagola {L} v {CAS_LOC}.", "A"),
        ("{CAS_NOM} glagola {L} v {SPOL_LOC} spolu?", "A"),
        ("Kaj je {SPOL_NOM} oblika glagola {L} v {CAS_LOC}?", "A"),
    ],
    # Every frame must ask for ALL THREE slots.  T7's answer contract is
    # fixed-arity 3 (`nedoločnik: … | namenilnik: … | velelnik: …`, spec.py), so
    # a frame that asks for one of them -- "Kako se glasi velelnik glagola
    # krasti?" -- is answered with the other two as well.  That teaches the model
    # to ignore which form the question named, which is the opposite of what a
    # form-selection task is for.  Found by reading generated items; the earlier
    # single-slot and two-slot frames are deleted rather than repaired.
    "T7": [
        ("Kako se glasijo nedoločnik, namenilnik in velelnik glagola {L}?", "train"),
        ("Izpiši nedoločnik, namenilnik in velelnik glagola {L}.", "train"),
        ("Navedi nedoločnik, namenilnik in velelnik za glagol {L}.", "train"),
        ("Katere tri oblike – nedoločnik, namenilnik in velelnik – ima glagol {L}?", "train"),
        ("Prikaži tri oblike glagola {L}: nedoločnik, namenilnik in velelnik.", "train"),
        ("Za glagol {L} zapiši nedoločnik, namenilnik in velelnik.", "train"),
        ("Glagol {L}: kako se glasijo nedoločnik, namenilnik in velelnik?", "train"),
        ("Naštej nedoločnik, namenilnik in velelnik glagola {L}.", "train"),
        ("Zanimajo me nedoločnik, namenilnik in velelnik glagola {L}.", "train"),
        ("Kaj so nedoločnik, namenilnik in velelnik pri glagolu {L}?", "train"),
        # Category-NEUTRAL, and tier `train` on purpose.  Every frame above names
        # `glagol`, and T7's negative is a word that is not one; `neutral_frames`
        # kept only the tier-A frame below, which train and dev may not use, so
        # the type rendered no negative outside the test split and said nothing.
        # A type needs at least one neutral frame its own splits can reach --
        # selftest C7 now checks that for all 34.
        ("Kako se glasijo nedoločnik, namenilnik in velelnik besede {L}?", "train"),
        ("Za {L} zapiši nedoločnik, namenilnik in velelnik.", "train"),
        ("Izpiši tri oblike besede {L}: nedoločnik, namenilnik in velelnik.", "train"),
        ("Nedoločnik, namenilnik in velelnik glagola {L}?", "A"),
        ("Ali ima {L} vse tri oblike – nedoločnik, namenilnik in velelnik? "
         "Kako se glasijo?", "A"),
        ("Pri glagolu {L} me zanimajo nedoločnik, namenilnik ter velelnik.", "A"),
    ],

    # ---- Group C: besedna vrsta ----------------------------------------
    "T8": [
        ("Kako slovnično opredelimo besedo {L}?", "train"),
        ("V katero vrsto besed uvrščamo besedo {L}?", "train"),
        ("Slovnično opredeli besedo {L}.", "train"),
        ("Katero besedno vrsto predstavlja {L}?", "train"),
        ("Kaj po besedni vrsti pomeni {L}?", "train"),
        ("Kakšne so slovnične lastnosti besede {L}?", "train"),
        ("Navedi slovnične značilnosti za lemo {L}.", "train"),
        ("Kateri besedni vrsti pripada beseda {L}?", "train"),
        ("Kaj je {L} po besedni vrsti?", "train"),
        ("Katera besedna vrsta je {L}?", "train"),
        # C24: T8's answer is the POS *and*, for nouns and verbs, the gender or
        # aspect.  "Rabim samo besednovrstno oznako" asked for strictly less than
        # the contract emits, so the gold contradicted the question.
        ("Rabim besednovrstno oznako za {L} z osnovnimi lastnostmi.", "A"),
        ("Zanima me, v katero besedno vrsto spada {L}.", "A"),
        ("Povej mi osnovne slovnične lastnosti besede {L}.", "A"),
    ],
    "T9": [
        ("Ali je samostalnik {L} moškega, ženskega ali srednjega spola?", "train"),
        ("Navedi spol za samostalnik {L}.", "train"),
        ("Kateri spol ima beseda {L}?", "train"),
        ("Kakšnega spola je samostalnik {L}?", "train"),
        ("Določi spol samostalnika {L}.", "train"),
        ("Povej, kakšnega spola je beseda {L}.", "train"),
        ("Kateri slovnični spol ima beseda {L}?", "train"),
        ("V kateri spol razvrstimo samostalnik {L}?", "train"),
        ("Zanima me spol besede {L}.", "train"),
        ("Povej spol za besedo {L}.", "train"),
        ("Spol samostalnika {L}?", "A"),
        ("Ne vem, ali je {L} moškega ali ženskega spola. Kaj je pravilno?", "A"),
        ("Pri besedi {L} — moški, ženski ali srednji spol?", "A"),
    ],
    "T10": [
        ("Ali je glagol {L} dovršni ali nedovršni?", "train"),
        ("Kateri vid izraža glagol {L}?", "train"),
        ("Kakšnega vida je glagol {L}?", "train"),
        ("Določi glagolski vid za besedo {L}.", "train"),
        ("Ali je {L} dovršni, nedovršni ali dvovidski glagol?", "train"),
        ("Opredeli glagolski vid besede {L}.", "train"),
        ("Ali spada glagol {L} med dovršne ali nedovršne glagole?", "train"),
        ("Kakšen vid ima {L}?", "train"),
        ("Zanima me vid besede {L}.", "train"),
        ("Določi vid besede {L}.", "train"),
        ("Kateri vid izraža {L}?", "train"),
        ("Vid glagola {L}?", "A"),
        ("Ali glagol {L} izraža dovršeno dejanje?", "A"),
        ("Kako je z vidom pri glagolu {L} — dovršni ali nedovršni?", "A"),
    ],
    "T11": [
        ("Izpiši stopnjevalni vzorec za besedo {L}.", "train"),
        # C24: the contract is arity 3 — osnovnik, primernik, presežnik — so a
        # frame may not ask for two of them.  Milder here than in T7 (the missing
        # slot is the lemma, which the question already names) but the same rule.
        ("Navedi vse tri stopnje besede {L}: osnovnik, primernik in presežnik.",
         "train"),
        ("Kako poteka stopnjevanje besede {L}?", "train"),
        ("Prikaži stopnjevanje pridevnika {L}.", "train"),
        ("Navedi vse stopnje za {L}.", "train"),
        ("Stopnjuj besedo {L}.", "train"),
        ("Izpiši osnovnik, primernik in presežnik za {L}.", "train"),
        ("Prikaži vse tri stopnje besede {L}.", "train"),
        ("Kako se stopnjuje beseda {L}?", "train"),
        ("Katere so tri stopnje besede {L}?", "train"),
        ("Ali se beseda {L} sploh stopnjuje? Če se, izpiši vse tri stopnje.", "A"),
        ("Kako rečem, da je nekaj bolj {L}, in kako najbolj? "
         "Navedi vse tri stopnje.", "A"),
        ("Ima {L} primernik in presežnik? Izpiši osnovnik, primernik in "
         "presežnik.", "A"),
    ],

    # ---- Group D: pomen -------------------------------------------------
    # Two halves, kept visibly apart because they must stay BALANCED: if the
    # "define it" phrasings dominate, the model learns to answer with one
    # definition and drop the later senses of a polysemous entry -- the exact
    # failure the T12/T13 merge exists to make measurable.
    "T12": [
        ("Podaj razlago pomena za besedo {L}.", "train"),
        ("Opiši pomen besede {L}.", "train"),
        ("Kakšen je pomen besede {L}?", "train"),
        ("Kaj pomeni beseda {L}?", "train"),
        ("Navedi definicijo besede {L}.", "train"),
        ("Razloži, kaj označuje beseda {L}.", "train"),
        ("Razvrsti in naštej pomene besede {L}.", "train"),
        ("Kateri so pomeni besede {L}?", "train"),
        ("Navedi različne pomene besede {L}.", "train"),
        ("Izpiši vse registrirane pomene besede {L}.", "train"),
        ("Katere pomene ima beseda {L}?", "train"),
        ("Naštej pomene besede {L}.", "train"),
        ("Kaj je {L}?", "A"),
        ("Ne poznam besede {L}. Kaj pomeni?", "A"),
        ("Zanimajo me vsi pomeni besede {L}.", "A"),
    ],
    "T14": [
        ("Koliko pomenov je za {L} zabeleženih v bazi?", "train"),
        ("Izpiši število pomenov besede {L}.", "train"),
        ("Navedi število pomenov za besedo {L}.", "train"),
        ("Preveri število registriranih pomenov besede {L}.", "train"),
        ("Koliko različnih pomenov je zabeleženih za {L}?", "train"),
        ("Koliko pomenov besede {L} je razloženih v slovarju?", "train"),
        ("Število zabeleženih pomenov za {L}?", "train"),
        ("Koliko pomenskih razlag ima geslo {L}?", "train"),
        ("Koliko razlag je v bazi pri besedi {L}?", "train"),
        ("Za koliko pomenov besede {L} obstaja razlaga?", "train"),
        ("Povej, koliko pomenov je pri {L} opisanih.", "A"),
        ("Zanima me, koliko pomenov je za {L} registriranih.", "A"),
        ("Koliko ločenih pomenov je pri geslu {L} popisanih?", "A"),
    ],

    # ---- Group E: sense relations --------------------------------------
    "T15": [
        ("Katere so sopomenke besede {L}?", "train"),
        ("Poišči sopomenke za {L}.", "train"),
        ("Katere besede imajo podoben pomen kot {L}?", "train"),
        ("Katere sopomenke obstajajo za izraz {L}?", "train"),
        ("S katerimi besedami lahko nadomestimo besedo {L}?", "train"),
        ("Navedi soznačice za besedo {L}.", "train"),
        ("Navedi sinonime za besedo {L}.", "train"),
        ("Prikaži seznam sopomenk za besedo {L}.", "train"),
        ("S čim lahko zamenjam besedo {L}?", "train"),
        ("Zanimajo me sinonimi za {L}.", "train"),
        ("Rabim drugo besedo za {L}.", "A"),
        ("V besedilu sem že uporabil {L}. Katero besedo lahko uporabim namesto nje?", "A"),
        ("Kako drugače rečem {L}?", "A"),
    ],
    # Tier C: EVERY antonym item is a test item and none of these strings may
    # appear anywhere in training (C6).  Written independently of T15's frames
    # rather than by swapping a word, so nothing about the relation leaks.
    # Every T16 frame is usable, and every T16 ITEM is a test item -- Tier C is a
    # held-out RELATION, not a held-out phrasing, so marking these "A" as well
    # would leave the type with no frame at all in the only split it appears in.
    "T16": [
        ("Kaj je nasprotje besede {L}?", "train"),
        ("Katera beseda pomeni nasprotno od {L}?", "train"),
        ("Rabim protipomenko za {L}.", "train"),
        ("Ali ima {L} nasprotni pomen?", "train"),
        ("Kako rečem nasprotno od {L}?", "train"),
        ("Protipomenka besede {L}?", "train"),
        ("Katera je nasprotna beseda od {L}?", "train"),
        ("Zanima me antonim besede {L}.", "train"),
        ("Napiši nasprotje za {L}.", "train"),
        ("Kateri antonim ustreza besedi {L}?", "train"),
    ],

    # ---- Group F: kolokacije, by quantity band -------------------------
    "T17/none": [
        ("Navedi primerne kolokacije za besedo {L}.", "train"),
        ("S katerimi besedami se najpogosteje povezuje beseda {L}?", "train"),
        ("Izpiši tipične besedne zveze z besedo {L}.", "train"),
        ("Katere besedne zveze so pogoste z besedo {L}?", "train"),
        ("S katerimi izrazi se kolocira beseda {L}?", "train"),
        ("Poišči pogoste kolokacije za besedo {L}.", "train"),
        ("Navedi tipične kolokacije z besedo {L}.", "train"),
        ("S katerimi besedami se povezuje {L}?", "train"),
        ("Katere besede se pogosto pojavljajo skupaj z {L}?", "train"),
        ("Navedi kolokativne zveze za lemo {L}.", "train"),
        ("Katere besedne povezave so zabeležene za {L}?", "A"),
        ("S čim se navadno kombinira beseda {L}?", "A"),
        ("Rabim seznam besed, ki gredo skupaj z {L}.", "A"),
    ],
    "T17/vague_small": [
        ("Navedi nekaj kolokacij besede {L}.", "train"),
        ("Daj mi par primerov besednih zvez z besedo {L}.", "train"),
        ("Naštej nekaj besednih zvez z besedo {L}.", "train"),
        ("Zanima me nekaj sopojavitev besede {L}.", "train"),
        ("Pokaži nekaj tipičnih zvez z besedo {L}.", "train"),
        ("Nekaj kolokacij za {L}, prosim.", "train"),
        ("Rabim par primerov, s čim se povezuje {L}.", "train"),
        ("Ali mi lahko daš nekaj kolokacij besede {L}?", "A"),
        ("Samo nekaj primerov: s čim se povezuje {L}?", "A"),
        ("Daj mi par zvez z besedo {L}.", "A"),
    ],
    "T17/vague_large": [
        ("Naštej čim več kolokacij besede {L}.", "train"),
        ("Rabim veliko primerov besednih zvez z besedo {L}.", "train"),
        ("Izpiši daljši seznam kolokacij za {L}.", "train"),
        ("Zanima me veliko sopojavitev besede {L}.", "train"),
        ("Pokaži mi obsežen seznam zvez z besedo {L}.", "train"),
        ("Navedi kar največ besednih zvez z besedo {L}.", "train"),
        ("Rabim čim daljši seznam kolokacij besede {L}.", "train"),
        ("Kar največ primerov, prosim: s čim se povezuje {L}?", "A"),
        ("Naštej veliko besed, ki gredo skupaj z {L}.", "A"),
        ("Daj mi obsežen nabor kolokacij za {L}.", "A"),
    ],
    # The counted noun agrees with the number: {N_KOL} renders "2 kolokaciji",
    # "3 kolokacije", "5 kolokacij".  See sl.counted().
    "T17/exact": [
        ("Izpiši {N_KOL} besede {L}.", "train"),
        ("Sestavi seznam {N_KOL} z besedo {L}.", "train"),
        ("Prikaži številčni seznam {N_KOL} besede {L}.", "train"),
        ("Poišči {N_TIPZVEZ} za besedo {L}.", "train"),
        ("Prikaži {N_POGKOL} z besedo {L}.", "train"),
        ("Izpiši točno {N_KOL} besede {L}.", "train"),
        ("Zapiši {N_KOLPRIM} za besedo {L}.", "train"),
        ("Navedi {N_KOLZVEZ} za besedo {L}.", "train"),
        ("Naštej {N_ZVEZ} z besedo {L}.", "train"),
        ("Daj mi {N_KOL} besede {L}.", "train"),
        ("Izpiši natanko {N_KOL} za {L}.", "A"),
        ("Ali mi lahko daš {N_KOL} za {L}?", "A"),
        ("Potrebujem {N_SOPOJ} besede {L}.", "A"),
    ],

    # ---- Group G: primeri uporabe --------------------------------------
    "T19": [
        ("Kako se beseda {L} pojavlja v besedilnem kontekstu?", "train"),
        ("Kako se beseda {L} uporablja v stavkih?", "train"),
        ("Prikaži primer rabe besede {L}.", "train"),
        ("Izpiši primer iz korpusa za besedo {L}.", "train"),
        ("Prikaži uporabo besede {L} v dejanski povedi.", "train"),
        ("Kako v praksi uporabimo besedo {L}?", "train"),
        ("Navedi stavčni primer za besedo {L}.", "train"),
        ("Zapiši primer stavka z besedo {L}.", "train"),
        ("Pokaži mi primer rabe besede {L}.", "train"),
        ("Rabim zgled uporabe za {L}.", "train"),
        ("V kakšnem stavku se pojavi {L}?", "A"),
        ("Ali imaš kakšen primer rabe besede {L}?", "A"),
        ("Kako bi {L} uporabil v stavku?", "A"),
    ],
    "T20": [
        ("Analiziraj rabo besede {F} v povedi: {S}", "train"),
        ("V katerem sklonu je beseda {F} v povedi {S}", "train"),
        ("Katero slovnično obliko izraža beseda {F} v stavku: {S}", "train"),
        ("Razloži oblikoslovno vlogo besede {F} v stavku {S}", "train"),
        ("Kakšna je slovnična oblika besede {F} v povedi: {S}", "train"),
        ("V kateri obliki nastopa beseda {F} v povedi: {S}", "train"),
        ("Določi sklon in število besede {F} v povedi {S}", "train"),
        ("Slovnično analiziraj besedo {F} v povedi {S}", "train"),
        ("V povedi {S} — v kateri obliki je beseda {F}?", "train"),
        ("Določi obliko besede {F} v povedi {S}", "train"),
        ("Napisal sem: {S} Kaj je {F} po sklonu in številu?", "A"),
        ("Poglej stavek {S} in povej, v kateri obliki nastopa {F}.", "A"),
        ("Katero slovnično obliko ima {F} tukaj: {S}", "A"),
    ],

    # ---- Group H: the first two ----------------------------------------
    # {REL_*} is one relation name in the case the frame needs -- see
    # sl.relation_slots().  Never write a relation name literally into a frame:
    # `protipomenka` is Tier C, and a literal would put its tag word into every
    # item of this type instead of only into the test items that ask about it.
    "T23": [
        ("Ali ima beseda {L} {REL_ACC}?", "train"),
        ("Ali baza vsebuje {REL_ACC_PL} za besedo {L}?", "train"),
        ("Preveri, ali so za besedo {L} {REL_REC_PL}.", "train"),
        ("Ali je pri besedi {L} {REL_NAV_NOM}?", "train"),
        ("Ali ima {L} {REL_ONE_ACC}?", "train"),
        ("Ali v bazi najdem {REL_ACC_PL} besede {L}?", "train"),
        ("Zanima me, ali ima beseda {L} {REL_ACC}.", "train"),
        ("Ali obstaja {REL_NOM} za besedo {L}?", "train"),
        ("Za besedo {L} — ali ima {REL_ANY_ACC}?", "train"),
        ("Ali baza pozna {REL_ANY_ACC} besede {L}?", "train"),
        ("Rabim samo potrditev: ima beseda {L} {REL_ACC}?", "A"),
        ("Ne najdem {REL_GEN} za besedo {L}. Je sploh v bazi?", "A"),
        ("Me zanima, ali baza za besedo {L} sploh ima {REL_ACC_PL}.", "A"),
    ],
    # T30's {L} is a MULTI-WORD phrase, so a frame must not call it "beseda":
    # "iz katerih besed je sestavljena besedna zveza pod drobnogledom" is a true
    # premise, "beseda pod drobnogledom" is not.
    "T30": [
        ("Iz katerih besed je sestavljena zveza {L}?", "train"),
        ("Katere iztočnice sestavljajo besedno zvezo {L}?", "train"),
        ("Razčleni zvezo {L} na posamezne besede.", "train"),
        ("Navedi sestavine besedne zveze {L}.", "train"),
        ("Katere osnovne oblike nastopajo v zvezi {L}?", "train"),
        ("Iz katerih iztočnic je sestavljena besedna zveza {L}?", "train"),
        ("Zapiši besede, iz katerih je sestavljena zveza {L}.", "train"),
        ("Na katere iztočnice razpade zveza {L}?", "train"),
        ("Kateri deli sestavljajo besedno zvezo {L}?", "train"),
        ("Razstavi besedno zvezo {L} na iztočnice.", "train"),
        ("Zveza {L} — iz česa je sestavljena?", "A"),
        ("Ne vem, kako razčleniti zvezo {L}. Katere besede jo sestavljajo?", "A"),
        ("Iz katerih osnovnih oblik je zveza {L}?", "A"),
    ],

    # ---- H.1: an arbitrary subset of the grid ---------------------------
    # Three families, on T1's rule exactly: a frame must name the axes the item
    # fixes and no others (0.1 clause 3), and `render_question` reads those off
    # the slots.  {IZBOR} is the selection itself, already declined by
    # `sl.selection_phrase` -- the frames never assemble it themselves.
    # The selection is a noun phrase in the nominative, so a frame must not put
    # a genitive after it: `{IZBOR} besede {L}` reads "the instrumental in the
    # singular and plural OF THE WORD X", where the possessive has crossed the
    # number list.  Every frame here therefore names the word with a preposition
    # (`za besedo {L}`, `pri besedi {L}`) or puts it first.
    "T22": [
        ("Za besedo {L} navedi {IZBOR}.", "train"),
        ("Izpiši {IZBOR} za besedo {L}.", "train"),
        ("Pri besedi {L} me zanima {IZBOR}.", "train"),
        ("Zapiši {IZBOR} za besedo {L}.", "train"),
        ("Prikaži {IZBOR} za {L}.", "train"),
        ("Beseda {L}: kako se glasi {IZBOR}?", "train"),
        ("Rabim {IZBOR} za besedo {L}.", "train"),
        ("Katere oblike ima beseda {L} — {IZBOR}?", "train"),
        ("Sklanjaj besedo {L}: {IZBOR}.", "train"),
        ("Daj mi {IZBOR} za besedo {L}.", "train"),
        ("{IZBOR} za besedo {L}, prosim.", "A"),
        ("Pišem besedilo in rabim {IZBOR} za besedo {L}. Lahko pomagaš?", "A"),
        ("Zapiši samo {IZBOR} za besedo {L}.", "A"),
        # gender axis
        ("Za besedo {L} v {SPOL_LOC} spolu navedi {IZBOR}.", "train"),
        ("Izpiši {IZBOR} za besedo {L} v {SPOL_LOC} spolu.", "train"),
        ("Pri besedi {L} v {SPOL_LOC} spolu me zanima {IZBOR}.", "train"),
        ("Zapiši {IZBOR} za besedo {L} v {SPOL_NOM} spolu.", "train"),
        ("Prikaži {IZBOR} za {L} v {SPOL_LOC} spolu.", "train"),
        ("Beseda {L} v {SPOL_LOC} spolu: kako se glasi {IZBOR}?", "train"),
        ("Sklanjaj {L} v {SPOL_LOC} spolu: {IZBOR}.", "train"),
        ("Daj mi {IZBOR} za besedo {L} v {SPOL_LOC} spolu.", "train"),
        ("{IZBOR} za besedo {L} v {SPOL_LOC} spolu, prosim.", "A"),
        ("Rabim {IZBOR} za besedo {L} v {SPOL_NOM} spolu. Lahko pomagaš?", "A"),
        ("Zapiši samo {IZBOR} za besedo {L} v {SPOL_LOC} spolu.", "A"),
        # gender + definiteness
        ("Za besedo {L} v {SPOL_LOC} spolu, v {DOLOCNOST_LOC}, navedi {IZBOR}.", "train"),
        ("Izpiši {IZBOR} za besedo {L} v {SPOL_LOC} spolu, {DOLOCNOST}.", "train"),
        ("Pri besedi {L} v {SPOL_LOC} spolu, v {DOLOCNOST_LOC}, me zanima {IZBOR}.", "train"),
        ("Zapiši {IZBOR} za besedo {L} v {SPOL_NOM} spolu, v {DOLOCNOST_LOC}.", "train"),
        ("Prikaži {IZBOR} za {L} v {SPOL_LOC} spolu, {DOLOCNOST}.", "train"),
        ("Sklanjaj {L} v {SPOL_LOC} spolu, {DOLOCNOST}: {IZBOR}.", "train"),
        ("Beseda {L}, {SPOL_NOM} spol, {DOLOCNOST}: {IZBOR}?", "train"),
        ("{IZBOR} za besedo {L} v {SPOL_LOC} spolu, v {DOLOCNOST_LOC}, prosim.", "A"),
        ("Rabim {IZBOR} za besedo {L} v {SPOL_LOC} spolu, {DOLOCNOST}.", "A"),
        ("Zapiši {IZBOR} za {L}, {SPOL_NOM} spol, {DOLOCNOST}.", "A"),
    ],

    # ---- H.2: how many? -------------------------------------------------
    # The counted noun agrees with the number, and the relation declines, so
    # both come from `sl.relation_slots` (C21) and never from an f-string here.
    "T24": [
        ("Koliko {REL_GEN_PL} je zabeleženih za besedo {L}?", "train"),
        ("Koliko {REL_GEN_PL} ima beseda {L}?", "train"),
        ("Koliko {REL_GEN_PL} je v bazi za besedo {L}?", "train"),
        ("Preštej {REL_ACC_PL} besede {L}.", "train"),
        ("Koliko {REL_GEN_PL} je navedenih pri besedi {L}?", "train"),
        ("Navedi število {REL_GEN_PL} za besedo {L}.", "train"),
        ("Koliko {REL_GEN_PL} beleži baza za {L}?", "train"),
        ("Kolikšno je število {REL_GEN_PL} besede {L}?", "train"),
        ("Preštej, koliko {REL_GEN_PL} je zapisanih za {L}.", "train"),
        ("Koliko {REL_GEN_PL} najdem pri besedi {L}?", "train"),
        ("Zanima me število {REL_GEN_PL} besede {L}.", "A"),
        ("Ali veš, koliko {REL_GEN_PL} je zabeleženih za {L}?", "A"),
        ("Število {REL_GEN_PL} za {L}, prosim.", "A"),
    ],

    # ---- H.3: two anchors -----------------------------------------------
    "T25": [
        ("Ali sta besedi {L} in {L2} {REL_NOM_PL}?", "train"),
        ("Sta {L} in {L2} {REL_NOM_PL}?", "train"),
        ("Ali je {L2} {REL_NOM} besede {L}?", "train"),
        ("Je beseda {L2} {REL_NOM} besede {L}?", "train"),
        ("Ali baza povezuje besedi {L} in {L2} kot {REL_ACC_PL}?", "train"),
        ("Preveri, ali sta {L} in {L2} {REL_NOM_PL}.", "train"),
        ("Ali sta {L} in {L2} v razmerju {REL_GEN}?", "train"),
        ("Je med besedama {L} in {L2} zabeležena {REL_NOM}?", "train"),
        ("Ali je pri besedi {L} navedena {REL_NOM} {L2}?", "train"),
        ("Sta besedi {L} in {L2} povezani kot {REL_NOM_PL}?", "train"),
        ("Zanima me, ali sta {L} in {L2} {REL_NOM_PL}.", "A"),
        ("Bi rekel, da sta {L} in {L2} {REL_NOM_PL}? Kaj pravi baza?", "A"),
        ("{L} in {L2} — {REL_NOM_PL} ali ne?", "A"),
    ],
    "T26": [
        ("Kaj imata besedi {L} in {L2} skupnega?", "train"),
        ("Katere skupne lastnosti imata {L} in {L2}?", "train"),
        ("V čem sta si besedi {L} in {L2} podobni?", "train"),
        ("Navedi skupne slovnične lastnosti besed {L} in {L2}.", "train"),
        ("Kaj je skupnega besedama {L} in {L2}?", "train"),
        ("Katere lastnosti si delita {L} in {L2}?", "train"),
        ("Primerjaj {L} in {L2}: kaj imata skupnega?", "train"),
        ("Izpiši, v čem se besedi {L} in {L2} ujemata.", "train"),
        ("Kje se besedi {L} in {L2} slovnično ujemata?", "train"),
        ("Kaj si delita besedi {L} in {L2}?", "train"),
        ("Zanima me, kaj imata {L} in {L2} skupnega.", "A"),
        ("{L} in {L2} — kaj jima je skupno?", "A"),
        ("Ali imata {L} in {L2} kakšno skupno lastnost?", "A"),
    ],
    # `{PRIM}` is the asked direction, `več` or `manj`.  A frame either names it
    # -- and then the answer's label repeats it -- or asks for the comparison
    # without a direction, which either label answers.  No frame may hard-code
    # `več`: half these items ask for the smaller side.
    "T27": [
        ("Katera beseda ima {PRIM} {REL_GEN_PL}, {L} ali {L2}?", "train"),
        ("Kdo ima {PRIM} {REL_GEN_PL}: {L} ali {L2}?", "train"),
        ("Primerjaj število {REL_GEN_PL} besed {L} in {L2}.", "train"),
        ("Pri kateri besedi je zabeleženih {PRIM} {REL_GEN_PL}, pri {L} ali pri {L2}?", "train"),
        ("Katera od besed {L} in {L2} ima {PRIM} {REL_GEN_PL}?", "train"),
        ("Kje je {PRIM} {REL_GEN_PL} — pri {L} ali pri {L2}?", "train"),
        ("Primerjaj {L} in {L2} po številu {REL_GEN_PL}.", "train"),
        ("Katera beseda beleži {PRIM} {REL_GEN_PL}, {L} ali {L2}?", "train"),
        ("Ugotovi, katera od besed {L} in {L2} ima {PRIM} {REL_GEN_PL}.", "train"),
        ("Pri {L} ali pri {L2} je {PRIM} {REL_GEN_PL}?", "train"),
        ("Zanima me, katera ima {PRIM} {REL_GEN_PL}: {L} ali {L2}.", "A"),
        ("{L} ali {L2} — katera ima {PRIM} {REL_GEN_PL}?", "A"),
        ("Bi znal primerjati {L} in {L2} po številu {REL_GEN_PL}?", "A"),
    ],

    # ---- H.4: sense-scoped ----------------------------------------------
    "T28": [
        ("Katere sopomenke ima beseda {L} v pomenu {POMEN}?", "train"),
        ("Navedi sopomenke besede {L} za pomen {POMEN}.", "train"),
        ("Katere sopomenke se navezujejo na pomen {POMEN} besede {L}?", "train"),
        ("Izpiši sopomenke, ki pripadajo pomenu {POMEN} besede {L}.", "train"),
        ("Beseda {L} v pomenu {POMEN} — katere sopomenke ima?", "train"),
        ("Za {ORD}. pomen besede {L} navedi sopomenke.", "train"),
        ("Katere sopomenke so zabeležene pri pomenu {POMEN} besede {L}?", "train"),
        ("Naštej sopomenke besede {L}, ki veljajo za pomen {POMEN}.", "train"),
        ("Samo za pomen {POMEN}: katere sopomenke ima {L}?", "train"),
        ("Sopomenke besede {L} v pomenu {POMEN}, prosim.", "train"),
        ("Zanimajo me sopomenke besede {L} v pomenu {POMEN}.", "A"),
        ("Kaj je sopomenka za {L}, kadar pomeni {POMEN}?", "A"),
        ("Pri {ORD}. pomenu besede {L} — katere sopomenke so navedene?", "A"),
    ],

    # The same type, asked of a sense that has an ORDINAL but no definition to
    # quote.  Most senses in this base are undefined, and requiring a definition
    # to name the scope by cost T28 two thirds of its population -- while the
    # ball numbers every sense, so the ordinal names one just as exactly.  A
    # separate pool because no frame here may reach for {POMEN}: it does not
    # exist for these items, and `render_question` would drop them silently.
    "T28/ord": [
        ("Za {ORD}. pomen besede {L} navedi sopomenke.", "train"),
        ("Katere sopomenke ima beseda {L} v {ORD}. pomenu?", "train"),
        ("Izpiši sopomenke, ki pripadajo {ORD}. pomenu besede {L}.", "train"),
        ("Katere sopomenke so zabeležene pri {ORD}. pomenu besede {L}?", "train"),
        ("Beseda {L}, {ORD}. pomen — katere sopomenke ima?", "train"),
        ("Naštej sopomenke besede {L}, ki veljajo za njen {ORD}. pomen.", "train"),
        ("Samo za {ORD}. pomen: katere sopomenke ima {L}?", "train"),
        ("Sopomenke besede {L} v {ORD}. pomenu, prosim.", "train"),
        ("Katere sopomenke se navezujejo na {ORD}. pomen besede {L}?", "train"),
        ("Navedi sopomenke, zabeležene pri {ORD}. pomenu besede {L}.", "train"),
        ("Zanimajo me sopomenke besede {L} v {ORD}. pomenu.", "A"),
        ("Pri {ORD}. pomenu besede {L} — katere sopomenke so navedene?", "A"),
        ("{L}, {ORD}. pomen: kaj je zabeleženo kot sopomenka?", "A"),
    ],
    "T29": [
        ("Kateremu pomenu besede {L} pripada {Z}?", "train"),
        ("V katerem pomenu besede {L} nastopa {Z}?", "train"),
        ("K kateremu pomenu besede {L} sodi {Z}?", "train"),
        ("Določi pomen besede {L}, ki mu pripada {Z}.", "train"),
        ("Pri katerem pomenu besede {L} je zabeleženo {Z}?", "train"),
        ("Kateri pomen besede {L} ponazarja {Z}?", "train"),
        ("Poišči pomen besede {L}, pod katerim je {Z}.", "train"),
        ("Pod katerim pomenom besede {L} najdem {Z}?", "train"),
        ("Kateremu pomenu je pri besedi {L} pripisano {Z}?", "train"),
        ("Navedi pomen besede {L}, h kateremu spada {Z}.", "train"),
        ("Zanima me, kateremu pomenu besede {L} pripada {Z}.", "A"),
        ("{Z} — h kateremu pomenu besede {L} to sodi?", "A"),
        ("Ali veš, kateri pomen besede {L} pokriva {Z}?", "A"),
    ],

    # ---- H.5: phrases ---------------------------------------------------
    # T31 bands, on T17's rule: the question asks for SOME phrases and never for
    # all, because the ball holds D5's top ten and not the true membership.
    "T31/none": [
        ("V katerih zvezah nastopa beseda {L}?", "train"),
        ("Navedi besedne zveze z besedo {L}.", "train"),
        ("Katere stalne zveze vsebujejo besedo {L}?", "train"),
        ("V katerih besednih zvezah se pojavlja {L}?", "train"),
        ("Izpiši zveze, v katerih nastopa beseda {L}.", "train"),
        ("Katerih zvez je beseda {L} sestavni del?", "train"),
        ("Poišči besedne zveze z besedo {L}.", "train"),
        ("Del katerih zvez je beseda {L}?", "train"),
        ("Naštej zveze, ki vsebujejo besedo {L}.", "train"),
        ("V čem vse nastopa beseda {L}?", "A"),
        ("Zanima me, v katerih zvezah je beseda {L}.", "A"),
        ("Kje vse se pojavi beseda {L} kot del zveze?", "A"),
    ],
    "T31/vague_small": [
        ("Navedi nekaj zvez z besedo {L}.", "train"),
        ("Daj mi par besednih zvez, v katerih nastopa {L}.", "train"),
        ("Naštej nekaj stalnih zvez z besedo {L}.", "train"),
        ("Pokaži nekaj zvez, ki vsebujejo besedo {L}.", "train"),
        ("Nekaj besednih zvez z besedo {L}, prosim.", "train"),
        ("Rabim par primerov zvez z besedo {L}.", "train"),
        ("Zanima me nekaj zvez, v katerih je beseda {L}.", "A"),
        ("Ali mi lahko daš nekaj zvez z besedo {L}?", "A"),
        ("Samo par primerov: kje nastopa beseda {L}?", "A"),
    ],
    "T31/vague_large": [
        ("Naštej čim več zvez z besedo {L}.", "train"),
        ("Rabim veliko besednih zvez, v katerih nastopa {L}.", "train"),
        ("Izpiši daljši seznam zvez z besedo {L}.", "train"),
        ("Pokaži mi obsežen seznam zvez z besedo {L}.", "train"),
        ("Navedi kar največ stalnih zvez z besedo {L}.", "train"),
        ("Rabim čim daljši seznam zvez z besedo {L}.", "train"),
        ("Kar največ primerov, prosim: kje nastopa beseda {L}?", "A"),
        ("Daj mi obsežen nabor zvez z besedo {L}.", "A"),
        ("Naštej veliko zvez, ki vsebujejo {L}.", "A"),
    ],
    "T31/exact": [
        ("Izpiši {N_ZVEZ} z besedo {L}.", "train"),
        ("Navedi {N_ZVEZ}, v katerih nastopa {L}.", "train"),
        ("Sestavi seznam {N_ZVEZ} z besedo {L}.", "train"),
        ("Poišči {N_STALZVEZ} z besedo {L}.", "train"),
        ("Prikaži {N_ZVEZA}, ki vsebujejo besedo {L}.", "train"),
        ("Izpiši točno {N_ZVEZ} z besedo {L}.", "train"),
        ("Daj mi {N_ZVEZ} z besedo {L}.", "train"),
        ("Naštej {N_ZVEZA} z besedo {L}.", "train"),
        ("Izpiši natanko {N_ZVEZ} za {L}.", "A"),
        ("Ali mi lahko daš {N_ZVEZ} z besedo {L}?", "A"),
        ("Rabim {N_STALZVEZ}, v katerih je {L}.", "A"),
    ],
    "T32": [
        ("Kaj pomeni zveza {L}?", "train"),
        ("Razloži pomen zveze {L}.", "train"),
        ("Kakšen je pomen besedne zveze {L}?", "train"),
        ("Navedi razlago zveze {L}.", "train"),
        ("Kaj označuje zveza {L}?", "train"),
        ("Pojasni, kaj pomeni besedna zveza {L}.", "train"),
        ("Kako je razložena zveza {L}?", "train"),
        ("Kaj se skriva za zvezo {L}?", "train"),
        ("Opiši pomen zveze {L}.", "train"),
        ("Katere pomene ima zveza {L}?", "train"),
        ("Zanima me, kaj pomeni zveza {L}.", "A"),
        ("Zveza {L} — kaj to pomeni?", "A"),
        ("Ne razumem zveze {L}. Kaj pomeni?", "A"),
    ],
    "T33": [
        ("Navedi poved z zvezo {L}.", "train"),
        ("Zapiši primer rabe zveze {L}.", "train"),
        ("V kakšni povedi se uporablja zveza {L}?", "train"),
        ("Daj primer stavka z zvezo {L}.", "train"),
        ("Kako se zveza {L} uporabi v povedi?", "train"),
        ("Navedi zgled za zvezo {L}.", "train"),
        ("Pokaži rabo zveze {L} v povedi.", "train"),
        ("Sestavi primer uporabe zveze {L}.", "train"),
        ("Izpiši poved, v kateri nastopa zveza {L}.", "train"),
        ("Kje se zveza {L} pojavi v besedilu?", "train"),
        ("Zanima me primer rabe zveze {L}.", "A"),
        ("Ali imaš kakšen zgled za zvezo {L}?", "A"),
        ("Rabim poved z zvezo {L}. Lahko pomagaš?", "A"),
    ],
    # T34's {L} already carries the blank, so the frame never writes one.
    "T34": [
        ("Dopolni zvezo: {L}", "train"),
        ("Katera zveza se skriva za {L}?", "train"),
        ("Dopolni manjkajočo besedo: {L}", "train"),
        ("Kako se glasi celotna zveza {L}?", "train"),
        ("Zapiši celotno zvezo za {L}.", "train"),
        ("Kaj manjka v zvezi {L}?", "train"),
        ("Dopolni izpuščeno besedo v zvezi {L}.", "train"),
        ("Katera besedna zveza je {L}?", "train"),
        ("Iz {L} sestavi celotno zvezo.", "train"),
        ("Dopolni: {L}", "train"),
        ("Ne spomnim se cele zveze: {L}", "A"),
        ("Zanima me, kaj manjka v {L}.", "A"),
        ("{L} — katera zveza je to?", "A"),
    ],
    "T35": [
        ("Kolokacija katere besede je {L}?", "train"),
        ("Pri kateri besedi je zabeležena kolokacija {L}?", "train"),
        ("Kateri iztočnici pripada kolokacija {L}?", "train"),
        ("Pod katero besedo najdem kolokacijo {L}?", "train"),
        ("Katere besede je {L} kolokacija?", "train"),
        ("Določi iztočnico, pri kateri je navedena kolokacija {L}.", "train"),
        ("H kateri besedi sodi kolokacija {L}?", "train"),
        ("Kolokacija {L} — pri kateri iztočnici je zapisana?", "train"),
        ("Poišči besedo, ki ji pripada kolokacija {L}.", "train"),
        ("Kateri besedi je pripisana zveza {L}?", "train"),
        ("Zanima me, čigava kolokacija je {L}.", "A"),
        ("Ali veš, pri kateri besedi je kolokacija {L}?", "A"),
        ("Kolokacijo {L} — kje jo najdem?", "A"),
    ],

    # ---- H.6: the second held-out relation ------------------------------
    "T36": [
        ("Kako se beseda {L} reče madžarsko?", "train"),
        ("Navedi madžarski prevod besede {L}.", "train"),
        ("Kako je {L} v madžarščini?", "train"),
        ("Kateri madžarski ustreznik ima beseda {L}?", "train"),
        ("Prevedi besedo {L} v madžarščino.", "train"),
        ("Kako bi {L} povedali po madžarsko?", "train"),
        ("Izpiši madžarske prevode besede {L}.", "train"),
        ("Kaj je madžarsko za {L}?", "train"),
        ("Navedi madžarske ustreznike besede {L}.", "train"),
        ("Kako se {L} zapiše v madžarščini?", "train"),
        ("Zanima me madžarski prevod besede {L}.", "A"),
        ("Ali veš, kako je {L} v madžarščini?", "A"),
        ("{L} — kako se to reče madžarsko?", "A"),
    ],
}

# A frame that names a word class -- "samostalnik {L}", "glagol {L}",
# "pridevnika {L}" -- gives a category-mismatch negative away for free: asking
# "kakšnega spola je samostalnik teči" has already asserted the false premise the
# item is testing.  Negatives therefore draw only from frames that name no
# category, which is a filter rather than a hand-kept list so it cannot go stale
# when a frame is added.
_CATEGORY_WORDS = (
    "samostalnik", "glagol", "pridevnik", "prislov", "števnik", "zaimek",
    "medmet", "predlog", "veznik", "členek",
)


def frame_category(frame):
    """The word class a frame asserts, or None."""
    f = frame.casefold()
    for w in _CATEGORY_WORDS:
        if w in f:
            return w
    return None


#: Question-metalanguage slots that name an AXIS of the paradigm rather than
#: describing the entry.  A frame using one is asking about that axis.
AXIS_SLOTS = ("SPOL_NOM", "SPOL_LOC", "DOLOCNOST", "DOLOCNOST_LOC")

_AXIS_OF = {"SPOL_NOM": "gender", "SPOL_LOC": "gender",
            "DOLOCNOST": "definiteness", "DOLOCNOST_LOC": "definiteness"}


def frame_axes(frame):
    """The paradigm axes a frame names, read off the slots it uses.

    Derived from the frame text rather than declared beside it, so a frame and
    its axis list cannot drift apart.
    """
    return frozenset(_AXIS_OF[s] for s in AXIS_SLOTS if "{" + s + "}" in frame)


def _axes_match(frame, axes):
    """A frame may name EXACTLY the axes its item fixes.

    Both directions matter.  A frame naming an axis the item does not fix would
    raise on the missing slot -- and `render_question` swallows that and returns
    None, so the item would vanish silently rather than fail loudly.  A frame
    NOT naming an axis the item does fix is worse: `sklanjaj pridevnik lep` is
    under-specified against a three-gender paradigm, so its answer would not be
    determined by its question, which is what 0.1 clause 3 forbids.
    """
    return frame_axes(frame) == axes


def frames_for_pos(type_key, pos, axes=frozenset()):
    """Frames whose asserted word class matches `pos` (or which assert none),
    and which name exactly the axes this item fixes.

    A frame that names a class is asserting a premise: "Katero obliko ima
    imenovalnik ednine samostalnika izbrisen?" tells the reader that *izbrisen*
    is a noun.  When it is an adjective, the question is false before the model
    has read anything.  This filter is what keeps the premise true.
    """
    return [f for f, _t in TEMPLATES[type_key]
            if frame_category(f) in (None, pos) and _axes_match(f, axes)]


def neutral_frames(type_key, axes=frozenset()):
    """Frames usable for a negative: they name no word class."""
    out = [f for f, _t in TEMPLATES[type_key]
           if frame_category(f) is None and _axes_match(f, axes)]
    # Every type must keep at least one, or its negatives would have no phrasing
    # at all; if this ever fires, the type needs a neutral frame written for it.
    if not out:
        raise ValueError(f"{type_key}: no category-neutral frame available "
                         f"naming exactly {sorted(axes) or 'no axis'}")
    return out


def frames(type_key, tier=None):
    rows = TEMPLATES[type_key]
    if tier is None:
        return [f for f, _t in rows]
    return [f for f, t in rows if t == tier]


def frame_id(type_key, frame):
    for i, (f, _t) in enumerate(TEMPLATES[type_key]):
        if f == frame:
            return f"{type_key}/{i:02d}"
    raise KeyError(frame)


def tier_of(type_key, frame):
    for f, t in TEMPLATES[type_key]:
        if f == frame:
            return t
    raise KeyError(frame)
