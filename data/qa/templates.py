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

Volume: ~13 frames per type, 3 of them withheld.  No paraphrase expansion in v1 --
at D16 scale that is ~70 items per frame with the slot values varying every time,
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
        # C24: arity 9 — every person in all three numbers.  The frame used to
        # name a single cell ("v 2. osebi dvojine") and got the whole table back.
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


def frames_for_pos(type_key, pos):
    """Frames whose asserted word class matches `pos` (or which assert none).

    A frame that names a class is asserting a premise: "Katero obliko ima
    imenovalnik ednine samostalnika izbrisen?" tells the reader that *izbrisen*
    is a noun.  When it is an adjective, the question is false before the model
    has read anything.  This filter is what keeps the premise true.
    """
    return [f for f, _t in TEMPLATES[type_key]
            if frame_category(f) in (None, pos)]


def neutral_frames(type_key):
    """Frames usable for a negative: they name no word class."""
    out = [f for f, _t in TEMPLATES[type_key] if frame_category(f) is None]
    # Every type must keep at least one, or its negatives would have no phrasing
    # at all; if this ever fires, the type needs a neutral frame written for it.
    if not out:
        raise ValueError(f"{type_key}: no category-neutral frame available")
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
