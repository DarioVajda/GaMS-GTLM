#!/usr/bin/env python3
"""Acceptance check: a freshly built store against a reference store.

The reference is normally the store the new one replaces, built with the same
tokenizer.  A rebuild is allowed to ADD collocation nodes -- the dedup key is
(member set, phrase), so a member set reified under several dependent senses
mints one node per distinct phrase -- but it may not lose or reshape anything
else.  Seven assertions, in order of how loudly they fail:

  1. THE REAL LAYER IS UNTOUCHED.  node_codes / ntype[:n_real] / kind[:n_real] /
     mwe_set[:n_real] byte-identical, and n_real unchanged.  Nothing about the
     IRI-backed graph may change; only minted nodes are added.

  2. THE SYNONYM AND ANTONYM LAYERS ARE UNTOUCHED.  They are minted BEFORE
     collocations, so their node ids are unaffected by extra collocation nodes:
     same count, same ids, same text, byte for byte.  If this fails, a change
     leaked outside the collocation path.

  3. THE COLLOCATION LAYER ONLY GREW.  For every member set, every reference
     collocation text must still be present.  This is the check that matters: a
     change is worthless if it adds phrases while dropping others, and a pure
     count comparison would not notice.

     Compared under the FOLD (case, whitespace, one trailing period), because
     folding case-only duplicates is intended rather than a loss -- a store
     holding both `Fotografiranje otrok` and `fotografiranje otrok` on one member
     set would make a gold list fail its own dedup invariant.  The count of
     phrases that survive only up to the fold is reported, so the fold cannot
     quietly hide a real loss.

  4. NO PAIR CARRIES THE SAME PHRASE TWICE.  (member set, text) must be unique
     across the whole collocation layer -- that IS the dedup key, so a violation
     means the key did not take effect.

  5. NON-COLLOCATION ADJACENCY IS UNCHANGED.  For every real node, the multiset
     of neighbours that are real nodes, synonyms or antonyms must match the
     reference.  Collocation neighbours are compared by text in check 3, not by
     id -- their ids necessarily shift.

  6. VERBALISATION DID NOT REGRESS.  The share of collocation nodes carrying a
     curated phrase rather than a `lemma + lemma` fallback must not fall.

  7. REFERENCE REPRODUCTION DID NOT REGRESS.  The share of the 1,307 collocation
     phrases in data/datasets/reference/Lexical-QA-SLO(in).csv present verbatim
     in the store.  Reported for both, and required only not to fall.

Usage:  check_text.py NEW_STORE REFERENCE_STORE
"""
import os, sys, csv, json, re, hashlib, random, collections, unicodedata
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import graph_store

K_ANCHOR, K_FORM, K_SENSE, K_EXAMPLE, K_TRANS = 0, 1, 2, 3, 4
K_COLLOC, K_SYN, K_ANT, K_OTHER = 5, 6, 7, 8
TAG_COLLOC = "kolokacija: "
DATA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFCSV = os.path.join(DATA, "datasets", "reference", "Lexical-QA-SLO(in).csv")

# The pair fallback renders as `kolokacija: A + B`; a curated phrase never
# contains " + " (verified over the whole collocation layer: 0 of 2,981,731).
PAIR_MARK = " + "

# What `manifest.meta.text_convention` must say for a store to be current: both
# builder switches on.  The builder derives the same string from the switches.
CURRENT_CONVENTION = "collocation-phrases+entry-gender"


def md5(path, blocks=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while (b := f.read(blocks)):
            h.update(b)
    return h.hexdigest()


def norm(s):
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip().rstrip(".").casefold()


def check_real_layer(new, old, Gn, Go):
    print("--- 1. the real (IRI-backed) layer is untouched ---")
    ok = Gn["n_real"] == Go["n_real"]
    print(f"    n_real {Go['n_real']:,} -> {Gn['n_real']:,}  identical={ok}")
    nr = Gn["n_real"]
    a = md5(os.path.join(new, "node_codes.npy"))
    b = md5(os.path.join(old, "node_codes.npy"))
    ok &= a == b
    print(f"    node_codes.npy   identical={a == b}  {a}")
    for name, arr_new, arr_old in (
            ("ntype", Gn["ntype"], Go["ntype"]),
            ("kind", Gn["kind"], Go["kind"]),
            ("mwe_set", Gn["mwe_set"], Go["mwe_set"])):
        same = np.array_equal(np.asarray(arr_new[:nr]), np.asarray(arr_old[:nr]))
        ok &= same
        print(f"    {name+'[:n_real]':16s} identical={same}")
    return ok


def check_syn_ant(Gn, Go):
    """Synonyms and antonyms are minted before collocations, so their ids hold."""
    print("\n--- 2. the synonym / antonym layers are untouched ---")
    ok = True
    for k, nm in ((K_SYN, "sopomenka"), (K_ANT, "protipomenka")):
        idn = np.flatnonzero(np.asarray(Gn["kind"]) == k)
        ido = np.flatnonzero(np.asarray(Go["kind"]) == k)
        same_ids = np.array_equal(idn, ido)
        tn, to = Gn["text"].__getitem__, Go["text"].__getitem__
        diff = 0
        if same_ids:
            for i in idn:
                if tn(int(i)) != to(int(i)):
                    diff += 1
        ok &= same_ids and diff == 0
        print(f"    {nm:14s} {len(ido):,} -> {len(idn):,} nodes  "
              f"same ids={same_ids}  text differs on {diff:,}")
    return ok


def colloc_by_pair(G):
    """{(sense_code_lo, sense_code_hi): [text, ...]} over the collocation layer."""
    kind = np.asarray(G["kind"]); indptr = G["indptr"]; indices = G["indices"]
    codes = G["node_codes"]; nr = G["n_real"]; text = G["text"].__getitem__
    out = collections.defaultdict(list)
    ids = np.flatnonzero(kind == K_COLLOC)
    for i in ids:
        i = int(i)
        nb = indices[indptr[i]:indptr[i + 1]]
        members = tuple(sorted(int(codes[v]) for v in nb if v < nr))
        out[members].append(text(i))
    return out, len(ids)


def check_colloc_growth(Gn, Go):
    print("\n--- 3/4. the collocation layer only grew, and holds no duplicate ---")
    new_by_pair, n_new = colloc_by_pair(Gn)
    old_by_pair, n_old = colloc_by_pair(Go)
    print(f"    collocation nodes {n_old:,} -> {n_new:,} "
          f"({100.0 * (n_new - n_old) / max(n_old, 1):+.1f}%)")
    print(f"    distinct member sets {len(old_by_pair):,} -> {len(new_by_pair):,}")

    lost_pairs = 0
    lost_texts = 0
    folded = 0
    shown = 0
    for pair, texts in old_by_pair.items():
        have = new_by_pair.get(pair)
        if have is None:
            lost_pairs += 1
            lost_texts += len(texts)
            continue
        exact = set(have)
        folded_set = {norm(t) for t in have}
        for t in texts:
            if t in exact:
                continue
            if norm(t) in folded_set:
                folded += 1                      # a case-only variant, folded on purpose
                continue
            lost_texts += 1
            if shown < 5:
                print(f"    !! phrase lost for {pair}: {t!r} (new has {have!r})")
                shown += 1
    print(f"    member sets in the reference but not the new store: {lost_pairs:,} (required 0)")
    print(f"    reference phrases missing from the new store:      {lost_texts:,} (required 0)")
    print(f"    reference phrases kept only up to the fold:        {folded:,} "
          f"(intended: case-only duplicates)")

    dup = 0
    for pair, texts in new_by_pair.items():
        if len(texts) != len(set(texts)):
            dup += 1
            if dup <= 3:
                print(f"    !! duplicate text on one member set {pair}: {texts!r}")
    print(f"    member sets carrying a repeated phrase: {dup:,} (required 0)")

    per_pair = np.array([len(v) for v in new_by_pair.values()])
    print(f"    phrases per member set: mean {per_pair.mean():.2f}  "
          f"max {per_pair.max()}  >1 on {int((per_pair > 1).sum()):,} sets "
          f"({100.0 * (per_pair > 1).mean():.1f}%)")
    return (lost_pairs == 0 and lost_texts == 0 and dup == 0), new_by_pair, n_new, n_old


def check_adjacency(Gn, Go, n_sample=300_000, seed=0):
    """Real nodes keep exactly their non-collocation neighbours."""
    print("\n--- 5. non-collocation adjacency is unchanged ---")
    kn = np.asarray(Gn["kind"]); ko = np.asarray(Go["kind"])
    ipn, iin = Gn["indptr"], Gn["indices"]
    ipo, iio = Go["indptr"], Go["indices"]
    nr = Gn["n_real"]
    rng = random.Random(seed)
    rows = rng.sample(range(nr), min(n_sample, nr))
    bad = 0
    for i in rows:
        a = np.asarray(iin[ipn[i]:ipn[i + 1]])
        b = np.asarray(iio[ipo[i]:ipo[i + 1]])
        a = np.sort(a[kn[a] != K_COLLOC])
        b = np.sort(b[ko[b] != K_COLLOC])
        if not np.array_equal(a, b):
            bad += 1
            if bad <= 3:
                print(f"    !! node {i}: non-collocation neighbours differ")
    print(f"    real nodes sampled: {len(rows):,}  differing: {bad:,} (required 0)")
    return bad == 0


def check_verbalisation(new_by_pair, Go, n_new, n_old):
    print("\n--- 6. verbalisation rate did not regress ---")
    new_texts = [t for v in new_by_pair.values() for t in v]
    new_pair_form = sum(1 for t in new_texts if PAIR_MARK in t)
    to = Go["text"].__getitem__
    old_ids = np.flatnonzero(np.asarray(Go["kind"]) == K_COLLOC)
    old_pair_form = sum(1 for i in old_ids if PAIR_MARK in to(int(i)))
    rn = 100.0 * (n_new - new_pair_form) / max(n_new, 1)
    ro = 100.0 * (n_old - old_pair_form) / max(n_old, 1)
    print(f"    verbalised: reference {ro:.3f}% ({n_old - old_pair_form:,})  "
          f"new {rn:.3f}% ({n_new - new_pair_form:,})")
    return rn >= ro - 1e-9


def check_reference(new_by_pair, Go):
    """How much of the reference file's collocation column each store reproduces."""
    print("\n--- 7. reference-file reproduction ---")
    if not os.path.exists(REFCSV):
        print(f"    (skipped: {REFCSV} not found)")
        return True
    new_phrases = {norm(t[len(TAG_COLLOC):]) for v in new_by_pair.values() for t in v}
    to = Go["text"].__getitem__
    old_phrases = {norm(to(int(i))[len(TAG_COLLOC):])
                   for i in np.flatnonzero(np.asarray(Go["kind"]) == K_COLLOC)}
    item = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*(.+?)\s*$")
    tot = hit_new = hit_old = 0
    with open(REFCSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("vrsta") != "kolokacije":
                continue
            for line in (r.get("odgovor") or "").splitlines():
                m = item.match(line)
                if not m:
                    continue
                it = norm(m.group(1).replace("**", ""))
                tot += 1
                hit_old += it in old_phrases
                hit_new += it in new_phrases
    if not tot:
        print("    (no kolokacije rows found)")
        return True
    print(f"    reference phrases: {tot}   "
          f"reference {hit_old} ({100.0 * hit_old / tot:.1f}%)   "
          f"new {hit_new} ({100.0 * hit_new / tot:.1f}%)")
    print(f"    distinct phrases in store: {len(old_phrases):,} -> {len(new_phrases):,}")
    return hit_new >= hit_old


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    new, old = sys.argv[1], sys.argv[2]
    print(f"new: {new}\nold: {old}\n")

    man = json.load(open(os.path.join(new, "manifest.json")))
    meta = man["meta"]
    print(f"manifest text_convention = {meta.get('text_convention')!r}  "
          f"tokenizer = {meta.get('tokenizer')!r}")
    cst = (man.get("stats") or {}).get("collocations", {})
    print(f"manifest dedup_key       = {cst.get('dedup_key')!r}  "
          f"member_sets={cst.get('member_sets')}  "
          f"pairings={cst.get('distinct_pairings')}")
    ok = meta.get("text_convention") == CURRENT_CONVENTION
    if not ok:
        print(f"    !! new store does not declare text_convention "
              f"{CURRENT_CONVENTION!r}")
    if cst.get("dedup_key") != "member_set+phrase":
        print("    !! new store does not declare the (member set, phrase) dedup key")
        ok = False

    Gn = graph_store.load_graph(new, verbose=False)
    Go = graph_store.load_graph(old, verbose=False)

    ok &= check_real_layer(new, old, Gn, Go)
    ok &= check_syn_ant(Gn, Go)
    ok_growth, new_by_pair, n_new, n_old = check_colloc_growth(Gn, Go)
    ok &= ok_growth
    ok &= check_adjacency(Gn, Go)
    ok &= check_verbalisation(new_by_pair, Go, n_new, n_old)
    ok &= check_reference(new_by_pair, Go)

    print("\nACCEPTANCE:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
