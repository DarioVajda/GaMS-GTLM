#!/usr/bin/env python3
"""Scan every *-words.nt for lexinfo:partOfSpeech and lexinfo:gender on
lexical units, and report gender coverage per POS.

Answers:
  Q1  does every noun lexical unit carry a gender?
  Q2  does every noun lexical unit carry exactly one?

Also reports, for every other POS, how many units carry gender -- so we know
the full scope of an entry-level gender feature.
"""
import os, re, sys, glob
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

ROOT = "/shared/workspace/povejmo/gams_gtlm/data/kg_raw/OntoLex DSB"
LEXINFO = "http://www.lexinfo.net/ontology/3.0/lexinfo#"

POS_RE = re.compile(
    r"^<https://www\.cjvt\.ddds\.si/(lexical-unit-\d+)> "
    r"<http://www\.lexinfo\.net/ontology/3\.0/lexinfo#partOfSpeech> "
    r"<http://www\.lexinfo\.net/ontology/3\.0/lexinfo#([A-Za-z]+)> \.$")
GEN_RE = re.compile(
    r"^<https://www\.cjvt\.ddds\.si/(lexical-unit-\d+)> "
    r"<http://www\.lexinfo\.net/ontology/3\.0/lexinfo#gender> "
    r"<http://www\.lexinfo\.net/ontology/3\.0/lexinfo#([A-Za-z]+)> \.$")
# gender on a word-form (would make the feature form-level, not entry-level)
GENF_RE = re.compile(
    r"^<https://www\.cjvt\.ddds\.si/(word-form-\d+)> "
    r"<http://www\.lexinfo\.net/ontology/3\.0/lexinfo#gender> ")


def scan(path):
    pos = {}                      # lu -> pos local name
    gen = defaultdict(set)        # lu -> {gender}
    gen_dupe = Counter()          # lu -> raw gender triple count
    formgender_by_pos = Counter() # how many word-forms carry gender
    n_formgender = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if "lexinfo#partOfSpeech>" in line:
                m = POS_RE.match(line)
                if m:
                    pos[m.group(1)] = m.group(2)
                continue
            if "lexinfo#gender>" in line:
                m = GEN_RE.match(line)
                if m:
                    gen[m.group(1)].add(m.group(2))
                    gen_dupe[m.group(1)] += 1
                elif GENF_RE.match(line):
                    n_formgender += 1
    return path, pos, dict(gen), dict(gen_dupe), n_formgender


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "*-words.nt")))
    print(f"[scan] {len(files)} words files", flush=True)
    n_by_pos = Counter()
    withgender_by_pos = Counter()
    multigender_by_pos = Counter()
    gendervals_by_pos = defaultdict(Counter)
    dupe_triples = Counter()
    no_gender_examples = defaultdict(list)
    multi_gender_examples = []
    total_formgender = 0
    lu_with_gender_no_pos = 0
    done = 0
    with ProcessPoolExecutor(max_workers=int(os.environ.get("W", "16"))) as ex:
        for path, pos, gen, dupe, nfg in ex.map(scan, files, chunksize=4):
            total_formgender += nfg
            for lu, p in pos.items():
                n_by_pos[p] += 1
                g = gen.get(lu)
                if g:
                    withgender_by_pos[p] += 1
                    gendervals_by_pos[p][",".join(sorted(g))] += 1
                    if len(g) > 1:
                        multigender_by_pos[p] += 1
                        if len(multi_gender_examples) < 25:
                            multi_gender_examples.append(
                                (lu, p, sorted(g), os.path.basename(path)))
                    if dupe.get(lu, 0) > len(g):
                        dupe_triples[p] += 1
                else:
                    if len(no_gender_examples[p]) < 15:
                        no_gender_examples[p].append(
                            (lu, os.path.basename(path)))
            for lu in gen:
                if lu not in pos:
                    lu_with_gender_no_pos += 1
            done += 1
            if done % 40 == 0:
                print(f"  ..{done}/{len(files)}", flush=True)

    print("\n=== gender coverage on lexical units, by POS ===")
    print(f"{'POS':<14}{'units':>10}{'w/ gender':>12}{'pct':>8}{'>1 gender':>11}")
    for p, n in n_by_pos.most_common():
        w = withgender_by_pos[p]
        print(f"{p:<14}{n:>10,}{w:>12,}{100.0*w/n:>7.2f}%{multigender_by_pos[p]:>11,}")

    print("\n=== gender value distribution, by POS ===")
    for p, c in sorted(gendervals_by_pos.items(),
                       key=lambda kv: -sum(kv[1].values())):
        print(f"  {p}: " + ", ".join(f"{v}={n:,}" for v, n in c.most_common()))

    print(f"\nword-form-level gender triples: {total_formgender:,}")
    print(f"lexical units with gender but NO partOfSpeech: {lu_with_gender_no_pos:,}")
    print(f"units with duplicate gender triples (same value repeated), by POS: "
          f"{dict(dupe_triples)}")

    n = n_by_pos.get("noun", 0)
    w = withgender_by_pos.get("noun", 0)
    print("\n=== ANSWERS ===")
    print(f"Q1 every noun has a gender?      {'YES' if n and w == n else 'NO'} "
          f"({w:,}/{n:,}, missing {n-w:,})")
    print(f"Q2 exactly one gender per noun?  "
          f"{'YES' if multigender_by_pos.get('noun',0)==0 else 'NO'} "
          f"(multi-gender nouns: {multigender_by_pos.get('noun',0):,})")

    if no_gender_examples.get("noun"):
        print("\nnouns WITHOUT gender (first 15):")
        for lu, f in no_gender_examples["noun"]:
            print(f"  {lu}  [{f}]")
    if multi_gender_examples:
        print("\nunits with MORE THAN ONE gender (first 25):")
        for lu, p, g, f in multi_gender_examples:
            print(f"  {lu}  {p}  {g}  [{f}]")


if __name__ == "__main__":
    main()
