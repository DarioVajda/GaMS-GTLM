#!/usr/bin/env python3
"""Which word-forms carry two DIFFERENT words as their writtenRep?

Found 2026-08-22 while checking the top-proxy anchors of the v7 store:

    iztočnica: prikazati (zaimek, naslonska oblika)     <- 423,458 MWE memberships

That is the reflexive clitic `se`, the second-largest hub in the whole graph,
wearing the wrong lemma.  The cause is in the source, not the builder:

    word-form-2773103  ontolex:writtenRep  \"\"\"se\"\"\"@sl
    word-form-2773103  ontolex:writtenRep  \"\"\"prikazati\"\"\"@sl

The builder already handles a form with several writtenRep values -- it prefers
the least-capitalised, then the lexicographically smallest, which is right for
the case that motivated it (`BOJ` / `Boj` / `boj`).  Both of these are lowercase,
so the tie-break falls to the alphabet and `prikazati` wins.

This scan separates the two populations, because only one of them is a defect:

  CASE VARIANTS   several spellings of ONE word (`BOJ`/`Boj`/`boj`).  The
                  builder's rule is correct here and nothing needs excluding.
  DISTINCT WORDS  two different lemmas on one form node.  There is no signal in
                  the RDF that says which is intended, so any choice the builder
                  makes is a coin flip -- and on a high-degree entry a wrong
                  lemma becomes wrong gold in every question about it.

Writes an exclusion list of the affected lexical-units, for the QA seed pool.

    python scan_multi_writtenrep.py [--kg-dir DIR] [--out FILE]
"""
import os
import re
import sys
import json
import argparse
import collections
import subprocess

DATA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KG = os.path.join(DATA, "kg_raw", "OntoLex DSB")

# The key is (PREFIX, id), not the id alone: `word-form-N` and
# `form-lexical-unit-N` are different nodes that share the number space, so
# keying on the number alone would merge two unrelated forms.  (Measured: on this
# dump it changes almost nothing -- the two id ranges barely overlap -- but the
# correctness of the result should not rest on that.)
WR = re.compile(r'^<https://www\.cjvt\.ddds\.si/(word-form|form-lexical-unit)-(\d+)> '
                r'<http://www\.w3\.org/ns/lemon/ontolex#writtenRep> '
                r'"""(.*)"""@sl \.$')
CANON = re.compile(r'^<https://www\.cjvt\.ddds\.si/lexical-unit-(\d+)> '
                   r'<http://www\.w3\.org/ns/lemon/ontolex#canonicalForm> '
                   r'<https://www\.cjvt\.ddds\.si/(word-form|form-lexical-unit)-(\d+)> \.$')


def scan(kg_dir, pattern, workers=32, tmp=None):
    """Parallel grep over the *-words.nt files, yielding one line per match.

    **Each worker writes to its own file** -- QA_DATASET_DESIGN.md 8's prescribed
    pattern.  A shared stdout interleaves and corrupts lines, and here the damage
    is not symmetric: a torn line can splice one form's id onto another form's
    writtenRep, inventing exactly the "two different words on one form" conflict
    this scan looks for, while other torn lines are simply lost.

    Measured, same corpus, same regex: the shared pipe saw 8,761,553 forms with
    246,116 lines unparsed and reported 958-1,155 affected entries, varying run
    to run.  Per-worker files see **8,960,228** forms with 25,591 unparsed and
    report a stable **958**.  The first figure matches 3.2's independently
    measured 8,960,506 and the canonicalForm count comes out at exactly the
    census 400,180, which is what says the scan is now complete.
    """
    tmp = tmp or os.path.join(os.environ.get("TMPDIR", "/tmp"),
                              f"scan_{os.getpid()}")
    os.makedirs(tmp, exist_ok=True)
    cmd = (f'cd "{kg_dir}" && ls *-words.nt | '
           f'xargs -P {workers} -n 1 -I{{}} '
           f'sh -c \'grep -h -E {pattern!r} "$1" > "{tmp}/$1.out"\' _ {{}}')
    subprocess.run(["bash", "-c", cmd], check=True)
    try:
        for name in sorted(os.listdir(tmp)):
            with open(os.path.join(tmp, name), "rb") as f:
                for raw in f:
                    yield raw.decode("utf-8", "replace").rstrip("\n")
    finally:
        for name in os.listdir(tmp):
            os.unlink(os.path.join(tmp, name))
        os.rmdir(tmp)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg-dir", default=KG)
    ap.add_argument("--out", default=os.path.join(
        DATA, "analysis", "results", "ambiguous_lemma_units.json"))
    args = ap.parse_args()

    print("[scan] writtenRep ...", flush=True)
    forms = collections.defaultdict(set)
    bad = 0
    for line in scan(args.kg_dir, "ontolex#writtenRep"):
        m = WR.match(line)
        if not m:
            bad += 1
            continue
        forms[(m.group(1), int(m.group(2)))].add(m.group(3))
    print(f"    forms with an @sl writtenRep: {len(forms):,}  unparsed: {bad:,}",
          flush=True)

    multi = {f: v for f, v in forms.items() if len(v) > 1}
    case_only = {f: v for f, v in multi.items()
                 if len({s.casefold() for s in v}) == 1}
    distinct = {f: v for f, v in multi.items()
                if len({s.casefold() for s in v}) > 1}
    print(f"    forms with >1 writtenRep:       {len(multi):,}")
    print(f"      case variants of one word:    {len(case_only):,}")
    print(f"      DISTINCT WORDS:               {len(distinct):,}")

    print("[scan] canonicalForm ...", flush=True)
    canon = {}
    for line in scan(args.kg_dir, "ontolex#canonicalForm"):
        m = CANON.match(line)
        if m:
            canon[int(m.group(1))] = (m.group(2), int(m.group(3)))
    print(f"    lexical-units with a canonical form: {len(canon):,}", flush=True)

    affected = {lu: sorted(distinct[f]) for lu, f in canon.items() if f in distinct}
    core = {lu: v for lu, v in affected.items() if lu < 1_000_000}
    print(f"\n    entries whose LEMMA is ambiguous at the source: {len(affected):,}")
    print(f"      of which core (id < 1M):                     {len(core):,}")
    for lu, v in sorted(core.items())[:15]:
        print(f"        lexical-unit-{lu}: {v}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"units": {str(k): v for k, v in affected.items()},
                   "n_forms_multi": len(multi),
                   "n_forms_case_only": len(case_only),
                   "n_forms_distinct": len(distinct),
                   "n_units": len(affected), "n_core_units": len(core)},
                  f, ensure_ascii=False, indent=1)
    print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
