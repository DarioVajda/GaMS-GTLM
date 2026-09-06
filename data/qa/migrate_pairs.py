#!/usr/bin/env python3
"""Convert a pre-reformat corpus to the labelled-pair rule (QA_TASKS.md 0.1).

    python -m qa.migrate_pairs --out datasets/work/pairs \
        datasets/generated datasets/balls

Why this exists rather than a rebuild.  The reformat changes the answer of all
19 existing types, so the only honest way to land it is to show that every one of
them survives the change -- 12,490 items, not a fixture of nine.  Regenerating
from the store would test the new generators; converting the corpus tests the
CONTRACT, on exactly the data every published number was measured on, and it does
it in seconds instead of hours.

It is the same code path either way: the conversion calls `qa/pairs.py`, which is
what `qa/build_dataset.py` now calls too, on the same positional list.  So a
converted item and a regenerated one are byte-identical by construction, and the
converted corpus is the fixture that says so when the generators are rewritten.

Nothing is overwritten.  The output is a new directory; the input corpus stays
exactly as the results that cite it left it.

**Ball rows carry no `slots`.**  T3 needs to know which case its question asked
for and T20/T21 which form they are analysing, and only the generated record
holds that.  So a ball directory is converted against the generated one, joined
on item id -- which is also a check: a ball row whose id is not in the generated
set is a corpus that was assembled from two different runs.
"""
import os
import re
import json
import argparse
import collections

from qa import gen, grade, pairs, sl, spec

SPLITS = ("train", "dev", "test")

#: `pomen 3: potrdilo, ki ...` in a ball node.
SENSE_NODE = re.compile(r"^pomen(?: (\d+))?: (.*)$", re.S)


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def already_pairs(row):
    """True if this row's gold is already `[[oznaka, vrednost], ...]`."""
    g = row.get("gold_items") or []
    return bool(g) and isinstance(g[0], (list, tuple))


def sense_ordinals(directory):
    """{item id -> {normalised definition -> the ball's own ordinal}}.

    T12's old gold is a bare definition; its label under 0.1 is `pomen N`, and N
    has to be the number the GRAPH gives that sense, not the answer's position in
    a filtered list (see `qa/gen.py:gen_T12`).  The generator reads it off the
    store; a corpus already on disk has thrown it away, so it is recovered from
    the ball the item ships -- which is the same number by construction, and is
    also the only number the check would accept.
    """
    out = {}
    for s in SPLITS:
        p = os.path.join(directory, f"{s}.jsonl")
        if not os.path.exists(p):
            continue
        for r in read(p):
            if r.get("type") != "T12" or r.get("negative"):
                continue
            m = {}
            for node in r.get("nodes") or []:
                hit = SENSE_NODE.match(node)
                if hit:
                    m[sl.norm(hit.group(2))] = hit.group(1)
            out[r["id"]] = m
    return out


def relabel_senses(row, ordinals):
    """Glue T12's graph ordinal onto each definition, as the generator now does."""
    got = ordinals.get(row["id"])
    if got is None:
        raise KeyError(
            f"{row['id']} is a T12 positive but no ball was given for it. Pass "
            f"--ordinals-from with the balls directory: the sense ordinal is "
            f"part of the label under 0.1 and cannot be reconstructed from the "
            f"answer alone.")
    out = []
    for body in row["gold_items"]:
        ordinal = got.get(sl.norm(body))
        if ordinal is None:
            raise KeyError(
                f"{row['id']} answers with a definition its own ball does not "
                f"hold: {body[:60]!r}. That is a containment failure (C18 d), "
                f"not a labelling one -- do not paper over it here.")
        out.append(f"{gen.sense_label(ordinal)}: {body}")
    row["gold_items"] = out


def convert(row, slots, stats, ordinals=None):
    """Rewrite one row's answer side in place.  Returns the row."""
    if already_pairs(row):
        stats["already"] += 1
        return row

    grading = row.get("grading")
    if isinstance(grading, dict):
        # 0.1 retired the per-type shape fields; `mode` is read from the spec.
        for key in grade.TYPE_LEVEL + grade.RETIRED_ROW_FIELDS:
            if grading.pop(key, None) is not None:
                stats["stale_fields_dropped"] += 1
        allow = grading.get("all_items")
        if allow is not None and not (allow and isinstance(allow[0], (list, tuple))):
            kind = spec.MEMBER_KIND[row["type"]]
            grading["all_items"] = [[kind, v] for v in allow]
            stats["allow_lists_paired"] += 1

    if row.get("negative"):
        # The sentinel is not a labelled pair and never was (0.2).
        stats["negatives_unchanged"] += 1
        return row

    if row["type"] == "T12":
        relabel_senses(row, ordinals or {})
    row["answer"], row["gold_items"] = pairs.answer_of(
        row["type"], slots, row["gold_items"])
    stats["converted"] += 1
    stats[f"pairs/{row['type']}"] += len(row["gold_items"])
    return row


def slots_index(directory):
    """{item id -> slots} from a generated-shaped corpus."""
    out = {}
    for s in SPLITS:
        p = os.path.join(directory, f"{s}.jsonl")
        if os.path.exists(p):
            for r in read(p):
                out[r["id"]] = r.get("slots") or {}
    return out


def convert_dir(src, dst, index, ordinals, stats):
    n = 0
    for s in SPLITS:
        p = os.path.join(src, f"{s}.jsonl")
        if not os.path.exists(p):
            continue
        rows = read(p)
        for r in rows:
            slots = r.get("slots")
            if slots is None:
                if r["id"] not in index:
                    raise KeyError(
                        f"{r['id']} in {p} has no slots and is not in the "
                        f"generated corpus given by --slots-from. A ball row is "
                        f"converted against the run that generated it; joining "
                        f"two runs would relabel gold with another item's "
                        f"question.")
                slots = index[r["id"]]
            convert(r, slots, stats, ordinals)
        write(os.path.join(dst, f"{s}.jsonl"), rows)
        n += len(rows)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+",
                    help="corpus directories holding {train,dev,test}.jsonl")
    ap.add_argument("--out", required=True,
                    help="output root; each input becomes a subdirectory of it")
    ap.add_argument("--slots-from",
                    help="the generated corpus to take slots from for rows that "
                         "carry none (default: the first input directory)")
    ap.add_argument("--ordinals-from",
                    help="the balls corpus to take T12's sense ordinals from "
                         "(default: the last input directory)")
    args = ap.parse_args(argv)

    index = slots_index(args.slots_from or args.dirs[0])
    print(f"slots for {len(index):,} items "
          f"from {args.slots_from or args.dirs[0]}")
    ordinals = sense_ordinals(args.ordinals_from or args.dirs[-1])
    print(f"sense ordinals for {len(ordinals):,} T12 items "
          f"from {args.ordinals_from or args.dirs[-1]}")

    stats = collections.Counter()
    for src in args.dirs:
        dst = os.path.join(args.out, os.path.basename(os.path.normpath(src)))
        n = convert_dir(src, dst, index, ordinals, stats)
        print(f"  {src:34} -> {dst:40} {n:,} rows")

    print(f"\nconverted:            {stats['converted']:,} positives")
    print(f"negatives unchanged:  {stats['negatives_unchanged']:,}")
    print(f"already pairs:        {stats['already']:,}")
    print(f"allow-lists paired:   {stats['allow_lists_paired']:,}")
    print(f"stale row fields:     {stats['stale_fields_dropped']:,} dropped")
    per = {k[6:]: v for k, v in stats.items() if k.startswith("pairs/")}
    print(f"pairs emitted:        {sum(per.values()):,} "
          f"over {len(per)} types")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
