#!/usr/bin/env python3
"""Look at generated items -- the human half of verification.

`selftest.py` asserts the properties a script can check.  This prints items for a
person to read, because the failures that matter most here are the ones no
assertion catches: a grammatical question with a false premise, a gold answer
that repeats the question, a "corpus sentence" that is a fragment.

    python -m qa.inspect DIR                     # a few of every type
    python -m qa.inspect DIR --type T5 -n 10     # one type, more of it
    python -m qa.inspect DIR --negatives         # only the sentinel items
    python -m qa.inspect DIR --band B6           # only the high-frequency end
    python -m qa.inspect DIR --stats             # no items, just the shape
"""
import os
import json
import random
import argparse
import collections

from . import grade


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load(d, split=None):
    out = []
    for s in (split,) if split else ("train", "dev", "test"):
        p = os.path.join(d, f"{s}.jsonl")
        if os.path.exists(p):
            out += read(p)
    return out


def show(it, width=300):
    tags = [it["type"], it["band"], it["split"], it["tier"], it["template_id"]]
    if it["negative"]:
        tags.append(f"NEG/{it['negative_flavour']}")
    # Through `contract`, never off the row: the type-level fields live in
    # `qa/spec.py` and a row carries only this item's own facts.
    g = grade.contract(it)
    tags.append(g["mode"])
    if g.get("quantity_band"):
        tags.append(f"band={g['quantity_band']}"
                    + (f" N={g['n_asked']}" if g.get("n_asked") else "")
                    + f" |ALL|={g.get('n_all')}")
    print(f"\n[{'  '.join(str(t) for t in tags)}]  {it['lemma']}")
    print(f"  Q: {it['question']}")
    print(f"  A: {it['answer'][:width]}" + ("…" if len(it["answer"]) > width else ""))


def stats(items):
    print(f"items: {len(items):,}")
    for key in ("split", "type", "band", "tier"):
        c = collections.Counter(i[key] for i in items)
        print(f"\nby {key}:")
        for k, v in sorted(c.items()):
            print(f"  {k:6s} {v:6,d}  {100.0 * v / len(items):5.1f}%")
    neg = [i for i in items if i["negative"]]
    print(f"\nnegatives: {len(neg):,} ({100.0 * len(neg) / len(items):.1f}%)  "
          f"{dict(collections.Counter(i['negative_flavour'] for i in neg))}")
    lens = collections.Counter(len(i["gold_items"]) for i in items if not i["negative"])
    print(f"gold item counts: {dict(sorted(lens.items())[:12])} …")
    print(f"distinct templates: {len({i['template_id'] for i in items})}")
    q = [len(i["question"]) for i in items]
    a = [len(i["answer"]) for i in items]
    print(f"question chars: min {min(q)} median {sorted(q)[len(q)//2]} max {max(q)}")
    print(f"answer chars:   min {min(a)} median {sorted(a)[len(a)//2]} max {max(a)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--split")
    ap.add_argument("--type", dest="type_")
    ap.add_argument("--band")
    ap.add_argument("--tier")
    ap.add_argument("--negatives", action="store_true")
    ap.add_argument("--positives", action="store_true")
    ap.add_argument("-n", type=int, default=3, help="items per type")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    items = load(args.dir, args.split)
    if args.type_:
        items = [i for i in items if i["type"] == args.type_]
    if args.band:
        items = [i for i in items if i["band"] == args.band]
    if args.tier:
        items = [i for i in items if i["tier"] == args.tier]
    if args.negatives:
        items = [i for i in items if i["negative"]]
    if args.positives:
        items = [i for i in items if not i["negative"]]
    if not items:
        raise SystemExit("no items match")

    if args.stats:
        stats(items)
        return

    rng = random.Random(args.seed)
    by_type = collections.defaultdict(list)
    for i in items:
        by_type[i["type"]].append(i)
    for t in sorted(by_type, key=lambda s: (len(s), s)):
        rows = by_type[t]
        for it in rng.sample(rows, min(args.n, len(rows))):
            show(it)


if __name__ == "__main__":
    main()
