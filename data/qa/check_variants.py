#!/usr/bin/env python3
"""Acceptance check: the variants are the same corpus, differently shown.

    python -m qa.check_variants datasets/balls \
        datasets/balls_noretrieval datasets/balls_serialised

Every variant must carry the same ids in the same splits with the same answers as
the reference ball directory.  If it does not, the arms are no longer comparable
and any gap between them is partly a difference of corpus -- which is the one
thing the baseline design exists to rule out.

The serialised variant additionally has to *contain* what the GTLM arm sees: the
check asserts that every node text of the reference ball appears verbatim in the
serialised prompt, so "identical information, different encoding" is a verified
claim and not a description of intent.
"""
import os
import sys
import json
import argparse
import collections


def read(d, split):
    path = os.path.join(d, f"{split}.jsonl")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run(reference, variants):
    """Check every variant against the reference balls.  0 if all match."""
    args = argparse.Namespace(reference=reference, variants=list(variants))

    bad = 0
    truncated = collections.Counter()
    for split in ("train", "dev", "test"):
        ref = read(args.reference, split)
        if ref is None:
            continue
        ref_by_id = {r["id"]: r for r in ref}
        for d in args.variants:
            got = read(d, split)
            name = os.path.basename(d.rstrip("/"))
            if got is None:
                print(f"FAIL {name}/{split}: missing")
                bad += 1
                continue
            problems = []
            if [r["id"] for r in got] != [r["id"] for r in ref]:
                problems.append("id list differs (order or membership)")
            for r in got:
                s = ref_by_id.get(r["id"])
                if s is None:
                    problems.append(f"{r['id']}: not in the reference")
                    break
                if r["split"] != s["split"] or r["split"] != split:
                    problems.append(f"{r['id']}: split differs")
                    break
                if r["answer"] != s["answer"] or r["type"] != s["type"]:
                    problems.append(f"{r['id']}: answer/type differs")
                    break
                if r.get("variant") == "serialised":
                    # Every node the serialisation claims to keep must be there
                    # verbatim -- that is the "identical information" half of the
                    # comparison.  A row above the prompt-token budget keeps a
                    # prefix instead, and says so in `serialised_truncated`; the
                    # count is reported below rather than passed over.
                    keep = r.get("serialised_nodes", len(s["nodes"]))
                    missing = [t for t in s["nodes"][:keep] if t not in r["question"]]
                    if missing:
                        problems.append(
                            f"{r['id']}: {len(missing)} node text(s) absent from "
                            f"the serialised prompt, e.g. {missing[0][:60]!r}")
                        break
                    if bool(r.get("serialised_truncated")) != (keep < len(s["nodes"])):
                        problems.append(f"{r['id']}: truncation flag disagrees "
                                        f"with the kept node count")
                        break
                    truncated[name] += bool(r.get("serialised_truncated"))
                if r.get("variant") == "noretrieval" and r["nodes"]:
                    problems.append(f"{r['id']}: no-retrieval ball is not empty")
                    break
            if problems:
                bad += 1
                print(f"FAIL {name}/{split}: {problems[0]}")
            else:
                print(f"  ok {name}/{split}: {len(got):,} items match the reference")
    for name, n in truncated.items():
        print(f"\n{name}: {n:,} row(s) keep only a prefix of the node list "
              f"(above the serialised prompt's token budget)")
    print("\nALL VARIANTS MATCH" if not bad else f"\n{bad} CHECK(S) FAILED")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference")
    ap.add_argument("variants", nargs="+")
    args = ap.parse_args()
    return run(reference=args.reference, variants=args.variants)


if __name__ == "__main__":
    sys.exit(main())
