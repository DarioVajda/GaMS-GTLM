"""Aggregate the four-arm sweep into the table the project actually wants.

    .venv/bin/python -m train.report_arms train/results/arms_v2/runs.jsonl

One row per arm, `accuracy` averaged over seeds with a spread, and the
majority-class baseline alongside -- because several of these types admit a cheap
constant answer (T9's gender is 42.6 %, T10's aspect 35.7 %) and a score printed
without its baseline is unreadable (QA_TASKS.md C13).

The per-type table is the point of reporting per type at all: a single aggregate
over 19 heterogeneous types hides which of them the graph is actually doing any
work for.
"""
import sys
import json
import argparse
import collections
import statistics

# (input tag, bias arm) -> the name the run matrix uses.
ARM_NAMES = {
    ("v2", "spd+magnetic"): "GTLM (spd + magnetic)",
    ("v2", "no-bias"): "GTLM, no bias",
    ("v2_serialised", "no-bias"): "serialised graph",
    ("v2_noretrieval", "no-bias"): "no retrieval",
}
ORDER = ["GTLM (spd + magnetic)", "GTLM, no bias", "serialised graph",
         "no retrieval"]


def arm_of(rec):
    key = (rec.get("input") or "?", rec.get("arm") or "?")
    return ARM_NAMES.get(key, f"{key[0]} / {key[1]}")


def agg(values):
    values = [v for v in values if v is not None]
    if not values:
        return float("nan"), float("nan"), 0
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.mean(values), sd, len(values)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs_jsonl")
    ap.add_argument("--markdown", default=None, help="also write a .md table here")
    a = ap.parse_args(argv)

    with open(a.runs_jsonl, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    recs = [r for r in recs if r.get("test_accuracy") is not None]
    if not recs:
        print(f"{a.runs_jsonl}: no finished runs with a test accuracy")
        return 1

    by_arm = collections.defaultdict(list)
    for r in recs:
        by_arm[arm_of(r)].append(r)

    baseline = recs[0].get("baselines", {}).get("test", {}).get("majority_overall")
    lines = []
    lines.append(f"| arm | seeds | test accuracy | test F1 | best dev accuracy |")
    lines.append("|---|--:|--:|--:|--:|")
    for name in ORDER + [k for k in sorted(by_arm) if k not in ORDER]:
        rs = by_arm.get(name)
        if not rs:
            continue
        m, sd, n = agg([r["test_accuracy"] for r in rs])
        f1, _, _ = agg([r.get("test_f1") for r in rs])
        dv, dsd, _ = agg([r.get("best_val_accuracy") for r in rs])
        lines.append(f"| **{name}** | {n} | {m:.4f} ± {sd:.4f} | {f1:.4f} | "
                     f"{dv:.4f} ± {dsd:.4f} |")
    if baseline is not None:
        lines.append(f"| majority-class baseline | — | {baseline:.4f} | — | — |")

    types = sorted({t for r in recs for t in (r.get("test_accuracy_per_type") or {})},
                   key=lambda x: (len(x), x))
    arms = [n for n in ORDER + sorted(by_arm) if n in by_arm]
    arms = list(dict.fromkeys(arms))
    per_type = ["", "Per type (test accuracy, mean over seeds):", "",
                "| type | " + " | ".join(arms) + " | majority |",
                "|---" * (len(arms) + 2) + "|"]
    base_pt = recs[0].get("baselines", {}).get("test", {}).get("majority_per_type", {})
    for t in types:
        cells = []
        for name in arms:
            m, _sd, _n = agg([(r.get("test_accuracy_per_type") or {}).get(t)
                              for r in by_arm[name]])
            cells.append(f"{m:.3f}" if m == m else "—")
        b = base_pt.get(t)
        per_type.append(f"| {t} | " + " | ".join(cells) + " | "
                        + (f"{b:.3f}" if b is not None else "—") + " |")

    out = "\n".join(lines + per_type)
    print(out)
    if a.markdown:
        with open(a.markdown, "w", encoding="utf-8") as f:
            f.write(out + "\n")
        print(f"\n[wrote] {a.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
