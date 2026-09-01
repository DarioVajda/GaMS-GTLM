"""Aggregate the arms sweep into the tables the study is actually about.

    .venv/bin/python -m train.analysis.report_arms train/results/arms_v3/runs.jsonl \
        --markdown train/results/arms_v3/report_arms.md

Four sections, in the order a reader needs them.

**Per arm** — `accuracy` averaged over seeds with a spread, and the
majority-class baseline alongside, because several of these types admit a cheap
constant answer (T9's gender is 42.6 %, T10's aspect 35.7 %) and a score printed
without its baseline is unreadable (QA_TASKS.md C13).

**Per type** — the point of reporting per type at all: a single aggregate over 19
heterogeneous types hides which of them the graph is doing any work for.

**Contrasts** — each of the comparisons the six configurations were chosen to
make, PAIRED BY SEED rather than as a difference of means.  The two arms in a
contrast share their seed, their data order and their schedule, so the paired
difference removes seed variance that a difference of means leaves in.

**Convergence** — the pre-registered check.  `arms_v2` was thrown away because
dev accuracy was still rising in every arm at the stop and the best checkpoint
was the LAST one in all twelve runs, which measures how fast an arm learns
rather than where it ends up.  The rule, agreed before the numbers existed: **if
the best checkpoint is the last one in more than a third of the runs, the study
is still measuring learning speed and needs more epochs before any claim is
final.**

A run is identified by `(input, arm, stack)`.  The stack is the third coordinate
because two pairs of arms differ in nothing else: serialised on GTLM vs plain,
and no-retrieval on GTLM vs plain.
"""
import os
import sys
import json
import argparse
import collections
import statistics

# Where the per-item dumps live, for the tier fallback below.
PRED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "predictions")
SENTINEL = "ni podatka v bazi"
# Reported in this order; anything else follows, sorted.
TIER_ORDER = ["core", "A", "B", "C"]
TIER_WHAT = {
    "core": "seen type, seen phrasing",
    "A": "seen type, **unseen phrasing**",
    "B": "unseen type over seen relations",
    "C": "**unseen relation** (antonyms)",
}

# (input tag, bias arm, stack) -> the name the run matrix uses.
#
# The input tag is the ball directory's basename, so it changed when the dataset
# directories lost their version suffixes on 2026-08-26 (`v2_clean` -> `balls`).
# Both spellings are listed: a record in `train/results/` was written by the run
# that produced it and still says `v2_clean`, and rewriting a results file to
# agree with a later rename would be editing the evidence.  New runs write the
# `balls*` tags; both resolve to the same arm.
ARM_NAMES = {
    # the six configurations, as named since 2026-08-26.
    ("balls", "spd+magnetic", "gtlm"): "GTLM (spd + magnetic)",
    ("balls", "no-bias", "gtlm"): "GTLM, no bias",
    ("balls_serialised", "no-bias", "gtlm"): "serialised — GTLM stack",
    ("balls_serialised", "no-bias", "plain"): "serialised — plain stack",
    ("balls_noretrieval", "no-bias", "plain"): "no retrieval — plain stack",
    ("balls_noretrieval", "no-bias", "gtlm"): "no retrieval — GTLM stack",
    # arms_v3 as recorded at the time — the same six arms under the old names.
    ("v2_clean", "spd+magnetic", "gtlm"): "GTLM (spd + magnetic)",
    ("v2_clean", "no-bias", "gtlm"): "GTLM, no bias",
    ("v2_clean_serialised", "no-bias", "gtlm"): "serialised — GTLM stack",
    ("v2_clean_serialised", "no-bias", "plain"): "serialised — plain stack",
    ("v2_clean_noretrieval", "no-bias", "plain"): "no retrieval — plain stack",
    ("v2_clean_noretrieval", "no-bias", "gtlm"): "no retrieval — GTLM stack",
    # arms_v2 — kept so the older sweep still reports.  Every v2 run was on the
    # GTLM stack (the plain path did not exist yet), which is why records
    # without a `stack` field default to "gtlm".
    ("v2", "spd+magnetic", "gtlm"): "GTLM (spd + magnetic)",
    ("v2", "no-bias", "gtlm"): "GTLM, no bias",
    ("v2_serialised", "no-bias", "gtlm"): "serialised — GTLM stack",
    ("v2_noretrieval", "no-bias", "gtlm"): "no retrieval — GTLM stack",
}
ORDER = ["GTLM (spd + magnetic)", "GTLM, no bias",
         "serialised — GTLM stack", "serialised — plain stack",
         "no retrieval — plain stack", "no retrieval — GTLM stack"]

# The comparisons the design exists to make (TODO.md §2).  `(name, a, b, note)`
# reports `a - b`.
CONTRASTS = [
    ("Retrieval", "serialised — plain stack", "no retrieval — plain stack",
     "is the subgraph worth anything at all (same stack)"),
    ("Structural bias", "GTLM (spd + magnetic)", "GTLM, no bias",
     "does the bias make the graph usable (same stack, same input)"),
    ("Graph encoding", "GTLM (spd + magnetic)", "serialised — GTLM stack",
     "THE HEADLINE: encoding, with the stack held fixed"),
    ("Stack cost", "serialised — GTLM stack", "serialised — plain stack",
     "what SDPA + sliding_window=512 is worth on a long input"),
    ("Stack control", "no retrieval — GTLM stack", "no retrieval — plain stack",
     "the same stack difference where it CANNOT act — expected ~0"),
]

# T9's pre-registered convergence rule.
CONVERGENCE_LIMIT = 1 / 3


def _wilson(k, n, z=1.96):
    """Wilson score interval for `k` successes in `n` -- usable at small n,
    where the normal approximation is not."""
    if not n:
        return float("nan"), float("nan")
    ph = k / n
    d = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / d
    half = z * ((ph * (1 - ph) + z * z / (4 * n)) / n) ** 0.5 / d
    return max(0.0, centre - half), min(1.0, centre + half)


def arm_of(rec):
    key = (rec.get("input") or "?", rec.get("arm") or "?",
           rec.get("stack") or ("plain" if rec.get("plain_llm") else "gtlm"))
    return ARM_NAMES.get(key, f"{key[0]} / {key[1]} / {key[2]}")


def agg(values):
    values = [v for v in values if v is not None]
    if not values:
        return float("nan"), float("nan"), 0
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.mean(values), sd, len(values)


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if x == x else "—"


def per_arm_table(by_arm, arms, baseline):
    lines = ["| arm | seeds | test accuracy | test F1 | best dev accuracy |",
             "|---|--:|--:|--:|--:|"]
    for name in arms:
        rs = by_arm[name]
        m, sd, n = agg([r["test_accuracy"] for r in rs])
        f1, _, _ = agg([r.get("test_f1") for r in rs])
        dv, dsd, _ = agg([r.get("best_val_accuracy") for r in rs])
        lines.append(f"| **{name}** | {n} | {fmt(m)} ± {fmt(sd)} | {fmt(f1)} | "
                     f"{fmt(dv)} ± {fmt(dsd)} |")
    if baseline is not None:
        lines.append(f"| majority-class baseline | — | {fmt(baseline)} | — | — |")
    return lines


def per_type_table(recs, by_arm, arms, base_pt):
    types = sorted({t for r in recs for t in (r.get("test_accuracy_per_type") or {})},
                   key=lambda x: (len(x), x))
    out = ["", "Per type (test accuracy, mean over seeds):", "",
           "| type | " + " | ".join(arms) + " | majority |",
           "|---" * (len(arms) + 2) + "|"]
    for t in types:
        cells = []
        for name in arms:
            m, _sd, _n = agg([(r.get("test_accuracy_per_type") or {}).get(t)
                              for r in by_arm[name]])
            cells.append(fmt(m, 3))
        b = base_pt.get(t)
        out.append(f"| {t} | " + " | ".join(cells) + " | "
                   + (fmt(b, 3) if b is not None else "—") + " |")
    return out


def tiers_from_items(items_root):
    """`{item id -> tier}` from the dataset the runs were graded against."""
    path = os.path.join(items_root, "test.jsonl")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r.get("tier") or "core"
    return out


def tier_stats_from_dump(run_name, tier_of):
    """Recover a run's per-tier block from its per-item dump.

    Runs recorded before `test_per_tier` existed (everything up to and including
    `arms_v3`) still dumped every item's verdict, and the tier is a property of
    the ITEM, not of the run -- so the block can be reconstructed exactly rather
    than the older sweep being left unreportable.  Joined on `id` against the
    dataset the run was graded against, which the caller passes in; a dump whose
    ids are not in that dataset yields nothing rather than a silent partial.
    """
    path = os.path.join(PRED_DIR, run_name, "test.jsonl")
    if not (tier_of and os.path.exists(path)):
        return {}
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t = r.get("tier") or tier_of.get(r["id"])
            if t is None:
                return {}
            rows.append((t, r))
    out = {}
    for tier in sorted({t for t, _ in rows}):
        sel = [r for t, r in rows if t == tier]
        pos = [r for r in sel if not r.get("negative")]
        out[tier] = {
            "accuracy": sum(r["success"] for r in sel) / len(sel),
            "n": len(sel),
            "false_sentinel": (sum(SENTINEL in (r.get("prediction") or "")
                                   for r in pos) / len(pos)
                               if pos else float("nan")),
        }
    return out


def per_tier_table(by_arm, arms, tier_of):
    """Accuracy per generalisation tier -- the study's actual claim.

    Kept separate from the per-type table because it answers a different
    question: per type says *what* the model can do, per tier says whether any
    of it survives contact with a phrasing or a relation it was not trained on.
    The test split is 73 % core, so the aggregate in the first table is very
    nearly the core number and says almost nothing about the other two.
    """
    resolved, missing = {}, []
    for name in arms:
        per_seed = []
        for r in by_arm[name]:
            block = r.get("test_per_tier") or tier_stats_from_dump(
                r.get("run_name", ""), tier_of)
            if block:
                per_seed.append(block)
            else:
                missing.append(r.get("run_name", "?"))
        resolved[name] = per_seed

    tiers = sorted({t for rs in resolved.values() for b in rs for t in b},
                   key=lambda x: (TIER_ORDER.index(x) if x in TIER_ORDER else 99, x))
    if not tiers:
        return ["", "Per tier: **not available** — no run in this file records "
                "`test_per_tier`, and no per-item dump could be joined to a tier.",
                ""]

    out = ["", "Per **generalisation tier** (test accuracy, mean over seeds):", "",
           "| tier | what it is | n | " + " | ".join(arms) + " |",
           "|---|---|--:" + "|--:" * len(arms) + "|"]
    for tier in tiers:
        ns = [b[tier]["n"] for rs in resolved.values() for b in rs if tier in b]
        cells = []
        for name in arms:
            m, sd, n = agg([b[tier]["accuracy"] for b in resolved[name] if tier in b])
            cells.append(fmt(m, 3) + (f" ± {fmt(sd, 3)}" if n > 1 else ""))
        out.append(f"| **{tier}** | {TIER_WHAT.get(tier, '—')} | "
                   f"{ns[0] if ns else '—'} | " + " | ".join(cells) + " |")

    out += ["", "False `ni podatka v bazi` on POSITIVE items, per tier "
            "(a refusal where an answer exists):", "",
            "| tier | " + " | ".join(arms) + " |",
            "|---" * (len(arms) + 1) + "|"]
    for tier in tiers:
        cells = []
        for name in arms:
            m, _sd, _n = agg([b[tier]["false_sentinel"] for b in resolved[name]
                              if tier in b])
            cells.append(fmt(m, 3))
        out.append(f"| **{tier}** | " + " | ".join(cells) + " |")
    if missing:
        out += ["", f"> {len(missing)} run(s) contributed no tier block "
                f"(no `test_per_tier` and no joinable dump): "
                + ", ".join(sorted(set(missing))[:4])
                + ("…" if len(set(missing)) > 4 else "") + "."]
    return out


def contrast_table(by_arm):
    """Each contrast as a per-seed paired difference.

    Paired, because the two arms in a contrast share a seed, a data order and a
    schedule; the difference of means throws that pairing away and reports a
    spread that is mostly seed variance common to both sides.
    """
    out = ["", "Contrasts (test accuracy, **paired by seed**):", "",
           "| contrast | arms | per-seed Δ | mean Δ | sd | what it isolates |",
           "|---|---|---|--:|--:|---|"]
    for name, a, b, note in CONTRASTS:
        ra = {r.get("seed"): r for r in by_arm.get(a, [])}
        rb = {r.get("seed"): r for r in by_arm.get(b, [])}
        seeds = sorted(s for s in ra.keys() & rb.keys()
                       if ra[s].get("test_accuracy") is not None
                       and rb[s].get("test_accuracy") is not None)
        if not seeds:
            out.append(f"| **{name}** | {a} − {b} | — | — | — | {note} |")
            continue
        diffs = [ra[s]["test_accuracy"] - rb[s]["test_accuracy"] for s in seeds]
        m, sd, _ = agg(diffs)
        per_seed = ", ".join(f"s{s}: {d:+.4f}" for s, d in zip(seeds, diffs))
        out.append(f"| **{name}** | {a} − {b} | {per_seed} | {m:+.4f} | "
                   f"{fmt(sd)} | {note} |")
    return out


def convergence_section(recs, by_arm, arms):
    """The pre-registered rule, checked and stated -- never silently skipped."""
    out = ["", "Convergence (T9's pre-registered rule):", "",
           "| arm | seed | evals | last three dev accuracies | best step | "
           "best = final? |", "|---|--:|--:|---|--:|---|"]
    n_final, n_known = 0, 0
    for name in arms:
        for r in sorted(by_arm[name], key=lambda x: (x.get("seed") or 0)):
            c = r.get("convergence") or {}
            tail = ", ".join(f"{s}: {v:.4f}" for s, v in (c.get("last_three") or []))
            best_final = c.get("best_is_final")
            if best_final is not None:
                n_known += 1
                n_final += bool(best_final)
            out.append(f"| {name} | {r.get('seed')} | {c.get('n_evals', '—')} | "
                       f"{tail or '—'} | {c.get('best_step', '—')} | "
                       f"{'YES' if best_final else ('no' if best_final is not None else '—')} |")
    out.append("")
    if not n_known:
        out.append("**Convergence rule: NOT CHECKABLE** — no run in this file "
                   "records a `convergence` block (it was added with `arms_v3`). "
                   "Re-run the sweep before treating any number here as final.")
        return out
    share = n_final / n_known
    verdict = ("**FAILS the pre-registered rule**" if share > CONVERGENCE_LIMIT
               else "**passes the pre-registered rule**")
    lo, hi = _wilson(n_final, n_known)
    out.append(
        f"The best checkpoint was the FINAL one in **{n_final} of {n_known}** "
        f"runs ({share:.0%}); the pre-registered limit is "
        f"{CONVERGENCE_LIMIT:.0%}.  This {verdict}.")
    # The interval, stated whether the rule passes or fails.  The rule is a
    # proportion test and its precision is entirely a function of how many runs
    # the file holds: at 18 runs it can separate a fifth from a third, at 6 it
    # cannot.  Printing it unconditionally keeps that from becoming an excuse
    # available only when the verdict is unwelcome.
    out.append("")
    out.append(f"Wilson 95 % interval on that proportion: **{lo:.2f} – {hi:.2f}** "
               f"(n = {n_known})."
               + ("  The interval spans the limit, so this file does not have "
                  "the runs to settle the rule either way."
                  if lo <= CONVERGENCE_LIMIT <= hi else ""))
    if share > CONVERGENCE_LIMIT:
        out.append("")
        out.append("> Under the rule agreed before these numbers existed, the "
                   "study is still measuring **learning speed** rather than "
                   "where each arm ends up, and needs more epochs before any "
                   "claim above is final.")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    # One or more.  A contrast's two arms do not always land in the same sweep
    # directory: the 12B pair was split when its GTLM arm was re-run under the
    # bias-LR schedule (SCALING.md decision 8) while its no-retrieval control,
    # which has no graph-bias modules to rescale, stayed valid where it was.
    # Taking several paths keeps that a reading-time join instead of a merged
    # file on disk, which would be a second copy of the evidence to keep honest.
    ap.add_argument("runs_jsonl", nargs="+")
    ap.add_argument("--markdown", default=None, help="also write a .md table here")
    ap.add_argument("--items-root", default="data/datasets/generated",
                    help="the dataset carrying each item's TIER, used only to "
                         "reconstruct the per-tier block for runs recorded "
                         "before `test_per_tier` existed")
    a = ap.parse_args(argv)

    recs = []
    for path in a.runs_jsonl:
        with open(path, encoding="utf-8") as f:
            recs += [json.loads(line) for line in f if line.strip()]
    recs = [r for r in recs if r.get("test_accuracy") is not None]
    if not recs:
        print(f"{', '.join(a.runs_jsonl)}: no finished runs with a test accuracy")
        return 1

    by_arm = collections.defaultdict(list)
    for r in recs:
        by_arm[arm_of(r)].append(r)
    arms = [n for n in ORDER if n in by_arm] + sorted(k for k in by_arm
                                                     if k not in ORDER)

    base = recs[0].get("baselines", {}).get("test", {})
    lines = per_arm_table(by_arm, arms, base.get("majority_overall"))
    lines += per_tier_table(by_arm, arms, tiers_from_items(a.items_root))
    lines += per_type_table(recs, by_arm, arms, base.get("majority_per_type", {}))
    lines += contrast_table(by_arm)
    lines += convergence_section(recs, by_arm, arms)

    # Anything that makes a reported number less than fully reproducible from the
    # config, surfaced rather than left in the JSON.
    flagged = [r["run_name"] for r in recs
               if any((r.get("final_eval_oom_splits") or {}).values())]
    if flagged:
        lines += ["", "> **OOM fallback fired during a FINAL evaluation** in: "
                  + ", ".join(flagged) + ".  Those numbers depend on a batch "
                  "grouping that was not pre-declared (train/evaluate.py)."]

    out = "\n".join(lines)
    print(out)
    if a.markdown:
        with open(a.markdown, "w", encoding="utf-8") as f:
            f.write(out + "\n")
        print(f"\n[wrote] {a.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
