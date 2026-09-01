#!/usr/bin/env python3
"""What the flex prefill actually buys, against eager, on this GPU.

    python -m ask.checks.bench_backends --impl flex  --out flex.json
    python -m ask.checks.bench_backends --impl eager --out eager.json
    python -m ask.checks.bench_backends --compare flex.json eager.json

D17 serves eager unless the GPU model has a warm compile cache, which makes the
fallback a *decision* rather than an accident -- and a decision needs a number.
This produces it: the same questions, from their on-disk balls, answered on each
backend, timed and measured for memory.

**Only the prefill differs.**  Serving is flex for the prefill and eager for the
decode (D16), so what a warm cache can change is the time to the FIRST token and
the memory the prefill peaks at.  Every token after it costs the same either way.
That is why this reports time-to-first-token separately from total time rather
than only the total, which would dilute the effect by the length of the answer
and say "8 % faster" about a difference that is entirely in one phase.

Three timings per run, because they answer different questions:

  * `batch`   -- building the input: tokenising, SPD / RRWP / magnetic, collating.
                 CPU, identical on both backends, and part of what a user waits
                 through.  Measured separately so it can be subtracted rather
                 than silently credited to whichever backend is being blamed.
  * `first`   -- the wait before anything appears: batch + prefill + one decode
                 step.  This is the number a person feels.
  * `total`   -- to the last token.

Repeats matter and are reported apart.  The first run of a shape in a process
pays dynamo's tracing and guard installation even when inductor's cache hits
(V5), so a benchmark that averaged repeat 1 with repeats 2-3 would understate a
warm cache and overstate a cold one.  Repeat 1 is reported as `cold`, the median
of the rest as `warm`.
"""
import os
import sys
import json
import time
import argparse

DEFAULT_SPLIT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "datasets", "balls", "test.jsonl")

# Where the corpus actually sits, as fractions of the token distribution: p50 is
# 1,318 tokens, p99 is 5,780, and the largest ball in the split is 11,421.  A
# benchmark of the median alone would miss that the two backends diverge with
# length; one of the maximum alone would describe a ball nobody asks about.
PERCENTILES = (0.05, 0.25, 0.50, 0.75, 0.90, 0.99, 1.00)


def pick(rows, percentiles=PERCENTILES):
    """One item per percentile of packed length, deduplicated."""
    ordered = sorted(rows, key=lambda r: int(r["n_tokens"]))
    out, seen = [], set()
    for p in percentiles:
        row = ordered[min(len(ordered) - 1, int(len(ordered) * p))]
        if row["id"] not in seen:
            seen.add(row["id"])
            out.append(row)
    return out


def spread(rows, n):
    """`n` items at an even stride through the file, for the drift question.

    File order, not sorted order: the split is written type by type, so the
    first `n` rows would be one question type and a size-sorted pick would be
    all short answers or all long ones.  A stride crosses both.
    """
    if n >= len(rows):
        return list(rows)
    step = len(rows) / n
    return [rows[min(len(rows) - 1, int(i * step))] for i in range(n)]


def measure(a):
    """Answer each item `--repeats` times on one backend, and record everything."""
    import torch

    from ask import backbone
    from ask.answer import answer, to_batch
    from ask.retrieve import Ball
    from ask.precompile import padded_n

    backbone.quiet_libraries()
    with open(a.split, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    items = spread(rows, a.items) if a.items else pick(rows)

    t0 = time.perf_counter()
    gtlm = backbone.load_gtlm(a.checkpoint, graph_attn_impl=a.impl)
    load_s = time.perf_counter() - t0
    torch.cuda.synchronize()
    # The model, resident and alone: every peak below is reported against this,
    # so the two backends are compared on what the ANSWER costs rather than on
    # weights they both carry.
    base = torch.cuda.memory_allocated()

    runs = []
    for row in items:
        ball = Ball(texts=list(row["nodes"]),
                    edges=[list(e) for e in row["edges"]],
                    anchors=list(row["anchors"]), strings=[],
                    n_tokens=int(row.get("n_tokens") or 0))
        for r in range(a.repeats):
            torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()
            batch = to_batch(row["question"], ball, gtlm)
            t_batch = time.perf_counter() - t0
            shape = [int(batch["input_ids"].shape[1]), padded_n(batch)]
            del batch

            t0 = time.perf_counter()
            stream = iter(answer(row["question"], ball, gtlm,
                                 max_new_tokens=a.max_new_tokens))
            first = next(stream, "")
            torch.cuda.synchronize()
            t_first = time.perf_counter() - t0
            chunks = [first, *stream]
            torch.cuda.synchronize()
            t_total = time.perf_counter() - t0

            runs.append({
                "id": row["id"], "repeat": r,
                "n_nodes": int(row["n_nodes"]), "n_tokens": int(row["n_tokens"]),
                "L": shape[0], "N": shape[1],
                "out_tokens": len(chunks),
                "batch_s": round(t_batch, 4),
                "first_s": round(t_first, 4),
                "total_s": round(t_total, 4),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "peak_over_model_gb": round(
                    (torch.cuda.max_memory_allocated() - base) / 2**30, 3),
                "reserved_gb": round(
                    torch.cuda.max_memory_reserved() / 2**30, 3),
                "answer": "".join(chunks),
            })
            print(f"  {a.impl:5s} {row['id']:>28}  rep{r}  L={shape[0]:>5} "
                  f"N={shape[1]:>4}  first {t_first:6.2f}s  total {t_total:6.2f}s"
                  f"  peak {runs[-1]['peak_over_model_gb']:6.2f} GiB",
                  file=sys.stderr, flush=True)

    out = {
        "impl": a.impl,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "checkpoint": a.checkpoint,
        "load_s": round(load_s, 2),
        "model_gb": round(base / 2**30, 3),
        "max_new_tokens": a.max_new_tokens,
        "runs": runs,
    }
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    else:
        print(json.dumps(out, ensure_ascii=False))
    return 0


MAX_TABLE_ROWS = 12


def agreement(x, y):
    """Where two answers stop being the same string, in characters."""
    i = 0
    for i, (cx, cy) in enumerate(zip(x, y)):
        if cx != cy:
            return i
    return min(len(x), len(y)) if len(x) != len(y) else -1


def drift_report(by_id):
    """How often the eager fallback changes the answer, and on what.

    D17 says the fallback is the same arithmetic in a different order, which bf16
    makes differ in the last bits and greedy decoding can turn into a different
    token.  `check_parity --drift` measures that at the *evaluator's* budget on
    16 items, which is the right question for "is ask the model the numbers
    describe" and the wrong one for serving: it is short answers on small balls,
    where nothing diverges.  This is the serving question -- the full 1024-token
    budget, across the split -- and it reports what the divergences have in
    common rather than only how many there were.
    """
    rows = []
    for item, per in sorted(by_id.items()):
        if len(per) != 2:
            continue
        f, e = per["flex"][0], per["eager"][0]
        rows.append((item, f, e, agreement(f["answer"], e["answer"])))

    diff = [r for r in rows if r[3] >= 0]
    print(f"\n{len(rows) - len(diff)}/{len(rows)} identical answers "
          f"({len(diff)} differ)\n")
    if diff:
        head = (f"{'item':>14} {'L':>6} {'N':>5} | {'tok flex':>8} {'tok eager':>9}"
                f" | {'agree chars':>11} {'of flex len':>11}")
        print(head)
        print("-" * len(head))
        for item, f, e, at in diff:
            print(f"{item[:14]:>14} {f['L']:>6} {f['N']:>5} | "
                  f"{f['out_tokens']:>8} {e['out_tokens']:>9} | "
                  f"{at:>11} {len(f['answer']):>11}")

    # What the two groups look like, so the answer is "large balls with long
    # answers" or "no pattern" rather than a bare rate.
    for label, group in (("differ", diff), ("identical", [r for r in rows
                                                          if r[3] < 0])):
        if not group:
            continue
        import statistics
        print(f"\n{label:>9}: n={len(group):<4} "
              f"median L={statistics.median(r[1]['L'] for r in group):>6.0f}  "
              f"median tokens out="
              f"{statistics.median(r[1]['out_tokens'] for r in group):>5.0f}  "
              f"max tokens out={max(r[1]['out_tokens'] for r in group):>5}")
    return 0


def stats(runs, key):
    """`(cold, warm)` for one item: repeat 0, and the median of the rest."""
    import statistics

    cold = next((r[key] for r in runs if r["repeat"] == 0), None)
    rest = [r[key] for r in runs if r["repeat"] > 0]
    return cold, (statistics.median(rest) if rest else cold)


def compare(paths):
    """Print the table, and refuse to compare runs that are not comparable."""
    data = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            data.append(json.load(f))
    a, b = data
    if a["gpu"] != b["gpu"]:
        print(f"different GPUs: {a['gpu']!r} vs {b['gpu']!r}", file=sys.stderr)
        return 2

    by_id = {}
    for d in data:
        for r in d["runs"]:
            by_id.setdefault(r["id"], {}).setdefault(d["impl"], []).append(r)

    print(f"\n{a['gpu']}  ·  torch {a['torch']}  ·  budget "
          f"{a['max_new_tokens']} tokens  ·  model {a['model_gb']:.2f} GiB")
    print(f"load: " + "  ".join(f"{d['impl']} {d['load_s']:.1f}s" for d in data))
    if len(by_id) > MAX_TABLE_ROWS:
        # A drift run is a hundred items; a hundred rows of timings is not a
        # table anyone reads, and the question there is agreement, not speed.
        return drift_report(by_id)
    print("\nwarm = median of repeats after the first; cold = the first\n")
    head = (f"{'item':>26} {'L':>6} {'N':>5} {'out':>4} | "
            f"{'first flex':>10} {'first eager':>11} {'Δ':>7} | "
            f"{'total flex':>10} {'total eager':>11} | "
            f"{'peak flex':>9} {'peak eager':>10}")
    print(head)
    print("-" * len(head))
    for item, per in by_id.items():
        if len(per) != 2:
            continue
        f_runs, e_runs = per.get("flex", []), per.get("eager", [])
        if not (f_runs and e_runs):
            continue
        r0 = f_runs[0]
        _, f_first = stats(f_runs, "first_s")
        _, e_first = stats(e_runs, "first_s")
        _, f_total = stats(f_runs, "total_s")
        _, e_total = stats(e_runs, "total_s")
        _, f_peak = stats(f_runs, "peak_over_model_gb")
        _, e_peak = stats(e_runs, "peak_over_model_gb")
        delta = f"{(e_first - f_first) / e_first * 100:+.0f}%" if e_first else "—"
        print(f"{item[:26]:>26} {r0['L']:>6} {r0['N']:>5} {r0['out_tokens']:>4} | "
              f"{f_first:>9.2f}s {e_first:>10.2f}s {delta:>7} | "
              f"{f_total:>9.2f}s {e_total:>10.2f}s | "
              f"{f_peak:>8.2f}G {e_peak:>9.2f}G")

    # The answers must not depend on the backend.  If they do, no timing on this
    # page means anything -- the two columns are then measuring two models.
    same = sum(1 for item, per in by_id.items()
               if len(per) == 2
               and per["flex"][0]["answer"] == per["eager"][0]["answer"])
    both = sum(1 for per in by_id.values() if len(per) == 2)
    print(f"\nidentical answers on both backends: {same}/{both}")
    return 0


def build_parser():
    from ask import backbone
    from ask.answer import MAX_NEW_TOKENS

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["flex", "eager"])
    p.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    p.add_argument("--checkpoint", default=backbone.DEFAULT_CHECKPOINT)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--items", type=int, default=0,
                   help="measure N items spread across the split instead of "
                        "the seven percentile picks; for the drift question, "
                        "where the sample size is the point")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--out", default="")
    return p


def main(argv=None):
    from ask import aliases

    a = build_parser().parse_args(argv)
    # Before anything is measured, so the JSON this writes records the arm and
    # not the alias it was reached by (D23) -- a result file outlives the table.
    a.checkpoint = aliases.resolve(a.checkpoint)
    if a.compare:
        return compare(a.compare)
    if not a.impl:
        print("give --impl flex|eager, or --compare A.json B.json", file=sys.stderr)
        return 2
    return measure(a)


if __name__ == "__main__":
    sys.exit(main())
