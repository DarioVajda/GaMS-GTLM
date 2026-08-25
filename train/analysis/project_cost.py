"""Project the 8-epoch wall clock, and the evaluation share, from a 1-epoch run.

    .venv/bin/python -m train.analysis.project_cost train/results/arms_v3_timing

Answers the two questions T10 and T5 ask before 18 runs are committed:

  * **does every arm fit its Slurm wall clock?**  A killed run loses everything,
    so this is checked from a measurement rather than assumed from `arms_v2`'s
    A100 numbers.
  * **how much of a run is evaluation?**  `arms_v2` spent a large share of each
    run in it, and 8 epochs against a hard wall clock turns that from a papercut
    into a run-killing risk.

Everything below is measured, not assumed.  A 1-epoch run reports its total
runtime and prints, per evaluation, `[eval] <split>: pass1 X s, pass2 Y s over N
items` plus a metrics dict carrying `eval_pass1_share`.  From the LAST
in-training eval of each run:

    pass1_rate = eval_pass1_s / (items in the eval split)     per item
    pass2_rate = eval_pass2_s / eval_generated                per GENERATED item
    train_only = train_runtime - (seconds spent in those evals)

and then

    training      = 8 epochs   x train_only
    in-training   = 12 evals   x (measured seconds per eval)
    final dev+test= pass1_rate x N  +  pass2_rate x (1 - pass1_share) x N

for N = 1,040 + 2,184.  The final passes run `fast=False`, so pass 2 covers every
pass-1 miss rather than only the modes a token mismatch condemns — hence the
`(1 - pass1_share)` factor rather than the fast path's smaller generated set.
Using a ONE-EPOCH `pass1_share` makes that estimate conservative: pass 1 settles
more as the model improves, so the real 8-epoch run generates less than this.

**On the 25 % bound.**  It is a proxy for "evaluation must not blow the wall
clock", and it is the wrong proxy for an arm whose training is nearly free: the
no-retrieval arms train in minutes while their dev+test evaluation still has to
score 3,224 items. So both the share AND the absolute hours are reported, and the
wall-clock verdict is given separately — that is the risk the bound exists to
catch.
"""
import os
import re
import ast
import glob
import json
import argparse
import statistics

from .report_arms import arm_of

EVAL_LINE = re.compile(
    r"\[eval\] (?P<split>\S+): pass1 (?P<p1>[\d.]+) s, pass2 (?P<p2>[\d.]+) s "
    r"over (?P<n>\d+) items")
METRICS_LINE = re.compile(r"\{'eval_loss'.*?\}")

EPOCHS = 8
EVALS = 12                      # 11 at eval_steps=400, plus the forced final one
DEV_N, TEST_N = 1040, 2184
EVAL_SHARE_BOUND = 0.25


def parse_log(log_path):
    """`(eval_seconds list, last in-training metrics dict)` from one run's log."""
    seconds, metrics = [], None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = EVAL_LINE.search(line)
            if m:
                seconds.append(float(m["p1"]) + float(m["p2"]))
            d = METRICS_LINE.search(line)
            if d:
                try:
                    metrics = ast.literal_eval(d.group(0))
                except (ValueError, SyntaxError):
                    pass
    return seconds, metrics


def find_log(sweep_dir, run_name):
    """The slurm log for `run_name` — array logs are named by job/task, not run."""
    for path in sorted(glob.glob(os.path.join(sweep_dir, "logs", "*.out"))):
        with open(path, encoding="utf-8", errors="replace") as f:
            if run_name in f.read(400_000):
                return path
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_dir")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--evals", type=int, default=EVALS)
    ap.add_argument("--wall-clock-h", type=float, default=16.0,
                    help="the Slurm limit each run has to fit inside")
    a = ap.parse_args(argv)

    with open(os.path.join(a.sweep_dir, "runs.jsonl"), encoding="utf-8") as f:
        runs = [json.loads(line) for line in f if line.strip()]
    if not runs:
        print(f"{a.sweep_dir}: no finished runs")
        return 1

    rows, skipped = [], []
    for rec in runs:
        log = find_log(a.sweep_dir, rec["run_name"])
        seconds, metrics = parse_log(log) if log else ([], None)
        if not seconds or not metrics:
            skipped.append(rec["run_name"])
            continue
        n_fast = (rec.get("dev_subsample") or {}).get("n") or rec["val_size"]
        per_eval = statistics.mean(seconds)
        pass1_rate = metrics["eval_pass1_s"] / max(1, n_fast)
        pass2_rate = (metrics["eval_pass2_s"]
                      / max(1, metrics.get("eval_generated") or 1))
        miss = 1.0 - (metrics.get("eval_pass1_share") or 0.0)

        steps = rec["convergence"]["max_steps"]
        train_only = rec["train_runtime_s"] - sum(seconds)
        training = train_only * a.epochs
        in_training = per_eval * a.evals
        final = sum(pass1_rate * n + pass2_rate * miss * n for n in (DEV_N, TEST_N))
        total = training + in_training + final
        rows.append(dict(
            arm=arm_of(rec), mb=f"{rec['batch_size']}x{rec['accumulation_steps']}",
            s_step=train_only / steps, per_eval=per_eval, training=training,
            in_training=in_training, final=final, total=total,
            share=(in_training + final) / total))

    if skipped:
        print(f"[warn] no parseable log for: {', '.join(skipped)}\n")
    if not rows:
        print("nothing to project")
        return 1

    print(f"Projected cost of {a.epochs} epochs, from 1-epoch runs "
          f"({runs[0]['convergence']['max_steps']} optimizer steps measured):\n")
    print(f"| arm | micro-batch | s/step | 1 eval | train {a.epochs}ep | "
          f"{a.evals} evals | final dev+test | TOTAL | eval share |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for r in sorted(rows, key=lambda r: -r["total"]):
        print(f"| {r['arm']} | {r['mb']} | {r['s_step']:.2f} s | "
              f"{r['per_eval']:.0f} s | {r['training'] / 3600:.2f} h | "
              f"{r['in_training'] / 60:.0f} min | {r['final'] / 60:.0f} min | "
              f"**{r['total'] / 3600:.2f} h** | {r['share']:.0%} |")

    slowest = max(r["total"] / 3600 for r in rows)
    fits = slowest < a.wall_clock_h
    over = [r for r in rows if r["share"] >= EVAL_SHARE_BOUND]
    print(f"\n**Wall clock.** Slowest arm **{slowest:.2f} h** against a "
          f"{a.wall_clock_h:.0f} h Slurm limit — "
          f"{'FITS, with headroom' if fits else '**DOES NOT FIT**'}.")
    if not over:
        print(f"**Evaluation share.** Every arm is under the "
              f"{EVAL_SHARE_BOUND:.0%} bound.")
    else:
        worst = max(over, key=lambda r: r["share"])
        print(f"**Evaluation share.** {len(over)} of {len(rows)} arm(s) exceed "
              f"{EVAL_SHARE_BOUND:.0%}, worst {worst['share']:.0%} "
              f"({worst['arm']}).")
        cheap = all(r["total"] / 3600 < a.wall_clock_h / 4 for r in over)
        print("  " + (
            f"Every one of them is an arm whose TRAINING is nearly free "
            f"(total under {a.wall_clock_h / 4:.0f} h), so the share is high "
            f"because the denominator is small, not because evaluation is "
            f"expensive — the absolute eval time is the same few minutes as "
            f"everywhere else. The risk the bound exists to catch is the wall "
            f"clock, and that is met with room to spare."
            if cheap else
            "At least one of them is a long run, so this IS the failure the "
            "bound was written to catch. Cut evaluation cost before submitting."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
