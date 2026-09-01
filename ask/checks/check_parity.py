#!/usr/bin/env python3
"""The serving path answers exactly what the evaluated path answered.

    python -m ask.checks.check_parity --n 32
    sbatch ask/slurm/run_ask.sbatch  # (wraps `-m ask`, not this; see --help)

Every number in `train/results` rests on one claim: **the model that was scored
is the model that answers**.  Nothing else in `ask/` proves it.  The retrieval
side is proved by construction -- `build()` and `_graph()` are imported from the
code that built the corpus, so a served ball and a trained ball are the same
object -- but the generation side is `ask/answer.py`'s own assembly of the
batch, the config and the decode, and an assembly can be wrong in ways that
still produce fluent Slovene.

So: take N test items, answer each **from its on-disk ball** through
`ask.answer.answer`, and assert the string equals what `evaluate.pass2` produces
for the same item.  What that catches -- each of these leaves the answer
plausible and different:

  * a generation config rebuilt instead of borrowed (V3): gemma-3-1b-it ships
    `do_sample=True`, so a "greedy" decode that forgot `use_model_defaults=False`
    samples, and the two runs differ per token;
  * `cache_implementation` carried over wrongly -- `"hybrid"` on this stack dies
    on the first decode step past a 512-token prompt;
  * the prompt node attached to one anchor instead of all of them;
  * the chat template applied differently, or the answer marker cut elsewhere;
  * a feature computed under a different `magnetic_m` or `max_spd` than the
    checkpoint trained with (V1).

**The budget is pinned to the evaluator's, per item.**  `evaluate` sizes
generation from the gold length (`min(320, 2*gold+16)`) and D12 serves at a flat
1024, so an unpinned comparison would report every long answer as a mismatch
when only the cap differed.  The check is about the decode, not the cap.

Batches of one on both sides, which makes the comparison exact rather than
approximate: with no padding there is no batch-composition effect on the
bf16 numerics, so two greedy decodes of the same item are token-identical.
"""
import sys
import argparse

import torch

from ask import backbone, aliases
from ask.answer import answer, MAX_NEW_TOKENS
from ask.retrieve import Ball


def ball_of(row):
    """The on-disk ball, in the shape `ask` passes around.

    `codes` and `n_tokens` stay empty: nothing in the answer path reads them,
    and inventing values here would test the report rather than the model.
    """
    return Ball(texts=list(row["nodes"]), edges=[list(e) for e in row["edges"]],
                anchors=list(row["anchors"]), strings=[],
                n_tokens=int(row.get("n_tokens") or 0))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=backbone.DEFAULT_CHECKPOINT,
                   help="an alias (ask --aliases) or a checkpoint directory")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=32, help="items to compare")
    p.add_argument("--impl", default="eager", choices=["flex", "eager"],
                   help="attention backend for BOTH sides")
    p.add_argument("--no-spread", dest="spread", action="store_false",
                   default=True,
                   help="take the split's first N items instead of N spread "
                        "across it; the file is written type by type, so the "
                        "first N are all one question type")
    p.add_argument("--drift", action="store_true",
                   help="also answer the same items on the OTHER backend and "
                        "report how many answers the fallback changes (D17)")
    p.add_argument("--verbose", action="store_true")
    return p


def drift(a, split, golds, first, cap):
    """How many answers change when the other attention backend serves them.

    D17 falls back to eager when a GPU has no warm compile cache, and every
    reported number was produced under the backend the checkpoint recorded.  The
    two are the same arithmetic in a different order, so bf16 makes them differ
    in the last bits and greedy decoding can turn that into a different token.
    This measures how often it actually does, rather than assuming either that
    it never happens or that it matters.
    """
    other = "flex" if a.impl == "eager" else "eager"
    print(f"\n[drift] the same {len(first)} items on {other}", flush=True)
    gtlm = backbone.load_gtlm(a.checkpoint, graph_attn_impl=other)
    same, changed = 0, []
    for i, want in first.items():
        row = split.rows[i]
        # The SAME budget the first pass used for this item, or a longer answer
        # would be reported as drift when only the cap differed.
        budget = min(cap, 2 * golds[i] + 16)
        got = "".join(answer(row["question"], ball_of(row), gtlm,
                             max_new_tokens=budget))
        if got == want:
            same += 1
        else:
            changed.append((row["id"], want, got))
            print(f"  differs {row['id']}\n    {a.impl}: {want!r}\n"
                  f"    {other}: {got!r}", flush=True)
    print(f"\n[drift] {same}/{len(first)} identical between {a.impl} and {other}")
    return changed


def main(argv=None):
    a = build_parser().parse_args(argv)
    # Up front, so the line this check prints names the arm it ran on and not
    # the alias it was reached by (D23).
    a.checkpoint = aliases.resolve(a.checkpoint)

    from train.data import load_split
    from train.evaluate import GradeEvaluator, scaled_budgets

    backbone.quiet_libraries()
    gtlm = backbone.load_gtlm(a.checkpoint, graph_attn_impl=a.impl)
    model, tok, cfg = gtlm["model"], gtlm["tokenizer"], gtlm["cfg"]

    # `max_items` caps the split; `spread` takes them across the whole file
    # rather than the first N, which would be one question type.
    cfg.max_items = a.n
    split = load_split(cfg, tok, a.split, with_generation=True, spread=a.spread)

    eval_budget, gen_budget, note = scaled_budgets(cfg.eval_token_budget,
                                                   cfg.gen_token_budget)
    print(f"[parity] {note}", flush=True)
    # `max_batch=1`: padding-free generation, so the reference is reproducible
    # item by item rather than only in aggregate.
    ev = GradeEvaluator(tok, gtlm["collator"], [split], eval_budget=eval_budget,
                        gen_budget=gen_budget, max_batch=1)
    golds = ev.gold_lens(split)

    same, diff, mine = 0, [], {}
    for i in range(len(split)):
        row = split.rows[i]
        with torch.no_grad():
            ref, _trunc = ev.pass2(model, split, [i])
        want = ref[i]
        # Exactly the budget `pass2` just used for this item (its batch is one
        # row), so only the DECODE is being compared.
        budget = min(ev.max_new_tokens, 2 * golds[i] + 16)
        got = "".join(answer(row["question"], ball_of(row), gtlm,
                             max_new_tokens=budget))
        mine[i] = got
        if got == want:
            same += 1
            if a.verbose:
                print(f"  ok  {row['id']}  {got[:70]!r}", flush=True)
        else:
            diff.append((row["id"], want, got))
            print(f"  MISMATCH {row['id']}\n    eval: {want!r}\n    ask:  {got!r}",
                  flush=True)

    n = len(split)
    print(f"\n{same}/{n} identical, {len(diff)} different "
          f"({a.impl}, {a.split}, {a.checkpoint.rstrip('/').split('/')[-1]})")
    if diff:
        print("\nThe serving path and the evaluated path do not agree.  Every "
              "accuracy number in train/results describes the evaluated path; "
              "until this is 0, `ask` is not serving the model those numbers "
              "are about.")
        return 1
    print(f"\nThe serving path IS the evaluated path, on {n} items.  "
          f"(Serving decodes at {MAX_NEW_TOKENS} tokens rather than this "
          f"per-item budget -- D12 -- which changes only where a long answer "
          f"stops, not what it says.)")
    if a.drift:
        drift(a, split, golds, mine, ev.max_new_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
