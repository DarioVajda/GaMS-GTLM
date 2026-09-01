"""The licence for the DDP path: N GPUs must score a split exactly as one does.

    .venv/bin/python -m train.checks.check_ddp --out /tmp/ddp_w1.jsonl
    torchrun --standalone --nproc_per_node 4 \\
        -m train.checks.check_ddp --out /tmp/ddp_w4.jsonl
    .venv/bin/python -m train.checks.check_ddp --compare /tmp/ddp_w1.jsonl /tmp/ddp_w4.jsonl

Running the study on more than one GPU is only worth doing if it changes the
wall clock and nothing else.  Two claims carry that, and this checks both.

**1. The effective batch survives the split.**  `batch_size x
accumulation_steps` is the binding quantity; HF multiplies whatever it is given
by the world size, so the declared product has to be divided by the rank count
before it is handed over.  `distributed.ddp_factorisation` does that, and the
table below asserts the product is invariant for every rank count this cluster
can offer.  Pure arithmetic -- it needs no GPU and runs in the `--compare` pass
too.

**2. Evaluation is invariant item for item.**  The evaluator groups items into
batches by token budget, and that grouping is a deterministic function of the
split's lengths, so every rank derives the *same* batch list and runs a strided
slice of it.  Nothing is re-padded, so the merged verdicts should be not merely
close to the single-GPU ones but **identical** -- which is what is asserted, on
`success`, on `pass1` and on the decoded string.

That exactness is the point.  `evaluate.py` is explicit that re-grouping a batch
changes its padding and that a bf16 near-tie can flip an argmax between two
groupings; an implementation that sharded *items* instead of *batches* would
have re-padded nearly every batch, and its disagreements with the single-GPU run
would be indistinguishable from a real bug.

Exit code 0 on success, 1 on any mismatch.
"""
import os
import json
import argparse

import torch
from transformers import AutoTokenizer, set_seed

from gtlm.utils import GraphCollatorV2

from .. import distributed as dd
from ..config import RunConfig
from ..data import load_split
from ..evaluate import GradeEvaluator, scaled_budgets
from ..batching import LeftPadCollator, PlainCollator

# (effective batch, declared micro-batch, ranks).  The arms in use plus the rank
# counts a single B200/B300/H100 node can supply.
FACTORISATION_CASES = [
    (16, 4, 1), (16, 4, 2), (16, 4, 4), (16, 4, 8),
    (16, 16, 1), (16, 16, 2), (16, 16, 4), (16, 16, 8),
]


def check_factorisation():
    """`micro x accum x ranks == effective`, for every rank count."""
    print("[test] effective-batch invariance under ddp_factorisation")
    bad = 0
    for effective, declared, ranks in FACTORISATION_CASES:
        micro, accum = dd.ddp_factorisation(effective, declared, ranks)
        got = micro * accum * ranks
        ok = got == effective
        bad += not ok
        print(f"    E={effective:>3} micro={declared:>2} ranks={ranks}"
              f"  ->  {micro} x {accum} x {ranks} = {got}"
              f"   {'ok' if ok else '*** WRONG'}")
    return bad


def build_parser():
    d = RunConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None,
                   help="diff two emitted files instead of running the model")
    p.add_argument("--out", default=None,
                   help="where rank 0 writes the scored rows")
    p.add_argument("--model-name", default=d.model_name)
    p.add_argument("--data-root", default=d.data_root)
    p.add_argument("--items-root", default=d.items_root)
    p.add_argument("--split", default="dev")
    p.add_argument("--max-items", type=int, default=96)
    p.add_argument("--max-length", type=int, default=d.max_length)
    p.add_argument("--types", default="")
    p.add_argument("--plain-llm", action="store_true", default=False)
    p.add_argument("--checkpoint", default=None,
                   help="a trained adapter, so pass 1 is not vacuous")
    p.add_argument("--no-spd", dest="spd", action="store_false", default=d.spd)
    p.add_argument("--no-magnetic", dest="magnetic", action="store_false",
                   default=d.magnetic)
    return p


# ── the comparison ───────────────────────────────────────────────────────────
def compare(path_a, path_b):
    a = [json.loads(l) for l in open(path_a, encoding="utf-8")]
    b = [json.loads(l) for l in open(path_b, encoding="utf-8")]
    print(f"[test] {os.path.basename(path_a)}: {len(a)} rows "
          f"(world_size={a[0].get('_world_size') if a else '?'})")
    print(f"[test] {os.path.basename(path_b)}: {len(b)} rows "
          f"(world_size={b[0].get('_world_size') if b else '?'})")
    if len(a) != len(b):
        print(f"*** row counts differ: {len(a)} vs {len(b)}")
        return 1
    ids_a = [r["id"] for r in a]
    ids_b = [r["id"] for r in b]
    if ids_a != ids_b:
        print("*** the two runs scored different items, or in a different order")
        return 1

    fields = ("success", "pass1", "prediction", "reason")
    bad = {f: [] for f in fields}
    for ra, rb in zip(a, b):
        for f in fields:
            if ra.get(f) != rb.get(f):
                bad[f].append((ra["id"], ra.get(f), rb.get(f)))

    n = len(a)
    print(f"\nitems compared                {n}")
    for f in fields:
        k = len(bad[f])
        print(f"  {f:<24} {n - k}/{n} identical"
              + ("" if not k else f"   *** {k} differ"))
    acc_a = sum(r["success"] for r in a) / max(1, n)
    acc_b = sum(r["success"] for r in b) / max(1, n)
    print(f"accuracy                      {acc_a:.6f} vs {acc_b:.6f} "
          f"(delta {acc_b - acc_a:+.6f})")
    for f in fields:
        for iid, va, vb in bad[f][:3]:
            print(f"\n  {iid} [{f}]\n    A: {va!r}\n    B: {vb!r}")
    return 1 if any(bad[f] for f in fields) else 0


# ── the run ──────────────────────────────────────────────────────────────────
def emit(a):
    cfg = RunConfig(model_name=a.model_name, data_root=a.data_root,
                    items_root=a.items_root, types=a.types,
                    max_items=a.max_items, max_length=a.max_length,
                    plain_llm=a.plain_llm,
                    spd=False if a.plain_llm else a.spd,
                    magnetic=False if a.plain_llm else a.magnetic).validate()

    dd.init_distributed()
    device = dd.device()
    set_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # Spread across the file so the slice carries all three grading modes: pass 2
    # is where the sharding actually has to be right, and a prefix would be one
    # type in one mode.
    split = load_split(cfg, tokenizer, a.split, with_generation=True, spread=True)

    if cfg.plain_llm:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=cfg.torch_dtype(), attn_implementation="sdpa")
    else:
        config_cls, model_cls = cfg.gtlm_classes()
        if a.checkpoint:
            model = model_cls.from_pretrained(
                a.checkpoint, graph_attn_impl=cfg.backend(),
                torch_dtype=cfg.torch_dtype())
        else:
            config = config_cls.from_pretrained(
                cfg.model_name, **cfg.bias_params(), k_hop=cfg.k_hop,
                graph_attn_impl=cfg.backend(), **cfg.flex_params())
            model = model_cls.from_pretrained(
                cfg.model_name, config=config, graph_attn_impl=cfg.backend(),
                torch_dtype=cfg.torch_dtype())
    model.to(device).eval()

    collator = GraphCollatorV2(
        tokenizer=tokenizer, k_hop=cfg.k_hop,
        magnetic_m=cfg.magnetic_m if cfg.magnetic else 0,
        pad_to_block=(cfg.backend() == "flex" and not cfg.plain_llm),
        max_spd=cfg.max_spd)
    if cfg.plain_llm:
        collator = PlainCollator(collator)
    collator = LeftPadCollator(collator)

    # The budgets the real runs use, so the batch GROUPING under test is the
    # grouping training would have produced -- not a smaller one that happens to
    # put every item in its own batch and would make the test vacuous.
    eval_budget, gen_budget, note = scaled_budgets()
    if dd.is_main():
        print(f"[test] {note}")
        print(f"[test] world_size={dd.world_size()}  items={len(split)}")
    ev = GradeEvaluator(tokenizer, collator, [split],
                        eval_budget=eval_budget, gen_budget=gen_budget,
                        max_batch=RunConfig().eval_max_batch)

    rows, loss, n_gen, truncated, timing = ev.score(model, split.ds, fast=False)

    if dd.is_main():
        n = len(rows)
        acc = sum(r["success"] for r in rows) / max(1, n)
        print(f"[test] scored {n} items: accuracy {acc:.6f}, "
              f"pass1 cleared {sum(r['pass1'] for r in rows)}, generated {n_gen}, "
              f"loss {loss:.6f}")
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps({**r, "_world_size": dd.world_size()},
                                       ensure_ascii=False) + "\n")
            print(f"[test] rows -> {a.out}")
    dd.barrier()
    return 0


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.compare:
        bad = check_factorisation()
        print()
        return 1 if (compare(*a.compare) or bad) else 0
    if dd.is_main():
        if check_factorisation():
            return 1
        print()
    return emit(a)


if __name__ == "__main__":
    raise SystemExit(main())
