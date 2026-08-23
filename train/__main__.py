"""Run ONE training configuration.

    .venv/bin/python -m train --types T3,T4,T9,T10 --num-epochs 8

Every flag defaults from ``RunConfig()``; nothing is required.  Flag names follow
the runner's convention (``some_key`` -> ``--some-key``) so the generic `sweep`
runner drives this module unchanged:

    .venv/bin/python -m sweep train train/configs/<name>.jsonc

Both must be run from the repo root: `sweep` refuses to submit when the calling
interpreter lives outside the project root (it forwards `SWEEP_VENV_BIN` to the
job, and a path outside the root would not resolve there).
"""
import argparse

from .config import RunConfig, IMPLS


def build_parser():
    d = RunConfig()
    p = argparse.ArgumentParser(
        prog="python3 -m train",
        description="Slovene lexicographical QA over the CJVT/DDDS knowledge graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    B = argparse.BooleanOptionalAction

    p.add_argument("--mode", choices=("train",), default=d.mode)

    p.add_argument("--model-name", default=d.model_name)
    p.add_argument("--impl", choices=IMPLS, default=d.impl)
    p.add_argument("--dtype", choices=("fp32", "bf16"), default=d.dtype)
    p.add_argument("--lora", action=B, default=d.lora)
    p.add_argument("--lora-r", type=int, default=d.lora_r)
    p.add_argument("--lora-dropout", type=float, default=d.lora_dropout)

    p.add_argument("--spd", action=B, default=d.spd)
    p.add_argument("--max-spd", type=int, default=d.max_spd)
    p.add_argument("--rrwp", action=B, default=d.rrwp)
    p.add_argument("--max-rw-steps", type=int, default=d.max_rw_steps)
    p.add_argument("--magnetic", action=B, default=d.magnetic)
    p.add_argument("--magnetic-dim", type=int, default=d.magnetic_dim)
    p.add_argument("--magnetic-q", type=float, default=d.magnetic_q)
    p.add_argument("--magnetic-m", type=int, default=d.magnetic_m)
    p.add_argument("--k-hop", type=int, default=d.k_hop)

    p.add_argument("--data-root", default=d.data_root)
    p.add_argument("--items-root", default=d.items_root,
                   help="the dataset directory carrying the GRADING contract "
                        "(datasets/generated/v2_final)")
    p.add_argument("--types", default=d.types,
                   help="comma-separated subset, e.g. T3,T4,T9,T10 ('' = all)")
    p.add_argument("--max-items", type=int, default=d.max_items,
                   help=">0 caps each split (smoke tests)")
    p.add_argument("--max-length", type=int, default=d.max_length)
    p.add_argument("--data-seed", type=int, default=d.data_seed)

    p.add_argument("--num-epochs", type=int, default=d.num_epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--accumulation-steps", type=int, default=d.accumulation_steps)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--bias-lr", type=float, default=d.bias_lr)
    p.add_argument("--eval-steps", type=int, default=d.eval_steps)
    p.add_argument("--max-steps", type=int, default=d.max_steps)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--gradient-checkpointing", action=B,
                   default=d.gradient_checkpointing)
    p.add_argument("--include-f1", action=B, default=d.include_f1)
    p.add_argument("--wandb-project", default=d.wandb_project)

    p.add_argument("--eval-token-budget", type=int, default=d.eval_token_budget)
    p.add_argument("--gen-token-budget", type=int, default=d.gen_token_budget)
    p.add_argument("--train-token-budget", type=int, default=d.train_token_budget,
                   help="0 falls back to a fixed --batch-size (which pads badly "
                        "on this corpus -- see train/batching.py)")
    p.add_argument("--max-batch", type=int, default=d.max_batch)
    p.add_argument("--final-eval", action=B, default=d.final_eval,
                   help="--no-final-eval skips the slow graded dev/test pass "
                        "(timing probes only -- a run without it has no result)")

    # sweep-runner bookkeeping
    p.add_argument("--runs-jsonl", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--sweep-id", default=None)
    return p


def config_from_args(a):
    return RunConfig(
        mode=a.mode,
        model_name=a.model_name, impl=a.impl, dtype=a.dtype,
        lora=a.lora, lora_r=a.lora_r, lora_dropout=a.lora_dropout,
        spd=a.spd, max_spd=a.max_spd,
        rrwp=a.rrwp, max_rw_steps=a.max_rw_steps,
        magnetic=a.magnetic, magnetic_dim=a.magnetic_dim,
        magnetic_q=a.magnetic_q, magnetic_m=a.magnetic_m,
        k_hop=a.k_hop,
        data_root=a.data_root, items_root=a.items_root,
        types=a.types, max_items=a.max_items,
        max_length=a.max_length, data_seed=a.data_seed,
        num_epochs=a.num_epochs, batch_size=a.batch_size,
        accumulation_steps=a.accumulation_steps,
        lr=a.lr, bias_lr=a.bias_lr, eval_steps=a.eval_steps,
        max_steps=a.max_steps, seed=a.seed, num_workers=a.num_workers,
        gradient_checkpointing=a.gradient_checkpointing,
        include_f1=a.include_f1, wandb_project=a.wandb_project,
        eval_token_budget=a.eval_token_budget,
        gen_token_budget=a.gen_token_budget,
        train_token_budget=a.train_token_budget, max_batch=a.max_batch,
        final_eval=a.final_eval,
    ).validate()


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    from .run import run_train_mode
    run_train_mode(cfg, runs_jsonl=args.runs_jsonl, run_name=args.run_name,
                   sweep_id=args.sweep_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
