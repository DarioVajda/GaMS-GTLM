"""The licence for left-padded training batches: same loss, a tiny logits slice.

    .venv/bin/python -m train.checks.check_left_pad --data-root data/datasets/balls
    .venv/bin/python -m train.checks.check_left_pad \
        --data-root data/datasets/balls_serialised --plain-llm \
        --no-spd --no-magnetic --max-length 17408

`GraphCollatorV2` right-pads and has no `padding_side` option, so training
batches are rolled after collation (`batching.LeftPadCollator`).  Two things
have to be true for that to be a fix rather than a change of results:

  1. **the loss must not move.**  A roll relabels absolute positions but the
     structural mask reads padding from `attention_mask` alone, `node_ids`
     travels with `input_ids`, and `position_ids` are per-node LOCAL positions
     under `node_position_mode='reset'` — so every real token should see exactly
     the same keys with exactly the same biases.  That is an argument; this
     measures it, on both the GTLM and the plain stacks, over batches that are
     short, long, and deliberately mixed (the mixed ones are where right padding
     hurts and therefore where a roll would break something if it were going to).
  2. **the realised `logits_to_keep` must collapse.**  Right-padded, the slice is
     set by whichever row's answer starts earliest in the padded sequence:
     measured on the worst GTLM batch at packed L=16,384 it was 6,880 tokens and
     a 100.5 GiB peak, ~43 GB of it logits at a 262 k vocabulary for answer spans
     a few hundred tokens long.  Left-padded every row ends at `L-1`, so the same
     `min` collapses to `longest answer + 1`.

Peak GPU memory is reported for both paddings on every batch, so the memory
probe this replaces does not have to be run separately.

Exit code 0 on success, 1 on any failure.
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from gtlm.utils import GraphCollatorV2

from ..config import RunConfig
from ..data import load_split
from ..batching import packed_lengths, LeftPadCollator, PlainCollator
from ..evaluate import MAX_NEW_TOKENS
from ..run import answer_tail_inputs

# How the loss tolerance is set, and why it is not a guessed constant.
#
# In exact arithmetic the roll cannot move the loss: every real token attends to
# the same keys with the same biases.  In bf16 it can, through exactly two
# channels, and both are pure rounding:
#
#   * it changes which tokens share a kernel TILE, so partial sums accumulate in
#     a different order;
#   * on the PLAIN stack it changes each token's ABSOLUTE position (that collator
#     ships no `position_ids`, so the model derives them from `arange`), and RoPE
#     is relative in exact arithmetic but not bitwise so in bf16.  On the GTLM
#     stack this channel is absent: `position_ids` are per-node local values that
#     travel with their token.
#
# Two things calibrate the bound, and neither is a chosen constant.
#
# **The control** is a batch with pad columns added at BOTH ends — the same
# content, the same right-padded layout between rows, every absolute position
# moved by `CONTROL_FRONT`, and every tile boundary redrawn.  That is the
# perturbation left padding applies, minus the change to the padding pattern the
# test is about, so its effect on the loss is rounding by construction.
#
#   * the front pad is the LARGEST shift left padding would apply to any row in
#     this batch, so the control moves real tokens as far as the thing it
#     calibrates does.  A fixed small pad does not: at a 128-column front pad the
#     shift is exactly one flex block and every real token keeps its offset
#     inside its block, and at 64 the short batches (whose real tokens all fit in
#     block 0) still never leave it — both measured 0.000000, calibrating nothing;
#   * front + back is rounded up to a multiple of the block size, because flex
#     requires a block-aligned total length.
#
# (An even earlier version permuted the batch's rows.  Also exactly 0.000000:
# rows are independent and the loss reduction is order-stable.)
#
# **fp32 is the decisive run**, and the reason the bf16 bound below is credible
# rather than convenient.  It is also the one comparison with no circularity: the
# control puts pads at the front too, so a hypothetical bug in how front padding
# is handled would move the control and the measurement together — whereas
# left-vs-RIGHT in fp32 shrinks by four orders of magnitude if the difference is
# rounding and does not move at all if it is semantic.  `run_preflight.sbatch`
# stages 2e/2f run it.
CONTROL_BLOCK = 128
NOISE_FACTOR = 3.0
# bf16 carries 8 mantissa bits, i.e. 2^-8 = 0.39 % per rounding; a 26-layer
# network's output differing by ~1 % between two accumulation orders is ordinary.
# fp32's 24 bits make the same argument four orders of magnitude tighter, which
# is what turns stages 2e/2f into a real test.
REL_TOL = {"bf16": 0.01, "fp32": 1e-4}
# The whole point of the change.  Left-padded, the slice is `longest answer + 1`,
# and the longest answer is bounded by what the evaluator may generate -- an
# answer past that is a failure of its own -- so the same cap bounds the slice.
# Right-padded it ran to thousands.  (A fixed 400 here went stale when answers
# reached 467 tokens, with memory still ~20 GiB.)
MAX_LOGITS_TO_KEEP = MAX_NEW_TOKENS


def build_parser():
    d = RunConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default=d.model_name)
    p.add_argument("--data-root", default=d.data_root)
    p.add_argument("--items-root", default=d.items_root)
    p.add_argument("--impl", default=d.impl)
    p.add_argument("--split", default="train")
    p.add_argument("--max-items", type=int, default=512,
                   help="how much of the split to load (a spread, so the length "
                        "range is covered)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=d.max_length)
    p.add_argument("--dtype", choices=("fp32", "bf16"), default=d.dtype,
                   help="fp32 shrinks the numerical noise floor by orders of "
                        "magnitude, which turns this into a much sharper test "
                        "of the claim (at 4x the weight memory)")
    p.add_argument("--plain-llm", action="store_true", default=False)
    p.add_argument("--no-spd", dest="spd", action="store_false", default=d.spd)
    p.add_argument("--no-magnetic", dest="magnetic", action="store_false",
                   default=d.magnetic)
    return p


def make_batches(lengths, bs):
    """>= 8 index batches: short, long, and deliberately mixed.

    The mixed ones pair the shortest items in the slice with the longest, which
    is exactly the composition that made the right-padded slice blow up (one
    90-token `ni v bazi` ball next to a 6,000-token one) — so they are the case
    the roll has to survive, not an incidental extra.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    short, long = order[:3 * bs], order[-3 * bs:]
    batches = [("short", short[i * bs:(i + 1) * bs]) for i in range(3)]
    batches += [("long", long[i * bs:(i + 1) * bs]) for i in range(3)]
    half = max(1, bs // 2)
    for j in range(3):
        mixed = (short[j * half:(j + 1) * half]
                 + long[len(long) - (j + 1) * half:][:half])
        batches.append(("mixed", mixed))
    # The worst case the memory probe cared about: the longest items in the
    # slice, at the configured micro-batch size.
    batches.append(("worst", order[-bs:]))
    return [(tag, b) for tag, b in batches if b]


def control_batch(batch, pad_token_id, block=CONTROL_BLOCK):
    """The rounding-floor control (see the tolerance note above).

    Every row gets the SAME number of pad columns prepended — the largest shift
    left padding would apply to any row here — and enough appended to keep the
    total length block-aligned.  So: identical content, identical right-padded
    layout *between* rows, every absolute position moved as far as left padding
    moves them, every tile boundary redrawn.  `(control, front)`.

    `node_ids` takes each row's own prompt node, which is what the collator
    already writes into padded positions; `labels` takes -100 so the supervised
    token set is unchanged; `position_ids` takes 0, as the collator's pads do.
    """
    am = batch["attention_mask"]
    front = int((am.shape[1] - am.sum(dim=1)).max())
    if front == 0:
        return batch, 0
    back = (-front) % block
    out = dict(batch)
    B = batch["input_ids"].shape[0]
    fills = {"input_ids": pad_token_id, "attention_mask": 0, "labels": -100,
             "position_ids": 0}
    for key, fill in fills.items():
        if key in batch and torch.is_tensor(batch[key]):
            t = batch[key]
            out[key] = torch.cat([torch.full((B, front), fill, dtype=t.dtype), t,
                                  torch.full((B, back), fill, dtype=t.dtype)], dim=1)
    if "node_ids" in batch and torch.is_tensor(batch["node_ids"]):
        p = batch["prompt_node"].view(B, 1)
        out["node_ids"] = torch.cat([p.expand(B, front).clone(), batch["node_ids"],
                                     p.expand(B, back).clone()], dim=1)
    return out, front


def loss_and_slice(model, batch, device):
    """`(loss, logits_to_keep, peak GiB)` for one collated batch."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    sliced = answer_tail_inputs(batch)
    k = int(sliced.get("logits_to_keep", sliced["labels"].shape[1]))
    moved = {key: (v.to(device) if torch.is_tensor(v) else v)
             for key, v in sliced.items() if v is not None}
    out = model(**moved)
    loss = float(out.loss)
    del out
    peak = (torch.cuda.max_memory_allocated() / 2 ** 30
            if torch.cuda.is_available() else float("nan"))
    return loss, k, peak


def main(argv=None):
    a = build_parser().parse_args(argv)
    cfg = RunConfig(model_name=a.model_name, data_root=a.data_root,
                    items_root=a.items_root, impl=a.impl, dtype=a.dtype,
                    plain_llm=a.plain_llm, max_length=a.max_length,
                    max_items=a.max_items,
                    spd=a.spd and not a.plain_llm,
                    magnetic=a.magnetic and not a.plain_llm).validate()

    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    split = load_split(cfg, tokenizer, a.split, with_generation=False, spread=True)

    if cfg.plain_llm:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=cfg.torch_dtype(),
            attn_implementation="sdpa")
    else:
        config_cls, model_cls = cfg.gtlm_classes()
        config = config_cls.from_pretrained(
            cfg.model_name, **cfg.bias_params(), k_hop=cfg.k_hop,
            graph_attn_impl=cfg.backend(), **cfg.flex_params())
        model = model_cls.from_pretrained(
            cfg.model_name, config=config, graph_attn_impl=cfg.backend(),
            torch_dtype=cfg.torch_dtype())
    model.to(device).eval()

    right = GraphCollatorV2(
        tokenizer=tokenizer, k_hop=cfg.k_hop,
        magnetic_m=cfg.magnetic_m if cfg.magnetic else 0,
        pad_to_block=(cfg.backend() == "flex" and not cfg.plain_llm),
        max_spd=cfg.max_spd)
    if cfg.plain_llm:
        right = PlainCollator(right)
    left = LeftPadCollator(right)

    pad_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
              else tokenizer.eos_token_id)
    lengths = packed_lengths(split.ds)
    batches = make_batches(lengths, a.batch_size)
    print(f"\n[test] stack={cfg.stack()} impl={cfg.impl} dtype={cfg.dtype} "
          f"biases={cfg.arm()} items={len(split)} "
          f"packed lengths {min(lengths)}..{max(lengths)}")
    print(f"[test] {len(batches)} batches at micro-batch {a.batch_size}.  `noise` "
          f"is the same right-padded batch with `shift` pad columns prepended to "
          f"EVERY row (and enough appended to stay block-aligned) -- same "
          f"content, same layout between rows, absolute positions moved as far "
          f"as left padding moves them, all tiles redrawn -- so it measures the "
          f"{cfg.dtype} rounding floor this comparison must be read against.  "
          f"Bound: max({NOISE_FACTOR}x noise, {REL_TOL[cfg.dtype]:g} x |loss|)\n")

    header = (f"{'batch':>6} {'L':>7} {'shift':>6} {'k right':>9} {'k left':>8} "
              f"{'loss right':>11} {'loss left':>10} {'d l-r':>9} {'noise':>9} "
              f"{'GiB right':>10} {'GiB left':>9}")
    print(header)
    print("-" * len(header))

    failures = []
    ratios = []
    for tag, idxs in batches:
        rows = [split.ds[i] for i in idxs]
        with torch.no_grad():
            b_left = left(rows)
            l_left, k_left, p_left = loss_and_slice(model, b_left, device)
            L = b_left["input_ids"].shape[1]
            try:
                b_right = right(rows)
                l_right, k_right, p_right = loss_and_slice(model, b_right, device)
                # The control: same content and same between-row layout, every
                # absolute position moved and every tile boundary redrawn.  Any
                # difference here is rounding by construction.
                b_ctrl, ctrl_front = control_batch(b_right, pad_id)
                l_ctrl, _k, _p = loss_and_slice(model, b_ctrl, device)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                # Not a failure of the fix -- it IS the fix's justification.
                torch.cuda.empty_cache()
                print(f"{tag:>6} {L:>7} {'-':>6} {'OOM':>9} {k_left:>8} "
                      f"{'OOM':>11} {l_left:>10.4f} {'-':>9} {'-':>9} "
                      f"{'OOM':>10} {p_left:>9.1f}")
                if k_left > MAX_LOGITS_TO_KEEP:
                    failures.append(f"{tag}: left-pad logits_to_keep={k_left}")
                continue
        d = abs(l_left - l_right)
        noise = abs(l_ctrl - l_right)
        print(f"{tag:>6} {L:>7} {ctrl_front:>6} {k_right:>9} {k_left:>8} "
              f"{l_right:>11.4f} {l_left:>10.4f} {d:>9.6f} {noise:>9.6f} "
              f"{p_right:>10.1f} {p_left:>9.1f}")
        bound = max(NOISE_FACTOR * noise, REL_TOL[cfg.dtype] * abs(l_right))
        ratios.append(d / noise if noise else float("inf") if d else 0.0)
        if d > bound:
            failures.append(
                f"{tag}: loss moved by {d:.6f} — more than both {NOISE_FACTOR}x "
                f"the measured rounding floor ({noise:.6f}) and "
                f"{REL_TOL[cfg.dtype]:g} of the loss ({abs(l_right):.4f}).  That "
                f"is a change in what the model computed, not in the order it "
                f"accumulated.")
        if k_left > MAX_LOGITS_TO_KEEP:
            failures.append(f"{tag}: left-pad logits_to_keep={k_left} "
                            f"(> {MAX_LOGITS_TO_KEEP})")
        if k_left > k_right:
            failures.append(f"{tag}: left padding made the slice WIDER "
                            f"({k_right} -> {k_left})")

    finite = [r for r in ratios if r == r and r != float("inf")]
    if finite:
        print(f"\n|d(left-right)| / |d(noise)|: median {sorted(finite)[len(finite)//2]:.2f}, "
              f"max {max(finite):.2f}  (1.0 = indistinguishable from rounding)")
    print()
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        print(f"\nFAILED ({len(failures)} check(s))")
        return 1
    print(f"PASS ({cfg.dtype}): the loss difference stays inside the rounding "
          f"floor, and the answer-tail slice collapses under left padding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
