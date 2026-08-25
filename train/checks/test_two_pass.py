"""The licence for the fast path: two-pass == generate-everything, item by item.

    .venv/bin/python -m train.checks.test_two_pass --max-items 32
    .venv/bin/python -m train.checks.test_two_pass --max-items 32 --checkpoint <dir>

Pass 1 declares an item correct without generating anything, on the grounds that
a teacher-forced forward whose argmax equals the gold token at every answer
position would have decoded to exactly that string.  That is an argument, not a
measurement, and if it is wrong the whole evaluation is quietly wrong in the
direction of flattering the model.  So it is checked directly: the same slice is
scored twice, once through the two-pass evaluator and once by generating every
item, and the per-item `success` verdicts must be **identical**.

Two things make the comparison exact rather than approximate:

  * `max_batch=1`, so generation is padding-free and the two runs feed the model
    bitwise identical tensors.  With mixed batches a near-tie in bf16 can flip an
    argmax between two groupings of the same items, and the test would fail for
    a reason that has nothing to do with the claim.
  * greedy decoding, which is the equivalence's own precondition.

The second assertion is the sharper one: for every item pass 1 *passed*,
generation must have produced the gold answer string.  That is the claim itself,
tested where it bites.  It is vacuous on an untrained model (nothing passes pass
1), which is why `--checkpoint` exists -- run it once against a trained adapter
and the branch is actually exercised.

Exit code 0 on success, 1 on any mismatch.
"""
import argparse
import collections

import torch
from transformers import AutoTokenizer, set_seed

from gtlm.utils import GraphCollatorV2

from ..config import RunConfig
from ..data import load_split
from ..evaluate import GradeEvaluator
from ..run import LeftPadCollator
from ..qa_contract import contract


def build_parser():
    d = RunConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default=d.model_name)
    p.add_argument("--data-root", default=d.data_root)
    p.add_argument("--items-root", default=d.items_root)
    p.add_argument("--split", default="dev")
    p.add_argument("--max-items", type=int, default=32)
    p.add_argument("--max-length", type=int, default=d.max_length)
    p.add_argument("--types", default="")
    p.add_argument("--checkpoint", default=None,
                   help="a trained checkpoint, so the pass-1 branch is not vacuous")
    p.add_argument("--no-spread", dest="spread", action="store_false", default=True,
                   help="take the file's first N items instead of a spread -- use "
                        "this to score exactly the items a --max-items run trained on")
    p.add_argument("--no-spd", dest="spd", action="store_false", default=d.spd)
    p.add_argument("--no-magnetic", dest="magnetic", action="store_false",
                   default=d.magnetic)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    cfg = RunConfig(model_name=a.model_name, data_root=a.data_root,
                    items_root=a.items_root, types=a.types,
                    max_items=a.max_items, max_length=a.max_length,
                    spd=a.spd, magnetic=a.magnetic).validate()

    set_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # Spread across the file: the items are written type by type, so a prefix of
    # 32 would be 32 T1s -- all `sequence`, the one mode where pass 2 changes
    # nothing.  The spread puts multiset and membership items in the slice, which
    # is where the fast path's shortcut actually has to be right.
    split = load_split(cfg, tokenizer, a.split, with_generation=True, spread=a.spread)
    modes = collections.Counter(contract(it)["mode"] for it in split.items)
    print(f"[test] slice modes: {dict(modes)}  "
          f"negatives: {sum(it['negative'] for it in split.items)}")

    config_cls, model_cls = cfg.gtlm_classes()
    if a.checkpoint:
        model = model_cls.from_pretrained(
            a.checkpoint, graph_attn_impl=cfg.backend(),
            torch_dtype=cfg.torch_dtype())
    else:
        config = config_cls.from_pretrained(
            cfg.model_name, **cfg.bias_params(), k_hop=cfg.k_hop,
            graph_attn_impl=cfg.backend())
        model = model_cls.from_pretrained(
            cfg.model_name, config=config, graph_attn_impl=cfg.backend(),
            torch_dtype=cfg.torch_dtype())
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    # The same wrapper the training path uses, so what is verified here is what
    # actually runs (`to_left_padding` is idempotent, so pass 2 is unaffected).
    collator = LeftPadCollator(GraphCollatorV2(
        tokenizer=tokenizer, k_hop=cfg.k_hop,
        magnetic_m=cfg.magnetic_m if cfg.magnetic else 0,
        pad_to_block=(cfg.backend() == "flex"), max_spd=cfg.max_spd))
    ev = GradeEvaluator(tokenizer, collator, [split], max_batch=1)

    probe = list(range(min(8, len(split))))
    print(f"\n[test] decoding {len(probe)} items twice -- greedy or not?", flush=True)
    with torch.no_grad():
        first, _ = ev.pass2(model, split, probe)
        second, _ = ev.pass2(model, split, probe)
    nondet = [i for i in probe if first[i] != second[i]]
    if nondet:
        print(f"*** {len(nondet)} of {len(probe)} items decoded DIFFERENTLY twice: "
              f"decoding is not greedy, and pass 1's equivalence does not hold.")
        for i in nondet[:2]:
            print(f"    {split.items[i]['id']}\n      1) {first[i]!r}\n      2) {second[i]!r}")
        return 1
    print("    identical -- decoding is deterministic")

    print(f"\n[test] {len(split)} items from {a.split}; two-pass ...", flush=True)
    two_pass, _loss, n_gen, _tr, _t = ev.score(model, split.ds, fast=False)
    print(f"[test] generated {n_gen}/{len(two_pass)}; generate-everything ...",
          flush=True)
    every, _l2, _g2, _t2, _tm2 = ev.score(model, split.ds, generate_all=True)

    mismatches = [(t["id"], t["success"], e["success"], t["prediction"],
                   e["prediction"])
                  for t, e in zip(two_pass, every) if t["success"] != e["success"]]
    n_pass1 = sum(t["pass1"] for t in two_pass)
    gold_mismatch = [t["id"] for t, e in zip(two_pass, every)
                     if t["pass1"] and e["prediction"] is not None
                     and e["prediction"].strip() != t["gold"].strip()]

    acc = sum(r["success"] for r in two_pass) / max(1, len(two_pass))
    print(f"\nitems                         {len(two_pass)}")
    print(f"cleared by pass 1             {n_pass1}")
    print(f"two-pass accuracy             {acc:.4f}")
    print(f"verdicts that disagree        {len(mismatches)}")
    print(f"pass-1 items whose generation differed from gold  {len(gold_mismatch)}")
    for iid, t, e, tp, ep in mismatches[:5]:
        print(f"\n  {iid}: two-pass={t} generate-all={e}"
              f"\n    two-pass  {tp!r}\n    generate  {ep!r}")
    for iid in gold_mismatch[:5]:
        print(f"  pass-1 claimed {iid} but generation said otherwise")

    if n_pass1 == 0:
        print("\nNOTE: nothing cleared pass 1, so the pass-1 branch was not "
              "exercised.  Re-run with --checkpoint against a trained adapter.")
    bad = bool(mismatches or gold_mismatch)
    print("\nFAILED" if bad else "\nPASS: the two-pass evaluator and "
          "generate-everything agree on every item")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
