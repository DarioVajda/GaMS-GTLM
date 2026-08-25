"""Is the KV cache actually live in pass 2?  (T5e — measure, don't assume.)

    .venv/bin/python -m train.checks.probe_eval --n 32

Pass 2 decodes autoregressively, so without a KV cache every new token re-runs
attention over the whole prompt — at this corpus's lengths that is the single
largest avoidable cost in the evaluation, and evaluation is what makes an
eight-epoch run fit or not fit its wall clock.

The reason to check rather than assume: training enables gradient checkpointing,
and HF then sets `config.use_cache = False` and warns about it.  `generate(
use_cache=True)` is supposed to override that per call, but "supposed to" is not
a measurement, and the failure is silent — the numbers come out identical and
only the clock moves.  So the same items are decoded twice, once with the cache
and once without, and the ratio is printed.  A ratio near 1.0 means the cache is
NOT doing anything and the evaluator needs fixing.

The model is put in the state a mid-training evaluation actually finds it in
(`gradient_checkpointing_enable()`, `config.use_cache = False`) before either
timing, so this measures the real situation rather than a clean one.
"""
import time
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from gtlm.utils import GraphCollatorV2

from ..config import RunConfig
from ..data import load_split
from ..evaluate import GradeEvaluator, scaled_budgets
from ..run import LeftPadCollator, PlainCollator


def build_parser():
    d = RunConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default=d.model_name)
    p.add_argument("--data-root", default=d.data_root)
    p.add_argument("--items-root", default=d.items_root)
    p.add_argument("--impl", default=d.impl)
    p.add_argument("--split", default="dev")
    p.add_argument("--n", type=int, default=32, help="items to decode")
    p.add_argument("--max-length", type=int, default=d.max_length)
    p.add_argument("--plain-llm", action="store_true", default=False)
    p.add_argument("--no-spd", dest="spd", action="store_false", default=d.spd)
    p.add_argument("--no-magnetic", dest="magnetic", action="store_false",
                   default=d.magnetic)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    cfg = RunConfig(model_name=a.model_name, data_root=a.data_root,
                    items_root=a.items_root, impl=a.impl,
                    plain_llm=a.plain_llm, max_length=a.max_length,
                    max_items=a.n,
                    spd=a.spd and not a.plain_llm,
                    magnetic=a.magnetic and not a.plain_llm).validate()

    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    split = load_split(cfg, tokenizer, a.split, with_generation=True, spread=True)

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
    # The state a mid-training eval finds the model in.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    collator = GraphCollatorV2(
        tokenizer=tokenizer, k_hop=cfg.k_hop,
        magnetic_m=cfg.magnetic_m if cfg.magnetic else 0,
        pad_to_block=(cfg.backend() == "flex" and not cfg.plain_llm),
        max_spd=cfg.max_spd)
    if cfg.plain_llm:
        collator = PlainCollator(collator)
    collator = LeftPadCollator(collator)

    eval_budget, gen_budget, note = scaled_budgets(cfg.eval_token_budget,
                                                   cfg.gen_token_budget)
    print(f"[probe] {note}")
    ev = GradeEvaluator(tokenizer, collator, [split],
                        eval_budget=eval_budget, gen_budget=gen_budget,
                        max_batch=cfg.eval_max_batch)

    idxs = list(range(len(split)))

    def timed(use_cache):
        # Both the generation config's flag and the per-call kwarg `pass2`
        # passes; the kwarg is applied last, so setting only one proves nothing.
        ev.use_cache = use_cache
        ev.gen_cfg.use_cache = use_cache
        # `pass2` reads a per-model COPY of `gen_cfg` (it carries the model's own
        # `cache_implementation` over); drop it so the flag above is picked up.
        ev._gen_cfg_cache.clear()
        with torch.no_grad():
            ev.pass2(model, split, idxs[:1])              # warm up / compile
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            preds, _ = ev.pass2(model, split, idxs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.perf_counter() - t0, preds

    on_s, on_preds = timed(True)
    print(f"\nitems decoded                {len(idxs)}")
    print(f"with    KV cache             {on_s:7.2f} s")
    try:
        off_s, off_preds = timed(False)
    except ValueError as exc:
        # On the flex backend the no-cache arm cannot run AT ALL, and that is
        # itself the answer.  `causal_lm.forward` takes the flex path only when
        # `q_len == kv_len`; a cached decode step feeds one token against a
        # cached prefix (q_len=1 < kv_len) and goes down the dense path, while an
        # UNcached step re-feeds the whole grown sequence (q_len == kv_len ==
        # prompt+1), which is no longer block-aligned and raises.  So if the
        # cache were not live, the run above would have raised the same error.
        ev.use_cache = True
        if "multiple of the block size" not in str(exc):
            raise
        print("without KV cache               n/a  -- it cannot run: "
              f"{str(exc).splitlines()[0][:90]}")
        print("\nPASS: the KV cache is live in pass 2.  The uncached path fails "
              "the flex block-alignment check because it re-feeds the whole "
              "grown sequence; the cached run above did not, so what executed "
              "was the q_len=1 cached decode.")
        return 0
    ev.use_cache = True

    same = sum(on_preds[i] == off_preds[i] for i in idxs)
    ratio = off_s / on_s if on_s else float("nan")
    print(f"without KV cache             {off_s:7.2f} s")
    print(f"speedup from the cache       {ratio:7.2f}x")
    print(f"identical predictions        {same}/{len(idxs)}")
    if ratio < 1.2:
        print("\nWARNING: the cache buys almost nothing, which means it is not "
              "live in pass 2.  Fix that before pricing the sweep -- it is the "
              "single largest available win in evaluation.")
        return 1
    print("\nPASS: the KV cache is live in pass 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
