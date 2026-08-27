"""Evaluate with the dataset's OWN grading contract, and report it as `accuracy`.

`qa/grade.py` is the contract: three modes, one constant per type.  Thirteen
types are `sequence` (ordered, exact), five are `multiset` (order-insensitive set
equality) and T17 is `membership` (any subset of the anchor's collocation set
whose size the quantity band allows).  Token-level exact match answers the wrong
question for the last two, so the headline number is

    accuracy = the proportion of items with `qa.grade.grade(...)["success"]`

reported overall AND per type, because a single aggregate over 19 heterogeneous
types hides everything interesting.  `f1` and the `reason` counts (`unparseable`,
`not_in_all`, `bad_count`, `mismatch`, `repeated_item`) come along as
diagnostics, so a bad run can be read without re-running it.

Two passes, because free-running generation over the whole dev split at every
`eval_steps` is the dominant cost of evaluation and most of it is avoidable.

  **Pass 1** -- one teacher-forced forward over `prompt + gold_answer`.  If the
  argmax at every answer position equals the gold token, greedy decoding would
  have emitted exactly that string, so the item succeeds.  This is sufficient in
  every mode: the gold answer trivially satisfies its own multiset equality, and
  for `membership` it is by construction a subset of `all_items` of an allowed
  size.  It settles ~75 % of the dev split before a token is generated, and the
  share grows as the model improves.

  **Pass 2** -- autoregressive greedy generation for the remainder, then
  `qa.grade.grade`.

Four conditions make the equivalence exact; all four are enforced here:

  * **greedy only** -- `do_sample=False`, no beams, no repetition/length penalty;
  * **the stop token is checked** -- the label span ends with the
    `<end_of_turn>` the chat template closes the model turn with, so "every
    answer position is argmax" already includes "and then it stops".  Generation
    stops on that token and on `<eos>`;
  * **tokenisation false negatives are safe** -- pass 1 compares against ONE
    tokenisation of gold, so a model emitting a different token sequence that
    decodes to the same text is a pass-1 MISS and falls through to pass 2.  Pass
    1 therefore never declares failure, only success;
  * **prefix alignment** -- answer positions come from the label mask the
    training loss itself uses, not from re-tokenising the answer separately.

`fast=True` (the intermediate evals) skips pass 2 for items a token mismatch
already condemns -- `sequence` mode and every negative -- giving a lower bound.
`fast=False` (the final evals and the test) runs pass 2 on every pass-1 miss and
is the number that gets reported.
"""
import os
import json
import time
import collections

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import GenerationConfig

from .config import ANSWER_PREFIX
from .chat import stop_token_ids
from .batching import to_left_padding
from .qa_contract import grade
from ._log import quiet_repeated_sliding_window_warning
from .batching import packed_lengths

# The band that decides how many items share one forward pass.  Padding is per
# batch, so a token budget (not a fixed batch size) is what actually bounds the
# activation memory when the corpus runs from 300 to 16,384 packed tokens.
#
# These are the values chosen for and measured on **A100-80GB**; `scaled_budgets`
# rescales them by the device's own memory so the same config is correct on a
# bigger or smaller card.  See there.
EVAL_TOKEN_BUDGET = 16_384
GEN_TOKEN_BUDGET = 8_192
MAX_BATCH = 16

# The card the two budgets above were chosen on, in GiB as `torch` reports it
# (`get_device_properties(0).total_memory`), which for an "80 GB" A100 is ~79.
REFERENCE_TOTAL_GIB = 79.0
EVAL_BUDGET_CLAMP = (8_192, 131_072)
GEN_BUDGET_CLAMP = (4_096, 65_536)

# Longest gold answer in the corpus is well under this (T5's 27 conjugated forms).
# A generation that runs past it is a failure anyway -- but it is counted and
# reported, so "the cap was too low" cannot hide inside the accuracy.
MAX_NEW_TOKENS = 320

# `torch.OutOfMemoryError` is the modern spelling and `torch.cuda.OutOfMemoryError`
# its alias; both subclass RuntimeError.  Some kernels (and cuBLAS) still raise a
# plain RuntimeError whose message is the only evidence, so the guard catches
# RuntimeError and re-raises anything `_is_oom` does not recognise -- retrying a
# real bug at half the batch size would just hide it.
_TORCH_OOM = tuple({e for e in (getattr(torch, "OutOfMemoryError", None),
                                getattr(torch.cuda, "OutOfMemoryError", None))
                    if e is not None})


def _is_oom(exc):
    if _TORCH_OOM and isinstance(exc, _TORCH_OOM):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


# SDPA backends allowed during pass-2 GENERATION.  `EFFICIENT_ATTENTION` (the
# cutlass kernel) requires the attention-bias pointer to be 16-byte aligned, and
# stock Gemma-3's sliding layers hand it a non-contiguous VIEW:
# `attention_mask[:, :, :, offset : offset + 512]` with
# `offset = max(0, kv_len - sliding_window)`.  During decode `offset` grows by one
# per step, so seven steps in eight the pointer is misaligned and the kernel
# raises `p.attn_bias_ptr is not correctly aligned`.
#
# Prefill is safe (there `offset == 0`), so this is a decode-only kernel-dispatch
# problem, and at decode `q_len == 1` — the math backend materialises a
# `(B, H, 1, kv)` score matrix, i.e. costs nothing.  Restricting the backend
# changes which kernel computes the attention, never what it computes, and it
# touches the plain stack only: the GTLM path does not go through SDPA at all.
# TRAINING is untouched and keeps the fused kernels, which is what arm 4's
# "what a standard LLM actually does" claim rests on.
_GEN_SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]


def _round_down(value, multiple=1024):
    return max(multiple, (int(value) // multiple) * multiple)


def scaled_budgets(eval_budget=EVAL_TOKEN_BUDGET, gen_budget=GEN_TOKEN_BUDGET,
                   device_index=0):
    """Rescale the A100-80GB reference budgets by THIS device's memory.

    `(eval_budget, gen_budget, note)`.  A constant token budget is a constant
    only in tokens: the activation memory it buys is a fixed fraction of an
    80 GiB card and less than half of that on a 178 GiB B200, which is a
    silent 2x throughput loss on the bigger machine and an OOM on a smaller one.
    Scaling linearly from the reference keeps the code GPU-agnostic instead of
    B200-specific -- an A100-80GB reproduces today's 16,384 / 8,192 exactly.

    Clamped at both ends, because the linear model is an approximation: the
    weights, the optimizer state and the KV cache do not scale with the budget,
    so the floor keeps a small card able to run at all and the ceiling stops a
    very large one from asking for a batch whose *logits* would not fit.
    """
    if not torch.cuda.is_available():
        return eval_budget, gen_budget, "cpu: budgets unscaled"
    total_gib = torch.cuda.get_device_properties(device_index).total_memory / 2 ** 30
    scale = total_gib / REFERENCE_TOTAL_GIB
    ev = _round_down(min(max(eval_budget * scale, EVAL_BUDGET_CLAMP[0]),
                         EVAL_BUDGET_CLAMP[1]))
    gen = _round_down(min(max(gen_budget * scale, GEN_BUDGET_CLAMP[0]),
                          GEN_BUDGET_CLAMP[1]))
    note = (f"{torch.cuda.get_device_name(device_index)} {total_gib:.0f} GiB "
            f"-> x{scale:.2f}: eval {eval_budget}->{ev}, gen {gen_budget}->{gen}")
    return ev, gen, note


def token_budget_batches(lengths, budget=EVAL_TOKEN_BUDGET, max_batch=MAX_BATCH,
                         group_keys=None):
    """Group indices into batches of `sum -> padded` cost at most `budget`.

    Sorted by length first, so a batch is homogeneous.  That matters twice over:
    it stops one 16k-token ball from padding three 300-token balls up to its own
    length, and it keeps `logits_to_keep` (below) tight -- with mixed lengths the
    slice needed to cover the earliest answer span in the batch is nearly the
    whole sequence, which is the memory blow-up this exists to avoid.

    `group_keys` (one hashable per item) adds a HARD partition: no batch spans
    two keys.  Pass 2 uses it to group by gold-answer length as well as by input
    length, because `max_new_tokens` is a batch-level bound and generation runs
    until every row stops -- one long-gold item otherwise makes the whole batch
    decode long.
    """
    if group_keys is None:
        group_keys = [0] * len(lengths)
    order = sorted(range(len(lengths)),
                   key=lambda i: (group_keys[i], lengths[i]))
    batches, cur, cur_max, cur_key = [], [], 0, None
    for i in order:
        nxt = max(cur_max, lengths[i])
        if cur and (group_keys[i] != cur_key
                    or nxt * (len(cur) + 1) > budget
                    or len(cur) >= max_batch):
            batches.append(cur)
            cur, cur_max, cur_key = [i], lengths[i], group_keys[i]
        else:
            cur.append(i)
            cur_max, cur_key = nxt, group_keys[i]
    if cur:
        batches.append(cur)
    return batches


def gold_length_key(n):
    """Coarse band for a gold-answer token count: the next power of two.

    `max_new_tokens = min(320, 2 * max(gold in batch) + 16)`, so grouping golds
    into power-of-two bands bounds the decode waste inside a batch at ~2x while
    keeping the number of groups small enough not to fragment the batching.
    """
    return 1 << max(0, int(n) - 1).bit_length()


class GradeEvaluator:
    """Score a split through `qa.grade`, two passes, and dump the predictions."""

    def __init__(self, tokenizer, collator, splits, dump_dir=None,
                 max_new_tokens=MAX_NEW_TOKENS,
                 eval_budget=EVAL_TOKEN_BUDGET, gen_budget=GEN_TOKEN_BUDGET,
                 max_batch=MAX_BATCH):
        self.tok = tokenizer
        self.collator = collator
        self.splits = {s.name: s for s in splits}
        self._by_ds = {id(s.ds): s for s in splits}
        self.dump_dir = dump_dir
        self.max_new_tokens = max_new_tokens
        self.eval_budget = eval_budget
        self.gen_budget = gen_budget
        # `max_batch=1` makes generation padding-free, and therefore bitwise
        # reproducible across two different groupings of the same items -- which
        # is what lets `check_two_pass.py` assert equality rather than closeness.
        self.max_batch = max_batch
        self._lengths = {}
        self._gold_lens = {}
        # OOM fallbacks, reset per `score()` call.  Reported in the metrics dict
        # so a final eval that needed one is visible in the run record: splitting
        # a batch changes its padding, and a bf16 near-tie can flip an argmax
        # between two groupings, so the affected number is not reproducible from
        # the config alone.
        self.oom_splits = 0
        # Always True in a real run; `probe_eval.py` flips it to time the
        # difference.  See `pass2`.
        self.use_cache = True
        quiet_repeated_sliding_window_warning()
        # Greedy, spelled out, and **pinned against the model's own defaults**.
        #
        # gemma-3-1b-it ships `do_sample=True`, `top_p=0.95`, `top_k=64` in its
        # generation_config.  Since transformers 4.50, passing a custom
        # `GenerationConfig` does NOT stop those from applying: any field still
        # equal to the library default is silently back-filled from the model's
        # config (`_prepare_generation_config`, `use_model_defaults`), and
        # `do_sample=False` IS the library default -- so an explicitly greedy
        # config decodes by *sampling*.  Caught by the smoke run, where the
        # warning naming `{'do_sample': True, ...}` is the only visible trace.
        #
        # `use_model_defaults=False` turns that back-fill off, and the flags are
        # passed as generate() kwargs as well, which are applied last of all.
        # This is not belt-and-braces pedantry: pass 1 declares an item correct
        # without generating it, on the argument that GREEDY decoding would have
        # emitted the gold string.  Under sampling that argument is simply false,
        # and every number the evaluator reports would be unsound.
        #
        # The cost of turning that back-fill off, and why it is paid explicitly:
        # `use_model_defaults=False` suppresses EVERY field, including
        # `cache_implementation`, which is not a decoding policy but an
        # architectural fact.  gemma-3-1b-it ships `cache_implementation:
        # "hybrid"`; without it `generate` builds a plain `DynamicCache`, whose
        # sliding layers keep growing while Gemma-3's sliding-layer mask is
        # sliced to `max(1, sliding_window) = 512` -- so the first decode step
        # past a ~512-token prompt dies with "The expanded size of the tensor
        # (594) must match the existing size (512) at non-singleton dimension 3".
        # That is invisible on the GTLM stack (which drops the window) and on
        # short prompts, and fatal on the plain serialised arm, whose prompts are
        # p50 2,166 tokens.  Caught by `run_preflight.sbatch` stage 4b.
        #
        # So the field is copied from the model itself in `gen_cfg_for` rather
        # than the blanket back-fill being re-enabled -- re-enabling it would
        # bring `do_sample=True` back with it and make every number unsound.
        self.greedy_kwargs = dict(do_sample=False, num_beams=1,
                                  use_model_defaults=False)
        self._gen_cfg_cache = {}
        # `<end_of_turn>` first, `<eos>` behind it -- NOT `tokenizer.eos_token_id`
        # alone.  The prompt is a chat turn now (`train/chat.py`) and the answer
        # the model is trained to produce ends by closing that turn; stopping only
        # on `<eos>` would let every generation run past its own ending to the
        # `max_new_tokens` cap, and pass 2 would grade a trailing hallucination
        # that pass 1 declared a clean stop.
        self.stop_ids = stop_token_ids(tokenizer)
        self.gen_cfg = GenerationConfig(
            do_sample=False, num_beams=1,
            repetition_penalty=1.0, length_penalty=1.0,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                          else self.stop_ids[0]),
            eos_token_id=self.stop_ids,
        )

    # ── plumbing ───────────────────────────────────────────────────────────
    def gen_cfg_for(self, model):
        """`self.gen_cfg` plus the ARCHITECTURAL fields the model itself names.

        Only `cache_implementation` today, and only because it is a property of
        the architecture rather than a decoding choice -- see the note in
        `__init__` for the failure it prevents.  Everything that decides *how*
        the model decodes stays pinned here, because pass 1's correctness
        argument depends on decoding being greedy.

        Read from `model.config`, and NOT from `model.generation_config`, because
        the model config is where both stacks state their requirement
        deliberately: stock Gemma-3 ships `"cache_implementation": "hybrid"` in
        `config.json`, and `GTLMGemma3ForCausalLM._sanitize_attn_config` sets it
        to `None` on purpose — a HybridCache's sliding layers would truncate the
        KV span the adapter's attention still expects to see, and would fix the
        cache length at the prefill size, which its decode path cannot work
        against.  The hub's `generation_config.json` says `"hybrid"` for both, so
        reading that would hand the GTLM path a cache it must not have.
        """
        key = id(model)
        if key in self._gen_cfg_cache:
            return self._gen_cfg_cache[key]
        cfg = self.gen_cfg
        impl = getattr(getattr(model, "config", None), "cache_implementation", None)
        if impl and impl != cfg.cache_implementation:
            cfg = GenerationConfig(**{**cfg.to_dict(), "cache_implementation": impl})
            print(f"[eval] generation cache_implementation={impl!r} "
                  f"(from the model; `use_model_defaults=False` suppresses it)",
                  flush=True)
        self._gen_cfg_cache[key] = cfg
        return cfg

    def split_of(self, ds):
        s = self._by_ds.get(id(ds))
        if s is None:
            raise KeyError(
                "evaluate() was handed a dataset this evaluator does not know; "
                "the grading contract is joined by split, so an unregistered "
                "split cannot be scored.")
        return s

    def gold_lens(self, split):
        """Token length of each item's gold answer, cached per split.

        Used to bound generation: a *correct* answer is gold's own content, in
        another order at most, so `2 x gold + 16` tokens cannot truncate a
        success in any of the three modes -- `sequence` and `multiset` demand the
        same items, and `membership`'s uncapped band still starts from a gold of
        fifteen phrases.  What it does cut short is the untrained model's
        babbling, which otherwise runs to the hard cap on every single item and
        makes an in-training evaluation cost more than the training does.
        """
        if split.name not in self._gold_lens:
            self._gold_lens[split.name] = [
                len(self.tok(it["answer"], add_special_tokens=False)["input_ids"])
                for it in split.items]
        return self._gold_lens[split.name]

    def lengths(self, split, which="ds"):
        key = (split.name, which)
        if key not in self._lengths:
            self._lengths[key] = packed_lengths(getattr(split, which))
        return self._lengths[key]

    @staticmethod
    def _move(batch, device):
        return {k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items() if v is not None}

    def _with_oom_retry(self, rows, work, what):
        """Run `work(rows)`; on CUDA OOM, halve the batch and retry, down to 1.

        A token budget sized from the device (`scaled_budgets`) is an estimate,
        and an estimate that is occasionally wrong must not kill an eight-epoch
        run at hour nine.  Bounded (log2 of the batch size), and every fallback
        is logged and counted -- re-grouping changes the padding, and a bf16
        near-tie can flip an argmax between two groupings, so a fallback during a
        FINAL eval means the reported number depends on a grouping that was not
        pre-declared.
        """
        try:
            return work(rows)
        except RuntimeError as exc:
            if not _is_oom(exc) or len(rows) <= 1:
                raise
            torch.cuda.empty_cache()
            half = len(rows) // 2
            self.oom_splits += 1
            print(f"[eval] OOM in {what} at batch {len(rows)}; splitting into "
                  f"{half} + {len(rows) - half} and retrying ({exc.__class__.__name__})",
                  flush=True)
            self._with_oom_retry(rows[:half], work, what)
            self._with_oom_retry(rows[half:], work, what)
            return None

    # ── pass 1: teacher forcing ────────────────────────────────────────────
    def pass1(self, model, split):
        """(per-item `argmax == gold` bool, mean answer-token cross-entropy).

        Only the tail of the logits is materialised.  The answer span sits at the
        very end of the packed sequence (the prompt node is packed last), so
        `logits_to_keep=k` covers every supervised position with k in the low
        hundreds instead of the full sequence -- at a 262 k vocabulary that is the
        difference between 0.3 GB and 30 GB for one batch.
        """
        device = next(model.parameters()).device
        ds = split.ds
        ok = [False] * len(ds)
        acc = {"loss_sum": 0.0, "loss_n": 0}

        def run(idxs):
            batch = self.collator([ds[i] for i in idxs])
            labels = batch.pop("labels")
            L = labels.shape[1]
            sup = labels != -100
            has = sup.any(dim=1)
            if not bool(has.any()):
                raise ValueError("a batch with no supervised answer token")
            first = int(sup.float().argmax(dim=1)[has].min())
            # One position before the earliest answer token -- clamped, because a
            # `logits_to_keep` past the sequence length is silently truncated by
            # the slice and would leave `lab` misaligned by one.
            k = min(L, L - first + 1)
            out = model(**self._move(batch, device), logits_to_keep=k)
            # The same shift `ForCausalLMLoss` applies: logits[:, j] predicts the
            # label one position later.  The slice starts one position BEFORE the
            # earliest answer token in the batch, so every supervised position
            # has its predictor inside it.
            lab = labels[:, L - k:].to(device)                    # (B, k)
            gold = torch.nn.functional.pad(lab, (0, 1), value=-100)[:, 1:]
            mask = gold != -100
            hit = ((out.logits.argmax(dim=-1) == gold) | ~mask).all(dim=1)
            for j, i in enumerate(idxs):
                ok[i] = bool(hit[j])
            flat = mask.reshape(-1)
            if bool(flat.any()):
                # Index the CONTIGUOUS (B, k, V) tensor and only then upcast, so
                # the fp32 copy is (answer tokens x V) rather than (B x k x V).
                sel = out.logits.reshape(-1, out.logits.shape[-1])[flat]
                ce = torch.nn.functional.cross_entropy(
                    sel.float(), gold.reshape(-1)[flat], reduction="sum")
                acc["loss_sum"] += float(ce)
                acc["loss_n"] += int(flat.sum())
            del out

        for idxs in token_budget_batches(self.lengths(split, "ds"), self.eval_budget,
                                         self.max_batch):
            self._with_oom_retry(idxs, run, "pass 1")
        return ok, (acc["loss_sum"] / acc["loss_n"] if acc["loss_n"] else float("nan"))

    # ── pass 2: greedy generation ──────────────────────────────────────────
    def pass2(self, model, split, idxs):
        """{dataset index -> predicted answer string} for `idxs`, greedily decoded."""
        if not idxs:
            return {}, 0
        device = next(model.parameters()).device
        ds = split.gen_ds
        all_lengths = self.lengths(split, "gen_ds")
        lengths = [all_lengths[i] for i in idxs]
        gold = self.gold_lens(split)
        # Group by gold-answer band as well as by input length: `max_new_tokens`
        # below is a batch-level bound and generation runs until EVERY row stops,
        # so one long-gold item makes the whole batch decode long.
        keys = [gold_length_key(gold[i]) for i in idxs]
        gen_cfg = self.gen_cfg_for(model)
        preds, state = {}, {"truncated": 0}

        def run(rows):
            batch = to_left_padding(self.collator([ds[i] for i in rows]))
            batch.pop("labels", None)
            inputs = self._move(batch, device)
            prompt_len = inputs["input_ids"].shape[1]
            budget = min(self.max_new_tokens, 2 * max(gold[i] for i in rows) + 16)
            # `use_cache` is an attribute rather than a literal so
            # `train/checks/probe_eval.py` can time the same call with it off -- HF sets
            # `config.use_cache = False` under gradient checkpointing and the
            # per-call override is exactly the thing that needs measuring.
            with sdpa_kernel(_GEN_SDPA_BACKENDS):
                out = model.generate(**inputs, generation_config=gen_cfg,
                                     **self.greedy_kwargs, max_new_tokens=budget,
                                     use_cache=self.use_cache)
            new = out[:, prompt_len:]
            for j, i in enumerate(rows):
                seq = new[j].tolist()
                if not any(t in seq for t in self.stop_ids):
                    state["truncated"] += 1
                text = self.tok.decode(seq, skip_special_tokens=True)
                # The marker itself was the last thing the model was shown, so it
                # is not in the continuation; the grader parses a full line.
                preds[i] = ANSWER_PREFIX.lstrip("\n") + text

        for group in token_budget_batches(lengths, self.gen_budget, self.max_batch,
                                          group_keys=keys):
            self._with_oom_retry([idxs[g] for g in group], run, "pass 2")
        return preds, state["truncated"]

    # ── the join, and the report ───────────────────────────────────────────
    def score(self, model, ds, fast=True, generate_all=False):
        """`(per-item rows, loss, n_generated, n_truncated)` for one split.

        `generate_all=True` bypasses the fast path entirely: every item is
        decoded and graded, with pass 1 still run so its verdict can be compared
        against generation's.  It is the reference implementation the two-pass
        one is verified against (`train/checks/check_two_pass.py`) and is far too slow to
        use for anything else."""
        split = self.split_of(ds)
        was_training = model.training
        model.eval()
        self.oom_splits = 0
        timing = {}
        try:
            with torch.no_grad():
                t0 = time.perf_counter()
                ok, loss = self.pass1(model, split)
                timing["pass1_s"] = time.perf_counter() - t0
                if generate_all:
                    need = list(range(len(split.items)))
                else:
                    need = [i for i, good in enumerate(ok)
                            if not good and (not fast or split.needs_generation[i])]
                t1 = time.perf_counter()
                preds, truncated = self.pass2(model, split, need)
                timing["pass2_s"] = time.perf_counter() - t1
        finally:
            if was_training:
                model.train()

        rows = []
        for i, item in enumerate(split.items):
            if ok[i] and not generate_all:
                r = {"success": True, "f1": 1.0, "reason": "ok", "pass1": True,
                     "prediction": item["answer"]}
            elif i in preds:
                r = dict(grade(item, preds[i]), pass1=ok[i], prediction=preds[i])
            else:
                # fast mode, and this item's mode admits no non-exact success.
                r = {"success": False, "f1": 0.0, "reason": "token_mismatch",
                     "pass1": False, "prediction": None}
            r.update(id=item["id"], type=item["type"], band=item["band"],
                     negative=bool(item.get("negative")), gold=item["answer"])
            rows.append(r)
        timing["oom_splits"] = self.oom_splits
        print(f"[eval] {split.name}: pass1 {timing['pass1_s']:.1f} s, "
              f"pass2 {timing['pass2_s']:.1f} s over {len(need)} items"
              + (f", {self.oom_splits} OOM fallback(s)" if self.oom_splits else ""),
              flush=True)
        return rows, loss, len(need), truncated, timing

    def evaluate(self, model, ds, prefix="eval", fast=True, dump_tag=None,
                 generate_all=False):
        """`score`, reported as a metrics dict the Trainer can log and select on."""
        rows, loss, n_generated, truncated, timing = self.score(
            model, ds, fast=fast, generate_all=generate_all)
        if self.dump_dir and dump_tag:
            self._dump(rows, dump_tag)
        return self._metrics(rows, prefix, loss, n_generated, truncated, fast,
                             timing)

    def _dump(self, rows, tag):
        os.makedirs(self.dump_dir, exist_ok=True)
        path = os.path.join(self.dump_dir, f"{tag}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[eval] per-item predictions -> {path}", flush=True)

    @staticmethod
    def _metrics(rows, prefix, loss, n_generated, truncated, fast, timing=None):
        def pct(sel):
            sel = list(sel)
            return sum(r["success"] for r in sel) / len(sel) if sel else float("nan")

        timing = timing or {}
        m = {f"{prefix}_loss": loss,
             f"{prefix}_accuracy": pct(rows),
             f"{prefix}_f1": (sum(r["f1"] for r in rows) / len(rows)) if rows else 0.0,
             f"{prefix}_accuracy_positive": pct(r for r in rows if not r["negative"]),
             f"{prefix}_accuracy_negative": pct(r for r in rows if r["negative"]),
             f"{prefix}_pass1_share": (sum(r["pass1"] for r in rows) / len(rows)) if rows else 0.0,
             f"{prefix}_generated": n_generated,
             f"{prefix}_gen_truncated": truncated,
             # The three below are what makes the next round of eval tuning
             # data-driven instead of a guess (T5f), and `oom_splits` is the
             # audit trail for the batch-splitting fallback.
             f"{prefix}_pass1_s": round(timing.get("pass1_s", float("nan")), 2),
             f"{prefix}_pass2_s": round(timing.get("pass2_s", float("nan")), 2),
             f"{prefix}_oom_splits": timing.get("oom_splits", 0),
             f"{prefix}_fast": int(fast)}
        for t in sorted({r["type"] for r in rows}):
            m[f"{prefix}_accuracy_{t}"] = pct(r for r in rows if r["type"] == t)
        for reason, c in collections.Counter(r["reason"] for r in rows).items():
            m[f"{prefix}_reason_{reason}"] = c
        return m
