"""Evaluate with the dataset's OWN grading contract, and report it as `accuracy`.

`qa/grade.py` is the contract: three modes, one constant per type.  Thirteen
types are `sequence` (ordered, exact), five are `multiset` (**order-insensitive**
set equality) and T17 is `membership` (**any** subset of the anchor's complete
collocation set whose size the quantity band allows).  A token-level exact match
answers the wrong question for the last two: it penalises a correct multiset in
another order, and scores a model naming five perfectly valid collocations at
zero.  So the headline number here is

    accuracy = the proportion of items with `qa.grade.grade(...)["success"]`

reported overall AND per type, because a single aggregate over 19 heterogeneous
types hides everything interesting.  `f1` and the `reason` counts (`unparseable`,
`not_in_all`, `bad_count`, `mismatch`, `repeated_item`) come along as
diagnostics -- they are how a bad run is diagnosed without re-running it.

Two passes, because free-running generation over the whole dev split at every
`eval_steps` is the dominant cost of evaluation and most of it is avoidable.

  **Pass 1** -- one teacher-forced forward over `prompt + gold_answer`.  If the
  argmax at every answer position equals the gold token, greedy decoding would
  have emitted exactly that string, so the item succeeds.  This is *sufficient*
  in every mode, not just `sequence`: the gold answer trivially satisfies its own
  multiset equality, and for `membership` it is by construction a subset of
  `all_items` of a size the band allows.  Measured on this corpus it settles
  74.8 % of the dev split before a single token is generated, and the share grows
  as the model improves.

  **Pass 2** -- autoregressive greedy generation for the remainder, then
  `qa.grade.grade`.

Four conditions make the equivalence exact; all four are enforced here:

  * **greedy only** -- `do_sample=False`, no beams, no repetition/length penalty;
  * **the stop token is checked** -- the label span produced by
    `train/data.py`'s masker ends with the appended EOS, so "every answer
    position is argmax" already includes "and then it stops";
  * **tokenisation false negatives are safe** -- pass 1 compares against ONE
    tokenisation of the gold string, so a model that would emit a different token
    sequence decoding to the same text is a pass-1 *miss*; it falls through to
    pass 2 and is graded properly.  Pass 1 therefore never declares failure, only
    success;
  * **prefix alignment** -- answer positions come from the label mask the
    training loss itself uses (`OffsetLabelMasker`, keyed on `\\nODGOVOR:`), not
    from re-tokenising the answer separately.

`fast=True` (the intermediate evals) skips pass 2 for items a token mismatch
already condemns -- `sequence` mode and every negative, whose sentinel is
exact-match in all three modes -- and so gives a lower bound that is exact except
for the tokenisation false negatives above.  `fast=False` (the final evals, and
the test) runs pass 2 on every pass-1 miss and is the number that gets reported.
"""
import os
import json
import collections

import torch
from transformers import GenerationConfig

from .config import ANSWER_PREFIX
from .qa_contract import grade
from ._log import quiet_repeated_sliding_window_warning

# The band that decides how many items share one forward pass.  Padding is per
# batch, so a token budget (not a fixed batch size) is what actually bounds the
# activation memory when the corpus runs from 300 to 14,055 packed tokens.
EVAL_TOKEN_BUDGET = 16_384
GEN_TOKEN_BUDGET = 8_192
MAX_BATCH = 16

# Longest gold answer in the corpus is well under this (T5's 27 conjugated forms).
# A generation that runs past it is a failure anyway -- but it is counted and
# reported, so "the cap was too low" cannot hide inside the accuracy.
MAX_NEW_TOKENS = 320


def token_budget_batches(lengths, budget=EVAL_TOKEN_BUDGET, max_batch=MAX_BATCH):
    """Group indices into batches of `sum -> padded` cost at most `budget`.

    Sorted by length first, so a batch is homogeneous.  That matters twice over:
    it stops one 14k-token ball from padding three 300-token balls up to its own
    length, and it keeps `logits_to_keep` (below) tight -- with mixed lengths the
    slice needed to cover the earliest answer span in the batch is nearly the
    whole sequence, which is the memory blow-up this exists to avoid.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        nxt = max(cur_max, lengths[i])
        if cur and (nxt * (len(cur) + 1) > budget or len(cur) >= max_batch):
            batches.append(cur)
            cur, cur_max = [i], lengths[i]
        else:
            cur.append(i)
            cur_max = nxt
    if cur:
        batches.append(cur)
    return batches


def to_left_padding(batch):
    """Turn the collator's RIGHT padding into LEFT padding, in place.

    `generate` appends to the END of the sequence, so every row has to *end* at
    its own last real token; with right padding a short row would grow its
    continuation on the far side of its pads.  Rolling each row by its pad width
    moves the trailing pads to the front and changes nothing else: the packed
    node order is preserved, `attention_mask` still marks the pads (nothing
    attends to them), and `position_ids[:, -1]` is still the prompt node's last
    local position, which is what `prepare_inputs_for_generation` continues from.
    """
    am = batch["attention_mask"]
    shift = (am.shape[1] - am.sum(dim=1)).tolist()
    for i, s in enumerate(shift):
        if not s:
            continue
        for key in ("input_ids", "position_ids", "node_ids", "attention_mask"):
            batch[key][i] = torch.roll(batch[key][i], int(s), dims=0)
    return batch


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
        # is what lets `test_two_pass.py` assert equality rather than closeness.
        self.max_batch = max_batch
        self._lengths = {}
        self._gold_lens = {}
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
        self.greedy_kwargs = dict(do_sample=False, num_beams=1,
                                  use_model_defaults=False)
        self.gen_cfg = GenerationConfig(
            do_sample=False, num_beams=1,
            repetition_penalty=1.0, length_penalty=1.0,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                          else tokenizer.eos_token_id),
            eos_token_id=tokenizer.eos_token_id,
        )

    # ── plumbing ───────────────────────────────────────────────────────────
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
            ds = getattr(split, which)
            col = ds._hf_dataset.select_columns("input_ids")["input_ids"]
            self._lengths[key] = [sum(len(x) for x in r) for r in col]
        return self._lengths[key]

    @staticmethod
    def _move(batch, device):
        return {k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items() if v is not None}

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
        n = len(ds)
        ok = [False] * n
        loss_sum, loss_n = 0.0, 0
        for idxs in token_budget_batches(self.lengths(split, "ds"), self.eval_budget,
                                         self.max_batch):
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
                loss_sum += float(ce)
                loss_n += int(flat.sum())
            del out
        return ok, (loss_sum / loss_n if loss_n else float("nan"))

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
        preds, truncated = {}, 0
        for group in token_budget_batches(lengths, self.gen_budget, self.max_batch):
            rows = [idxs[g] for g in group]
            batch = to_left_padding(self.collator([ds[i] for i in rows]))
            batch.pop("labels", None)
            inputs = self._move(batch, device)
            prompt_len = inputs["input_ids"].shape[1]
            budget = min(self.max_new_tokens, 2 * max(gold[i] for i in rows) + 16)
            out = model.generate(**inputs, generation_config=self.gen_cfg,
                                 **self.greedy_kwargs,
                                 max_new_tokens=budget, use_cache=True)
            new = out[:, prompt_len:]
            for j, i in enumerate(rows):
                seq = new[j].tolist()
                if self.tok.eos_token_id not in seq:
                    truncated += 1
                text = self.tok.decode(seq, skip_special_tokens=True)
                # The marker itself was the last thing the model was shown, so it
                # is not in the continuation; the grader parses a full line.
                preds[i] = ANSWER_PREFIX.lstrip("\n") + text
        return preds, truncated

    # ── the join, and the report ───────────────────────────────────────────
    def score(self, model, ds, fast=True, generate_all=False):
        """`(per-item rows, loss, n_generated, n_truncated)` for one split.

        `generate_all=True` bypasses the fast path entirely: every item is
        decoded and graded, with pass 1 still run so its verdict can be compared
        against generation's.  It is the reference implementation the two-pass
        one is verified against (`train/test_two_pass.py`) and is far too slow to
        use for anything else."""
        split = self.split_of(ds)
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                ok, loss = self.pass1(model, split)
                if generate_all:
                    need = list(range(len(split.items)))
                else:
                    need = [i for i, good in enumerate(ok)
                            if not good and (not fast or split.needs_generation[i])]
                preds, truncated = self.pass2(model, split, need)
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
        return rows, loss, len(need), truncated

    def evaluate(self, model, ds, prefix="eval", fast=True, dump_tag=None,
                 generate_all=False):
        """`score`, reported as a metrics dict the Trainer can log and select on."""
        rows, loss, n_generated, truncated = self.score(
            model, ds, fast=fast, generate_all=generate_all)
        if self.dump_dir and dump_tag:
            self._dump(rows, dump_tag)
        return self._metrics(rows, prefix, loss, n_generated, truncated, fast)

    def _dump(self, rows, tag):
        os.makedirs(self.dump_dir, exist_ok=True)
        path = os.path.join(self.dump_dir, f"{tag}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[eval] per-item predictions -> {path}", flush=True)

    @staticmethod
    def _metrics(rows, prefix, loss, n_generated, truncated, fast):
        def pct(sel):
            sel = list(sel)
            return sum(r["success"] for r in sel) / len(sel) if sel else float("nan")

        m = {f"{prefix}_loss": loss,
             f"{prefix}_accuracy": pct(rows),
             f"{prefix}_f1": (sum(r["f1"] for r in rows) / len(rows)) if rows else 0.0,
             f"{prefix}_accuracy_positive": pct(r for r in rows if not r["negative"]),
             f"{prefix}_accuracy_negative": pct(r for r in rows if r["negative"]),
             f"{prefix}_pass1_share": (sum(r["pass1"] for r in rows) / len(rows)) if rows else 0.0,
             f"{prefix}_generated": n_generated,
             f"{prefix}_gen_truncated": truncated,
             f"{prefix}_fast": int(fast)}
        for t in sorted({r["type"] for r in rows}):
            m[f"{prefix}_accuracy_{t}"] = pct(r for r in rows if r["type"] == t)
        for reason, c in collections.Counter(r["reason"] for r in rows).items():
            m[f"{prefix}_reason_{reason}"] = c
        return m
