"""Train ONE configuration and log ONE record.

Same protocol as `graph_model`'s `our_tests`: train, evaluate + checkpoint every
`eval_steps`, reload the best-validation checkpoint (adapter AND graph-bias
tensors — see `gtlm.utils.text_graph_trainer_v2._load_best_model`), then report
that checkpoint's held-out score.

Two structural differences from `our_tests`.

**The model classes come from `cfg.gtlm_classes()`** rather than being imported at
module scope, because this experiment runs a *gemma-3* backbone and `our_tests`
imports the Llama classes directly.

**Evaluation goes through the dataset's own grader.**  `gtlm.utils`'s
`make_compute_metrics` reports token-level exact match over teacher-forced
predictions, which is the wrong question for six of the nineteen types: five are
`multiset` (order-insensitive) and T17 is `membership` (any subset of the
anchor's collocation set whose size the band allows), so EM penalises correct
answers in another order and scores a model naming five perfectly valid
collocations at zero.  `train/evaluate.py` implements the real contract and
reports it as `accuracy`; `METRIC` selects the best checkpoint on that same
number, so the run cannot train against one objective and select against
another.  Everything else — the optimizer split (`bias_lr` for the graph-bias
parameters, `lr` for LoRA), the cosine-with-min-lr schedule, the 10 % warmup —
is kept identical, so a number from here is comparable with one from there.
"""
import os
import json
import collections

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainingArguments, set_seed

from gtlm.utils import GraphCollatorV2, GraphTrainerV2, set_wandb_project
from gtlm.train import (select_active_params, print_trainable_parameters,
                        get_device)

from .config import EXPERIMENT_NAME, CHECKPOINT_ROOT
from .data import load_data
from .evaluate import GradeEvaluator
from .batching import TokenBudgetBatchSampler, packed_lengths
from ._io import append_jsonl

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_RUNS_JSONL = os.path.join(EXPERIMENT_DIR, "results", "train_runs.jsonl")
_DEFAULT_PRED_DIR = os.path.join(EXPERIMENT_DIR, "results", "predictions")

ACTIVE_PARAMS = ["graph_bias"]
# The grade-based accuracy from evaluate.py — NOT the token-level EM the shared
# stack computes.  This string is what the Trainer uses to pick the checkpoint to
# reload at the end of a run, and on the multiset and membership types the two
# objectives actively disagree: EM would select the checkpoint that best
# reproduces gold *ordering* rather than the one that answers most questions.
METRIC = "eval_accuracy"


class GradeTrainer(GraphTrainerV2):
    """`GraphTrainerV2`, with `evaluate()` replaced by the two-pass grader.

    Not a `compute_metrics` hook, for two reasons.  The contract needs the item's
    `grading` block and its id, which the `(preds, labels)` pair HF hands to
    `compute_metrics` does not carry; and pass 2 has to *generate*, which is not
    something a metric function is given a model to do.  Overriding `evaluate` is
    also what keeps the logits small — the shared evaluation loop materialises
    `(B, L, 262144)` logits for the whole packed sequence, which at this corpus's
    p99 of 5,605 tokens is tens of gigabytes per batch; `evaluate.py` asks the
    model for only the answer-span tail.
    """

    def __init__(self, *args, evaluator=None, batch_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator
        self.batch_sampler = batch_sampler
        self.fast_eval = True

    def get_train_dataloader(self):
        """The stock loader, with a token-budget batch sampler in place of a
        fixed batch size.  See `train/batching.py` for why -- in short, attention
        is quadratic in the PADDED length and this corpus spans 300 to 14,055
        packed tokens, so a shuffled fixed-size batch spends most of its compute
        on padding and occasionally OOMs outright."""
        if self.batch_sampler is None:
            return super().get_train_dataloader()
        loader = DataLoader(
            self.train_dataset, batch_sampler=self.batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )
        return self.accelerator.prepare(loader)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """The stock loss, computed on the answer-span tail of the logits only.

        The supervised span is the last thing in the packed sequence (the prompt
        node is packed last) and every earlier position is already `-100`, so
        slicing both the logits and the labels to that tail is *arithmetically
        the same loss* -- `ForCausalLMLoss` right-pads the labels by one and
        shifts, so a length-k logits/labels pair scores exactly the positions a
        length-L pair would have scored, minus the ones that were ignored.

        What changes is the memory.  Gemma-3's vocabulary is 262 k, so full-
        sequence logits cost `B x L x 262144 x 2` bytes and again that much when
        the loss upcasts to fp32: at this corpus's p99 of 5,605 packed tokens
        that is ~24 GB for a batch of four, and the 14,055-token maximum ball
        cannot be trained at all.  On the tail it is a few hundred megabytes,
        which is what makes a batch bigger than one possible.
        """
        labels = inputs.get("labels")
        if labels is not None and "logits_to_keep" not in inputs:
            L = labels.shape[1]
            sup = labels != -100
            rows = sup.any(dim=1)
            if bool(rows.any()):
                first = int(sup.float().argmax(dim=1)[rows].min())
                k = L - first + 1
                if k < L:
                    inputs = dict(inputs)
                    inputs["labels"] = labels[:, L - k:]
                    inputs["logits_to_keep"] = k
        return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval",
                 fast=None, dump_tag=None):
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        metrics = self.evaluator.evaluate(
            self.model, ds, prefix=metric_key_prefix,
            fast=self.fast_eval if fast is None else fast,
            dump_tag=dump_tag)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics)
        return metrics


def _baselines(cfg):
    """Majority-class accuracy per split, from the balls themselves.

    Reported with the result and never separately: check C13 in `QA_TASKS.md`
    exists because several of these types admit a cheap constant answer (T9's
    gender is 42.6 %, T10's aspect 35.7 %), and a score printed without its
    baseline is unreadable.
    """
    out = {}
    types = cfg.type_list()
    for split in ("train", "dev", "test"):
        path = os.path.join(cfg.data_root, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        per_type = collections.defaultdict(collections.Counter)
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if types and r["type"] not in types:
                    continue
                per_type[r["type"]][r["answer"]] += 1
        allc = collections.Counter()
        for c in per_type.values():
            allc.update(c)
        n = sum(allc.values()) or 1
        out[split] = {
            "majority_overall": round(max(allc.values()) / n, 4) if allc else 0.0,
            "majority_per_type": {t: round(max(c.values()) / sum(c.values()), 4)
                                  for t, c in sorted(per_type.items())},
            "n": n,
        }
    return out


def _save_train_record(cfg, run_name, sizes, results, runs_jsonl, sweep_meta=None):
    record = {
        "mode": "train",
        **(sweep_meta or {}),
        "run_name": run_name,
        "types": list(cfg.type_list()), "arm": cfg.arm(),
        "input": cfg.input_tag(),
        "spd": cfg.spd, "rrwp": cfg.rrwp, "magnetic": cfg.magnetic,
        "model_name": cfg.model_name, "impl": cfg.impl, "dtype": cfg.dtype,
        "k_hop": cfg.k_hop, "max_spd": cfg.max_spd,
        "magnetic_dim": cfg.magnetic_dim, "magnetic_q": cfg.magnetic_q,
        "magnetic_m": cfg.magnetic_m,
        "lora": cfg.lora, "lora_r": cfg.lora_r, "lora_dropout": cfg.lora_dropout,
        "lr": cfg.lr, "bias_lr": cfg.bias_lr, "num_epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size, "accumulation_steps": cfg.accumulation_steps,
        "train_token_budget": cfg.train_token_budget, "max_batch": cfg.max_batch,
        "eval_steps": cfg.eval_steps, "max_steps": cfg.max_steps,
        "max_length": cfg.max_length, "seed": cfg.seed,
        "data_root": cfg.data_root, "items_root": cfg.items_root,
        "train_size": sizes[0], "val_size": sizes[1], "test_size": sizes[2],
        "baselines": _baselines(cfg),
        **(results or {}),
    }
    append_jsonl(runs_jsonl, record)
    print(f"[results] appended training run to {runs_jsonl}")


def _per_type(metrics, prefix):
    return {k[len(prefix) + 10:]: v for k, v in metrics.items()
            if k.startswith(f"{prefix}_accuracy_T")}


def run_train_mode(cfg, runs_jsonl=None, run_name=None, sweep_id=None):
    runs_jsonl = runs_jsonl or _DEFAULT_RUNS_JSONL
    sweep_meta = {}
    if sweep_id:
        sweep_meta["sweep_id"] = sweep_id
    if run_name:
        sweep_meta["sweep_run"] = run_name
    internal_run = f"{sweep_id}_{run_name}" if (sweep_id and run_name) else cfg.run_name()

    report_to = "wandb" if cfg.wandb_project else "none"
    if cfg.wandb_project:
        set_wandb_project(cfg.wandb_project)

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train, val, test = load_data(cfg, tokenizer)
    sizes = (len(train), len(val), len(test))

    set_seed(cfg.seed)
    config_cls, model_cls = cfg.gtlm_classes()
    config = config_cls.from_pretrained(
        cfg.model_name, **cfg.bias_params(),
        k_hop=cfg.k_hop, graph_attn_impl=cfg.backend())
    model = model_cls.from_pretrained(
        cfg.model_name, config=config, graph_attn_impl=cfg.backend(),
        torch_dtype=cfg.torch_dtype())
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False

    model = select_active_params(model, active_params=ACTIVE_PARAMS,
                                 lora=cfg.lora_config())
    print_trainable_parameters(model)

    collator = GraphCollatorV2(
        tokenizer=tokenizer, k_hop=cfg.k_hop,
        magnetic_m=cfg.magnetic_m if cfg.magnetic else 0,
        pad_to_block=(cfg.backend() == "flex"), max_spd=cfg.max_spd)

    evaluator = GradeEvaluator(
        tokenizer=tokenizer, collator=collator, splits=(val, test),
        dump_dir=os.path.join(_DEFAULT_PRED_DIR, internal_run),
        eval_budget=cfg.eval_token_budget, gen_budget=cfg.gen_token_budget,
        max_batch=cfg.max_batch)

    batch_sampler = None
    if cfg.train_token_budget > 0:
        batch_sampler = TokenBudgetBatchSampler(
            packed_lengths(train.ds), budget=cfg.train_token_budget,
            max_batch=cfg.max_batch, seed=cfg.seed)
        print(f"[data] train batching: {batch_sampler.describe()}", flush=True)
        n_batches = len(batch_sampler)
    else:
        n_batches = max(1, sizes[0] // cfg.batch_size)
    steps_per_epoch = max(1, n_batches // cfg.accumulation_steps)
    gc = cfg.gradient_checkpointing
    args = TrainingArguments(
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        # Absolute, and inside THIS repo: a relative `./checkpoints` writes
        # wherever the job happened to cd to, which is how a gigabyte of this
        # experiment's checkpoints ended up in the graph_model tree.
        output_dir=os.path.join(CHECKPOINT_ROOT, EXPERIMENT_NAME, internal_run),
        logging_steps=5,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.accumulation_steps,
        gradient_checkpointing=gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gc else None,
        dataloader_num_workers=cfg.num_workers,
        dataloader_persistent_workers=(cfg.num_workers > 0),
        report_to=report_to,
        run_name=internal_run,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": cfg.lr / 10},
        warmup_steps=max(1, (steps_per_epoch * cfg.num_epochs) // 10),
        weight_decay=0.1,
        eval_strategy="steps", eval_steps=cfg.eval_steps,
        save_strategy="steps", save_steps=cfg.eval_steps,
        metric_for_best_model=METRIC, greater_is_better=True,
        save_total_limit=1, load_best_model_at_end=True,
        seed=cfg.seed, data_seed=cfg.seed,
    )

    trainer = GradeTrainer(
        model=model, args=args,
        train_dataset=train.ds, eval_dataset=val.ds,
        data_collator=collator,
        active_params=ACTIVE_PARAMS, bias_lr=cfg.bias_lr,
        evaluator=evaluator, batch_sampler=batch_sampler,
    )

    train_output = trainer.train()
    # The final numbers are the SLOW pass: pass 2 runs on every pass-1 miss, not
    # only on the modes where a token mismatch is survivable, so `accuracy` here
    # is the graded number rather than the lower bound the in-training evals use.
    trainer.fast_eval = False
    if cfg.final_eval:
        val_metrics = trainer.evaluate(eval_dataset=val.ds, metric_key_prefix="eval",
                                       fast=False, dump_tag="dev")
        test_metrics = trainer.evaluate(eval_dataset=test.ds, metric_key_prefix="test",
                                        fast=False, dump_tag="test")
    else:
        print("[results] --no-final-eval: this run measures cost, not accuracy")
        val_metrics = test_metrics = {}

    results = {
        "test_accuracy": test_metrics.get("test_accuracy"),
        "test_f1": test_metrics.get("test_f1"),
        "test_accuracy_per_type": _per_type(test_metrics, "test"),
        "test_accuracy_positive": test_metrics.get("test_accuracy_positive"),
        "test_accuracy_negative": test_metrics.get("test_accuracy_negative"),
        "test_reasons": {k[len("test_reason_"):]: v for k, v in test_metrics.items()
                         if k.startswith("test_reason_")},
        "best_val_accuracy": val_metrics.get(METRIC),
        "val_accuracy_per_type": _per_type(val_metrics, "eval"),
        "test_loss": test_metrics.get("test_loss"),
        "train_runtime_s": train_output.metrics.get("train_runtime"),
    }
    base = _baselines(cfg).get("test", {})
    print(f"[results] {cfg.input_tag()} / {cfg.arm()} "
          f"types={cfg.type_list() or 'ALL'} "
          f"test_accuracy={results['test_accuracy']} "
          f"(best-val={results['best_val_accuracy']}, "
          f"majority-class baseline={base.get('majority_overall')})")
    for t, v in sorted(results["test_accuracy_per_type"].items()):
        print(f"    {t:>5}  {v:.4f}")

    _save_train_record(cfg, internal_run, sizes, results, runs_jsonl=runs_jsonl,
                       sweep_meta=sweep_meta)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results
