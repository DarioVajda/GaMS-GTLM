"""Train ONE configuration and log ONE record.

Same protocol as `graph_model`'s `our_tests`: train, evaluate + checkpoint every
`eval_steps`, reload the best-validation checkpoint (adapter AND graph-bias
tensors), then report that checkpoint's held-out score.  The optimizer split
(`bias_lr` for the graph-bias parameters, `lr` for LoRA), the cosine-with-min-lr
schedule and the 10 % warmup are kept identical, so the two are comparable.

Two things differ.  The model classes come from `cfg.gtlm_classes()` rather than
module scope, so the backbone can be chosen by name.  And evaluation goes through
the dataset's own grader: the shared stack reports token-level exact match, which
is the wrong question for six of the nineteen types -- five are order-insensitive
and T17 is membership, so EM scores a model naming five perfectly valid
collocations at zero.  `train/evaluate.py` implements the real contract and
reports it as `accuracy`.
"""
import os
import json
import collections

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainerCallback, TrainingArguments, set_seed)

from gtlm.utils import GraphCollatorV2, GraphTrainerV2, set_wandb_project
from gtlm.train import (select_active_params, print_trainable_parameters,
                        get_device)

from .config import EXPERIMENT_NAME, CHECKPOINT_ROOT
from .data import load_data, load_dev_subsample
from .evaluate import GradeEvaluator, scaled_budgets, to_left_padding
from ._io import append_jsonl

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_RUNS_JSONL = os.path.join(EXPERIMENT_DIR, "results", "train_runs.jsonl")
_DEFAULT_PRED_DIR = os.path.join(EXPERIMENT_DIR, "results", "predictions")

ACTIVE_PARAMS = ["graph_bias"]


def answer_tail_inputs(inputs):
    """`inputs`, sliced to the answer-span tail of the sequence.

    Both the labels and `logits_to_keep` are set from the EARLIEST supervised
    position in the batch, one position before it so every supervised token still
    has its predictor inside the slice.  Left-padded (`LeftPadCollator`) that
    earliest position is `L - (longest answer)`, so the slice is a few hundred
    tokens wide; right-padded it is dragged back by whichever row is shortest.

    Separate from `GradeTrainer.compute_loss` so `train/checks/check_left_pad.py`
    measures the slice the trainer actually takes rather than a copy of it.
    """
    labels = inputs.get("labels")
    if labels is None or "logits_to_keep" in inputs:
        return inputs
    L = labels.shape[1]
    sup = labels != -100
    rows = sup.any(dim=1)
    if not bool(rows.any()):
        return inputs
    first = int(sup.float().argmax(dim=1)[rows].min())
    k = L - first + 1
    if k >= L:
        return inputs
    inputs = dict(inputs)
    inputs["labels"] = labels[:, L - k:]
    inputs["logits_to_keep"] = k
    return inputs


class LeftPadCollator:
    """Wrap a collator and move every row's padding to the FRONT.

    `GraphCollatorV2` right-pads and has no `padding_side` option, so this is a
    post-collation roll -- see `evaluate.to_left_padding` for why it is safe.

    It exists for the answer-tail logits slice.  `GradeTrainer.compute_loss`
    slices to the tail using the EARLIEST supervised position in the batch;
    right-padded, a shorter row's answer span sits earlier in the padded sequence
    and drags that slice back for every row -- on the worst GTLM batch at packed
    L=16,384 that is `logits_to_keep=6,880` and a 100.5 GiB peak.  Left padding
    makes every row end at `L-1`, so the same `min` collapses to
    `longest answer + 1` with no extra slicing logic.

    One collator object serves both training and evaluation, which also makes
    `to_left_padding` in pass 2 a no-op -- it is idempotent by construction.
    """

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, features):
        return to_left_padding(self.inner(features))


class PlainCollator:
    """Wrap GraphCollatorV2 and keep only what a stock causal LM reads.

    The graph collator ships `node_ids`, `prompt_node`, `num_nodes` and the bias
    feature tensors; stock Gemma-3 accepts none of them.  `position_ids` is
    dropped too -- a baseline ball has exactly one node, so the graph collator's
    per-node reset produces a plain `arange`, and letting the model derive its
    own keeps left-padded generation correct.
    """

    KEEP = ("input_ids", "attention_mask", "labels")

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, features):
        batch = self.inner(features)
        return {k: v for k, v in batch.items() if k in self.KEEP}


def assert_plain_arm(split, cfg):
    """Refuse `--plain-llm` on an arm whose balls actually carry a graph.

    Silently dropping the graph would turn the GTLM arm into a no-retrieval run
    reported under the wrong name, which is the kind of error that survives all
    the way into a results table.
    """
    withgraph = sum(1 for r in split.rows if r.get("nodes"))
    if withgraph:
        raise ValueError(
            f"--plain-llm was set but {withgraph:,} of {len(split.rows):,} "
            f"{split.name} balls in {cfg.data_root} carry graph nodes. The plain "
            f"backend cannot see them, so this arm would silently train without "
            f"its graph. Use it only for the baselines, whose balls are empty.")


# The grade-based accuracy from evaluate.py, NOT the token-level EM the shared
# stack computes.  The Trainer selects the reload checkpoint on this, so training
# and selection share one objective; on the multiset and membership types the two
# disagree, and EM would select for reproducing gold ORDERING.
METRIC = "eval_accuracy"


class GradeTrainer(GraphTrainerV2):
    """`GraphTrainerV2`, with `evaluate()` replaced by the two-pass grader.

    Not a `compute_metrics` hook: the contract needs the item's `grading` block
    and its id, which the `(preds, labels)` pair HF passes does not carry, and
    pass 2 has to generate, which a metric function has no model to do.
    Overriding `evaluate` also keeps the logits small -- the shared loop
    materialises `(B, L, 262144)` logits for the whole packed sequence, tens of
    gigabytes per batch at this corpus's p99 of 5,605 tokens.
    """

    def __init__(self, *args, evaluator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator
        self.fast_eval = True

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
        the loss upcasts to fp32: ~24 GB for a batch of four at this corpus's p99
        of 5,605 packed tokens, and the 14,055-token maximum ball cannot be
        trained at all.  On the tail it is a few hundred megabytes, which is what
        makes a batch bigger than one possible.

        Batches reach here LEFT-padded (`LeftPadCollator`), so every row ends at
        `L-1` and the slice collapses to `longest answer + 1`.
        """
        return super().compute_loss(model, answer_tail_inputs(inputs),
                                    return_outputs=return_outputs,
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


class EvaluateOnFinalStep(TrainerCallback):
    """Force one eval + save on the very last optimizer step.

    `eval_steps` does not divide `max_steps` and HF's Trainer neither evaluates
    nor saves at the end of training, so the tail of the run would be trained and
    then discarded -- and that tail is where the best checkpoint tends to fall.

    Only ever sets the flags to True, and runs after `DefaultFlowCallback`, so a
    step that was already an eval step is unaffected rather than doubled.
    """

    def on_step_end(self, args, state, control, **kwargs):
        if state.max_steps and state.global_step >= state.max_steps:
            control.should_evaluate = True
            control.should_save = True
        return control


def _baselines(cfg):
    """Majority-class accuracy per split, from the balls themselves.

    Reported with the result and never separately (QA_TASKS.md C13): several
    types admit a cheap constant answer -- T9's gender is 42.6 %, T10's aspect
    35.7 % -- so a score printed without its baseline is unreadable.
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


def _save_train_record(cfg, run_name, sizes, results, runs_jsonl, sweep_meta=None,
                       stack_meta=None):
    record = {
        "mode": "train",
        **(sweep_meta or {}),
        "run_name": run_name,
        "types": list(cfg.type_list()), "arm": cfg.arm(),
        "input": cfg.input_tag(),
        # A run is identified by `(input, arm, stack)`: two pairs of arms share an
        # input and the same (absent) bias flags and differ only in the stack, so
        # without this two of the study's contrasts are unreadable from this file.
        "stack": cfg.stack(), "plain_llm": cfg.plain_llm,
        # How the prompt was spelled, so a chat-template run is never read next to
        # a bare-string one as a comparable measurement.  A record without this
        # key predates `train/chat.py`.
        "prompt_format": "chat_template",
        "flex_compile_mode": cfg.flex_compile_mode,
        "flex_cache_size_limit": cfg.flex_cache_size_limit,
        "spd": cfg.spd, "rrwp": cfg.rrwp, "magnetic": cfg.magnetic,
        "model_name": cfg.model_name, "impl": cfg.impl, "dtype": cfg.dtype,
        "k_hop": cfg.k_hop, "max_spd": cfg.max_spd,
        "magnetic_dim": cfg.magnetic_dim, "magnetic_q": cfg.magnetic_q,
        "magnetic_m": cfg.magnetic_m,
        "lora": cfg.lora, "lora_r": cfg.lora_r, "lora_dropout": cfg.lora_dropout,
        "lr": cfg.lr, "bias_lr": cfg.bias_lr, "num_epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size, "accumulation_steps": cfg.accumulation_steps,
        "effective_batch": cfg.effective_batch(),
        "train_token_budget": cfg.train_token_budget,
        "eval_max_batch": cfg.eval_max_batch,
        "eval_steps": cfg.eval_steps, "max_steps": cfg.max_steps,
        "max_length": cfg.max_length, "seed": cfg.seed,
        "data_root": cfg.data_root, "items_root": cfg.items_root,
        "train_size": sizes[0], "val_size": sizes[1], "test_size": sizes[2],
        "baselines": _baselines(cfg),
        **(stack_meta or {}),
        **(results or {}),
    }
    append_jsonl(runs_jsonl, record)
    print(f"[results] appended training run to {runs_jsonl}")


def _per_type(metrics, prefix):
    return {k[len(prefix) + 10:]: v for k, v in metrics.items()
            if k.startswith(f"{prefix}_accuracy_T")}


def _convergence(trainer):
    """Was the run still improving when it stopped?  (The pre-registered rule.)

    A sweep whose best checkpoint is the last one in most runs is measuring how
    fast an arm learns rather than where it ends up, so the tail of the
    in-training dev curve and the selected step are recorded here and the
    question is answerable from `runs.jsonl` alone.

    Read BEFORE the final dev/test evaluations, which log into the same history.
    """
    curve = [(int(h["step"]), float(h["eval_accuracy"]))
             for h in trainer.state.log_history
             if "eval_accuracy" in h and "step" in h]
    # What the in-training evals cost, so the share of the run spent evaluating
    # is in the record rather than in a log someone has to parse.
    spent = [float(h.get("eval_pass1_s") or 0) + float(h.get("eval_pass2_s") or 0)
             for h in trainer.state.log_history if "eval_pass1_s" in h]
    best_ckpt = trainer.state.best_model_checkpoint
    best_step = None
    if best_ckpt:
        tail = os.path.basename(best_ckpt.rstrip("/"))
        if tail.startswith("checkpoint-") and tail[11:].isdigit():
            best_step = int(tail[11:])
    return {
        "n_evals": len(curve),
        "last_three": curve[-3:],
        "in_training_eval_s": round(sum(spent), 1),
        "mean_eval_s": round(sum(spent) / len(spent), 1) if spent else None,
        "max_steps": int(trainer.state.max_steps),
        "best_step": best_step,
        "best_is_final": (best_step is not None
                          and best_step == int(trainer.state.max_steps)),
    }


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
    # A fixed, stratified ~50 % of dev, from a CONSTANT seed, for the in-training
    # evals and checkpoint selection only.  The final dev and test evaluations
    # below run over everything.  See `data.stratified_subset`.
    dev_fast, dev_subsample = load_dev_subsample(cfg, tokenizer)
    sizes = (len(train), len(val), len(test))

    set_seed(cfg.seed)
    if cfg.plain_llm:
        # A baseline arm: its balls carry no graph nodes, so the packed sequence
        # IS an ordinary prompt and stock Gemma-3 reads it directly, with SDPA
        # and with the sliding window the weights were trained under.
        assert_plain_arm(train, cfg)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=cfg.torch_dtype(),
            attn_implementation="sdpa")
        print(f"[model] stock {type(model).__name__} + sdpa, "
              f"sliding_window={getattr(model.config, 'sliding_window', None)}",
              flush=True)
    else:
        config_cls, model_cls = cfg.gtlm_classes()
        config = config_cls.from_pretrained(
            cfg.model_name, **cfg.bias_params(),
            k_hop=cfg.k_hop, graph_attn_impl=cfg.backend(),
            **cfg.flex_params())
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
        pad_to_block=(cfg.backend() == "flex" and not cfg.plain_llm),
        max_spd=cfg.max_spd)
    if cfg.plain_llm:
        collator = PlainCollator(collator)
    # LAST, so it sees the final key set on both stacks.
    collator = LeftPadCollator(collator)

    eval_budget, gen_budget, budget_note = scaled_budgets(
        cfg.eval_token_budget, cfg.gen_token_budget)
    print(f"[eval] token budgets: {budget_note}", flush=True)
    eval_splits = [s for s in (dev_fast, val, test) if s is not None]
    evaluator = GradeEvaluator(
        tokenizer=tokenizer, collator=collator, splits=eval_splits,
        dump_dir=os.path.join(_DEFAULT_PRED_DIR, internal_run),
        eval_budget=eval_budget, gen_budget=gen_budget,
        max_batch=cfg.eval_max_batch)

    # ── the schedule, identical in every arm ──────────────────────────────
    # `max_steps` is derived from the effective batch rather than left to HF's
    # `len(dataloader) // accumulation_steps`, which depends on the
    # factorisation: 16x1 over 9,266 items gives 580 steps per epoch where 4x4
    # gives 579.  `drop_last` then makes every optimizer step exactly
    # `effective` items rather than "16, except at each epoch boundary".
    effective = cfg.effective_batch()
    steps_per_epoch = max(1, sizes[0] // effective)
    derived = steps_per_epoch * cfg.num_epochs
    max_steps = cfg.max_steps if cfg.max_steps > 0 else derived
    print(f"[schedule] effective batch {effective} "
          f"({cfg.batch_size} x {cfg.accumulation_steps}), "
          f"{steps_per_epoch} optimizer steps/epoch x {cfg.num_epochs} epochs "
          f"= {derived}"
          + (f", CAPPED at --max-steps {max_steps}" if max_steps != derived else "")
          + f"; eval every {cfg.eval_steps} + the final step", flush=True)
    gc = cfg.gradient_checkpointing
    args = TrainingArguments(
        num_train_epochs=cfg.num_epochs,
        max_steps=max_steps,
        # Absolute and inside this repo: a relative `./checkpoints` would
        # write wherever the job happened to cd to.
        output_dir=os.path.join(CHECKPOINT_ROOT, EXPERIMENT_NAME, internal_run),
        logging_steps=5,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.accumulation_steps,
        dataloader_drop_last=True,
        gradient_checkpointing=gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gc else None,
        dataloader_num_workers=cfg.num_workers,
        dataloader_persistent_workers=(cfg.num_workers > 0),
        report_to=report_to,
        run_name=internal_run,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": cfg.lr / 10},
        warmup_steps=max(1, max_steps // 10),
        weight_decay=0.1,
        eval_strategy="steps", eval_steps=cfg.eval_steps,
        save_strategy="steps", save_steps=cfg.eval_steps,
        metric_for_best_model=METRIC, greater_is_better=True,
        save_total_limit=1, load_best_model_at_end=True,
        seed=cfg.seed, data_seed=cfg.seed,
    )

    trainer = GradeTrainer(
        model=model, args=args,
        train_dataset=train.ds,
        # Checkpoint selection runs on the dev SUBSAMPLE when there is one.
        eval_dataset=(dev_fast.ds if dev_fast is not None else val.ds),
        data_collator=collator,
        active_params=ACTIVE_PARAMS, bias_lr=cfg.bias_lr,
        evaluator=evaluator,
    )
    trainer.add_callback(EvaluateOnFinalStep())

    train_output = trainer.train()
    # Read the in-training history BEFORE the final evals log into it.
    convergence = _convergence(trainer)
    # The final numbers are the SLOW pass: pass 2 runs on every pass-1 miss, so
    # `accuracy` here is the graded number, not the lower bound the in-training
    # evals report.
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
        "eval_seconds": {
            "dev_pass1_s": val_metrics.get("eval_pass1_s"),
            "dev_pass2_s": val_metrics.get("eval_pass2_s"),
            "test_pass1_s": test_metrics.get("test_pass1_s"),
            "test_pass2_s": test_metrics.get("test_pass2_s"),
        },
        # Non-zero means the corresponding FINAL number depends on a batch
        # grouping that was not pre-declared -- see evaluate.py.
        "final_eval_oom_splits": {
            "dev": val_metrics.get("eval_oom_splits"),
            "test": test_metrics.get("test_oom_splits"),
        },
        "convergence": convergence,
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
                       sweep_meta=sweep_meta,
                       stack_meta={"dev_subsample": dev_subsample,
                                   "eval_token_budget_used": eval_budget,
                                   "gen_token_budget_used": gen_budget,
                                   "device": budget_note})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results
