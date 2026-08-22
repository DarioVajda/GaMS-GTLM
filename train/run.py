"""Train ONE configuration and log ONE record.

Same protocol as `graph_model`'s `our_tests`: train, evaluate + checkpoint every
`eval_steps`, reload the best-validation checkpoint (adapter AND graph-bias
tensors — see `gtlm.utils.text_graph_trainer_v2._load_best_model`), then report
that checkpoint's test exact match over the answer span.

The one structural difference is that the model classes come from
`cfg.gtlm_classes()` rather than being imported at module scope, because this
experiment runs a **gemma-3-1b-it** backbone and `our_tests` imports the Llama
classes directly.  Everything else — the optimizer split (`bias_lr` for the
graph-bias parameters, `lr` for LoRA), the cosine-with-min-lr schedule, the 10 %
warmup — is kept identical, so a number from here is comparable with one from
there.
"""
import os
import json
import collections

import torch
from transformers import AutoTokenizer, TrainingArguments, set_seed

from gtlm.utils import GraphCollatorV2, GraphTrainerV2, set_wandb_project
from gtlm.utils.text_graph_trainer_v2 import (make_compute_metrics,
                                              shift_logits_for_metrics)
from gtlm.train import (select_active_params, print_trainable_parameters,
                        get_device)

from .config import EXPERIMENT_NAME, CHECKPOINT_ROOT
from .data import load_data
from ._io import append_jsonl

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_RUNS_JSONL = os.path.join(EXPERIMENT_DIR, "results", "train_runs.jsonl")

ACTIVE_PARAMS = ["graph_bias"]
METRIC = "eval_em_accuracy"


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
        "spd": cfg.spd, "rrwp": cfg.rrwp, "magnetic": cfg.magnetic,
        "model_name": cfg.model_name, "impl": cfg.impl, "dtype": cfg.dtype,
        "k_hop": cfg.k_hop, "max_spd": cfg.max_spd,
        "magnetic_dim": cfg.magnetic_dim, "magnetic_q": cfg.magnetic_q,
        "magnetic_m": cfg.magnetic_m,
        "lora": cfg.lora, "lora_r": cfg.lora_r, "lora_dropout": cfg.lora_dropout,
        "lr": cfg.lr, "bias_lr": cfg.bias_lr, "num_epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size, "accumulation_steps": cfg.accumulation_steps,
        "eval_steps": cfg.eval_steps, "max_steps": cfg.max_steps,
        "max_length": cfg.max_length, "seed": cfg.seed,
        "data_root": cfg.data_root,
        "train_size": sizes[0], "val_size": sizes[1], "test_size": sizes[2],
        "baselines": _baselines(cfg),
        **(results or {}),
    }
    append_jsonl(runs_jsonl, record)
    print(f"[results] appended training run to {runs_jsonl}")


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
    train_ds, val_ds, test_ds = load_data(cfg, tokenizer)
    sizes = (len(train_ds), len(val_ds), len(test_ds))

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

    steps_per_epoch = max(1, sizes[0] // cfg.batch_size // cfg.accumulation_steps)
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

    trainer = GraphTrainerV2(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=make_compute_metrics(include_f1=cfg.include_f1),
        preprocess_logits_for_metrics=shift_logits_for_metrics,
        active_params=ACTIVE_PARAMS, bias_lr=cfg.bias_lr,
    )

    train_output = trainer.train()
    val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="eval")
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    results = {
        "test_accuracy": test_metrics.get("test_em_accuracy"),
        "best_val_accuracy": val_metrics.get(METRIC),
        "test_loss": test_metrics.get("test_loss"),
        "train_runtime_s": train_output.metrics.get("train_runtime"),
    }
    if cfg.include_f1:
        results["test_f1"] = test_metrics.get("test_em_f1")
    base = _baselines(cfg).get("test", {})
    print(f"[results] {cfg.arm()} types={cfg.type_list() or 'ALL'} "
          f"test_accuracy={results['test_accuracy']} "
          f"(best-val={results['best_val_accuracy']}, "
          f"majority-class baseline={base.get('majority_overall')})")

    _save_train_record(cfg, internal_run, sizes, results, runs_jsonl=runs_jsonl,
                       sweep_meta=sweep_meta)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results
