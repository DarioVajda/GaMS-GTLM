"""Every knob a training run reads, in one dataclass.

The task: answer a Slovene lexicographical question about a headword, given that
headword's neighbourhood in the CJVT/DDDS knowledge graph as a text graph.  Items
come from `data/datasets/generated/`, balls from `data/datasets/balls/`.

Three things distinguish this from `graph_model`'s `our_tests` kg_qa arm:

  * the backbone is chosen by name (`BACKBONES` below), so the same code runs a
    gemma-3 or a llama checkpoint;
  * graph features are recomputed at load rather than cached, so no cache key can
    go stale against a rebuilt store.  Building SPD and magnetic features for the
    whole corpus takes a few minutes on the GPU against hours of training;
  * the loss is computed on the answer-span tail of the logits -- at a 262 k
    vocabulary, full-sequence logits for a p99 ball cost tens of gigabytes.  See
    `GradeTrainer.compute_loss` in run.py.
"""
import os
import json

from dataclasses import dataclass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Copied from graph_model's graphqa/config.py rather than imported: that module is
# not exported by the installed `gtlm` package.  LoRA targets are named per
# backbone because PEFT's defaults are model_type-keyed and do not know the
# gtlm_* types.
IMPLS = ("v2-eager", "v2-flex")
BACKBONES = {
    "llama": dict(
        classes=("GTLMLlamaConfig", "GTLMLlamaForCausalLM"),
        lora_targets=("q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"),
    ),
    "gemma-3": dict(
        classes=("GTLMGemma3Config", "GTLMGemma3ForCausalLM"),
        # Gemma-3 uses Llama's projection names, so the same targets apply.
        lora_targets=("q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"),
    ),
}
# Checkpoints that look like a wired backbone but are not.  Checked BEFORE the
# substring match so the error names the real obstacle.
#
# The three multimodal entries are refused as PUBLISHED, not as architectures:
# `train/extract_text_tower.py` writes their text tower out as a flat
# `gemma3_text` directory, and such a directory loads through the adapter
# exactly as `GaMS3-12B-Instruct` does.  `_is_text_only_dir` below is what tells
# the two apart -- by reading the config, not by trusting a naming convention.
UNWIRED_BACKBONES = {
    "gemma-2": ("Gemma-2 ships attn/final logit softcapping, which the GTLM stack "
                "applies at neither site; GTLMGemma3ForCausalLM refuses it."),
    "gemma-3-4b": ("Multimodal Gemma-3 nests its text config under `text_config`. "
                   "Run `python -m train.extract_text_tower <repo>` and point "
                   "--model-name at the directory it writes."),
    "gemma-3-12b": ("Multimodal Gemma-3 nests its text config under `text_config`. "
                    "Run `python -m train.extract_text_tower <repo>` and point "
                    "--model-name at the directory it writes."),
    "gemma-3-27b": ("Multimodal Gemma-3 nests its text config under `text_config`. "
                    "Run `python -m train.extract_text_tower <repo>` and point "
                    "--model-name at the directory it writes."),
}


# What a checkpoint's own `model_type` says about which backbone wires it.  The
# fallback when the NAME says nothing -- `cjvt/GaMS3-12B-Instruct` is a Gemma-3
# model whose name contains no "gemma-3", and it is the study's target.
MODEL_TYPE_BACKBONES = {
    "gemma3_text": "gemma-3",   # flat text-only: what the adapter loads
    "gtlm_gemma3": "gemma-3",   # a checkpoint this experiment already wrote
    "llama": "llama",
}


def _config_dict(model_name):
    """`config.json` for a local directory or a hub id; `{}` if unreadable.

    A local read first, so an extracted text tower resolves without touching the
    network.  `AutoConfig` behind it, because the target checkpoint
    (`cjvt/GaMS3-12B-Instruct`) lives on the hub and names no backbone in its id.
    Only ever called when the NAME alone has not settled the question, so the
    common path stays offline and free.

    Any failure answers `{}` rather than raising: the caller's job is to produce
    a good error message about the backbone, not to surface a network fault as
    one.
    """
    local = os.path.join(model_name, "config.json")
    if os.path.isfile(local):
        try:
            with open(local, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}
    try:
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(model_name).to_dict()
    except Exception:
        return {}


def _is_text_only(cfg):
    """Is this config the flat `gemma3_text` the adapter can load directly?"""
    return cfg.get("model_type") in ("gemma3_text", "gtlm_gemma3") \
        and "text_config" not in cfg

MODEL_NAME = "google/gemma-3-1b-it"
# Names the EXPERIMENT, not the directory: it is baked into `run_name` and the
# checkpoint path, so it stays put even though the package moved.
EXPERIMENT_NAME = "sl_qa"
# The balls and the items are ONE pairing and move together.  A ball carries the
# answer; the dataset carries the answer and the grading contract, and the two
# must agree -- `train/data.py:_read_items` asserts it.  Mixing generations is
# self-consistent enough to run and produces numbers that are not comparable.
DATA_ROOT = os.path.join(REPO_ROOT, "data", "datasets", "balls")
ITEMS_ROOT = os.path.join(REPO_ROOT, "data", "datasets", "generated")
# Checkpoints are ~500 MB per run and belong next to the results record rather
# than wherever the job happened to cd to.
CHECKPOINT_ROOT = os.path.join(REPO_ROOT, "checkpoints")

WIRED_FEATURES = ("spd", "rrwp", "magnetic")
UNWIRED_FEATURES = ("laplacian", "rwse")

# Where the label mask cuts.  It must be a string the QUESTIONS never contain:
# every question is Slovene prose and every answer line opens `ODGOVOR:`
# (QA_TASKS.md 0.1), so this appears exactly once.  The chat template preserves
# it -- gemma writes `<start_of_turn>model\nODGOVOR: …`, keeping the leading
# newline -- so cutting here yields the template's own generation prompt.
ANSWER_PREFIX = "\nODGOVOR:"


@dataclass
class RunConfig:
    mode: str = "train"

    # ── model ──────────────────────────────────────────────────────────────
    model_name: str = MODEL_NAME
    impl: str = "v2-flex"
    # torch.compile mode for the FlexAttention kernel (`impl == "v2-flex"` only).
    # gtlm defaults to "max-autotune-no-cudagraphs", which buys ~1.47x per step
    # for a ~320 s compile PER DISTINCT (L, N) SHAPE.  This corpus presents many
    # shapes, so the autotune time would dwarf the saving; "default" compiles in
    # seconds and keeps the flex win.
    flex_compile_mode: str = "default"
    # torch._dynamo recompile budget for the flex kernel.  gtlm defaults to 32,
    # which the GTLM arm exhausts (2-706 nodes x many lengths); past the budget
    # it falls back to uncompiled python at 14-18 s/step against eager's 3.4.
    flex_cache_size_limit: int = 512
    # Run on STOCK Gemma-3 (AutoModelForCausalLM + SDPA) instead of the GTLM
    # stack.  Only valid for the BASELINE arms, whose balls carry zero graph
    # nodes -- there is no graph for the GTLM machinery to act on.
    #
    # A baseline has to show what a standard LLM does, so it gets the standard
    # kernels and the pretrained configuration: FlashAttention, and Gemma-3's
    # sliding_window=512 on 22 of 26 layers, which GTLMGemma3ForCausalLM drops.
    # Routing it through the GTLM path instead would flatter GTLM's throughput
    # for a reason that has nothing to do with the model.
    plain_llm: bool = False
    dtype: str = "bf16"
    lora: bool = True
    lora_r: int = 32
    lora_dropout: float = 0.05
    active_params: tuple = ("graph_bias",)

    # ── graph bias ─────────────────────────────────────────────────────────
    # SPD and magnetic on, RRWP off: the arm is named "SPD and magnetic bias",
    # and leaving RRWP on would report a three-feature run as a two-feature one.
    spd: bool = True
    max_spd: int = 8
    rrwp: bool = False
    max_rw_steps: int = 16
    magnetic: bool = True
    magnetic_dim: int = 32
    magnetic_q: float = 0.25
    magnetic_m: int = 0
    laplacian: bool = False
    rwse: bool = False
    k_hop: int = 0

    # ── data ───────────────────────────────────────────────────────────────
    data_root: str = DATA_ROOT
    items_root: str = ITEMS_ROOT
    types: str = ""            # "" = every type present in data_root
    max_items: int = 0         # >0 caps each split (smoke tests)
    max_length: int = 2048     # per-node tokenization cap
    data_seed: int = 42

    # ── schedule ───────────────────────────────────────────────────────────
    num_epochs: int = 6
    # Only the PRODUCT `batch_size x accumulation_steps` is scientifically
    # binding: accumulation normalised by `num_items_in_batch` gives the same
    # average gradient however the items are split into micro-batches.  So the
    # factorisation is a memory/throughput knob and may vary per arm; the product
    # may not, and `run.py` derives `max_steps` from it so every arm takes the
    # same number of optimizer steps.
    batch_size: int = 4
    accumulation_steps: int = 4
    lr: float = 5e-4
    # For the graph-bias tensors only, so it binds on the SPD+magnetic arm alone
    # (every other arm reports `Custom Graph Biases: 0`).  `lr` stays shared and
    # untuned across arms, so the study is not two tuned knobs against zero.
    bias_lr: float = 1e-2
    eval_steps: int = 50
    max_steps: int = -1
    seed: int = 42
    num_workers: int = 4
    gradient_checkpointing: bool = True
    include_f1: bool = False
    wandb_project: str = None

    # ── evaluation ─────────────────────────────────────────────────────────
    # Evaluation batches are formed by TOKEN budget, not item count: the corpus
    # runs from ~300 to 16,384 packed tokens, so a fixed batch size is either
    # wasteful at the bottom or an OOM at the top.  These are the A100-80GB
    # reference values; `evaluate.scaled_budgets` rescales them by the device's
    # own memory.  Set them explicitly to pin a budget instead.
    eval_token_budget: int = 16_384
    gen_token_budget: int = 8_192
    # Items per evaluation batch, independent of the TRAINING `batch_size` so
    # that lowering the training micro-batch for memory does not also quarter
    # evaluation throughput.
    eval_max_batch: int = 16
    # Retired -- see train/batching.py.  Kept only so a config that still sets it
    # fails loudly instead of silently running a different schedule.
    train_token_budget: int = 0
    # In-training evals and checkpoint selection run on a fixed, stratified ~50 %
    # subsample of dev; the FINAL dev and test evaluations run on everything.
    # The subsample seed is a CONSTANT, never `cfg.seed`, so every arm and every
    # seed selects the identical subset.  0 evaluates on the whole split.
    dev_subsample: float = 0.5
    dev_subsample_seed: int = 20260823
    # The final evaluations run pass 2 on EVERY pass-1 miss, which on an
    # untrained model is nearly the whole split.  Off for timing probes.
    final_eval: bool = True

    # ── derived ────────────────────────────────────────────────────────────
    def torch_dtype(self):
        import torch
        return {"fp32": torch.float32, "bf16": torch.bfloat16}[self.dtype]

    def backend(self):
        return self.impl.split("-", 1)[1]

    def backbone(self):
        """Which backbone wires this checkpoint -- by name, then by its config.

        The name is tried first and settles the common cases offline
        (`google/gemma-3-1b-it`).  The config is consulted only when the name
        leaves something open, which is exactly twice:

          * `cjvt/GaMS3-12B-Instruct`, a Gemma-3 model whose id contains no
            "gemma-3".  It is the target of this whole study, and refusing it on
            a substring match would be refusing it for its name.
          * an extracted text tower, which keeps its multimodal source's name
            and so trips the `UNWIRED_BACKBONES` guard that it exists to satisfy.

        In both directions the config, not the name, is the authority.
        """
        name = self.model_name.lower()
        unwired = [r for k, r in UNWIRED_BACKBONES.items() if k in name]
        wired = [k for k in BACKBONES if k in name]
        if wired and not unwired:
            return wired[0]

        cfg = _config_dict(self.model_name)
        text_only = _is_text_only(cfg)
        if unwired and not text_only:
            raise ValueError(
                f"model_name={self.model_name!r} is not wired: {unwired[0]}")
        if wired:
            return wired[0]
        # A multimodal config whose NAME did not give it away -- same obstacle,
        # same remedy, so it gets the same message rather than a vaguer one.
        if "text_config" in cfg:
            raise ValueError(
                f"model_name={self.model_name!r} nests its text config under "
                f"`text_config` (model_type={cfg.get('model_type')!r}), which "
                f"GTLMGemma3Config cannot load.  Run "
                f"`python -m train.extract_text_tower {self.model_name}` and "
                f"point --model-name at the directory it writes.")
        by_type = MODEL_TYPE_BACKBONES.get(cfg.get("model_type"))
        if by_type:
            return by_type
        raise ValueError(
            f"model_name={self.model_name!r} names no backbone this experiment "
            f"wires (expected one of {tuple(BACKBONES)} in the name), and its "
            f"config says model_type={cfg.get('model_type')!r}, which is not "
            f"one of {tuple(MODEL_TYPE_BACKBONES)}.")

    def gtlm_classes(self):
        from gtlm import models
        return tuple(getattr(models, n)
                     for n in BACKBONES[self.backbone()]["classes"])

    def lora_config(self):
        if not self.lora:
            return None
        return {"r": self.lora_r, "lora_alpha": self.lora_r * 2,
                "lora_dropout": self.lora_dropout,
                "target_modules": list(BACKBONES[self.backbone()]["lora_targets"])}

    def flex_params(self):
        """FlexAttention knobs, empty on the eager backend so it is untouched."""
        if self.backend() != "flex":
            return {}
        return dict(flex_compile_mode=self.flex_compile_mode,
                    flex_cache_size_limit=self.flex_cache_size_limit)

    def bias_params(self):
        cfg = {}
        if self.spd:
            cfg.update(spd=True, max_spd=self.max_spd)
        if self.rrwp:
            cfg.update(rrwp=True, max_rw_steps=self.max_rw_steps)
        if self.magnetic:
            cfg.update(magnetic=True, magnetic_dim=self.magnetic_dim,
                       magnetic_q=self.magnetic_q)
        return cfg

    def arm(self):
        on = [f for f in WIRED_FEATURES if getattr(self, f)]
        return "+".join(on) if on else "no-bias"

    def stack(self):
        """Which model stack runs this arm -- `"gtlm"` or `"plain"`.

        The third coordinate of a run's identity: the serialised and the
        no-retrieval inputs are each run on BOTH stacks, so `(input_tag, arm)`
        names two runs and would collide in the checkpoint path.
        """
        return "plain" if self.plain_llm else "gtlm"

    def effective_batch(self):
        """Items per optimizer step.  The scientifically binding quantity."""
        return self.batch_size * self.accumulation_steps

    def type_list(self):
        return tuple(t.strip() for t in self.types.split(",") if t.strip())

    def input_tag(self):
        """Which INPUT this arm reads, from the ball directory's own name.

        Most arms share a backbone, a schedule and a seed and differ only in what
        they are shown, so the bias arm name alone does not identify a run.
        """
        return os.path.basename(self.data_root.rstrip("/")) or "balls"

    def run_name(self):
        tag = "-".join(self.type_list()) or "all"
        model = self.model_name.split("/")[-1]
        return (f"{EXPERIMENT_NAME}_{model}_{tag}_{self.input_tag()}"
                f"_{self.arm()}_{self.stack()}_s{self.seed}")

    def validate(self):
        if self.mode != "train":
            raise ValueError(f"Unknown mode {self.mode!r} (only 'train').")
        self.backbone()                       # raises with a reason if unwired
        if self.impl not in IMPLS:
            raise ValueError(f"Unknown impl {self.impl!r} (expected {IMPLS}).")
        bad = [f for f in UNWIRED_FEATURES if getattr(self, f)]
        if bad:
            raise ValueError(
                f"Bias feature(s) {bad} are in the schema but not produced by "
                f"data.py (which computes {WIRED_FEATURES}).")
        # `arm()` reads the bias flags, which do NOTHING under `plain_llm` --
        # stock Gemma-3 has no graph-bias parameters.  Refuse the combination
        # rather than file a run with zero bias parameters under the bias arm's
        # name in the results table.
        if self.plain_llm:
            on = [f for f in WIRED_FEATURES if getattr(self, f)]
            if on:
                raise ValueError(
                    f"--plain-llm is set together with bias feature(s) {on}, "
                    f"which stock Gemma-3 cannot use: the run would train with "
                    f"NO bias and be recorded under the arm name "
                    f"{self.arm()!r}.  Turn the flags off explicitly.")
        if self.train_token_budget:
            raise ValueError(
                f"train_token_budget={self.train_token_budget} but the "
                f"token-budget TRAINING sampler is retired (see "
                f"train/batching.py): it made the effective batch and the "
                f"update count a function of sequence length, so two arms ran "
                f"different optimisations.  Set it to 0 and use "
                f"batch_size x accumulation_steps.")
        # All features off is ALLOWED, and is the control arm: the model reads
        # the same packed node texts in the same order, so the gap to the
        # SPD+magnetic run is attributable to the structural bias alone.
        return self
