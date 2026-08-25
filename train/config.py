"""Every knob a training run reads, in one dataclass.

The task: answer a Slovene lexicographical question about a headword, given that
headword's neighbourhood in the CJVT/DDDS knowledge graph as a text graph.  Items
come from `data/datasets/generated/`, balls from `data/datasets/balls/` (built by
`data/qa/build_balls.py`, which records the node and edge lists so the graph the
model saw is an artefact, not a side effect of whatever the extractor did that
day).

Structurally this is `graph_model`'s `our_tests` kg_qa arm with a real graph and
a real backbone.  Two things differ, both deliberate:

  * **the backbone is chosen by name** (`BACKBONES` below, the same dispatch
    `graphqa` uses), because the point of the run is a *gemma-3-1b-it* baseline
    and `our_tests` imports the Llama classes at module scope;
  * **no data_prep mode.**  Features are recomputed at load rather than cached,
    and a cache key is one more thing that can silently go stale against a
    rebuilt store.

    The original argument for this was that "these graphs are ~32 nodes", which
    died with balls/v2: the D4 policy gives p50 = 77, p90 = 154, p99 = 293 and
    max 705, and the magnetic eigendecomposition is cubic.  Re-measured on
    balls/v2 (2026-08-22, timing probe): the whole corpus's
    features -- SPD and magnetic for 12,490 graphs, plus the generation copies of
    dev and test -- are built once per run in a few minutes on the GPU, against
    hours of training.  So the trade still holds, for a different reason: it is
    now small relative to the run, rather than small in absolute terms.

  * **the loss is computed on the answer-span tail of the logits.**  Not an
    optimisation of taste -- see `GradeTrainer.compute_loss` in run.py.  At a
    262 k vocabulary, full-sequence logits for a p99 ball cost tens of gigabytes
    and the 14,055-token maximum ball cannot be trained at all.
"""
import os

from dataclasses import dataclass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Copied from graph_model's graphqa/config.py rather than imported: that module is
# not exported by the installed `gtlm` package (only `gtlm.models`, `gtlm.train`
# and `gtlm.utils` are), and it carries a lot of GraphQA-specific validation.
# This is the whole of what is needed here.  The LoRA targets are named per
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
UNWIRED_BACKBONES = {
    "gemma-2": ("Gemma-2 ships attn/final logit softcapping, which the GTLM stack "
                "applies at neither site; GTLMGemma3ForCausalLM refuses it."),
    "gemma-3-4b": "Multimodal Gemma-3 nests its text config under `text_config`.",
    "gemma-3-12b": "Multimodal Gemma-3 nests its text config under `text_config`.",
    "gemma-3-27b": "Multimodal Gemma-3 nests its text config under `text_config`.",
}

MODEL_NAME = "google/gemma-3-1b-it"
# Names the EXPERIMENT, not the directory -- it is baked into `run_name` and the
# checkpoint path, and the first two runs were recorded under it, so it stays put
# even though the package moved.
EXPERIMENT_NAME = "sl_qa"
DATA_ROOT = os.path.join(REPO_ROOT, "data", "datasets", "balls", "v2_clean")
# The GRADING contract lives only in the dataset, never in the ball: for T17 and
# T19 the item-level facts (`all_items`, `n_all`, `n_asked`, `quantity_band`);
# the TYPE-level fields (`mode`, `sep`, `arity`, `regex`) now live in
# `data/qa/spec.py` rather than in every row (commit 85011db), because an in-row
# constant is exactly how T19 kept being graded `sequence` for a whole run after
# the spec said otherwise.
#
# `v2_clean` is the current pipeline output and is the authority for BOTH the
# target and the contract.  It is **not** a re-spelling of the older
# `v2_graded`, and the two must never be mixed:
#
#   * 76 of 2,184 test ANSWERS differ (T19 x62, T17 x14) -- these are training
#     targets, not just grading.  `T17-000584` goes from
#     `ODGOVOR: lutkar Majarona` to `ODGOVOR: bazilika in majaron | divji
#     majaron | listki majarona | lutkar Majarona`;
#   * the grading block was restructured as above, so 1,966 of 2,184 `v2_clean`
#     rows now carry `{}`.
#
# `train/data.py:_read_items` asserts ball-answer == dataset-answer, so a
# MISMATCHED pairing raises loudly.  The hazard runs the other way: the old
# default pairing (`balls/v2` + `generated/v2_graded`) is self-consistent, so a
# hand-run `python -m train` against it executes cleanly and produces numbers
# that look directly comparable to the arms_v3 sweep and are not.  Hence the
# defaults move together.
ITEMS_ROOT = os.path.join(REPO_ROOT, "data", "datasets", "generated", "v2_clean")
# Checkpoints are large (~500 MB per run) and belong to this repo, next to the
# results record rather than wherever the job happened to cd to.
CHECKPOINT_ROOT = os.path.join(REPO_ROOT, "checkpoints")

WIRED_FEATURES = ("spd", "rrwp", "magnetic")
UNWIRED_FEATURES = ("laplacian", "rwse")

# The answer prefix the label mask keys on.  It has to be a string the *questions*
# never contain, or the mask would cut at the wrong place; every generated question
# is Slovene prose and every answer line starts `ODGOVOR:` (QA_TASKS.md 0.1), so
# the prefix below is the one token sequence guaranteed to appear exactly once.
ANSWER_PREFIX = "\nODGOVOR:"


@dataclass
class RunConfig:
    mode: str = "train"

    # ── model ──────────────────────────────────────────────────────────────
    model_name: str = MODEL_NAME
    impl: str = "v2-flex"
    # torch.compile mode for the FlexAttention kernel, used only when
    # `impl == "v2-flex"`.  gtlm's own default is "max-autotune-no-cudagraphs",
    # which buys ~1.47x per step for a ~320 s one-time compile PER DISTINCT
    # (L, N) SHAPE.  This corpus batches by token budget over sequences from 300
    # to 16,382 tokens, so it presents many shapes and that trade inverts: the
    # autotune time would dwarf the kernel saving.  Plain "default" compiles in
    # seconds and keeps the flex win.
    flex_compile_mode: str = "default"
    # torch._dynamo recompile budget for the flex kernel.  gtlm defaults to 32;
    # the GTLM arm blew straight through that (2-706 nodes x token-budget
    # lengths = many distinct (L, N) shapes), fell back to UNCOMPILED python and
    # ran at 14-18 s/step against eager's 3.4.  The serialised arm never hit it:
    # one node per graph, so only L varies.
    flex_cache_size_limit: int = 512
    # Run the arm on STOCK Gemma-3 (AutoModelForCausalLM + SDPA/flash) instead of
    # the GTLM stack.  Only meaningful for the BASELINE arms, whose balls carry
    # zero graph nodes -- the whole subgraph is flattened into the prompt, so
    # there is no graph for the GTLM machinery to do anything with.
    #
    # Why it matters: routing a plain-text baseline through the GTLM interface
    # forces it onto the custom attention path, where FlashAttention cannot be
    # used.  Comparing GTLM's throughput against a baseline nerfed that way
    # flatters GTLM for a reason that has nothing to do with the model.  A
    # baseline is meant to show what a STANDARD LLM does, so it gets the
    # standard stack and the standard kernels.
    #
    # This also restores Gemma-3's sliding_window=512 (22 of 26 layers), which
    # GTLMGemma3ForCausalLM drops -- see train/README.md.  That is the
    # pretrained configuration, so the baseline runs on-distribution.
    plain_llm: bool = False
    dtype: str = "bf16"
    lora: bool = True
    lora_r: int = 32
    lora_dropout: float = 0.05
    active_params: tuple = ("graph_bias",)

    # ── graph bias ─────────────────────────────────────────────────────────
    # SPD and magnetic ON, RRWP off: the run is asked for specifically as
    # "SPD and magnetic bias", and leaving RRWP on would make it a three-feature
    # run reported as a two-feature one.
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
    # `batch_size x accumulation_steps` is the EFFECTIVE batch, and only the
    # PRODUCT is scientifically binding: gradient accumulation normalised by
    # `num_items_in_batch` yields the same average gradient however the 16 items
    # are split into micro-batches (equal up to floating-point summation order
    # and dropout RNG -- not a systematic difference).  So the factorisation is a
    # pure memory/throughput knob and may be set per arm; the product may not.
    # `run.py` derives `max_steps` from the product, so every arm runs the same
    # number of optimizer steps whatever factorisation it uses.
    batch_size: int = 4
    accumulation_steps: int = 4
    lr: float = 5e-4
    # LoRA learning rate is `lr`; this one is for the graph-bias tensors only, so
    # it binds on the SPD+magnetic arm alone (every other arm reports
    # `Custom Graph Biases: 0`).  3e-2 was the old default; 1e-2 is the better
    # choice more often in practice.  It is a hyperparameter *of* the thing being
    # ablated, which is legitimate, but it must be disclosed -- and `lr` stays
    # shared and untuned across arms so the study is not two tuned knobs
    # against zero.
    bias_lr: float = 1e-2
    eval_steps: int = 50
    max_steps: int = -1
    seed: int = 42
    num_workers: int = 4
    gradient_checkpointing: bool = True
    include_f1: bool = False
    wandb_project: str = None

    # ── evaluation ─────────────────────────────────────────────────────────
    # Evaluation batches are formed by TOKEN budget, not by item count: the
    # corpus runs from ~300 to 16,384 packed tokens, so a fixed batch size is
    # either wasteful at the bottom or an OOM at the top.  See
    # `evaluate.token_budget_batches`.
    #
    # These two are the A100-80GB REFERENCE values.  `evaluate.scaled_budgets`
    # rescales them linearly by the device's own total memory (clamped), so the
    # same config runs on a 178 GiB B200 at ~36k/18k and on a smaller card at
    # proportionally less.  Set them explicitly to pin a budget instead.
    eval_token_budget: int = 16_384
    gen_token_budget: int = 8_192
    # Items per evaluation batch, independent of the TRAINING `batch_size`.  It
    # used to share `max_batch` with the training sampler, which meant dropping
    # the training micro-batch for memory quartered evaluation throughput as a
    # side effect -- with 8 epochs and a hard Slurm wall clock that is a
    # run-killing coupling, not a papercut.
    eval_max_batch: int = 16
    # The token-budget TRAINING sampler is retired (see train/batching.py): every
    # arm now runs the same fixed effective batch, because under token-budget
    # batching the item count per batch falls out of sequence length and each arm
    # got a different effective batch AND a different number of updates.  The
    # knob survives only so a config that still sets it fails loudly.
    train_token_budget: int = 0
    # The in-training evals and checkpoint selection run on a fixed, stratified
    # ~50 % subsample of dev; the FINAL dev and test evaluations run on
    # everything.  Selected from `dev_subsample_seed`, which is a CONSTANT and
    # never `cfg.seed`, so every arm and every seed selects on the identical
    # subset.  0 disables the subsample and evaluates on the whole split.
    dev_subsample: float = 0.5
    dev_subsample_seed: int = 20260823
    # The final dev+test evaluations run pass 2 on EVERY pass-1 miss, which on an
    # untrained model is nearly the whole split.  Off for timing probes, which
    # only want the training step and the in-training eval measured.
    final_eval: bool = True

    # ── derived ────────────────────────────────────────────────────────────
    def torch_dtype(self):
        import torch
        return {"fp32": torch.float32, "bf16": torch.bfloat16}[self.dtype]

    def backend(self):
        return self.impl.split("-", 1)[1]

    def backbone(self):
        name = self.model_name.lower()
        for key, reason in UNWIRED_BACKBONES.items():
            if key in name:
                raise ValueError(f"model_name={self.model_name!r} is not wired: {reason}")
        for key in BACKBONES:
            if key in name:
                return key
        raise ValueError(
            f"model_name={self.model_name!r} names no backbone this experiment "
            f"wires (expected one of {tuple(BACKBONES)} in the name).")

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

        The third coordinate of a run's identity.  `arms_v3` runs the serialised
        input on BOTH stacks and the no-retrieval input on both, so
        `(input_tag, arm)` names two runs each and neither the checkpoint
        directory nor the results record would distinguish them.
        """
        return "plain" if self.plain_llm else "gtlm"

    def effective_batch(self):
        """Items per optimizer step.  The scientifically binding quantity."""
        return self.batch_size * self.accumulation_steps

    def type_list(self):
        return tuple(t.strip() for t in self.types.split(",") if t.strip())

    def input_tag(self):
        """Which INPUT this arm reads, from the ball directory's own name.

        Three of the four arms in the run matrix share a backbone, a schedule and
        a seed and differ only in what they are shown, so the bias arm name alone
        does not identify a run -- `balls/v2_serialised` with SPD off and
        `balls/v2_noretrieval` with SPD off would otherwise write to the same
        checkpoint directory.
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
        # The inert-flag trap.  `arm()` reads the bias flags, which do NOTHING
        # under `plain_llm` -- stock Gemma-3 has no graph-bias parameters at all.
        # A plain smoke run has already been recorded as
        # `..._v2_clean_serialised_spd+magnetic_s0`, i.e. a run with zero bias
        # parameters filed under the name of the bias arm.  Refuse the
        # combination rather than silently mislabel a row in the results table.
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
        # Unlike `our_tests`, all-features-off is ALLOWED here, and is the point:
        # it is the control arm.  The model still reads the same packed node texts
        # in the same order, so the difference between that run and the SPD +
        # magnetic run is attributable to the structural bias and to nothing else.
        # Without it, "GTLM scores X on T9" is unreadable — X could be the
        # backbone's Slovene and the flat text alone.
        return self
