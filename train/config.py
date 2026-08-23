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
    balls/v2 (2026-08-22, `train/run_smoke.sbatch` stage 4): the whole corpus's
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
DATA_ROOT = os.path.join(REPO_ROOT, "data", "datasets", "balls", "v2")
# The GRADING contract lives only in the dataset, never in the ball: `mode`,
# `sep`, `arity`, and for T17 the quantity band and the full `all_items` set.
# `v2_final` is stage 4's second output -- the same rows as `v2_relabelled` with
# the membership target re-verbalised from the ball, so the answer it ships and
# the answer the ball ships are the same string.  It is the authority
# for the target; nothing downstream should read `v2_relabelled`.
#
# `v2_graded` is `v2_final` with two grading contracts repaired (2026-08-23,
# `data/qa/repair_grading.py`): T19 no longer grades on an arbitrary tie-break
# among the recorded examples of a sense, and T17's `all_items` allow-list no
# longer omits collocations the ball actually shows the model.  **Only the
# `grading` block differs** -- every `answer` is byte-identical, so this changes
# scoring and checkpoint selection, never the training target.
#
# It is the default because a run should be selected on the contract we believe
# is correct.  Point `--items-root` at `v2_final` to reproduce the numbers in the
# "as graded" column of train/README.md.
ITEMS_ROOT = os.path.join(REPO_ROOT, "data", "datasets", "generated", "v2_graded")
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
    impl: str = "v2-eager"
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
    batch_size: int = 4
    accumulation_steps: int = 4
    lr: float = 5e-4
    bias_lr: float = 3e-2
    eval_steps: int = 50
    max_steps: int = -1
    seed: int = 42
    num_workers: int = 4
    gradient_checkpointing: bool = True
    include_f1: bool = False
    wandb_project: str = None

    # ── evaluation ─────────────────────────────────────────────────────────
    # Evaluation batches are formed by TOKEN budget, not by item count: the
    # corpus runs from ~300 to 14,055 packed tokens, so a fixed batch size is
    # either wasteful at the bottom or an OOM at the top.  See
    # `evaluate.token_budget_batches`.
    eval_token_budget: int = 16_384
    gen_token_budget: int = 8_192
    # Training batches too -- `batch_size` is IGNORED unless this is 0.  Attention
    # is quadratic in the PADDED length, so a shuffled fixed-size batch over a
    # 300..14,055-token corpus spends most of its compute padding short items up
    # to the longest one, and its logits slice up with them.  Measured: 3-8 s per
    # item at batch 4 shuffled, against 0.55 s at batch 1 where nothing is padded
    # (and an OOM at step 11).  See train/batching.py.
    train_token_budget: int = 8_192
    max_batch: int = 16
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
                f"_{self.arm()}_s{self.seed}")

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
        # Unlike `our_tests`, all-features-off is ALLOWED here, and is the point:
        # it is the control arm.  The model still reads the same packed node texts
        # in the same order, so the difference between that run and the SPD +
        # magnetic run is attributable to the structural bias and to nothing else.
        # Without it, "GTLM scores X on T9" is unreadable — X could be the
        # backbone's Slovene and the flat text alone.
        return self
