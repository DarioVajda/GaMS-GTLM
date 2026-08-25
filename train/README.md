# `train/` — fine-tuning GTLM on the generated lexicographical QA dataset

Answer a Slovene question about a headword given that headword's neighbourhood in
the CJVT/DDDS dictionary graph, encoded as a text graph.

```bash
# THE study: six configurations x three seeds, one Slurm array, all on B200
.venv/bin/python -m sweep train train/configs/arms_v3.jsonc

# everything that must pass on the real hardware first (B200, in the container)
sbatch train/slurm/run_preflight.sbatch

# 1 epoch per arm, no final eval -- prices the wall clock and the evaluation
.venv/bin/python -m sweep train train/configs/arms_v3_timing.jsonc

# the tables: per arm, per type, the five contrasts, and the convergence rule
.venv/bin/python -m train.analysis.report_arms train/results/arms_v3/runs.jsonl

# one run
.venv/bin/python -m train --num-epochs 8                     # (needs a GPU)

# individual checks
.venv/bin/python -m train.checks.check_labels --n 10       # what the mask supervises
.venv/bin/python -m train.checks.test_left_pad             # left padding is a no-op
.venv/bin/python -m train.checks.test_two_pass --max-items 32   # the fast evaluator
.venv/bin/python -m train.checks.probe_eval --n 32         # is the KV cache live?
```

Everything runs **from the repo root**. `sweep` refuses to submit when the calling
interpreter lives outside the project root — it forwards `SWEEP_VENV_BIN` to the
job, and a path outside the root would not resolve there.

## Layout

The top level holds only what the training path itself imports; everything that
merely *runs against* it sits one level down.

```
train/
  config.py  data.py  run.py  evaluate.py    the training path
  batching.py  qa_contract.py                 shared by it
  _io.py  _log.py                             helpers
  __main__.py                                 one run: python -m train
  checks/     check_labels  test_left_pad  test_two_pass  probe_eval
  analysis/   report_arms  analyse_t17  length_stats  project_cost  rescore
  slurm/      run_train  run_preflight  run_length_stats
  configs/    the sweep definitions
  results/    sweep output (gitignored)
```

The split is by **direction of dependency**, not by topic. Nothing in `checks/`,
`analysis/` or `slurm/` is imported by the training path — the arrows all point
inward — so a module's directory tells you whether breaking it can break a run.

## How this uses `graph_model` without living in it

The GTLM stack is a **dependency**, editable-installed into this repo's `.venv`:

| import | resolves to |
|---|---|
| `gtlm.models` | `graph_model/src/models` |
| `gtlm.train` | `graph_model/src/train` |
| `gtlm.utils` | `graph_model/src/utils` |
| `sweep` | `graph_model/sweep` |

Note `graph_model/src/experiments` is **not** exported — experiments are not part
of the package, which is the argument for lexicography-specific ones living here.
`sweep` supports that directly: `slurm_launch.sh` reads `SWEEP_PROJECT_ROOT`,
`SWEEP_VENV_BIN` and `SWEEP_LOGIN` from the submitting process precisely so a
sweep runs against the project it was submitted *from*.

Two consequences worth knowing:

* **`results_dir` must be set in every sweep config.** Its default is
  `<graph_model>/sweeps`, which would put our results back in the other repo.
* **A `container` key is required for Blackwell.** `ixb*` are Ubuntu 24.04 /
  python3.12 and this repo's venv is 3.10, so a bare job there dies on imports;
  the pyxis image is a py3.10 base and the venv resolves inside it. See "Why
  these runs are on B200" below.

`gtlm.train` (shared training utilities, over there) and this `train` package are
different things despite the name; the import is always spelled in full.

## Where the data comes from

Three artefacts, all built by `data/qa/` and all explicit files on disk:

| artefact | built by | what it is |
|---|---|---|
| `data/datasets/balls/v2_clean` | `qa/build_balls.py` | one text graph per item — node texts verbatim from the KG, plus an edge list |
| `data/datasets/generated/v2_clean` | the same pass | the dataset rows, carrying the **grading contract** and the same `answer` the ball ships |
| `data/datasets/balls/v2_clean_{serialised,noretrieval}` | `qa/build_variants.py` | the two baseline inputs, derived from the ball file itself |

> **`v2_clean` is not a re-spelling of the older `v2_graded`, and the two must
> never be mixed.** 76 of 2,184 test **answers** differ (T19 ×62, T17 ×14) —
> these are *training targets*, not just grading. `T17-000584` goes from
> `ODGOVOR: lutkar Majarona` to `ODGOVOR: bazilika in majaron | divji majaron |
> listki majarona | lutkar Majarona`. The grading block was also restructured:
> the type-level fields (`mode`, `sep`, `arity`, `regex`) moved out of every row
> into `qa/spec.py` (commit `85011db`), because an in-row constant is exactly how
> T19 kept being graded `sequence` for a whole run after the spec said otherwise,
> so 1,966 of 2,184 `v2_clean` rows now carry `{}`.
>
> `data.py` asserts ball-answer == dataset-answer, so a *mismatched* pairing
> raises loudly. The hazard runs the other way: the old default pairing
> (`balls/v2` + `generated/v2_graded`) is self-consistent, so a hand-run
> `python -m train` against it executes cleanly and produces numbers that look
> directly comparable to `arms_v3` and are not. **The defaults in
> `train/config.py` now point at `v2_clean`, and both move together.**

Nothing here opens the 37.5 M-node store. The ball is already an artefact, which
is what lets a GPU job start cold and what makes the graph the model saw part of
the record rather than a side effect of whatever the extractor did that day.

Graph layout per item:

```
node 0 .. N-1   the ball  (iztočnica: gora (…) / oblika: gore (…) / pomen: … / …)
node N          the PROMPT node,  "{question}\nODGOVOR: {answer}"
edges           the ball's parent->child edges, plus prompt -> EVERY anchor
```

The prompt attaches to every anchor, not just the first: D3's lookup returns a
**union** when a surface string owns several lexical units (9.34 % of items), and
choosing among them is the model's job.

**The ball is uniform across task types on purpose.** Giving T3 the paradigm and
T9 the anchor alone would be cheaper and would also make every score meaningless,
because the model would be told which relation to read before it read anything.
The caps in `build_balls.py` are a token budget, not a relevance judgement.

Sizes (`balls/v2`, packed tokens): p50 1,314, p90 2,610, p99 5,605, max 14,055 —
**not** v1's ~32 nodes / 550–780 tokens. Two things follow, and both are handled
rather than assumed: see "Cost" below.

## The join: ball + contract

`qa.grade.grade(item, prediction)` needs **both** the answer and the grading
block, and they live in different files — the ball carries `question`, `answer`,
`nodes`, `edges`, `anchors`; only the dataset carries `grading` (`mode`, `sep`,
`arity`, and for T17 `quantity_band`, `n_asked`, `n_all`, `all_items`).
`data.py` joins them on `id` and **asserts the two answers are identical**. That
is not a formality: T17's target was verbalised at generation time from a K=15
collocation sample while the ball holds K=10, so 26.5 % of its target phrases
named collocations the model cannot see until stage 4 re-verbalised them
(`qa/build_balls.py:reverbalise`). `generated/v2_clean` is the authority; nothing
downstream should read `v2_relabelled`, `v2_final` or `v2_graded`.

## Evaluation is the dataset's own grader

`train/evaluate.py`, reported as **`accuracy` = the share of items whose answer
satisfies `qa/grade.py`**, overall and **per type**.

The shared stack's `make_compute_metrics` reports token-level exact match, which
answers the wrong question for six of the nineteen types:

| grading mode | types | the contract | what EM does |
|---|---|---|---|
| `sequence` | T1 T2 T3 T5 T6 T7 T8 T9 T10 T11 T12 T14 T19 | ordered, exact | correct |
| `multiset` | T4 T15 T16 T20 T21 | order-**insensitive** set equality | penalises a correct answer in another order |
| `membership` | T17 | **any** subset of `all_items` the band allows | a model naming five valid collocations scores zero |

`train/run.py`'s `METRIC` selects the best checkpoint on this same `accuracy`, so
no run trains against one objective and selects against another.

### Two passes

Free-running generation over the whole dev split at every `eval_steps` is the
dominant cost of evaluation, and most of it is avoidable.

* **Pass 1** — one teacher-forced forward. If the argmax at every answer position
  equals the gold token, greedy decoding would have emitted exactly that string.
  This is *sufficient* in every mode: gold trivially satisfies its own multiset
  equality, and is by construction an allowed subset for `membership`. It settles
  **74.8 % of the dev split** before a token is generated, and the share grows as
  the model improves.
* **Pass 2** — greedy generation for the remainder, then `qa.grade.grade`.

`fast=True` (the in-training evals) skips pass 2 where a token mismatch already
condemns the item — `sequence` mode and every negative, whose sentinel is
exact-match in all modes. `fast=False` (the final dev/test numbers) generates for
every pass-1 miss.

Four further things keep evaluation inside its budget without making the number
depend on the machine it ran on:

* **The eval batch knob is separate from the training one.** `eval_max_batch`
  (default 16) is independent of `batch_size`; they used to share one field, so
  lowering the training micro-batch for memory quartered evaluation throughput as
  a side effect.
* **Token budgets are sized from the device, not from a constant.** 16,384 /
  8,192 were chosen for A100-80GB; `evaluate.scaled_budgets` rescales them
  linearly by `torch.cuda.get_device_properties(0).total_memory`, clamped. An
  A100-80GB reproduces those values exactly; a 178 GiB B200 measured
  **36,864 / 18,432**. The realised values go in the run record.
* **An OOM guard**, because a sized-from-the-device estimate that is occasionally
  wrong must not kill an eight-epoch run at hour nine: on `OutOfMemoryError` the
  batch is halved and retried down to size 1, and **every fallback is logged and
  counted**. Re-grouping changes padding and a bf16 near-tie can flip an argmax,
  so a fallback during a *final* eval is surfaced in the report — that number then
  depends on a grouping that was not pre-declared.
* **Pass 2 groups by gold-answer length as well as input length.**
  `max_new_tokens = min(320, 2·max(gold in batch) + 16)` is a batch-level bound
  and generation runs until every row stops, so one long-gold item otherwise makes
  the whole batch decode long.

`eval_pass1_s`, `eval_pass2_s`, `eval_generated` and `eval_oom_splits` are in the
metrics dict, so the next round of tuning is data-driven rather than a guess.

**The KV cache is verified, not assumed** (`train/checks/probe_eval.py`). Training
enables gradient checkpointing and HF then sets `config.use_cache = False`;
`generate(use_cache=True)` is supposed to override that per call, and if it did
not, pass 2 would be decoding without a cache and the numbers would be identical
while only the clock moved. On the flex backend the check is decisive by
construction: `causal_lm.forward` takes the flex path only when
`q_len == kv_len`, so a *cached* decode step (q_len=1) goes down the dense path
while an *uncached* one re-feeds the whole grown sequence and fails the
block-alignment check. The uncached arm cannot run at all — which is exactly what
proves the cached one is what executed. On the plain stack, where the A/B is a
real timing comparison, 32 serialised items decode in **26.8 s with the cache and
163.8 s without — 6.10×** — with **32/32 identical predictions**, so the cache is
both live and inert as far as the numbers are concerned.

### Two generation bugs the plain stack hid, and how they are fixed

Both were found by `run_preflight.sbatch` stage 4b and both were **fatal to arm 4
only** — the serialised prompt on stock Gemma-3, whose prompts are p50 2,166
tokens. Neither can fire on the GTLM stack (which drops the sliding window) or on
a prompt under 512 tokens, which is why nothing before this caught them.

* **The hybrid cache was being suppressed.** `use_model_defaults=False` is what
  stops gemma-3-1b-it's `do_sample=True` from silently overriding our greedy
  config — it has to stay, or pass 1's correctness argument is false. But it
  suppresses *every* model default, including `cache_implementation="hybrid"`,
  which is not a decoding policy but an architectural fact. Without it `generate`
  builds a plain `DynamicCache`, whose sliding layers keep growing while
  Gemma-3's sliding-layer mask is sliced to `max(1, 512)` — so the first decode
  step past a ~512-token prompt dies with *"The expanded size of the tensor (594)
  must match the existing size (512) at non-singleton dimension 3"*. The fix
  copies that one field from **`model.config`** (`evaluate.gen_cfg_for`), not
  from `model.generation_config`: the model config is where both stacks state
  their requirement deliberately — stock Gemma-3 ships `"hybrid"` in
  `config.json`, and `GTLMGemma3ForCausalLM._sanitize_attn_config` sets it to
  `None` on purpose, because a hybrid cache would truncate the KV span its
  attention still expects. The hub's `generation_config.json` says `"hybrid"` for
  both, so reading *that* would hand the GTLM path a cache it must not have.
* **A misaligned attention-bias pointer during decode.** With the hybrid cache in
  place, the sliding layers hand SDPA a non-contiguous *view* —
  `attention_mask[:, :, :, offset : offset + 512]` with
  `offset = max(0, kv_len − 512)` — and the cutlass `EFFICIENT_ATTENTION` kernel
  requires that pointer 16-byte aligned. `offset` grows by one per decode step,
  so seven steps in eight it raises `p.attn_bias_ptr is not correctly aligned`.
  Prefill is safe (`offset == 0` there), so pass-2 generation runs inside
  `sdpa_kernel([FLASH_ATTENTION, MATH])`. At decode `q_len == 1`, so the math
  backend materialises a `(B, H, 1, kv)` score matrix and costs nothing. This
  changes *which kernel* computes the attention, never what it computes, and
  **training is untouched** — it keeps the fused kernels, which is what arm 4's
  "what a standard LLM actually does" claim rests on.

Four conditions make the equivalence exact, and all four bit during development:

* **greedy only.** Since transformers 4.50 a custom `GenerationConfig` does *not*
  stop the model's own defaults from applying: any field still at the library
  default is back-filled from `model.generation_config`, and `do_sample=False`
  **is** the library default — so gemma-3-1b-it's `do_sample=True` silently won.
  `use_model_defaults=False` plus the flags as `generate()` kwargs pins it, and
  `test_two_pass.py` decodes a slice twice and compares, so a regression here
  fails loudly instead of quietly invalidating every number.
* **the stop token is checked** — the supervised span ends with the appended EOS,
  so "every answer position is argmax" already includes "and then it stops".
* **tokenisation false negatives are safe** — pass 1 compares one tokenisation of
  the gold string, so a model emitting a different token sequence for the same
  text is a pass-1 *miss*; it falls through to pass 2 and is graded properly.
  Pass 1 never declares failure, only success.
* **prefix alignment** — answer positions come from the same offset-based mask the
  training loss uses (`OffsetLabelMasker`, keyed on `\nODGOVOR:`).

### The acceptance test

```bash
.venv/bin/python -m train.checks.test_two_pass --max-items 32                 # untrained
.venv/bin/python -m train.checks.test_two_pass --max-items 32 --checkpoint …  # trained
```

Scores the same slice twice — once through the two-pass evaluator, once by
generating every item — and requires **identical per-item `success`**, plus
`prediction == gold` for every item pass 1 passed. `max_batch=1` makes generation
padding-free so the two runs feed the model bitwise identical tensors; otherwise
a bf16 near-tie could flip an argmax between two groupings and fail the test for
a reason that has nothing to do with the claim. The trained-checkpoint form
matters: on an untrained model nothing clears pass 1 and the sharper assertion is
vacuous, which is why `run_preflight.sbatch` overfits 32 items first (stage 6).

## Label masking

Done on **character offsets**, not by searching for the delimiter's token-id
subsequence. `"ODGOVOR:"` alone and the same string preceded by a newline need
not tokenize identically, and that failure is silent — it masks the wrong span
rather than raising. `check_labels.py` decodes the supervised span so a person
can confirm it.

## The schedule, and why it is identical in every arm

* **Effective batch 16 everywhere**, as `batch_size × accumulation_steps`. Only
  the **product** is scientifically binding: gradient accumulation normalised by
  `num_items_in_batch` yields the same average gradient however the 16 items are
  split (equal up to floating-point summation order and dropout RNG — not a
  systematic difference), so the factorisation is a pure memory/throughput knob
  and is set per arm from the timing runs. The record carries both, so the report
  can show the product was constant.
* **`max_steps` is derived from the product**, not from HF's
  `len(dataloader) // accumulation_steps`. That expression depends on the
  factorisation — 16×1 over 9,266 items gives 580 optimizer steps per epoch where
  4×4 gives 579 — and two arms differing by 8 steps is exactly the kind of small
  unaccounted difference this ablation exists to remove. 9,266 // 16 = **579
  steps/epoch × 8 epochs = 4,632 steps**, in every arm. `dataloader_drop_last`
  then makes every step exactly 16 items rather than "16, except at each epoch
  boundary".
* **12 evaluations, the last at step 4,632.** `eval_steps=400` gives 400 … 4,400;
  400 does not divide 4,632 and HF's Trainer does not evaluate or save at the end
  of training, so without `run.py`'s `EvaluateOnFinalStep` callback the last
  **232 steps** would be trained and silently discarded, with the best checkpoint
  chosen from step 4,400. That matters precisely because `arms_v2` found the best
  checkpoint was the last one in all 12 runs — the region this would drop is the
  region the run is most likely still improving in.
* **The token-budget training sampler is retired** (`train/batching.py` records
  why). Its entire justification was the answer-tail logits blow-up, which left
  padding removes at source; and it made length — which correlates with task type
  — decide batch composition, so whole gradient steps were made of one kind of
  item.

## Cost: what balls/v2 changed

`train/config.py` used to justify skipping a feature cache on the grounds that
"these graphs are ~32 nodes, so the magnetic Laplacian is a 32×32
eigendecomposition". `balls/v2` is p50 77 / p99 293 / max 705 nodes, so that
premise is dead. Three consequences, all handled:

* **Training batches are LEFT-padded** (`run.LeftPadCollator`). `GraphCollatorV2`
  right-pads and has no `padding_side` option, and `compute_loss` slices the
  logits to the answer tail using the *earliest* supervised position in the batch
  — so right-padded, a shorter row's answer span sits earlier in the padded
  sequence and drags that slice back for everyone. Measured on B200: the worst
  mixed GTLM batch needed `logits_to_keep = 8,166` and peaked at **82.8 GiB**;
  left-padded the same batch needs **13** and peaks at **4.7 GiB**. Every row now
  ends at `L−1`, so the existing `min` collapses on its own to
  `longest answer + 1` and no new slicing logic was needed.
  `train/checks/test_left_pad.py` is the licence for it — see "Is left padding really a
  no-op?" below.

* **Logits, not attention, were the memory wall.** Gemma-3's vocabulary is 262 k,
  so full-sequence logits cost `B × L × 262144 × 2` bytes and again that much when
  the loss upcasts to fp32 — ~24 GB for a batch of four at the p99 length, and the
  14,055-token maximum ball could not be trained at all. The supervised span is
  the tail of the packed sequence (the prompt node is packed last) and everything
  earlier is `-100`, so `GradeTrainer.compute_loss` slices **both** the logits and
  the labels to that tail and lets `ForCausalLMLoss` shift as usual. The loss is
  arithmetically unchanged; the memory is a few hundred megabytes. The evaluator
  does the same thing with `logits_to_keep`.
* **Evaluation batches are formed by token budget**, not item count — a corpus
  running from 300 to 14,055 packed tokens has no single sensible batch size, and
  a mixed batch also inflates the `logits_to_keep` slice (it must reach the
  earliest answer span in the batch).

## Why these runs are on B200

Every run is on **B200, inside the pyxis container**, and that is a requirement
rather than a preference: a gap measured across two hardware or software stacks
is not a measurement of the thing being compared, so one tier has to serve all 18
runs.

Blackwell hosts are Ubuntu 24.04 / python3.12 and this repo's venv is 3.10, so a
**bare** job there dies on imports. The container is a py3.10 base and the venv
resolves inside it. The one thing that used to block that: `.venv/bin/python` was
an absolute symlink to `/opt/deepops/venv/bin/python`, a host path with no
counterpart in the image. It now points at `/usr/bin/python3` (the same 3.10.12),
matching `graph_model`'s venv, and the container probe passes.

Two things not to undo:

* **Do not substitute `PYTHONPATH` for running the venv interpreter directly.**
  `PYTHONPATH` appends our packages while leaving the image's
  `/usr/local/lib/python3.10/dist-packages` on the path, so the container's
  torchvision loads against our torch and transformers dies on
  `operator torchvision::nms does not exist` (job 130961).
* **B200 only, not `B200|B300`.** Of the four B300 hosts `ixb5` is drained (GPU7
  unrecoverable memory error) and `ixb7`'s enroot mellanox hook still fails on a
  missing `/dev/infiniband/rdma_cm` — a host-config fault no user can fix.
  Listing B300 would let part of the array land on a node where the container
  cannot start. `ixb1`–`ixb4` are healthy. Re-add B300 once `ixb7` is repaired.

`run_smoke.sbatch` and `run_timing.sbatch` were **deleted** rather than updated.
Both were pinned to `ana` (A100-80GB, bare, no container), and a harness job that
runs cleanly on different hardware from the study is precisely the "produces
numbers that look comparable and are not" hazard. `run_container_probe.sbatch`
went the same way once its question was answered: it existed to prove the
`.venv/bin/python` symlink fix worked, and that fix is now in place and recorded
in `configs/arms_v3.jsonc`. `slurm/run_preflight.sbatch` covers
everything the smoke job did plus the new checks, and
`configs/arms_v3_timing.jsonc` prices each arm on the hardware it will actually
run on. Git history keeps the originals.

## The run matrix

`train/configs/arms_v3.jsonc` — **six configurations × three seeds = 18 runs**, in
one file, submitted with one `sweep` call, writing one `runs.jsonl`.

| # | configuration | input | stack | biases |
|---|---|---|---|---|
| 1 | **GTLM (spd + magnetic)** | `balls/v2_clean` | GTLM | spd + magnetic |
| 2 | **GTLM, no bias** | `balls/v2_clean` | GTLM | none |
| 3 | **serialised — GTLM stack** | `balls/v2_clean_serialised` | GTLM | none |
| 4 | **serialised — plain stack** | `balls/v2_clean_serialised` | plain | n/a |
| 5 | **no retrieval — plain stack** | `balls/v2_clean_noretrieval` | plain | n/a |
| 6 | **no retrieval — GTLM stack** | `balls/v2_clean_noretrieval` | GTLM | none |

Arms 3–6 ship a ball with **zero graph nodes** — the subgraph is flattened into
the prompt (3, 4) or absent entirely (5, 6) — so arms 4 and 5 differ in exactly
one thing: whether the graph's text is in the prompt. The serialised prompt is
bounded at 16,384 tokens, which costs 12 rows of 12,490 (0.10 %) a suffix of their
node list; `qa/check_variants.py` verifies every kept node text is present
verbatim and reports the truncations.

The comparisons the matrix exists to make, each reported **paired by seed**:

| contrast | arms | what it isolates |
|---|---|---|
| Retrieval | 4 vs 5 | is the subgraph worth anything at all (same stack) |
| Structural bias | 1 vs 2 | does the bias make the graph usable (same stack, same input) |
| **Graph encoding** | **1 vs 3** | encoding, **stack held fixed** — the headline claim |
| Stack cost | 3 vs 4 | what SDPA + `sliding_window=512` is worth on a long input |
| Stack control | 5 vs 6 | the same stack difference where it *cannot* act — expected ≈ 0 |

**Why the stack is worth three runs.** Serialised prompts are p50 2,166 / p99
10,014 tokens, and stock Gemma-3 applies `sliding_window=512` in 22 of its 26
layers, which `GTLMGemma3ForCausalLM` drops. That is a large capability
difference sitting directly on top of the encoding claim, and without arm 3 there
is no way to tell the two apart. Arm 4 is also the correct arm for any throughput
or cost comparison: routing a plain-text baseline through the GTLM interface
denies it FlashAttention and would flatter GTLM for a reason that is not the
model.

**Why arm 6 is a control, not a fifth measurement.** The no-retrieval input is
p50 97 characters / p90 240 / max 1,014, i.e. roughly p50 ~35 tokens and a maximum
comfortably under 512, so the sliding window *cannot* bite — the whole sequence
fits inside it. Crossing a factor with an input where the factor has no room to
act buys no information about the factor. What it buys is a falsification check
on the stack contrast: if the plain/GTLM difference really is a length-and-window
interaction, arms 5 and 6 must land on top of each other. If they do not, contrast
4 is not measuring the window and the reading of contrast 3 has to change.

### What `arms_v2` got wrong, and what changed

Four arms × three seeds completed on A100-80GB (array `130369`), and three defects
make those numbers unusable as a final result:

1. **Nothing converged.** 3 epochs; dev accuracy was still rising in every arm at
   the stop and the best checkpoint was the last one in **all 12 runs**. That
   measures how fast an arm learns, not where it ends up. → `arms_v3` runs 8
   epochs and `report_arms.py` checks a **pre-registered** convergence rule.
2. **The arms were not running the same optimisation.** Under token-budget
   batching the item count per batch falls out of sequence length, so each arm got
   a different effective batch and a different number of updates: 16.5 items ×
   4,480 steps (GTLM), 10.9 × 6,824 (serialised), 63.9 × 1,160 (no-retrieval) — a
   6× spread on both axes. → the sampler is retired; every arm runs effective
   batch **16** and `run.py` derives `max_steps` from that product so the update
   count is identical by construction.
3. **Two grading contracts were wrong** and were repaired post hoc, forcing both
   an "as graded" and a "repaired" column. → fixed at source in `qa/spec.py`,
   `qa/grade.py` and `qa/build_balls.py`, and `v2_clean` is the only data the
   defaults point at.

> The previously recorded **+13.0-point** bias gap was measured on `balls/v1` — a
> hand-capped ~32-node ball built by a different and methodologically flawed
> policy. It was **re-measured on `balls/v2` and superseded**: the gap is
> **+32.3 points**. Do not cite the +13.0 figure.

### Measured cost, before committing 18 runs

`configs/arms_v3_timing.jsonc` — one 1-epoch run per arm, seed 0, `--no-final-eval`,
on the same B200 + container stack the study uses. `train.analysis.project_cost` turns each
into an 8-epoch projection; every number below is measured except the final
dev+test column, which is extrapolated from the last in-training eval's own
per-item rates and a **one-epoch** `pass1_share` (conservative — pass 1 settles
more as the model improves, so the real run generates less).

| arm | micro-batch | s/step | 1 eval | train 8ep | 12 evals | final dev+test | TOTAL | eval share |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| serialised — plain stack | 4×4 | 5.47 s | 78 s | 7.04 h | 16 min | 11 min | **7.49 h** | 6 % |
| GTLM (spd + magnetic) | 4×4 | 3.87 s | 93 s | 4.98 h | 19 min | 10 min | **5.45 h** | 9 % |
| serialised — GTLM stack | 4×4 | 3.43 s | 97 s | 4.42 h | 19 min | 12 min | **4.94 h** | 11 % |
| GTLM, no bias | 4×4 | 2.39 s | 74 s | 3.07 h | 15 min | 14 min | **3.55 h** | 13 % |
| no retrieval — GTLM stack | 16×1 | 0.38 s | 28 s | 0.48 h | 6 min | 6 min | **0.67 h** | 28 % |
| no retrieval — plain stack | 16×1 | 0.22 s | 30 s | 0.28 h | 6 min | 7 min | **0.50 h** | 44 % |

Slowest arm **7.49 h** against the 16 h Slurm limit; ~68 GPU-hours for all 18 runs.

**On the 25 % evaluation bound (T5).** Two arms exceed it, and both are the
no-retrieval arms — whose *training* is nearly free while their dev+test
evaluation still has to score 3,224 items. The share is large because the
denominator is small, not because evaluation is expensive: in absolute terms
those runs spend the same 12–13 minutes in evaluation as everyone else. The bound
is a proxy for "evaluation must not blow the wall clock", and that risk is met
with an order of magnitude to spare on every arm. Reported as measured rather
than redefined.

**An unexpected result, reported because it is one.** The *plain* stack is
**1.6× slower per step** than the GTLM stack on the identical serialised input
(5.47 vs 3.43 s/step). The premise for giving the baseline the standard stack was
that routing plain text through the GTLM interface would deny it FlashAttention;
in practice stock Gemma-3 does not get flash here either — our batches are padded
and Gemma-3's sliding layers need a 4-D mask, so SDPA dispatches to the
memory-efficient kernel, while GTLM's flex path exploits block sparsity in its
mask. Arm 4 is still the right reference for "what a standard LLM does with this
input", but **it is not a demonstration that the standard stack is faster**, and
any throughput claim in the write-up has to say so.

### Disclosures

Three deliberate choices that a reader is entitled to know about before reading
any number.

* **`bias_lr` binds on arm 1 only.** It is 1e-2 (was 3e-2), and arm 1 is the sole
  configuration with `graph_bias` parameters — every other arm reports
  `Custom Graph Biases: 0`. Tuning a hyperparameter *of* the thing being ablated
  is legitimate, but `lr` (LoRA) stays **shared and untuned** across all six arms,
  so the study is not two tuned knobs against zero.
* **Checkpoint selection runs on half of dev.** In-training evals score a fixed,
  stratified 50 % subsample (527 of 1,040 items, all 18 dev types intact) chosen
  from a *constant* seed, so every arm and every seed selects on the identical
  subset. Selecting on ~527 items and taking the best of 12 evaluations gives the
  reported **dev** figure a mild optimistic bias of roughly ±2 points of sampling
  noise. The **test** figure is untouched by this and is the number to quote. The
  final dev (1,040) and test (2,184) evaluations run over everything.
* **v3 and v2 numbers are not item-for-item comparable.** 76 of 2,184 test answers
  differ between `v2_graded` and `v2_clean` (T19 ×62, T17 ×14) — targets, not just
  grading. See "Where the data comes from".

### What the contrasts do and do not isolate

Three properties measured *after* the sweep was launched. None of them changes the
design or invalidates a run; all three change how a contrast should be read, so
they are recorded here rather than discovered by a reader.

**1. Contrast 1-vs-3 is confounded by RoPE range.** It is meant to isolate
*encoding* with the stack held fixed — both arms run the GTLM stack, which drops
Gemma-3's `sliding_window`. But the two inputs put wildly different demands on
position:

| | arm 1 (graph balls) | arm 3 (serialised) |
|---|--:|--:|
| position ids | reset per node | continuous |
| longest prefix node | 157 tok | — |
| longest prompt node | 324 tok | — |
| test sequence length (median / p99 / max) | — | 2,174 / 10,013 / 16,380 |
| fraction of items past 512 | **0 / 2,184** | **96.2 %** |

22 of Gemma-3-1b's 26 layers are sliding layers using `rope_local_base_freq`
= 10,000, trained to attend only within a 512-token window. With
`node_position_mode='reset'` and nodes this short, **no query–key pair in arm 1
ever sees a relative offset past 512** — the resets keep it inside the trained
range for free. Arm 3 has no node structure and no resets, so with the window
dropped those same 22 layers attend across the whole sequence at local-RoPE
frequencies, up to ~32× beyond their training range. Arm 3 is therefore not
"arm 1's input through the same machinery"; it is *also* a length-extrapolation
stress test, and any 1-vs-3 gap mixes the two effects. Arm 4 (same input, stock
stack) keeps `sliding_window=512` and is not affected — which is one candidate
explanation for why arm 4 beats arm 3 on identical input.

> **This is inferred from the design, not measured.** Isolating it needs an arm
> that decouples them — serialised input *with* the window restored, or graph
> balls without position resets — and no such arm is in `arms_v3`. The size of
> the effect is unknown; only its direction and its presence are established.

**2. The question is last in the serialised prompt, so this contrast is *not*
confounded by question placement.** The serialised `question` field is
`PODATKI IZ BAZE:` → the numbered node list → `POVEZAVE:` → the edge list → a
blank line → the bare question. Checked over the whole test split: **2,184/2,184**
serialised prompts end with the bare question and **0/2,184** begin with it. Under
the causal mask the serialised arm's graph tokens therefore cannot see the
question either, so both arms encode the neighbourhood question-blind and neither
gets question-conditioned retrieval. Note this is a *choice* in the variant
builder, not a constraint: putting the question first would give the serialised
baseline a question-conditioned encoding it currently lacks, so arm 4's lead is
measured against a deliberately conservative version of itself.

**3. GTLM's graph tokens get bidirectional context; the serialised arm's get
causal only.** In `structural_mask.py` the rule is
`allowed = causal | both_prefix`, so a graph-token query and a prompt-node key are
disallowed (the prompt node is packed last, so `causal` is false and `both_prefix`
is false) — confirming point 2 for GTLM — while any two graph tokens see each
other in both directions. The serialised arm's graph tokens see only left context.
On graph context GTLM has strictly more, not less; the offsetting cost is that
bidirectional attention is itself off-distribution for a causally pretrained
decoder.

## Results

### `arms_v3` (the study)

<!-- ARMS_V3_RESULTS -->
*Not yet filled in — the sweep is running. `report_arms.py` writes
`train/results/arms_v3/report_arms.md`; this section carries its tables plus the
convergence verdict.*

### `arms_v2` — SUPERSEDED (2026-08-23)

> **Do not quote these numbers as a result.** They are kept because the
> diagnostics below (T19, T17, the length analysis) are still the best account of
> what the arms do per task, and because the reasons they were replaced are part
> of the record. Three defects, each fixed in `arms_v3`: nothing converged (best
> checkpoint was the last one in **all 12** runs at 3 epochs), the arms were not
> running the same optimisation (a 6× spread in effective batch and update count),
> and two grading contracts were wrong and were repaired post hoc. See
> "What `arms_v2` got wrong" above. The data also differs: 76 of 2,184 test
> answers changed between `v2_graded` and `v2_clean`.

Array `130369`, 12/12 completed on `ana`; scored by `qa/grade.py`'s contract, best
checkpoint on `eval_accuracy`, seeds `[0, 1, 2]`, train 9,266 / dev 1,040 / test 2,184.

Two grading contracts were repaired after the runs (`data/qa/repair_grading.py`,
reasoning below). **Both columns are reported**: the repair is a post-hoc change
and the reader is entitled to see what it moved.

| arm | test acc (as graded) | **test acc (repaired)** | best dev (repaired) |
|---|--:|--:|--:|
| **GTLM (spd + magnetic)** | 0.6972 ± 0.0053 | **0.7219 ± 0.0063** | 0.8045 ± 0.0101 |
| serialised graph | 0.6799 ± 0.0095 | **0.6824 ± 0.0095** | 0.7635 ± 0.0186 |
| no retrieval | 0.3805 ± 0.0107 | **0.3805 ± 0.0107** | 0.4279 ± 0.0118 |
| GTLM, no bias | 0.3747 ± 0.0071 | **0.3747 ± 0.0071** | 0.4410 ± 0.0099 |
| majority-class baseline | 0.1071 | 0.1071 | — |

Three things this says:

* **Retrieval is worth ~34 points.** Both arms that see the subgraph reach ~0.68–0.72;
  both that do not sit at ~0.38.
* **The bias, not the encoder, is what makes the graph usable.** `GTLM, no bias` is
  statistically indistinguishable from `no retrieval` — paired by seed the difference is
  −0.0087, −0.0147, +0.0060, straddling zero — yet it trains in 2.11 h against
  no-retrieval's 0.20 h, so it is demonstrably being fed the graph and getting nothing
  out of it. Unaffected by the repair (both arms move 0.0000).
* **Graph encoding beats the serialised prompt, by ~4 points.** Paired by seed the gap is
  +0.0472, +0.0449, +0.0266 — mean **+0.0395**, sd 0.0092, same sign on 3/3 seeds, and
  the dev split reproduces it independently (+0.0410). Under the *original* contract this
  gap was +0.0172 and I reported it as "within noise, do not claim it"; the repair is
  what changed that reading, which is exactly why both columns are above.
* **A ceiling no arm clears.** T12, T15, T16, T21 are weak everywhere and T16 is at or
  below majority in all four arms — no arm has an advantage, so it is not architectural.

**Caveat: none of these runs converged.** Dev accuracy is still rising in every
arm at the 3-epoch stop — over the final eval interval, GTLM **+0.041**,
serialised **+0.023**, no-bias **+0.049** — and the best checkpoint is the last
one in all 12 runs. So this matrix partly measures *how fast each arm learns*,
not where each ends up. It bears hardest on the second claim: "the bias-off arm
gets nothing from the graph" is much weaker if that arm is merely slower, and it
has the steepest remaining slope of the three. Raise `num_epochs` and re-run
before treating any of this as final.

**On the honesty of a post-hoc repair.** The change was specified by question
semantics, not by results: T19's question admits many correct answers and the
grader accepted one; T17's allow-list omitted collocations the model was shown.
It was applied to all four arms from a single arm-independent source (`balls/v2`),
and `train/analysis/rescore.py` **asserts monotonicity** — no answer that was correct
becomes incorrect — which is also what licenses re-scoring the dumps instead of
retraining. It happens to help GTLM most, because GTLM is the arm that was
answering T19 correctly and being marked wrong; on T17 it helps the serialised
arm more (+0.029 vs +0.014). Anyone who prefers the unrepaired contract can read
the first column and reach the weaker conclusion.

### Per task: GTLM vs the serialised prompt

Test accuracy, mean over seeds `[0, 1, 2]`. Task names are `data/QA_TASKS.md`'s.

| task | GTLM | serialised |
|---|--:|--:|
| T1 · full declension table | **0.823** | 0.788 |
| T2 · declension, one number | 0.921 | **0.927** |
| T3 · single case cell | **0.959** | 0.722 |
| T4 · lemmatisation | 0.788 | **0.849** |
| T5 · full conjugation | **0.977** | 0.954 |
| T6 · conjugation, one tense | **0.841** | 0.806 |
| T7 · non-finite verb forms | **0.983** | 0.977 |
| T8 · word class + property | **0.820** | 0.725 |
| T9 · noun gender | **0.959** | 0.838 |
| T10 · verb aspect and type | **0.974** | 0.965 |
| T11 · gradation, all degrees | **0.930** | 0.899 |
| T12 · sense definitions | **0.377** | 0.287 |
| T14 · count of senses | **0.632** | 0.426 |
| T15 · synonyms | **0.328** | 0.258 |
| T16 · antonyms *(Tier C, held out)* | 0.067 | **0.072** |
| T17 · collocations | 0.301 | **0.452** |
| T19 · example sentences | 0.965 | **0.991** |
| T20 · form analysis in a sentence | **0.762** | 0.652 |
| T21 · form analysis (inverse of T3) | 0.310 | **0.380** |

Repaired contract (T17 and T19 rows differ from the original; every other row is
untouched by the repair). GTLM wins 13 of 19. Its biggest wins — T3 (+0.238) and
T14 (+0.206) — are *selection* and *counting* over the ball, which is what the
structural bias is for. Its remaining losses are T17 (−0.151), T4, T21, T2, T16.

#### T19 is a grading artefact, not a capability gap

The 46-point T19 gap is **not** GTLM failing to copy text. Diagnosed from the
per-item dumps (`train/results/predictions/`):

* Of GTLM's 52 T19 failures, **49 are a real `zgled:` node from the item's own
  ball** — a genuine recorded sentence containing the target word. Only 3 are
  hallucinations or the sentinel.
* T19 asks *"V kakšnem stavku se pojavi X?"* — "in what sentence does X appear?"
  **82 of 115 test items carry more than one `zgled:` node**, every one of which
  answers the question as asked. The question does not say which.
* The gold is, in **101 of 103** cases, simply the **first** `zgled:` in file
  order. The tie is broken by a KG ordering convention the question never states.
* The serialised arm picks the first candidate 102/115 times; GTLM 51/115,
  scattering over the rest. That is the whole gap.

The reason GTLM cannot follow that convention is architectural and deliberate.
`zgled:` nodes hang off sense nodes, which are themselves interchangeable
siblings of the anchor — equal degree, equal SPD, same edge type — so the
candidates are structurally symmetric, and `position_ids` are reset per node.
Sibling order is not represented in the graph at all. The serialised prompt, by
contrast, is a numbered list under one linear position encoding, so "the first
example" is trivially readable.

**Fixed** in `data/qa/repair_grading.py`: T19 becomes `membership` over the
recorded examples with an exact count of one — any `zgled:` the KG records for
the lemma is accepted, which is what the question asks for.

| arm | strict (first-only) | repaired (any recorded sentence) |
|---|--:|--:|
| GTLM | 0.510 | **0.965** |
| serialised | 0.974 | 0.991 |

The 46-point gap collapses to 2.6. T19 is 5.27 % of test, so as graded it dragged
GTLM's margin down by 2.44 points — the repair widens the aggregate gap rather
than closing it, which is why the encoding claim above is now stated positively.

The deeper point is not about one type. **Order is not a property of this data.**
The KG stores the examples of a sense as a set; the file order is an artefact of
serialisation. Grading on it rewards an architecture for preserving an accident
and penalises one for discarding it — and discarding it is *permutation
equivariance*, a property a graph model is designed to have. Any future type
whose answer is drawn from a set must be `membership` or `multiset`, never
`sequence`. `sequence` is correct only where position carries meaning that the
data itself defines: a paradigm cell, a tense, a dictionary sense ordinal.

#### T17 is a different bug — an incomplete allow-list

T17 was checked for the same failure mode and **does not have it**: it was already
`membership`, so order never mattered. Its failures split into two kinds:

* **`not_in_all` — a real grader defect.** The model returns a collocation that is
  a `kolokacija:` node *in its own ball*, but `all_items` is built from a
  deduplicated member set that drops phrases the KG does record. 13 of GTLM's and
  24 of the serialised arm's offending strings were in the ball. Repaired by
  widening `all_items` to every collocation present in the ball. This helps the
  **serialised** arm more (+0.029 vs GTLM's +0.014).
* **`repeated_item` — a genuine model failure, not repaired.** 63 of GTLM's 115
  T17 items list the same phrase two or more times (`A | A | A | B`). These are
  verbatim repeats, not a normalisation collapse — the raw strings are identical.
  Greedy decoding with `repetition_penalty=1.0` is required for pass 1's
  equivalence argument, and this is what it costs on list-valued types. The
  grader is right to fail them; it is the decoding that should be explained.

T17 is therefore still the type where GTLM trails most (0.301 vs 0.452), and the
reason is degenerate repetition rather than retrieval. Worth a look before any
write-up, but it is a real finding, not an artefact.

Note T16 is **Tier C, held out of training**, so its near-majority score in every
arm is expected and is not evidence about any architecture.

### Input sequence length

Packed tokens per item — the sum over a graph's nodes of their `input_ids`, i.e.
what the model actually attends over. `train/analysis/length_stats.py`, all 12,490 items.

| arm | mean | min | p50 | p90 | p99 | max |
|---|--:|--:|--:|--:|--:|--:|
| GTLM (graph ball) | 1,597 | 32 | 1,361 | 2,695 | 5,760 | 14,123 |
| serialised graph | 2,547 | 50 | 2,166 | 4,464 | 9,532 | 16,382 |
| ratio | 1.59× | — | 1.59× | 1.66× | 1.65× | — |

| arm | nodes/item mean | p50 | p90 | p99 | max |
|---|--:|--:|--:|--:|--:|
| GTLM (graph ball) | 89.1 | 78 | 157 | 300 | 706 |
| serialised graph | 1.0 | 1 | 1 | 1 | 1 |

**The same ball costs ~60 % more tokens once flattened**, consistently across the
whole distribution. The overhead is structure that the graph gets for free: the
serialised form has to spell out an index per node and then an explicit
`POVEZAVE:` edge list, whereas GTLM carries adjacency in the bias rather than in
tokens. So the graph encoding is the more token-efficient of the two at equal
information — which is a real result and, unlike the accuracy comparison, is not
within noise.

Two consequences worth keeping in view:

* The serialised arm's max (16,382) is pressed against `build_variants.py`'s
  16,384-token bound — the 12 truncated rows are exactly the tail of this
  distribution, not a random sample.
* Length and the bias cost **about the same**, which the pre-run timing probe got
  wrong. Holding the bias off, runtime tracks length: 2.11 h at 1,597 tokens
  (`no bias`) against 4.16 h at 2,547 tokens (`serialised`) — 1.97× the time for
  1.6× the tokens, i.e. slightly superlinear, as quadratic attention over a
  length-sorted batcher should be. Holding length fixed, the bias costs the same
  again: 2.11 h → 4.16 h at 1,597 tokens. So GTLM and the serialised arm land on
  an identical 4.16 h by two different routes, and neither factor "dominates".

Full per-type breakdown: `train/results/arms_v2/report_arms.md`; length detail:
`train/results/arms_v2/length_stats.json`. Regenerate with:

```bash
.venv/bin/python -m train.analysis.report_arms train/results/arms_v2/runs.jsonl
sbatch train/slurm/run_length_stats.sbatch
```

## Is left padding really a no-op?

`train/checks/test_left_pad.py`, run on B200 across all three inputs and both stacks
(`run_preflight.sbatch` stage 2). Two claims, checked separately.

**The slice collapses, and the memory with it.** Ten batches per configuration —
short, long, and deliberately *mixed* (the shortest items paired with the
longest, which is the composition that made right padding blow up):

| configuration | worst mixed batch | `logits_to_keep` right → left | peak GiB right → left |
|---|--:|--:|--:|
| GTLM, graph balls | L = 8,192 | 8,166 → **13** | 82.8 → **4.7** |
| GTLM, serialised | L = 16,384 | 16,340 → **13** | 163.3 → **6.8** |
| plain, serialised | L = 13,965 | 13,921 → **13** | 138.2 → **8.5** |

The serialised GTLM row is the point: 163.3 GiB is inside a B200's 178 and
outside anything smaller, for a batch of four whose answers are a dozen tokens
each.

**The loss does not move — measured against a rounding floor, not a guessed
threshold.** In exact arithmetic a roll cannot change the loss: every real token
attends to the same keys with the same biases, `attention_mask` is where padding
is read from, `node_ids` travels with `input_ids`, and `position_ids` are
per-node *local* values under `node_position_mode='reset'`. In bf16 it can move
it, through two rounding channels — different kernel tiles, and (on the plain
stack, whose collator ships no `position_ids`) different absolute RoPE angles.

So the bound is calibrated rather than chosen. Each batch is also scored with
the *same* number of pad columns prepended to every row — same content, same
right-padded layout between rows, positions moved as far as left padding moves
them, all tiles redrawn — which is a rounding difference by construction. The
left-vs-right difference has to sit inside a small multiple of it.

| configuration | median &#124;Δ left−right&#124; / &#124;Δ control&#124; | max |
|---|--:|--:|
| GTLM, graph balls | 2.39 | 5.45 |
| GTLM, serialised | 0.94 | 9.16 |
| plain, serialised | 0.86 | 41.08 |
| plain, no retrieval | 0.92 | 1.86 |

A ratio of 1.0 means "indistinguishable from rounding". In absolute terms the
largest difference anywhere was 0.040 on a loss of 7.32 — 0.54 %, against bf16's
0.39 % per rounding.

Two earlier versions of this control measured exactly `0.000000` and are recorded
in the source so they are not re-invented: permuting the batch's rows (rows are
independent and the loss reduction is order-stable) and prepending a fixed 128
columns (exactly one flex block, so every real token keeps its offset inside its
block and the block-sparse kernel skips the added block outright).

The control is not fully non-circular — it puts pads at the front too, so a
hypothetical bug in front-padding would move the control and the measurement
together. **fp32 is the run that settles it** (stages 2e/2f): it shrinks every
rounding channel by four orders of magnitude, so a difference that is rounding
collapses with it and a difference that is semantic does not. It collapses:

| configuration | max &#124;Δ left−right&#124; in bf16 | in fp32 |
|---|--:|--:|
| GTLM, graph balls | 0.039643 | **0.000025** |
| plain, no retrieval | 0.027164 | **0.000003** |

Three orders of magnitude on the GTLM stack and four on the plain one — where
left padding also moves every absolute RoPE angle, so if anything were going to
break semantically that is where it would show. It does not. **Left padding is a
no-op on the loss and the answer-tail slice collapses from thousands of tokens to
tens.**

## Known caveat: Gemma-3's sliding window

`GTLMGemma3ForCausalLM` drops Gemma-3's `sliding_window=512`, because it is
defined over packed serialization order and would hide most of the graph from 22
of the 26 layers. Packed lengths here are 300–14,055, so it fires on nearly every
batch and those layers attend more widely than the pretrained model does.
Intended by the adapter, but it means logits do not match stock Gemma-3. The
warning is emitted through `warning_once` keyed on a message that embeds the
sequence length, so during decoding it re-fires every token; `train/_log.py`
keeps the first and drops the rest.

Dropping the window has a second consequence that is *not* uniform across arms:
those 22 layers use `rope_local_base_freq` = 10,000 and were trained only on
relative offsets within the window. Position resets keep the graph arms inside
that range (no node is longer than 324 tokens, so 0/2,184 items reach 512), while
the serialised arm — no nodes, no resets, 96 % of items past 512 — runs them far
outside it. Packed *length* exceeds 512 in both cases; relative *offset* does not.
See "What the contrasts do and do not isolate".
