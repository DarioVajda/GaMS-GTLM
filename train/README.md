# `train/` — fine-tuning GTLM on the lexicographical QA dataset

Answer a Slovene question about a headword, given that headword's neighbourhood in
the CJVT/DDDS dictionary graph encoded as a text graph. The backbone is
`google/gemma-3-1b-it` with LoRA adapters; the graph reaches it as attention bias
rather than as tokens.

```bash
# the study: six configurations x three seeds, one Slurm array, all on B200
.venv/bin/python -m sweep train train/configs/arms_v3.jsonc

# everything that must pass on the real hardware first
sbatch train/slurm/run_preflight.sbatch

# one run, standalone (needs a GPU)
.venv/bin/python -m train --num-epochs 8

# the tables: per arm, per type, the contrasts, the convergence verdict
.venv/bin/python -m train.analysis.report_arms train/results/arms_v3/runs.jsonl
```

Everything runs **from the repo root**, with the repo's own `.venv/bin/python`:
`sweep` refuses to submit when the calling interpreter lives outside the project
root, because it forwards that path to the job and it would not resolve there.

## Layout

```
train/
  config.py  data.py  run.py  evaluate.py    the training path
  batching.py  chat.py  qa_contract.py        shared by it
  _io.py  _log.py                             jsonl records, log noise
  __main__.py                                 one run: python -m train
  checks/     check_labels  check_left_pad  check_two_pass  probe_eval
  analysis/   report_arms  analyse_t17  length_stats  project_cost
  slurm/      run_train  run_preflight (+ preflight_stages.sh)  run_length_stats
  configs/    the sweep definitions
  results/    sweep output (gitignored)
```

The split is by **direction of dependency**: nothing in `checks/`, `analysis/` or
`slurm/` is imported by the training path, so a module's directory tells you whether
breaking it can break a run.

## How this uses `graph_model`

The GTLM stack is a dependency, editable-installed into this repo's `.venv`:
`gtlm.models`, `gtlm.train`, `gtlm.utils` and `sweep` all resolve to the sibling
`graph_model` checkout. Experiments are not part of that package, which is the
argument for lexicography-specific ones living here. Two consequences:

* **`results_dir` must be set in every sweep config**, or results land in the other
  repo.
* **A `container` key is required**, because the Blackwell hosts are Ubuntu 24.04 /
  python3.12 and this venv is 3.10.

`gtlm.train` and this `train` package are different things despite the name; the
import is always spelled in full.

## The data

Three artefacts, all built by `data/qa/` (see `data/README.md`) and all explicit
files on disk:

| artefact | what it is |
|---|---|
| `data/datasets/balls/v2_clean` | one text graph per item — node texts verbatim from the KG, plus an edge list |
| `data/datasets/generated/v2_clean` | the items: the same answer, plus the **grading contract** |
| `data/datasets/balls/v2_clean_{serialised,noretrieval}` | the two baseline inputs, derived from the ball file itself |

Nothing here opens the 37.5 M-node store. `data.py` joins ball and item on `id` and
asserts the two answers are identical, so a mismatched pairing raises rather than
producing numbers that look comparable and are not.

Graph layout per item:

```
node 0 .. N-1   the ball  (iztočnica: gora (…) / oblika: gore (…) / pomen: … / …)
node N          the PROMPT node, the item as one chat turn pair
edges           the ball's parent->child edges, plus prompt -> EVERY anchor
```

The prompt attaches to every anchor, not just the first: entity linking returns a
union when a surface string owns several lexical units (9.34 % of items), and
choosing among them is the model's job.

## Prompt format

The backbone is instruction-tuned, so the prompt node is written in its own turn
format, by its own template (`train/chat.py`):

```
<bos><start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
ODGOVOR: {answer}<end_of_turn>
```

Where it is applied differs by stack, and that difference is the point. On the plain
stack the ball has zero graph nodes, so the template wraps the whole sequence. On
the GTLM stack it wraps **the prompt node only** — the ball's nodes stay verbatim KG
text, because they are not dialogue but retrieved context reached through the
structural bias; wrapping each would assert hundreds of turns that never happened.

That puts `<bos>` mid-sequence on the GTLM stack, which is deliberate:
`node_position_mode='reset'` gives every node its own positions from 0, so the
prompt node genuinely is a sequence start as the model sees it.

Two consequences, both handled: the stop token is `<end_of_turn>` rather than
`<eos>`, and it is part of the *text*, so `max_length` could truncate it away —
`OffsetLabelMasker` checks the prompt node's last id and raises instead. And the
generation copy is produced by truncating the full text at the last `\nODGOVOR:`,
which yields exactly the template's own generation prompt and guarantees that the
teacher-forced and generation copies share a byte-identical prefix.

Label masking is done on **character offsets**, never by searching for a token-id
subsequence: `"ODGOVOR:"` and the same string preceded by a newline need not
tokenize identically, and that failure is silent — it masks the wrong span rather
than raising. `train.checks.check_labels` decodes the supervised span so a person
can confirm it.

## Evaluation is the dataset's own grader

`accuracy` is the share of items whose answer satisfies `qa/grade.py`, reported
overall and per type, and `run.py` selects the best checkpoint on that same number —
no run trains against one objective and selects against another.

The shared stack's token-level exact match answers the wrong question for six of the
nineteen types: `multiset` types (T4 T15 T16 T20 T21) would be penalised for a
correct answer in another order, and T17's `membership` contract would score zero
for a model naming five valid collocations.

Generation over the whole dev split at every eval is the dominant cost, and most of
it is avoidable, so evaluation runs in **two passes**:

* **Pass 1** — one teacher-forced forward. If the argmax at every answer position is
  the gold token, greedy decoding would have emitted exactly that string. This is
  sufficient in every grading mode, and settles ~75 % of the dev split before a
  token is generated.
* **Pass 2** — greedy generation for the remainder, then grade.

Pass 1 never declares failure, only success: a model emitting a different
tokenisation of the same text falls through to pass 2 and is graded properly. In
training evals, pass 2 is skipped where a token mismatch already condemns the item.

Three things keep this inside its budget without making the number depend on the
machine: `eval_max_batch` is independent of the training micro-batch; token budgets
are scaled from the device's memory rather than a constant, and the realised values
go in the run record; and an OOM guard halves and retries, logging every fallback —
a fallback during a *final* eval is surfaced, since re-grouping changes padding and
a bf16 near-tie can flip an argmax.

Greedy decoding is pinned explicitly (`use_model_defaults=False` plus the flags as
`generate()` kwargs), because since transformers 4.50 any field left at the library
default is back-filled from the model's own generation config — and this
checkpoint's is `do_sample=True`.

**The acceptance test** scores the same slice twice, once through the two-pass
evaluator and once by generating every item, and requires identical per-item
success:

```bash
.venv/bin/python -m train.checks.check_two_pass --max-items 32                 # untrained
.venv/bin/python -m train.checks.check_two_pass --max-items 32 --checkpoint …  # trained
```

The trained form is the one that matters — on an untrained model nothing clears pass
1 and the assertion is vacuous — which is why `run_preflight.sbatch` overfits 32
items first.

## The schedule

Identical in every arm, so that the matrix below measures the arms and nothing else.

* **Effective batch 16 everywhere**, as `batch_size × accumulation_steps`. Only the
  product is scientifically binding; the factorisation is a memory/throughput knob
  set per arm, and the record carries both.
* **`max_steps` is derived from the product**, not from HF's
  `len(dataloader) // accumulation_steps` — that expression depends on the
  factorisation and would give two arms different step counts. 9,266 // 16 = 579
  steps/epoch × 8 epochs = **4,632 steps**, with `dataloader_drop_last` so every step
  is exactly 16 items.
* **12 evaluations, the last at step 4,632.** HF's Trainer does not evaluate at the
  end of training, so without `run.py`'s `EvaluateOnFinalStep` the last 232 steps
  would be trained and silently discarded.

## Memory: three things that make long balls trainable

* **Batches are LEFT-padded** (`run.LeftPadCollator`). `compute_loss` slices the
  logits to the answer tail using the earliest supervised position in the batch, so
  right-padded, one short row drags that slice back for everyone: the worst mixed
  GTLM batch needed `logits_to_keep = 8,166` and peaked at 82.8 GiB, against **13**
  and **4.7 GiB** left-padded.
* **Logits, not attention, are the memory wall.** At a 262 k vocabulary,
  full-sequence logits cost tens of gigabytes for a batch of four at the p99 length,
  and the longest ball could not be trained at all. The supervised span is the tail
  of the packed sequence, so the loss slices both logits and labels to it — the loss
  is arithmetically unchanged and the memory is a few hundred megabytes.
* **Evaluation batches are formed by token budget**, not item count: a corpus running
  from 300 to 14,055 packed tokens has no single sensible batch size.

Left padding being a no-op is checked, not assumed (`train.checks.check_left_pad`, run
across both stacks and all three inputs). In bf16 the left-vs-right difference sits
inside a calibrated rounding control; in fp32 it collapses by three to four orders of
magnitude, which is what distinguishes rounding from semantics.

## The run matrix

`train/configs/arms_v3.jsonc` — **six configurations × three seeds = 18 runs**, one
sweep call, one `runs.jsonl`.

| # | configuration | input | stack | biases |
|---|---|---|---|---|
| 1 | **GTLM (spd + magnetic)** | `balls/v2_clean` | GTLM | spd + magnetic |
| 2 | **GTLM, no bias** | `balls/v2_clean` | GTLM | none |
| 3 | **serialised — GTLM stack** | `balls/v2_clean_serialised` | GTLM | none |
| 4 | **serialised — plain stack** | `balls/v2_clean_serialised` | plain | n/a |
| 5 | **no retrieval — plain stack** | `balls/v2_clean_noretrieval` | plain | n/a |
| 6 | **no retrieval — GTLM stack** | `balls/v2_clean_noretrieval` | GTLM | none |

Arms 3–6 ship a ball with **zero graph nodes** — the subgraph is flattened into the
prompt (3, 4) or absent (5, 6) — so arms 4 and 5 differ in exactly one thing:
whether the graph's text is in the prompt.

The comparisons the matrix exists to make, each reported **paired by seed**:

| contrast | arms | what it isolates |
|---|---|---|
| Retrieval | 4 vs 5 | is the subgraph worth anything at all |
| Structural bias | 1 vs 2 | does the bias make the graph usable |
| **Graph encoding** | **1 vs 3** | encoding, stack held fixed — the headline claim |
| Stack cost | 3 vs 4 | what SDPA + `sliding_window=512` is worth on a long input |
| Stack control | 5 vs 6 | the same stack difference where it *cannot* act — expect ≈ 0 |

**Why the stack is worth three runs.** Serialised prompts are p50 2,166 / p99 10,014
tokens, and stock Gemma-3 applies `sliding_window=512` in 22 of its 26 layers, which
the GTLM adapter drops. That is a large capability difference sitting directly on top
of the encoding claim, and without arm 3 there is no way to tell the two apart. Arm 4
is also the right arm for any throughput comparison: routing a plain-text baseline
through the GTLM interface would deny it FlashAttention and flatter GTLM for a reason
that is not the model.

**Why arm 6 is a control.** The no-retrieval input is p50 ~35 tokens, comfortably
inside the window, so the factor has no room to act. What it buys is falsification:
if the plain/GTLM difference really is a length-and-window interaction, arms 5 and 6
must land on top of each other.

## Results

The sweep is complete: 18 records in `train/results/arms_v3/runs.jsonl`, six arms ×
three seeds, all carrying `"prompt_format": "chat_template"`. Read them with

```bash
.venv/bin/python -m train.analysis.report_arms train/results/arms_v3/runs.jsonl
```

which prints the per-arm table, the per-type breakdown, the five contrasts paired by
seed, and the convergence verdict. The pre-registered convergence rule **passes**:
the best checkpoint was the final one in 4 of 18 runs (22 %), under the 33 % limit,
so the sweep is measuring where the arms end up rather than how fast they learn.

`train/results/arms_v3_preformat/` holds the superseded sweep of the same six arms,
trained on a bare `"{question}\nODGOVOR: {answer}"` string — no `<bos>`, no turn
markers, the wrong terminator — against an instruction-tuned checkpoint. Those
records carry no `prompt_format` key; that absence is what marks them, and the
adapters under `checkpoints/sl_qa/arms_v3_*` from that sweep must not be evaluated
with the new format.

One measurement is a property of the inputs rather than of any run. Over all 12,490
items, the same ball costs **~60 % more tokens once flattened**
into a prompt — p50 1,361 packed tokens against 2,166, consistently across the whole
distribution. The overhead is structure the graph gets for free: the serialised form
spells out an index per node and an explicit edge list, where GTLM carries adjacency
in the bias.

## Why these runs are on B200

Every run is on B200 **inside the pyxis container**, and that is a requirement: a gap
measured across two hardware or software stacks is not a measurement of the thing
being compared, so one tier has to serve all 18 runs. Blackwell hosts are Ubuntu
24.04 / python3.12 and this venv is 3.10, so a bare job dies on imports; the
container is a py3.10 base and the venv resolves inside it.

Two things not to undo:

* **Do not substitute `PYTHONPATH` for running the venv interpreter directly.** It
  leaves the image's own `dist-packages` on the path, so the container's torchvision
  loads against our torch and transformers dies on `operator torchvision::nms does
  not exist`.
* **B200 only, not `B200|B300`.** Of the B300 hosts one is drained and one cannot
  start the container (a host-config fault); listing them would let part of the array
  land where nothing runs.

## Known caveat: Gemma-3's sliding window

The GTLM adapter drops Gemma-3's `sliding_window=512`, because the window is defined
over packed serialization order and would hide most of the graph from 22 of the 26
layers. Packed lengths here are 300–14,055, so it fires on nearly every batch: those
layers attend more widely than the pretrained model does, and logits do not match
stock Gemma-3.

Dropping it has a second consequence that is **not** uniform across arms. Those 22
layers use `rope_local_base_freq` = 10,000 and were trained only on relative offsets
within the window. Position resets keep the graph arms inside that range — no node is
longer than 324 tokens — while the serialised arm has no nodes and no resets, and 96 %
of its items run those layers far outside it. Packed *length* exceeds 512 in both
cases; relative *offset* does not.
