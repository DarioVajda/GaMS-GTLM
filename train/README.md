# `train/` — fine-tuning GTLM on the generated lexicographical QA dataset

Answer a Slovene question about a headword given that headword's neighbourhood in
the CJVT/DDDS dictionary graph, encoded as a text graph.

```bash
# one run
.venv/bin/python -m train --num-epochs 4                     # (needs a GPU)
sbatch train/run_train.sbatch

# the four-arm matrix, one Slurm job per configuration
.venv/bin/python -m sweep train train/configs/arms_v2.jsonc

# the two cheap checks + the evaluator's acceptance test
sbatch train/run_smoke.sbatch

# what the label mask actually supervises, decoded, for a human to read
.venv/bin/python -m train.check_labels --n 10
```

Everything runs **from the repo root**. `sweep` refuses to submit when the calling
interpreter lives outside the project root — it forwards `SWEEP_VENV_BIN` to the
job, and a path outside the root would not resolve there.

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
* **No `container` key** in `train/configs/*.jsonc`, and therefore no Blackwell
  arm. See "Why these runs are on A100-80GB" below — this is a property of this
  repo's `.venv`, not a preference.

`gtlm.train` (shared training utilities, over there) and this `train` package are
different things despite the name; the import is always spelled in full.

## Where the data comes from

Three artefacts, all built by `data/qa/` and all explicit files on disk:

| artefact | built by | what it is |
|---|---|---|
| `data/datasets/balls/v2` | `qa/build_balls.py` | one text graph per item — node texts verbatim from the KG, plus an edge list |
| `data/datasets/generated/v2_final` | the same pass | the dataset rows, carrying the **grading contract** and the same `answer` the ball ships |
| `data/datasets/balls/v2_{serialised,noretrieval}` | `qa/build_variants.py` | the two baseline inputs, derived from the ball file itself |

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
(`qa/build_balls.py:reverbalise`). `generated/v2_final` is the authority; nothing
downstream should read `v2_relabelled`.

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
.venv/bin/python -m train.test_two_pass --max-items 32                 # untrained
.venv/bin/python -m train.test_two_pass --max-items 32 --checkpoint …  # trained
```

Scores the same slice twice — once through the two-pass evaluator, once by
generating every item — and requires **identical per-item `success`**, plus
`prediction == gold` for every item pass 1 passed. `max_batch=1` makes generation
padding-free so the two runs feed the model bitwise identical tensors; otherwise
a bf16 near-tie could flip an argmax between two groupings and fail the test for
a reason that has nothing to do with the claim. The trained-checkpoint form
matters: on an untrained model nothing clears pass 1 and the sharper assertion is
vacuous, which is why `run_smoke.sbatch` overfits 32 items first.

## Label masking

Done on **character offsets**, not by searching for the delimiter's token-id
subsequence. `"ODGOVOR:"` alone and the same string preceded by a newline need
not tokenize identically, and that failure is silent — it masks the wrong span
rather than raising. `check_labels.py` decodes the supervised span so a person
can confirm it.

## Cost: what balls/v2 changed

`train/config.py` used to justify skipping a feature cache on the grounds that
"these graphs are ~32 nodes, so the magnetic Laplacian is a 32×32
eigendecomposition". `balls/v2` is p50 77 / p99 293 / max 705 nodes, so that
premise is dead. Two consequences, both handled:

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

## Why these runs are on A100-80GB

B300/B200 would be faster, and `sweep`'s pyxis wrap exists for exactly that. It
does not work for **this** repo today, for two independent reasons, both re-tested
on 2026-08-22:

* `ixb7` still fails to start any container — its enroot mellanox hook exits 1 on
  a missing `/dev/infiniband/rdma_cm` (unchanged since its 09:36 reboot).
* `ixb1` starts the container fine, and then cannot run our interpreter:
  `.venv/bin/python` is an absolute symlink to `/opt/deepops/venv/bin/python`,
  which does not exist inside the image. `graph_model`'s venv points at
  `/usr/bin/python3` instead, which is why the stage-2 extraction job works there
  and this would not.

The same symlink is why bare (containerless) jobs on `ixh` and `axa` die with
`ModuleNotFoundError: No module named 'torch'`. `ana` resolves it, so `ana` is
where the arms run — and every arm runs on the same node type with the same
environment, which is a requirement rather than a convenience: a gap measured
across two software stacks is not a measurement of the thing being compared.

## The run matrix

`train/configs/arms_v2.jsonc` — four arms × three seeds, everything else shared.

| arm | input | biases |
|---|---|---|
| **GTLM** | `balls/v2` | spd + magnetic |
| **GTLM, no bias** | `balls/v2` | none |
| **serialised graph** | `balls/v2_serialised` | none (one prompt node) |
| **no retrieval** | `balls/v2_noretrieval` | none (one prompt node) |

The two baselines ship a ball with **zero graph nodes**, so they differ from each
other in exactly one thing — whether the graph's text is in the prompt — and from
GTLM in how that same information is encoded. The serialised prompt is bounded at
16,384 tokens, which costs 12 rows of 12,490 (0.10 %) a suffix of their node list;
`qa/check_variants.py` verifies every kept node text is present verbatim and
reports the truncations.

> The previously recorded **+13.0-point** bias gap was measured on `balls/v1` — a
> hand-capped ~32-node ball built by a different and methodologically flawed
> policy. It has now been **re-measured on `balls/v2` and superseded**: the gap is
> **+32.3 points**. Do not cite the +13.0 figure.

## Results (2026-08-23)

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
and `train/rescore.py` **asserts monotonicity** — no answer that was correct
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
what the model actually attends over. `train/length_stats.py`, all 12,490 items.

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
.venv/bin/python -m train.report_arms train/results/arms_v2/runs.jsonl
sbatch train/run_length_stats.sbatch
```

## Known caveat: Gemma-3's sliding window

`GTLMGemma3ForCausalLM` drops Gemma-3's `sliding_window=512`, because it is
defined over packed serialization order and would hide most of the graph from 22
of the 26 layers. Packed lengths here are 300–14,055, so it fires on nearly every
batch and those layers attend more widely than the pretrained model does.
Intended by the adapter, but it means logits do not match stock Gemma-3. The
warning is emitted through `warning_once` keyed on a message that embeds the
sequence length, so during decoding it re-fires every token; `train/_log.py`
keeps the first and drops the rest.
