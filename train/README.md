# `train/` — fine-tuning GTLM on the generated lexicographical QA dataset

Answer a Slovene question about a headword given that headword's neighbourhood in
the CJVT/DDDS dictionary graph, encoded as a text graph.

```bash
# one run
.venv/bin/python -m train --types T3,T4,T9,T10 --num-epochs 8      # (needs a GPU)
sbatch train/run_train.sbatch --num-epochs 8 --include-f1

# a grid, one Slurm job per configuration
.venv/bin/python -m sweep train train/configs/bias_ablation.jsonc

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
* **No `container` key** in `train/configs/*.jsonc`. The runner only wraps in
  pyxis when one is given, and `ana`/`aga` run Ubuntu 22.04 / python3.10
  natively, which is what this `.venv` is built against. A B200/B300 arm would
  need the container back.

`gtlm.train` (shared training utilities, over there) and this `train` package are
different things despite the name; the import is always spelled in full.

## Where the data comes from

Two artefacts, both built by `data/qa/` and both explicit files on disk:

| artefact | built by | what it is |
|---|---|---|
| `data/datasets/generated/v1` | `qa/build_dataset.py` | 12,493 QA items over 19 task types, with gold answers, bands, splits and a grader |
| `data/datasets/balls/v1` | `qa/build_balls.py` | one text graph per item — node texts verbatim from the KG, plus an edge list |

Nothing here opens the 37.5 M-node store. The ball is already an artefact, which
is what lets a GPU job start cold and what makes the graph the model saw part of
the record rather than a side effect of whatever the extractor did that day.

Graph layout per item:

```
node 0 .. N-1   the ball  (iztočnica: gora (…) / oblika: gore (…) / pomen: … / …)
node N          the PROMPT node,  "{question}\nODGOVOR: {answer}"
edges           the ball's parent->child edges, plus prompt -> anchor
```

Typical size is **32 nodes** (p90 42, max 62), packing to 550–780 tokens.

**The ball is uniform across task types on purpose.** Giving T3 the paradigm and
T9 the anchor alone would be cheaper and would also make every score meaningless,
because the model would be told which relation to read before it read anything.
The caps in `build_balls.py` are a token budget, not a relevance judgement.

## Label masking

Done on **character offsets**, not by searching for the delimiter's token-id
subsequence. `"ODGOVOR:"` alone and the same string preceded by a newline need
not tokenize identically, and that failure is silent — it masks the wrong span
rather than raising. `check_labels.py` decodes the supervised span so a person
can confirm it; it currently reports 10/10.

## Results so far (2026-08-22, seed 42, one run per arm)

Four types — T3 (single case form), T4 (lemmatisation), T9 (noun gender),
T10 (verb aspect). 2,084 train / 235 dev / 460 test graphs. gemma-3-1b-it,
LoRA r=32, 8 epochs, ~80 min on one A100-80GB.

| arm | test EM | test F1 | best val EM |
|---|--:|--:|--:|
| **SPD + magnetic** | **0.7283** | 0.4653 | 0.7617 |
| no bias (control) | 0.5978 | 0.2808 | 0.6553 |
| majority-class baseline | 0.1065 | — | — |

The control is the same data, the same backbone and the same schedule with the
bias modules off — the model still reads the same packed node texts in the same
order — so the **+13.0 point** gap is attributable to the structural bias and not
to the backbone's Slovene. Both arms are far above the majority-class baseline,
which is the only reason the comparison means anything; per-type baselines
(T9 0.426, T10 0.357, T3 and T4 0.087) are written into every record.

One run per arm, one seed: this is a **verification that the pipeline works**,
not a measured effect size. `train/configs/bias_ablation.jsonc` re-runs both arms
over three seeds to put an error bar on that gap.

## Known caveat: Gemma-3's sliding window

`GTLMGemma3ForCausalLM` drops Gemma-3's `sliding_window=512`, because it is
defined over packed serialization order and would hide most of the graph from 22
of the 26 layers. Packed lengths here are 550–780, so the warning fires on nearly
every batch and those layers attend more widely than the pretrained model does.
Intended by the adapter, but it means logits do not match stock Gemma-3.
