# GaMS-KGQA — Slovenian lexicographical QA over a knowledge graph with GTLM

Given a Slovene question about a word — its declension, its senses, its
collocations — and the neighbourhood of that word in the CJVT lexicographical
knowledge graph, answer it by reasoning over text and graph structure together.

The architecture is **GTLM** ([preprint](https://arxiv.org/abs/2605.10247)): instead
of compressing a node's text into a single token through a GNN encoder, it injects
graph-aware attention biases directly into a pretrained LLM's attention modules.
Node semantics survive in full, the parameter cost is negligible, and the model stays
node-permutation equivariant and backward compatible with its backbone. The backbone
here is Gemma 3, and the target is [**GaMS3-12B**](https://huggingface.co/cjvt/GaMS3-12B), a Slovenian model built on it.

```bash
# 0. the source KG: https://nas.cjvt.si/s/aJE6243jd8iRXfc
#    unpack it to data/kg_raw/OntoLex DSB/ (83 GB, gitignored)
sbatch data/run_pipeline.sbatch                      # 1. the store AND the QA
                                                     #    dataset, one job, ~50 min
.venv/bin/python -m sweep train train/configs/arms_v3.jsonc   # 2. train
```

Step 1 is six stages in one job — graph store, items, entity linking, relabel,
balls, baselines — and it is the preferred way to build the data. Each stage can
also be run on its own while you iterate on it; both paths are documented in
[`data/README.md`](data/README.md).

A fresh clone carries code and documentation only — the raw KG, the built store,
the generated dataset and the checkpoints are all gitignored, and steps 0–2 above
are what produce them.

| | |
|---|---|
| [`data/README.md`](data/README.md) | the graph store and the QA dataset — how to build both, and what they contain |
| [`data/QA_TASKS.md`](data/QA_TASKS.md) | the 19 question types: what each asks, how it is generated, how it is graded |
| [`data/QA_DATASET_DESIGN.md`](data/QA_DATASET_DESIGN.md) | dataset design decisions and the KG answerability census |
| [`train/README.md`](train/README.md) | fine-tuning, evaluation, and the six-arm study |

## Where it stands

* **Graph store** — built. 37.5 M nodes / 50.1 M edges of untyped, self-describing
  text nodes, persisted so the ~15-minute, ~48 GB rebuild is paid once and loads in
  seconds under a gigabyte.
* **QA dataset** — built. 12,490 items over 19 types, each paired with the subgraph
  holding its evidence, plus two baseline inputs (the same graph flattened into the
  prompt, and no graph at all). There is one dataset and it carries no version
  suffix: `data/datasets/generated` and `data/datasets/balls` are it.
* **Training** — the six-arm × three-seed study has run to completion on B200, and
  passes its pre-registered convergence rule. `train/analysis/report_arms` prints the
  tables from `train/results/arms_v3/runs.jsonl`. See `train/README.md`.

## Model scope

**Gemma 3 only.** The upstream adapter (`GTLMGemma3ForCausalLM`) targets Gemma 3 and
refuses Gemma 2 by design, so GaMS-2B and GaMS-9B are out of scope: both ship attention
and final logit softcapping, which the shared GTLM stack applies at neither site, and
the adapter raises rather than silently train a backbone that no longer matches its
pretrained weights. Gemma 3 sets both to `null`, so the omission is exact.

| model | role |
|---|---|
| `gemma-3-1b-it` | current backbone — what the study runs on |
| **GaMS3-12B** | the target; published as a text-only `gemma3_text` config, so it loads through the adapter as-is |
| `gemma-3-4b/12b/27b` (multimodal) | not wired: they nest their text config under `text_config`, which needs an unwrap and weights from the `language_model` submodule |

Scaling from the 1b backbone to GaMS3-12B is a config change, not a port.

## Repository setup

Requires **Python 3.10**, matching the sibling [`graph_model`](../graph_model)
checkout that provides the `gtlm` library (see the [GitHub repo](https://github.com/DarioVajda/graph_model)). Dependencies are managed with `pip-tools`:
`requirements.in` is the spec, `requirements.txt` the lockfile.

```bash
python -m venv .venv
source .venv/bin/activate
pip install pip-tools
pip-compile && pip-sync
```

`pip-sync` installs `gtlm` in editable mode from `../graph_model`, so a `git pull`
there is picked up on the next run. Adjust the `-e` path in `requirements.in` if your
layout differs. `sweep`, the harness step 3 uses, comes from the same checkout.

`pyproject.toml` declares the project and the tooling config. `train` is imported
from the repo root; the `data/` pipeline packages (`qa`, `build`, `lookup`, `lib`,
`analysis`) are run with `-m` from `data/`, which is where their job scripts `cd` to.

GaMS models are gated on the Hugging Face Hub, so an authenticated login is required
to download them — copy the templates, insert your tokens, then run `./login.sh`:

```bash
cp hf_login.example.sh hf_login.sh
cp wandb_login.example.sh wandb_login.sh
```

From here on, every command assumes the activated `.venv`.

## A note on nodes

The shared venv is built against Ubuntu 22.04 / Python 3.10, which on this cluster
means **`aga`, `ana` and `apl`** for CPU work; the other nodes ship Python 3.12 and
cannot see its `site-packages`. GPU work runs on B200 inside the pyxis container,
which is a py3.10 base. Every job script already pins the right target — this is here
so an unexplained `ModuleNotFoundError: numpy` is recognisable.
