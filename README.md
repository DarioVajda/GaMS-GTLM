# GaMS-KGQA: Slovenian Lexicographical Question Answering over Knowledge Graphs with GTLM

Applying the **GTLM** (Graph Transformer Language Model) architecture to **Slovenian
lexicographical question answering from knowledge graphs**. The goal is to adapt
**GaMS** — a Slovenian large language model with a Gemma backbone — into the GTLM
architecture, and train it on a dataset of knowledge graphs paired with
question–answer examples, where each question carries an extracted subgraph of the
global lexicographical KG.

## Motivation

Standard LLM approaches to graph reasoning compress rich textual node attributes into
single tokens via GNN encoders. GTLM instead injects graph-aware attention biases
directly into a pretrained LLM's attention modules, preserving full node semantics
while adding a negligible number of parameters and providing node-permutation
equivariance and backward compatibility with the base model
(see the [GTLM preprint](https://arxiv.org/abs/2605.10247)).

This project brings that capability to Slovenian lexicography: given a natural-language
question about the language (definitions, relations between lemmas, senses, etc.) and a
relevant subgraph of a large lexicographical knowledge graph, the adapted GaMS model
answers by reasoning jointly over text and graph structure.

## Relationship to `graph_model` (the `gtlm` library)

This repository is an **application** built on top of the core GTLM research codebase,
which lives in the sibling repository [`graph_model`](../graph_model) and is packaged as
the importable `gtlm` library. This repo depends on it as an **editable local install**,
so architecture, dataset tooling, and training utilities are reused rather than copied:

```python
from gtlm.utils import GraphCollatorV2          # collation / batching
from gtlm.models import ...                       # GTLM model classes
from gtlm.train import select_active_params, get_device
```

The dataset class used to adapt raw data for GTLM (`TextGraphDataset` and its
feature-computation methods — shortest-path distances, RRWP, magnetic Laplacian, …)
comes from `gtlm.utils`. See `graph_model`'s README for the full library surface.

> **Note on the Gemma/GaMS adapter.** The Gemma adapter **landed upstream on 2026-07-25**
> (`src/models/modeling_gtlm_gemma3.py` — `GTLMGemma3ForCausalLM` / `GTLMGemma3Config`).
> It targets **Gemma 3**, and it deliberately does **not** support Gemma 2; see the model
> scope below.

## Model scope

**We target Gemma 3 only.** This reverses the original plan, which named the Gemma 2–based
GaMS-2B as the primary target: the upstream adapter that actually exists is a Gemma 3 one,
and it refuses Gemma 2 by design.

| Model | Backbone | `model_type` | Logit softcapping | Status |
|-------|----------|--------------|-------------------|--------|
| **gemma-3-1b-pt** | Gemma 3 | `gemma3_text` | none | Plumbing/iteration proxy — loads today; Slovenian quality expected to be poor, to be measured |
| **gemma-3-4b-pt** | Gemma 3 | `gemma3` (multimodal) | none | Preferred iteration target — needs a small config unwrap, see below |
| **GaMS3-12B** | Gemma 3 | `gemma3_text` | none | **Primary target** — loads through the adapter as-is |
| **GaMS-2B** | Gemma 2 | `gemma2` | 50.0 / 30.0 | **Out of scope** — adapter raises |
| **GaMS-9B** | Gemma 2 | `gemma2` | 50.0 / 30.0 | **Out of scope** — adapter raises |

**Why Gemma 2 is excluded.** The shared GTLM stack applies neither softcapping site: the
registered `gtlm_*` attention functions ignore the `softcap` kwarg, and the causal-LM mixin
calls `lm_head` directly. Gemma 3 sets `attn_logit_softcapping` and
`final_logit_softcapping` to `null`, so both omissions are exact. Gemma 2 ships them at
`50.0` and `30.0`, so `GTLMGemma3ForCausalLM._sanitize_attn_config` **raises** rather than
silently train a backbone that no longer matches its pretrained weights. Supporting GaMS-2B
would mean adding both softcapping sites to the shared stack upstream — real work in a
repository that deliberately chose not to carry them.

**Iteration ladder.** Develop against a small Gemma 3, then scale to GaMS3-12B — the
adapter is the same, so scaling is a config change. Prefer **gemma-3-4b-pt** over
gemma-3-1b-pt: 1b is likely too weak at Slovenian out of the box to tell a broken pipeline
apart from a weak model. **To be settled experimentally** — run both on a Slovenian sample
before committing.

> **Gotcha for 4b.** `gemma-3-4b` and larger *multimodal* checkpoints nest their text config
> under `text_config` and will not load through `GTLMGemma3Config` directly
> (`modeling_gtlm_gemma3.py:106`). Loading one means pulling `Gemma3TextConfig` out of the
> nested field and taking weights from the `language_model` submodule. Small, but not free.
> GaMS3-12B is published as a text-only `gemma3_text` config and is unaffected.

## Dat

- **Knowledge graph:** the CJVT lexicographical knowledge graph
  ([source](https://nas.cjvt.si/s/aJE6243jd8iRXfc)).
- **QA dataset:** generated from the graph, not reused from the reference samples. Design
  decisions, the KG answerability census, and the frequency banding are recorded in
  [`data/QA_DATASET_DESIGN.md`](data/QA_DATASET_DESIGN.md). Two internal reference files
  (`data/datasets/reference/Lexical-QA-SLO-test.json`,
  `data/datasets/reference/Lexical-QA-SLO(in).csv`, both not public) supply
  the question-type inventory and phrasing style only.
- **Pipeline:** the raw data requires **heavy preprocessing** first, then adaptation
  into GTLM-ready examples using the `TextGraphDataset` class from `gtlm.utils`. Each
  training example is a graph (a NetworkX `DiGraph`) with per-node `text` attributes, a
  designated `prompt_node`, and question+answer text on that prompt node; question tokens
  are masked to `-100` so the model is supervised on the answer only. Per-question
  subgraphs are extracted from the global KG.

> The repository layout (e.g. `src/` with `scripts/`, `data/`, `train/`) will be added
> as the preprocessing and training code is written.

## Repository setup

Requires **Python 3.10** (matching `graph_model`). Dependencies are managed with
`pip-tools`: `requirements.in` is the high-level spec, `requirements.txt` is the
generated lockfile. This project's `requirements.in` mirrors `graph_model`'s and adds an
editable install of the `gtlm` library from the sibling checkout.

```bash
# From the repository root (/shared/workspace/povejmo/gams_kgqa):
python -m venv .venv          # create the environment (Python 3.10)
source .venv/bin/activate      # activate it
pip install pip-tools          # install pip-tools
pip-compile                    # compile requirements.in -> requirements.txt
pip-sync                       # install the locked requirements (incl. editable gtlm)
```

`pip-sync` installs `gtlm` in editable mode from `../graph_model`, so a `git pull` in
that repository is picked up on the next run without reinstalling. This assumes the
`graph_model` checkout sits next to this one (`../graph_model`); adjust the `-e` path in
`requirements.in` if your layout differs.

From here on, assume every command runs inside the activated `.venv`.

### Authentication (Hugging Face + Weights & Biases)

`hf_login.sh`, `wandb_login.sh`, and `login.sh` are provided as placeholders and are
**gitignored** (they hold secrets — never commit real tokens). Insert your tokens, then:

```bash
chmod +x *.sh
./login.sh        # runs hf_login.sh (HF Hub) then wandb_login.sh (W&B)
```

GaMS models are gated on the Hugging Face Hub, so an authenticated HF login is required
to download them.

## Status

Environment, dependency lock, and the KG construction study are in place; no training has
started. The upstream Gemma 3 adapter has landed, so the backbone is no longer a blocker.

- **Done.** Raw KG downloaded and characterised; GTLM graph construction settled at **v3**
  (36.7 M nodes / 48.5 M edges, untyped edges with self-describing node text) together with
  a k-hop input-sizing study — see [`data/README.md`](data/README.md).
  The graph is now **persisted** — current store `data/stores/kg_graph_v4_gemma3/`, built
  by `data/build/run_save_v4.sbatch` — so the ~35-minute, ~70 GB rebuild is paid
  once instead of per run; `data/lib/graph_store.py` loads it in seconds. Its
  `token_len` uses the Gemma 3 tokenizer, which every Gemma 3 size shares with
  GaMS3-12B, so one store serves the whole iteration ladder below.
- **Next.** Write the subgraph extractor on top of the store, with a cap on the
  `sestavina` MWE↔word hub, the one structure that explodes; adapt into
  `TextGraphDataset`; then train on a small Gemma 3 before scaling to GaMS3-12B.
- **Open.** The lexicographical QA dataset — questions and answers anchored to KG entities —
  is not yet generated. Its design is **locked** as of 2026-08-18; see
  [`data/QA_DATASET_DESIGN.md`](data/QA_DATASET_DESIGN.md) for the decisions, the open
  items, and the measurements behind them.
