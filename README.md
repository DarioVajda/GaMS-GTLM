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

> **Note on the Gemma/GaMS adapter.** GTLM currently ships a Llama modeling
> implementation. The Gemma-backbone GTLM adapter needed for GaMS is being developed
> **upstream in `graph_model`**; this repository consumes it once available.

## Model scope

| Model | Backbone | Status |
|-------|----------|--------|
| **GaMS-2B** | Gemma 2 | Primary development target (smaller/faster to iterate) |
| **GaMS-9B** | Gemma 2 | Scale-up target — same architecture, so straightforward once 2B works |
| **GaMS3-12B** | Gemma 3 | Future goal — different architecture, requires a new GTLM adapter |

We focus on **GaMS-2B first**, then **GaMS-9B**. The Gemma 3–based GaMS3-12B is a longer-term
objective that will need a separate adapter and is out of scope for the initial work.

## Data

- **Knowledge graph:** the CJVT lexicographical knowledge graph
  ([source](https://nas.cjvt.si/s/aJE6243jd8iRXfc)).
- **QA dataset:** an initial internal sample of question–answer pairs (not public).
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

Early scaffolding. Current repo provides the environment, dependency lock, and
documentation. Next steps: raw KG/QA preprocessing, subgraph extraction, adaptation into
`TextGraphDataset`, and training the Gemma-adapted GaMS-2B once the upstream GTLM Gemma
adapter lands.
