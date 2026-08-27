#!/usr/bin/env python3
"""Loading: the extractor, the GTLM checkpoint, and the store (D3, D4, D5, D17).

The GTLM side is one call.  `GTLMGemma3ForCausalLM.from_pretrained(ckpt,
graph_attn_impl=...)` (graph_model, `src/models/causal_lm.py:490`) reads the
checkpoint's `base_model_name_or_path`, loads that base, wraps it in the LoRA
adapters and calls `load_bias_parameters` -- base, adapters and graph-bias in a
single line, with the attention implementation overridable, which is what D17's
eager fallback needs.

What the checkpoint does *not* carry is `magnetic_m` (V1): it self-describes the
model, not the features.  Every `arms_v3` arm used the `RunConfig` default of 0,
and a future mismatch fails loudly as a shape error rather than degrading, so
the default is read from `RunConfig` and printed at startup.

**Why two models, when one would do.**  The extractor is `GaMS3-12B-Instruct`
and today's GTLM is `gemma-3-1b-it`, so they are simply different weights.  Once
a GaMS3-12B-based GTLM exists, the two collapse: load it once, run the extraction
turn with `disable_adapter()` (plain backbone, no graph bias) and the answering
turn with the adapters on, saving ~24 GB and one load.  The check is on the
GTLM's `base_model_name_or_path`; the dispatch point is `load()` below, and this
note is deliberately duplicated there so it is found by whoever next reads that
code and not only by whoever reads this docstring.
"""

# The checkpoint answered from unless --checkpoint says otherwise: the best test
# accuracy of the arms_v3 sweep (0.7917, 1729/2184).  Seed 0 ties on test and
# loses on dev (0.8683 vs 0.8769); the 0.0028 spread is within seed noise.
# A path to a real checkpoint directory, loaded directly -- `ask` never searches
# a run directory for a best-of.
DEFAULT_CHECKPOINT = (
    "checkpoints/sl_qa/"
    "arms_v3_0002_data_rootdata-datasets-balls-v2_c_plain_llmFalse_spdTrue_"
    "magneticTrue_max_length2048_batch_size4_accumulation_steps4_seed2/"
    "checkpoint-4400")

DEFAULT_EXTRACTOR = "cjvt/GaMS3-12B-Instruct"


def load_store(path, ui):
    """Open the processed graph store and its surface index."""
    raise NotImplementedError("ask.backbone.load_store is a stub; run --demo")


def load_extractor(name, ui):
    """Load the extraction model.  Splits out of `measure_extraction.generate`
    (D15), so the serving path and the corpus that trained it stay one code
    path."""
    raise NotImplementedError("ask.backbone.load_extractor is a stub; run --demo")


def load_gtlm(checkpoint, ui, graph_attn_impl=None):
    """Load a GTLM checkpoint: base + LoRA + graph bias, in one call.

    Refuses a `plain_llm` checkpoint (D3): it has no graph parameters, so a ball
    would be assembled, padded, and then ignored -- an answer that looks grounded
    and is not.

    SHARED-BACKBONE NOTE (D4): when `base_model_name_or_path` names the same
    model as the extractor, this and `load_extractor` should return one set of
    weights, with extraction running under `disable_adapter()`.  Deferred until
    a GaMS3-12B-based GTLM exists; see this module's docstring.
    """
    raise NotImplementedError("ask.backbone.load_gtlm is a stub; run --demo")
