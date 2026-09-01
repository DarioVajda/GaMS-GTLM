"""Write a multimodal Gemma-3 checkpoint's TEXT tower out as a text-only one.

    .venv/bin/python -m train.extract_text_tower google/gemma-3-4b-it
    .venv/bin/python -m train.extract_text_tower google/gemma-3-12b-it

`GTLMGemma3Config` subclasses `Gemma3TextConfig`, so it loads a checkpoint whose
`model_type` is `gemma3_text` and nothing else.  `GaMS3-12B-Instruct` is
published that way and needs no help.  `gemma-3-4b/12b/27b-it` are not: they are
`model_type: gemma3`, they nest the text fields under `text_config`, and their
weights sit under a `language_model.` prefix behind a vision tower.

The conversion is done ONCE, offline, and written to disk rather than being
patched into the loader:

  * **the adapter is not touched.**  `graph_model` is shared with other
    experiments, and an unwrap branch inside `from_pretrained` would be a change
    to every one of them for the sake of two runs here.
  * **every cell then loads through the identical code path.**  This is the
    scientific reason and it is the stronger one.  The Slovene contrast is
    `GaMS3-12B-Instruct` against `gemma-3-12b-it`; if one of those went through a
    loader branch the other did not, the contrast would carry a code difference
    as well as a pretraining one.  After this step both are plain `gemma3_text`
    directories and the run configs differ in the model path alone.

In transformers 4.50 `Gemma3ForConditionalGeneration.language_model` already IS a
`Gemma3ForCausalLM` carrying the `Gemma3TextConfig`, so the extraction is a
submodule save rather than a state-dict rename.  The tokenizer is saved
alongside, because the run config names one directory for both.

The result is byte-for-byte the text half of the original: no quantisation, no
dtype change beyond the `--dtype` the weights are read in, and no retraining.
"""
import os
import json
import argparse

import torch
from transformers import (AutoTokenizer, Gemma3ForCausalLM,
                          Gemma3ForConditionalGeneration)

from .config import REPO_ROOT

DEFAULT_OUT = os.path.join(REPO_ROOT, "checkpoints", "text_towers")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_name", help="a multimodal Gemma-3 repo id")
    p.add_argument("--out", default=None,
                   help=f"target directory (default {DEFAULT_OUT}/<name>)")
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    out = a.out or os.path.join(DEFAULT_OUT, a.model_name.split("/")[-1] + "-text")

    print(f"[extract] {a.model_name} -> {out}  ({a.dtype})", flush=True)
    mm = Gemma3ForConditionalGeneration.from_pretrained(
        a.model_name, torch_dtype=dtype, device_map="cpu")
    lm = mm.language_model
    if not isinstance(lm, Gemma3ForCausalLM):
        raise TypeError(
            f"expected `.language_model` to be a Gemma3ForCausalLM, got "
            f"{type(lm).__name__} -- the transformers layout changed and this "
            f"script needs updating rather than guessing.")

    n_text = sum(p.numel() for p in lm.parameters())
    n_all = sum(p.numel() for p in mm.parameters())
    print(f"[extract] text tower {n_text/1e9:.2f} B of {n_all/1e9:.2f} B total "
          f"({n_text/n_all:.1%}); {lm.config.num_hidden_layers} layers, "
          f"hidden {lm.config.hidden_size}, vocab {lm.config.vocab_size}",
          flush=True)

    os.makedirs(out, exist_ok=True)
    lm.save_pretrained(out, safe_serialization=True)
    # The run config names ONE directory for the model and the tokenizer, so the
    # text tokenizer has to travel with the weights.  `AutoTokenizer` on a
    # multimodal repo already returns the text tokenizer, not the processor.
    AutoTokenizer.from_pretrained(a.model_name).save_pretrained(out)

    cfg = json.load(open(os.path.join(out, "config.json"), encoding="utf-8"))
    if cfg.get("model_type") != "gemma3_text" or "text_config" in cfg:
        raise ValueError(
            f"the written config is model_type={cfg.get('model_type')!r} "
            f"(text_config present: {'text_config' in cfg}) -- it must be a flat "
            f"`gemma3_text` for GTLMGemma3Config to load it.")
    # Recorded next to the weights: which upstream checkpoint this is the text
    # half of, so a directory found later is self-describing.
    with open(os.path.join(out, "EXTRACTED_FROM.json"), "w", encoding="utf-8") as f:
        json.dump({"source": a.model_name, "dtype": a.dtype,
                   "text_parameters": n_text, "total_parameters": n_all,
                   "transformers": __import__("transformers").__version__}, f,
                  indent=2)
    print(f"[extract] wrote {out} (model_type={cfg['model_type']}, "
          f"{cfg['num_hidden_layers']} layers)")
    print(f"[extract] use it with:  --model-name {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
