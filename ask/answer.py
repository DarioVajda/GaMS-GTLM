#!/usr/bin/env python3
"""Stage 3 -- question plus ball in, the model's answer out (D11, D12, D16).

Everything that turns a ball into model input is imported from `train/`:
`data._graph` builds the graph and attaches the prompt node to *every* anchor,
`data._features` computes SPD / RRWP / magnetic Laplacian, and the collator
chain is the trained one (`LeftPadCollator(GraphCollatorV2(pad_to_block=True))`).
The generation settings are borrowed from `evaluate.GradeEvaluator`, which can
be constructed with `splits=[]` -- not rebuilt, because `gen_cfg_for()` exists to
strip the `cache_implementation="hybrid"` the hub config ships and the GTLM
stack must not have, and a hand-rolled GenerationConfig gets that wrong quietly.

Serving is the trained shape: flex for the prefill, eager for the decode,
`pad_to_block=True` so both L and N land on a bucket a kernel was compiled for.
The budget is 1024 rather than evaluation's `min(320, 2·max(gold)+16)`, since at
serve time there is no gold to size it from; 5 of 12,490 corpus answers exceed
320 tokens.
"""
from dataclasses import dataclass

try:                                  # the D13 import seam, when it is present
    from qa import spec
    SENTINEL, PREFIX = spec.SENTINEL, spec.PREFIX
except ImportError:                   # the demo runs on stdlib alone
    SENTINEL, PREFIX = "ni podatka v bazi", "ODGOVOR: "

MAX_NEW_TOKENS = 1024


@dataclass
class Answer:
    text: str
    n_tokens: int = 0

    @property
    def sentinel(self):
        return self.text.strip().removeprefix(PREFIX).strip() == SENTINEL


def answer(question, ball, model, ui):
    """Generate one answer, streaming tokens to `ui` as they are decoded.

    Streaming is HF's `TextIteratorStreamer` with `generate` on a worker thread:
    `GTLMGemma3ForCausalLM.generate` (graph_model, causal_lm.py:413) is a thin
    wrapper over `super().generate`, so `streamer=` passes straight through.
    """
    raise NotImplementedError(
        "ask.answer.answer is a stub; run with --demo, or see build step 3 in "
        "ask/PLAN.md")
