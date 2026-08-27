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


def serve_row(question, ball):
    """A ball, in the shape `train/data._graph` reads.

    `answer` is a DUMMY: `_graph(…, with_answer=False)` still writes the answer
    into the chat template and truncates the string at the marker, so that the
    training copy and the generation copy share a byte-identical prefix (V2).
    `spec.PREFIX` is the shortest string that survives that truncation intact --
    what remains after the cut is the generation prompt, whatever followed it.

    `id` and `type` land on the graph as attributes and are never read back
    here; they exist because `_graph` records them.
    """
    return {"id": "ask", "type": "ask", "question": question,
            "answer": PREFIX,
            "nodes": list(ball.texts), "edges": [list(e) for e in ball.edges],
            "anchors": list(ball.anchors)}


def to_batch(question, ball, gtlm):
    """One question plus its ball, collated exactly as evaluation collates it.

    Every step is `train/`'s own: `_graph` builds the graph and attaches the
    prompt node to *every* anchor, `_features` computes SPD / RRWP / magnetic
    under the checkpoint's settings, and the collator is
    `LeftPadCollator(GraphCollatorV2(...))` -- the object the trained runs used.
    Nothing about the input shape is decided in this file.
    """
    from gtlm.utils import TextGraphDataset
    from train.data import _graph, _features
    from train.batching import to_left_padding

    tok, cfg = gtlm["tokenizer"], gtlm["cfg"]
    ds = TextGraphDataset([_graph(serve_row(question, ball), tok,
                                  with_answer=False)])
    ds.tokenize(tok, max_length=cfg.max_length, add_eos=False)
    _features(ds, cfg)
    ds.cast_float_features_to_fp32()

    batch = to_left_padding(gtlm["collator"]([ds[0]]))
    batch.pop("labels", None)      # there is no gold at serve time
    return batch


def answer(question, ball, gtlm, ui=None, max_new_tokens=MAX_NEW_TOKENS):
    """Generate one answer, yielding decoded text as it arrives.

    Streaming is HF's `TextIteratorStreamer` with `generate` on a worker thread:
    `GTLMGemma3ForCausalLM.generate` (graph_model, causal_lm.py:413) is a thin
    wrapper over `super().generate`, so `streamer=` passes straight through.
    Decode is eager and single-sequence either way (D16), so the tokens exist
    one at a time already; the thread only stops the process from sitting on
    them until the last one.

    The generation settings come from the evaluator (V3), never from a fresh
    `GenerationConfig`: `greedy_kwargs` suppresses gemma-3-1b-it's shipped
    `do_sample=True`, and `gen_cfg_for` carries the model's own
    `cache_implementation` -- `None` on this stack, `"hybrid"` on stock Gemma-3,
    and the wrong one dies on the first decode step past a 512-token prompt.
    """
    import torch
    from threading import Thread
    from transformers import TextIteratorStreamer
    from torch.nn.attention import sdpa_kernel

    from train import evaluate as ev_mod

    model, tok, ev = gtlm["model"], gtlm["tokenizer"], gtlm["evaluator"]
    batch = to_batch(question, ball, gtlm)
    inputs = {k: (v.to(gtlm["device"]) if torch.is_tensor(v) else v)
              for k, v in batch.items() if v is not None}

    streamer = TextIteratorStreamer(tok, skip_prompt=True,
                                    skip_special_tokens=True)
    error = {}

    def run():
        try:
            with torch.no_grad(), sdpa_kernel(ev_mod._GEN_SDPA_BACKENDS):
                model.generate(**inputs, generation_config=ev.gen_cfg_for(model),
                               **ev.greedy_kwargs,
                               max_new_tokens=max_new_tokens,
                               use_cache=True, streamer=streamer)
        except BaseException as e:                 # noqa: BLE001
            # The streamer would otherwise just stop, and the caller would see a
            # short answer instead of a failure.  Re-raised on this thread below.
            error["e"] = e
            streamer.end()

    worker = Thread(target=run, daemon=True)
    worker.start()
    # The marker was the last thing the model was SHOWN, so it is not in the
    # continuation -- `evaluate.pass2` rebuilds the answer line the same way,
    # and the grader parses a full line.  It is emitted WITH the first token
    # rather than before it, so that the caller's first `next()` blocks until
    # the prefill is actually done and the progress line covers it (D20).
    it = iter(streamer)
    first = next(it, "")
    yield ev_mod.ANSWER_PREFIX.lstrip("\n") + first
    for text in it:
        yield text
    worker.join()
    if "e" in error:
        raise error["e"]
