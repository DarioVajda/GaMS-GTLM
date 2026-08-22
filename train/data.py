"""Read the extracted balls and hand back three ``TextGraphDataset`` splits.

Input is `datasets/balls/<version>/{train,dev,test}.jsonl`, written by
`gams_gtlm/data/qa/build_balls.py`: one JSON object per item carrying the node
texts, the edge list, and the question/answer pair.  Nothing here opens the
37 M-node store — the ball is already an artefact, which is what makes a run
reproducible from the dataset directory alone and lets a GPU job start cold.

Graph layout, one item:

    node 0 .. N-1   the ball, texts verbatim from the KG
                    (`iztočnica: gora (…)`, `oblika: gore (…)`, `pomen: …`, …)
    node N          the PROMPT node, `"{question}\\nODGOVOR: {answer}"`
    edges           the ball's own induced edges, plus prompt -> every anchor

Label masking is done on **character offsets**, not on a token-id subsequence.
The usual trick — encode the delimiter alone and search for that id sequence — is
unreliable across tokenizers, because `"ODGOVOR:"` on its own and the same string
preceded by a newline need not tokenize to the same ids, and the failure is
silent: it masks the wrong span rather than raising.  A fast tokenizer gives
exact offsets, so the boundary is read off the string itself.
"""
import os
import json

import networkx as nx

from gtlm.utils import TextGraphDataset

from .config import ANSWER_PREFIX


def _read(path, types=(), cap=0):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if types and r["type"] not in types:
                continue
            rows.append(r)
            if cap and len(rows) >= cap:
                break
    return rows


def _graph(row):
    g = nx.DiGraph()
    for i, t in enumerate(row["nodes"]):
        g.add_node(i, text=t)
    for u, v in row["edges"]:
        g.add_edge(u, v)
    prompt = len(row["nodes"])
    # `answer` already carries the `ODGOVOR: ` tag (QA_TASKS.md 0.1), so the
    # prefix is not re-added -- it is joined with the newline the mask keys on.
    g.add_node(prompt, text=f"{row['question']}\n{row['answer']}")
    # One edge per matched lexical unit.  D3's lookup returns a UNION when a
    # surface string owns several units (9.3 % of items), and choosing between
    # them is the model's job -- attaching the prompt to only the first would
    # answer that question for it, in the builder, using knowledge the pipeline
    # does not have.  `anchor` is the pre-union field, still written as
    # `anchors[0]`.
    for a in row.get("anchors") or [row["anchor"]]:
        g.add_edge(prompt, a)
    g.graph["prompt_node"] = prompt
    g.graph["item_id"] = row["id"]
    g.graph["item_type"] = row["type"]
    return g


class OffsetLabelMasker:
    """Mask every token up to and including the `ODGOVOR:` marker.

    Uses the LAST occurrence: a question could in principle quote the marker (a
    corpus sentence in a T20 question is arbitrary text), and the delimiter that
    matters is the one the answer follows.
    """

    def __init__(self, tokenizer, max_length, marker=ANSWER_PREFIX):
        if not tokenizer.is_fast:
            raise ValueError(
                "A fast tokenizer is required: the answer boundary is read from "
                "character offsets, which the slow tokenizers do not provide.")
        self.tok = tokenizer
        self.max_length = max_length
        self.marker = marker

    def __call__(self, example):
        p = example["prompt_node"]
        text = example["text"][p]
        ids = list(example["input_ids"][p])
        cut = text.rfind(self.marker)
        if cut < 0:
            raise ValueError(f"No {self.marker!r} in the prompt node: {text[:200]!r}")
        cut += len(self.marker)                      # first supervised character
        enc = self.tok(text, add_special_tokens=False, truncation=True,
                       max_length=self.max_length, return_offsets_mapping=True)
        labels = list(ids)
        supervised = 0
        for i, (_s, e) in enumerate(enc["offset_mapping"]):
            if i >= len(labels):
                break
            if e <= cut:
                labels[i] = -100
            else:
                supervised += 1
        # `tokenize(add_eos=True)` appends one id past the offsets, and it must
        # stay supervised or the model is never taught to stop.
        if not supervised and len(labels) <= len(enc["offset_mapping"]):
            raise ValueError(
                f"The answer span is empty after masking: {text[:200]!r}")
        return labels


def load_data(cfg, tokenizer):
    """(train, val, test) TextGraphDatasets, features per ``cfg``."""
    types = cfg.type_list()
    out = []
    for split in ("train", "dev", "test"):
        path = os.path.join(cfg.data_root, f"{split}.jsonl")
        rows = _read(path, types, cfg.max_items)
        if not rows:
            raise FileNotFoundError(
                f"no items for types={types or 'ALL'} in {path} -- build the balls "
                f"first (gams_gtlm/data/qa/run_build_balls.sbatch)")
        ds = TextGraphDataset([_graph(r) for r in rows])
        if cfg.spd:
            ds.compute_shortest_path_distances(cutoff=cfg.max_spd)
        if cfg.rrwp:
            ds.compute_rrwp(max_rrwp_steps=cfg.max_rw_steps)
        if cfg.magnetic:
            ds.compute_magnetic_lap(q=cfg.magnetic_q, m=cfg.magnetic_m)
        ds.tokenize(tokenizer, max_length=cfg.max_length, add_eos=True)
        ds.compute_labels(OffsetLabelMasker(tokenizer, cfg.max_length),
                          num_proc=1)
        ds.cast_float_features_to_fp32()
        print(f"[data] {split}: {len(ds)} graphs", flush=True)
        out.append(ds)
    return tuple(out)
