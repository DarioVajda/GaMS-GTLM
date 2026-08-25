#!/usr/bin/env python3
"""The two baseline INPUT variants, derived from the balls the GTLM arm reads.

    python -m qa.build_variants datasets/balls/v2 datasets/balls

Both are built **from the exact ball file**, never re-extracted from the store.
That is the whole point: three of the four arms in the run matrix share their
data, schedule, backbone and seed, so the only thing that may differ between them
is the input.  Re-running the extractor for a baseline would confound the
comparison with whatever the extractor did that day.

  `v2_noretrieval`  question and answer only -- no graph at all.  Measures how
      much of the corpus a 1B model already answers from its own Slovene, which
      is the floor every other number has to clear before it says anything about
      retrieval.

  `v2_serialised`   the SAME subgraph, flattened into the prompt as text: the
      node list, then the edge list, then the question.  Isolates *structure* --
      this arm and the GTLM arm receive identical information and differ only in
      how it is encoded.

**Both ship a ball with zero graph nodes**, so the model reads a single prompt
node either way and the two baselines differ from each other in exactly one
thing: whether the graph's text is present.  Putting the serialisation inside the
prompt (rather than leaving it as a separate graph node) is what makes this the
ordinary text baseline it is meant to be -- a plain LM reading a verbalised
graph, with real sequential positions over the whole thing, against a GTLM whose
node positions reset per node and whose attention carries the structural bias.

The answer, `gold_items`, `negative`, `band`, `targets` and the split are copied
verbatim, so every id and every split matches `balls/v2` exactly and the grading
join (`datasets/generated/v2_final`) is unchanged.
"""
import os
import json
import argparse
import collections

NODES_HEADER = "PODATKI IZ BAZE:"
EDGES_HEADER = "POVEZAVE:"

# The serialised prompt's ceiling, in tokens, question and answer included.
#
# Flattening costs about 65 % on top of the ball's own token count -- the node
# numbering the edge list has to refer to, and the edge list itself -- so the
# 14,055-token maximum ball serialises to 34,176.  Attention is quadratic in that
# and eager (GTLM cannot use a fused kernel), so the longest item alone wants
# ~9 GB of transient activation per layer and is a live risk of killing a
# multi-hour run at the one batch that contains it.
#
# 16,384 puts the serialised arm's ceiling next to the GTLM arm's own 14 k and
# costs **8 of 9,266 train items** (0.09 %) some trailing nodes.  Every truncated
# row is flagged `serialised_truncated` and counted in this script's output, so
# the caveat is auditable rather than implicit.
#
# What this budget does NOT count: the trainer wraps the prompt in the backbone's
# chat turns (`train/chat.py`), which adds exactly 9 tokens per item -- `<bos>`,
# the two `<start_of_turn>` headers with their newlines, and the two
# `<end_of_turn>`s.  Left uncounted deliberately: the balls on disk were fitted
# against the bare string, and re-measuring here would move `serialised_nodes`
# for the handful of items sitting exactly on the boundary, i.e. change the
# dataset without changing what it is for.  The 9 tokens are absorbed by the
# arm's own `max_length` (17,408, so 1,024 of slack), and `train/data.py`'s label
# masker raises if a prompt node ever does lose its turn end to truncation.
MAX_PROMPT_TOKENS = 16_384


def serialise(row, keep=None):
    """The ball as prompt text: a numbered node list, then the edge list.

    Numbered because the edge list has to refer to something, and the numbering
    is the ball's own node order -- which `build_balls.py` fixes as sorted store
    node id, so it is a pure function of (store, policy) and not of the order the
    extractor happened to return its strings in.

    `keep` truncates to the first `keep` nodes; the edge list then keeps exactly
    the edges with both ends still present, so the text never refers to a node
    the reader cannot see.
    """
    texts = row["nodes"] if keep is None else row["nodes"][:keep]
    n = len(texts)
    nodes = "\n".join(f"{i}: {t}" for i, t in enumerate(texts))
    edges = ", ".join(f"{u}-{v}" for u, v in row["edges"] if u < n and v < n)
    return f"{NODES_HEADER}\n{nodes or '/'}\n{EDGES_HEADER} {edges or '/'}"


def _fit(row, tok, budget):
    """(prompt question text, nodes kept) for the largest prefix that fits.

    Shrinks by the measured overshoot ratio rather than one node at a time: the
    only items that need more than one pass are the handful above the budget, and
    those are exactly the ones a linear scan would be slowest on.
    """
    keep = len(row["nodes"])
    while True:
        q = f"{serialise(row, keep)}\n\n{row['question']}"
        if tok is None:
            return q, keep
        n = len(tok(f"{q}\n{row['answer']}", add_special_tokens=False)["input_ids"])
        if n <= budget or keep == 0:
            return q, keep
        keep = min(keep - 1, int(keep * budget / n))


def variant(row, kind, tok=None, budget=MAX_PROMPT_TOKENS):
    out = dict(row)
    out["nodes"] = []
    out["edges"] = []
    out["anchors"] = []
    out["anchor"] = None
    out["n_nodes"] = 0
    out["n_tokens"] = 0
    out["variant"] = kind
    if kind == "serialised":
        q, keep = _fit(row, tok, budget)
        out["question"] = q
        out["serialised_nodes"] = keep
        out["serialised_truncated"] = keep < len(row["nodes"])
    elif kind != "noretrieval":
        raise ValueError(f"unknown variant {kind!r}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("balls", help="the GTLM arm's ball directory (datasets/balls/v2)")
    ap.add_argument("out_root", help="where the <name>_<variant> dirs are written")
    ap.add_argument("--tokenizer", default="google/gemma-3-1b-it",
                    help="the backbone's tokenizer -- it decides where the "
                         "serialised prompt's token budget falls")
    ap.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    base = os.path.basename(args.balls.rstrip("/"))
    dirs = {k: os.path.join(args.out_root, f"{base}_{k}")
            for k in ("noretrieval", "serialised")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    total = collections.Counter()
    for split in ("train", "dev", "test"):
        src = os.path.join(args.balls, f"{split}.jsonl")
        if not os.path.exists(src):
            continue
        handles = {k: open(os.path.join(d, f"{split}.jsonl"), "w", encoding="utf-8")
                   for k, d in dirs.items()}
        lens, cut_items, cut_nodes = [], 0, 0
        try:
            with open(src, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    for k, h in handles.items():
                        v = variant(row, k, tok, args.max_prompt_tokens)
                        h.write(json.dumps(v, ensure_ascii=False) + "\n")
                        if k == "serialised":
                            lens.append(len(tok(
                                f"{v['question']}\n{v['answer']}",
                                add_special_tokens=False)["input_ids"]))
                            if v["serialised_truncated"]:
                                cut_items += 1
                                cut_nodes += len(row["nodes"]) - v["serialised_nodes"]
                    total[split] += 1
        finally:
            for h in handles.values():
                h.close()
        lens.sort()

        def q(p):
            return lens[min(len(lens) - 1, int(p * len(lens)))]

        print(f"[{split}] {total[split]:,} items   serialised prompt tokens "
              f"p50 {q(.50):,} p90 {q(.90):,} p99 {q(.99):,} max {lens[-1]:,}\n"
              f"         truncated at {args.max_prompt_tokens:,}: {cut_items:,} items "
              f"({100 * cut_items / max(1, total[split]):.2f} %), "
              f"{cut_nodes:,} nodes dropped in total", flush=True)

    for k, d in dirs.items():
        print(f"[wrote] {d}")


if __name__ == "__main__":
    main()
