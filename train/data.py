"""Read the extracted balls and hand back one ``Split`` per dataset split.

Input is `datasets/balls/<version>/{train,dev,test}.jsonl`, written by
`gams_gtlm/data/qa/build_balls.py`: one JSON object per item carrying the node
texts, the edge list, and the question/answer pair.  Nothing here opens the
37 M-node store — the ball is already an artefact, which is what makes a run
reproducible from the dataset directory alone and lets a GPU job start cold.

Graph layout, one item:

    node 0 .. N-1   the ball, texts verbatim from the KG
                    (`iztočnica: gora (…)`, `oblika: gore (…)`, `pomen: …`, …)
    node N          the PROMPT node, the item as ONE CHAT TURN PAIR in the
                    backbone's own format (`train/chat.py`):
                    `<bos><start_of_turn>user\\n{question}<end_of_turn>\\n`
                    `<start_of_turn>model\\nODGOVOR: {answer}<end_of_turn>`
    edges           the ball's own induced edges, plus prompt -> every anchor

The template goes on the prompt node ONLY -- the ball's nodes are retrieved KG
text, not dialogue, and stay verbatim.  See `train/chat.py` for why this is the
default rather than an option, and why `<bos>` belongs there even though on the
GTLM stack that puts it mid-sequence.

Label masking is done on **character offsets**, not on a token-id subsequence.
The usual trick — encode the delimiter alone and search for that id sequence — is
unreliable across tokenizers, because `"ODGOVOR:"` on its own and the same string
preceded by a newline need not tokenize to the same ids, and the failure is
silent: it masks the wrong span rather than raising.  A fast tokenizer gives
exact offsets, so the boundary is read off the string itself.

Two things travel alongside the training dataset:

  * **the grading contract**, joined on `id` from `datasets/generated/v2_clean`.
    The ball carries the answer; only the dataset carries the item-level
    `grading` facts (for T17 and T19: `quantity_band`, `n_asked`, `n_all`,
    `all_items`), and `qa.grade.grade` needs both.  The TYPE-level fields
    (`mode`, `sep`, `arity`, `regex`) come from `qa/spec.py`, not from the row —
    an in-row constant is how T19 kept being graded `sequence` for a whole run
    after the spec said otherwise.  The join asserts the two answers agree — the
    ball and the dataset must not disagree about what is being supervised,
    enforced on every run rather than checked once.  See `train/config.py`'s
    `ITEMS_ROOT` for why `v2_clean` and `v2_graded` are not interchangeable.
  * **a generation copy** of each eval split, identical in every way except that
    its prompt node stops at `ODGOVOR:`.  Pass 2 of the evaluator decodes from
    it; building it here means the structural features are computed with the
    same code as the training copy's.
"""
import os
import json
import random
import hashlib
import collections
from dataclasses import dataclass, field

import networkx as nx

from gtlm.utils import TextGraphDataset

from .config import ANSWER_PREFIX
from .chat import chat_prompt, stop_token_ids
from .qa_contract import needs_generation


@dataclass
class Split:
    name: str
    rows: list                       # ball rows, in dataset order
    items: list                      # {id, type, band, negative, answer, grading}
    ds: TextGraphDataset             # teacher-forced (labels on the answer span)
    gen_ds: TextGraphDataset = None  # prompt truncated at `ODGOVOR:`, no labels
    needs_generation: list = field(default_factory=list)

    def __len__(self):
        return len(self.rows)


def stratified_subset(rows, frac, seed):
    """A fixed ~`frac` subsample of `rows`, stratified by task type.

    Used for the in-training dev evaluations and for checkpoint selection (T6):
    the full 1,040-item dev split is evaluated twelve times per run and eighteen
    runs pay for it, and half of it selects the same checkpoint at half the cost.

    Stratified so all 19 types survive with their share intact -- an unstratified
    half would thin the small types to single digits and make the per-type dev
    column unreadable.  `seed` is a CONSTANT, never `cfg.seed`: every arm and
    every seed must select on the *identical* subset, or the comparison between
    two arms includes a difference in which items they were selected on.

    Returns `(kept_ids, descriptor)`.  The descriptor goes in the run record so
    the subset is recoverable from the results file alone.
    """
    by_type = collections.defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r["id"])
    rng = random.Random(seed)
    keep = []
    for t in sorted(by_type):
        ids = sorted(by_type[t])
        k = max(1, round(len(ids) * frac))
        keep.extend(rng.sample(ids, min(k, len(ids))))
    keep = sorted(keep)
    digest = hashlib.sha256("\n".join(keep).encode("utf-8")).hexdigest()[:16]
    return keep, {"n": len(keep), "n_full": len(rows), "frac": frac,
                  "seed": seed, "sha256_16": digest}


def _read(path, types=(), cap=0, spread=False, keep_ids=None):
    """`cap` takes the first N; `spread` takes N evenly across the whole file.

    The items are written type by type, so the first 24 rows are 24 T1 items --
    one grading mode, one question shape.  A smoke test wants the prefix (it is
    cheap and deterministic); the two-pass acceptance test wants the spread, or
    it never sees a `multiset` or `membership` item and checks nothing about the
    modes the fast path actually exists for.
    """
    keep_ids = set(keep_ids) if keep_ids is not None else None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if types and r["type"] not in types:
                continue
            if keep_ids is not None and r["id"] not in keep_ids:
                continue
            rows.append(r)
            if cap and not spread and len(rows) >= cap:
                break
    if cap and spread and len(rows) > cap:
        rows = rows[::len(rows) // cap][:cap]
    return rows


def _read_items(path, rows):
    """The grading contract for `rows`, joined on `id` and order-aligned."""
    by_id = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_id[r["id"]] = r
    items = []
    for row in rows:
        src = by_id.get(row["id"])
        if src is None:
            raise KeyError(
                f"item {row['id']} has a ball but no dataset row in {path} -- the "
                f"ball directory and the dataset directory are out of step.")
        if src["answer"] != row["answer"]:
            raise ValueError(
                f"item {row['id']}: the ball and the dataset disagree about the "
                f"answer.  Rebuild both in one pass (qa/run_build_balls.sbatch "
                f"writes datasets/balls/... and datasets/generated/v2_clean "
                f"together); ONE artefact has to be the authority for the target.\n"
                f"  ball:    {row['answer']!r}\n  dataset: {src['answer']!r}")
        items.append({"id": src["id"], "type": src["type"], "band": src["band"],
                      "negative": bool(src["negative"]), "answer": src["answer"],
                      "gold_items": src.get("gold_items"),
                      "grading": src["grading"]})
    return items


def _prompt_text(row, tokenizer, with_answer=True):
    """The prompt node's text: the item written as the backbone's own chat turns.

    The tokenizer is a parameter because the FORMAT is the tokenizer's -- it ships
    the jinja template, and reproducing its output here by hand is how the
    trainer would end up with a second opinion about what a turn looks like.
    `train/chat.py` holds the one call and the reasoning.
    """
    return chat_prompt(tokenizer, row["question"], row["answer"],
                       with_answer=with_answer)


def _graph(row, tokenizer, with_answer=True):
    g = nx.DiGraph()
    # The ball's own nodes are NOT wrapped: verbatim KG text is the context the
    # model retrieves through, not a conversation it had.
    for i, t in enumerate(row["nodes"]):
        g.add_node(i, text=t)
    for u, v in row["edges"]:
        g.add_edge(u, v)
    prompt = len(row["nodes"])
    g.add_node(prompt, text=_prompt_text(row, tokenizer, with_answer))
    # One edge per matched lexical unit.  D3's lookup returns a UNION when a
    # surface string owns several units (9.3 % of items), and choosing between
    # them is the model's job -- attaching the prompt to only the first would
    # answer that question for it, in the builder, using knowledge the pipeline
    # does not have.  `anchor` is the pre-union field, still written as
    # `anchors[0]`.
    #
    # `anchors` may be legitimately EMPTY: the no-retrieval baseline ships a ball
    # with no nodes at all, and then the graph is the prompt node alone.  Hence
    # the explicit key test rather than an `or` fallback, which would read the
    # empty list as "absent" and index into a ball that is not there.
    anchors = row["anchors"] if "anchors" in row else [row["anchor"]]
    for a in anchors:
        g.add_edge(prompt, a)
    g.graph["prompt_node"] = prompt
    g.graph["item_id"] = row["id"]
    g.graph["item_type"] = row["type"]
    return g


class OffsetLabelMasker:
    """Mask every token up to and including the `ODGOVOR:` marker.

    Uses the LAST occurrence: a question could in principle quote the marker (a
    corpus sentence in a T20 question is arbitrary text), and the delimiter that
    matters is the one the answer follows.  The chat markers do not disturb this
    -- the template writes `<start_of_turn>model\\nODGOVOR: …`, so the marker is
    still preceded by exactly the newline `ANSWER_PREFIX` carries.

    The supervised span therefore runs from the first answer character through
    the `<end_of_turn>` the template closes the model turn with, which is what
    teaches the model to stop.
    """

    def __init__(self, tokenizer, max_length, marker=ANSWER_PREFIX):
        if not tokenizer.is_fast:
            raise ValueError(
                "A fast tokenizer is required: the answer boundary is read from "
                "character offsets, which the slow tokenizers do not provide.")
        self.tok = tokenizer
        self.max_length = max_length
        self.marker = marker
        self.stop_id = stop_token_ids(tokenizer)[0]

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
        if not supervised:
            raise ValueError(
                f"The answer span is empty after masking: {text[:200]!r}")
        # The stop token is no longer appended after tokenization -- it is part
        # of the text, written by the chat template -- so `max_length` can now
        # truncate it away.  That would train an item whose answer never ends,
        # silently, and pass 1's "and then it stops" would be false for it.  So
        # it is checked on every item instead of assumed.
        if ids[-1] != self.stop_id:
            raise ValueError(
                f"the prompt node does not end on the stop token "
                f"(id {self.stop_id}): it needs "
                f"{len(self.tok(text, add_special_tokens=False)['input_ids'])} "
                f"tokens and max_length={self.max_length} cut it short, so the "
                f"model would never be taught to stop on this item.  Raise "
                f"`max_length` for this data_root, or lower "
                f"qa/build_variants.py's prompt budget.  Answer: "
                f"{text[cut:cut + 120]!r}")
        return labels


def _features(ds, cfg):
    if cfg.spd:
        ds.compute_shortest_path_distances(cutoff=cfg.max_spd)
    if cfg.rrwp:
        ds.compute_rrwp(max_rrwp_steps=cfg.max_rw_steps)
    if cfg.magnetic:
        ds.compute_magnetic_lap(q=cfg.magnetic_q, m=cfg.magnetic_m)
    return ds


def load_split(cfg, tokenizer, split, with_generation, spread=False,
               keep_ids=None, name=None):
    path = os.path.join(cfg.data_root, f"{split}.jsonl")
    rows = _read(path, cfg.type_list(), cfg.max_items, spread, keep_ids)
    if not rows:
        raise FileNotFoundError(
            f"no items for types={cfg.type_list() or 'ALL'} in {path} -- build the "
            f"balls first (gams_gtlm/data/qa/run_build_balls.sbatch)")
    items = _read_items(os.path.join(cfg.items_root, f"{split}.jsonl"), rows)

    # `add_eos=False` on BOTH copies.  The teacher-forced copy carries its own
    # terminator -- the `<end_of_turn>` the chat template closes the model turn
    # with -- and appending `<eos>` after it would teach the model to emit a
    # two-token ending its own checkpoint never produces.  `OffsetLabelMasker`
    # verifies the turn end survived tokenization on every item.
    ds = _features(TextGraphDataset([_graph(r, tokenizer) for r in rows]), cfg)
    ds.tokenize(tokenizer, max_length=cfg.max_length, add_eos=False)
    ds.compute_labels(OffsetLabelMasker(tokenizer, cfg.max_length), num_proc=1)
    ds.cast_float_features_to_fp32()

    gen_ds = None
    if with_generation:
        gen_ds = _features(
            TextGraphDataset([_graph(r, tokenizer, with_answer=False)
                              for r in rows]), cfg)
        gen_ds.tokenize(tokenizer, max_length=cfg.max_length, add_eos=False)
        gen_ds.cast_float_features_to_fp32()

    name = name or split
    print(f"[data] {name}: {len(ds)} graphs"
          f"{' (+ generation copy)' if gen_ds is not None else ''}", flush=True)
    return Split(name=name, rows=rows, items=items, ds=ds, gen_ds=gen_ds,
                 needs_generation=[needs_generation(it) for it in items])


def load_dev_subsample(cfg, tokenizer):
    """The `dev_fast` split the in-training evals and checkpoint selection use.

    `(Split, descriptor)`, or `(None, None)` when `cfg.dev_subsample` is off.
    A second, independently built dataset over a subset of the same rows -- not a
    view -- so the structural features and the tokenisation come from the same
    code path as the full split's.
    """
    if not cfg.dev_subsample or cfg.dev_subsample >= 1.0:
        return None, None
    path = os.path.join(cfg.data_root, "dev.jsonl")
    rows = _read(path, cfg.type_list(), cfg.max_items)
    keep, descriptor = stratified_subset(rows, cfg.dev_subsample,
                                         cfg.dev_subsample_seed)
    split = load_split(cfg, tokenizer, "dev", with_generation=True,
                       keep_ids=keep, name="dev_fast")
    print(f"[data] dev_fast: {descriptor['n']}/{descriptor['n_full']} items, "
          f"stratified over {len({r['type'] for r in rows})} types, "
          f"seed {descriptor['seed']}, sha256:{descriptor['sha256_16']}",
          flush=True)
    return split, descriptor


def load_data(cfg, tokenizer, generation_splits=("dev", "test")):
    """One ``Split`` per split, in (train, dev, test) order."""
    return tuple(load_split(cfg, tokenizer, s, s in generation_splits)
                 for s in ("train", "dev", "test"))
