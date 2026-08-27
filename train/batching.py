"""How a batch is formed and padded: token counts, padding side, key set.

Three things live here, in dependency order: `packed_lengths`, which sizes an
evaluation batch; `to_left_padding`, which decides which end of a row the pads
sit on; and the two collator wrappers that put `GraphCollatorV2`'s output into
the shape the training, evaluation and serving stacks each read.

They are together, and away from `run.py`, because they are the batch contract
rather than the training entry point.  `train/README.md` splits this package by
direction of dependency, and anything that consumes a batch -- the trainer, the
evaluator, the checks, `ask/` -- needs the same collator without also importing
argument parsing, wandb setup and model construction.

`packed_lengths` gives the number of tokens a graph occupies once its node texts
are concatenated.  `train/evaluate.py` groups evaluation batches by it: attention
is quadratic in the PADDED length and this corpus runs from ~300 to 16,384 packed
tokens, so a fixed eval batch is either wasteful at the bottom or an OOM at the
top.  No gradients are taken there, so batch composition carries no statistical
cost.

TRAINING batches are deliberately not formed this way.  Under a token budget the
item count per batch falls out of sequence length, so each arm gets a different
effective batch and a different number of updates -- a gap between two arms would
be partly a gap between two schedules.  Length also correlates with task type
(full paradigms and high-polysemy words are the long tail), so length-sorted
batches make whole gradient steps out of one kind of item.  A fixed
`batch_size x accumulation_steps` with random shuffling avoids both.
"""
import torch


def packed_lengths(ds):
    """Packed token count per graph: the sum over its nodes' `input_ids`.

    Read straight off the Arrow column rather than through `__getitem__`, which
    would rebuild edges and the original-id mapping for every one of 9,266 items
    to answer a question about token counts.
    """
    col = ds._hf_dataset.select_columns("input_ids")["input_ids"]
    return [sum(len(x) for x in row) for row in col]


def to_left_padding(batch):
    """Move each row's padding to the FRONT, in place.  Idempotent.

    Two callers, one reason each.

    **Generation** appends to the END of the sequence, so every row has to *end*
    at its own last real token; with right padding a short row would grow its
    continuation on the far side of its pads.

    **Training** (`LeftPadCollator`) needs it because
    `GradeTrainer.compute_loss` slices the logits to the answer tail using the
    EARLIEST supervised position in the batch.  Right-padded, a shorter row's
    answer span sits earlier in the padded sequence and drags that slice back for
    everyone -- measured, the worst GTLM batch at packed L=16,384 needed
    `logits_to_keep=6,880` and peaked at 100.5 GiB, ~43 GB of which was logits
    for answer spans a few hundred tokens long.  Left-padded, every row ends at
    `L-1`, so the same `min` collapses on its own to `longest answer + 1`.

    The roll changes nothing else: the packed node order is preserved,
    `attention_mask` still marks the pads (nothing attends to them, and the
    structural mask reads padding from that mask alone), `node_ids` travels with
    `input_ids` so the token->node->bias mapping is unchanged, and the leading
    pads keep `node_ids = prompt_node`, which is not a prefix node and therefore
    gains nothing from the bidirectional-prefix relaxation.  `position_ids` are
    per-node LOCAL positions under `node_position_mode='reset'`, and the value
    travels with its token.  `pad_to_block` bucketises the total `L`, which a
    roll does not change.

    The shift is the number of TRAILING pads, not the total pad count, which is
    what makes this idempotent: a row that is already left-padded has none and is
    left alone.  (`generate`'s own left-padding path calls this on batches the
    training collator may already have rolled.)
    """
    am = batch["attention_mask"]
    if am.shape[1] == 0:
        return batch
    # Trailing zeros per row = how far right the last real token has to move.
    # `argmax` on the flipped mask is the index of the last real token counted
    # from the end, i.e. exactly that count.
    shift = (am.flip(1) != 0).float().argmax(dim=1).tolist()
    for i, s in enumerate(shift):
        if not s:
            continue
        # `node_ids`/`position_ids` are absent on the plain-LLM baselines, whose
        # collator ships only what a stock causal model reads; `labels` is absent
        # on the generation path, which pops it.
        for key in [k for k in ("input_ids", "position_ids", "node_ids",
                                "attention_mask", "labels") if k in batch]:
            batch[key][i] = torch.roll(batch[key][i], int(s), dims=0)
    return batch


class LeftPadCollator:
    """Wrap a collator and move every row's padding to the FRONT.

    `GraphCollatorV2` right-pads and has no `padding_side` option, so this is a
    post-collation roll -- see `to_left_padding` for why it is safe.

    It exists for the answer-tail logits slice.  `GradeTrainer.compute_loss`
    slices to the tail using the EARLIEST supervised position in the batch;
    right-padded, a shorter row's answer span sits earlier in the padded sequence
    and drags that slice back for every row -- on the worst GTLM batch at packed
    L=16,384 that is `logits_to_keep=6,880` and a 100.5 GiB peak.  Left padding
    makes every row end at `L-1`, so the same `min` collapses to
    `longest answer + 1` with no extra slicing logic.

    One collator object serves both training and evaluation, which also makes
    `to_left_padding` in pass 2 a no-op -- it is idempotent by construction.
    """

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, features):
        return to_left_padding(self.inner(features))


class PlainCollator:
    """Wrap GraphCollatorV2 and keep only what a stock causal LM reads.

    The graph collator ships `node_ids`, `prompt_node`, `num_nodes` and the bias
    feature tensors; stock Gemma-3 accepts none of them.  `position_ids` is
    dropped too -- a baseline ball has exactly one node, so the graph collator's
    per-node reset produces a plain `arange`, and letting the model derive its
    own keeps left-padded generation correct.
    """

    KEEP = ("input_ids", "attention_mask", "labels")

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, features):
        batch = self.inner(features)
        return {k: v for k, v in batch.items() if k in self.KEEP}
