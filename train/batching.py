"""Packed token counts per graph, and why there is no training-side batcher.

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


def packed_lengths(ds):
    """Packed token count per graph: the sum over its nodes' `input_ids`.

    Read straight off the Arrow column rather than through `__getitem__`, which
    would rebuild edges and the original-id mapping for every one of 9,266 items
    to answer a question about token counts.
    """
    col = ds._hf_dataset.select_columns("input_ids")["input_ids"]
    return [sum(len(x) for x in row) for row in col]
