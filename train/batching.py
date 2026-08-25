"""Packed token counts per graph — and why the training-side batcher is gone.

`packed_lengths` is the one thing left here: the number of tokens a graph
occupies once its node texts are concatenated.  `train/evaluate.py` groups
evaluation batches by it (attention is quadratic in the PADDED length, and this
corpus runs from ~300 to 16,384 packed tokens, so a fixed eval batch is either
wasteful at the bottom or an OOM at the top).

**The `TokenBudgetBatchSampler` that used to live here has been retired from the
training path** (2026-08-23).  Its entire justification was the answer-tail
blow-up in `GradeTrainer.compute_loss` — its own docstring said so: *"the loss
slice is set by the EARLIEST answer position in the batch"*, so one 90-token
`ni v bazi` ball next to a 6,000-token one forced a 6,000-wide logits slice, 26 GB
in fp32 at a 262 k vocabulary.  Left-padding the training batches removes that
at the source (the slice collapses to the longest ANSWER in the batch, a few
hundred tokens), and once the slice is tight, length-sorting only buys attention
padding.

Two reasons not to keep it anyway:

  * **it made the arms run different optimisations.**  Under a token budget the
    item count per batch falls out of sequence length, so `arms_v2` gave the GTLM
    arm 16.5 items x 4,480 steps, the serialised arm 10.9 x 6,824 and the
    no-retrieval arm 63.9 x 1,160 — a 6x spread on both axes.  A gap between two
    arms was partly a gap between two schedules.  A fixed
    `batch_size x accumulation_steps` is identical in every arm by construction.
  * **length correlates with task type** (full paradigms and high-polysemy words
    are the long tail), so deterministic length-sorted batches make whole
    gradient steps out of one kind of item.  Random shuffling is the cleaner
    choice for a study whose output is a per-type table.

The evaluator keeps its own length-sorted grouping, and that is correct and
unaffected: no gradients are taken there, so batch composition carries no
statistical cost.
"""


def packed_lengths(ds):
    """Packed token count per graph: the sum over its nodes' `input_ids`.

    Read straight off the Arrow column rather than through `__getitem__`, which
    would rebuild edges and the original-id mapping for every one of 9,266 items
    to answer a question about token counts.
    """
    col = ds._hf_dataset.select_columns("input_ids")["input_ids"]
    return [sum(len(x) for x in row) for row in col]
