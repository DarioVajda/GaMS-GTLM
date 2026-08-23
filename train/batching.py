"""Batch by TOKEN BUDGET, not by item count.

`balls/v2` runs from 300 to 14,055 packed tokens (p50 1,314, p99 5,605).  A fixed
`per_device_train_batch_size` over that distribution is not a batch size, it is a
lottery: every row in a batch is padded up to the batch's longest, and attention
is quadratic in that length.

Measured on `ana` (A100-80GB, 2026-08-22, `run_smoke.sbatch` stage 4), batch 4 in
the shuffled order the stock sampler produces:

    48 - 131 s per optimizer step of 16 items  ==  3 - 8 s per item

against **0.55 s per item** at batch 1, where nothing is padded at all.  The gap
is the padding: a batch whose longest member is 6,000 tokens costs 4 x 6000^2 of
attention no matter how short the other three are.  The same shuffle also OOMs --
the loss slice `GradeTrainer.compute_loss` computes is set by the EARLIEST answer
position in the batch, so one 90-token `ni v bazi` ball next to a 6,000-token one
forces a 6,000-wide logits slice: 26 GB in fp32 at a 262 k vocabulary.

Sorting by length before batching fixes both at once.  Padding nearly vanishes,
and the answer spans line up so the logits slice is the longest answer in the
batch (a few hundred tokens) rather than the longest ball.

**Batch composition is deterministic; only the ORDER is shuffled**, per epoch.
That keeps `__len__` exact -- the Trainer computes the LR schedule from it before
the first epoch runs -- and makes a run reproducible from its seed.  Items are
bucketed by length before the tie-break, so which items share a batch is still a
function of the seed rather than of file order.
"""
import random

import torch


def packed_lengths(ds):
    """Packed token count per graph: the sum over its nodes' `input_ids`.

    Read straight off the Arrow column rather than through `__getitem__`, which
    would rebuild edges and the original-id mapping for every one of 9,266 items
    to answer a question about token counts.
    """
    col = ds._hf_dataset.select_columns("input_ids")["input_ids"]
    return [sum(len(x) for x in row) for row in col]


class TokenBudgetBatchSampler(torch.utils.data.Sampler):
    """Yields lists of indices whose padded cost stays under `budget` tokens.

    `budget` bounds `len(batch) * max_length_in_batch` -- the padded token count,
    which is what actually gets allocated. An item longer than the budget on its
    own still gets its own batch: dropping it would silently shrink the corpus.
    """

    def __init__(self, lengths, budget=8192, max_batch=16, seed=0, bucket=128):
        self.lengths = list(lengths)
        self.budget = int(budget)
        self.max_batch = int(max_batch)
        self.seed = int(seed)
        self.bucket = int(bucket)
        self.epoch = 0
        self._batches = self._build()

    def _build(self):
        rng = random.Random(self.seed)
        order = list(range(len(self.lengths)))
        rng.shuffle(order)                       # tie-break inside a length bucket
        order.sort(key=lambda i: self.lengths[i] // self.bucket)
        batches, cur, cur_max = [], [], 0
        for i in order:
            nxt = max(cur_max, self.lengths[i])
            if cur and (nxt * (len(cur) + 1) > self.budget
                        or len(cur) >= self.max_batch):
                batches.append(cur)
                cur, cur_max = [i], self.lengths[i]
            else:
                cur.append(i)
                cur_max = nxt
        if cur:
            batches.append(cur)
        return batches

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        order = list(range(len(self._batches)))
        random.Random(self.seed * 1000 + self.epoch).shuffle(order)
        for j in order:
            yield self._batches[j]

    def __len__(self):
        return len(self._batches)

    def describe(self):
        sizes = [len(b) for b in self._batches]
        padded = [len(b) * max(self.lengths[i] for i in b) for b in self._batches]
        real = sum(self.lengths)
        return (f"{len(self._batches):,} batches, "
                f"items/batch min {min(sizes)} mean {sum(sizes) / len(sizes):.1f} "
                f"max {max(sizes)}; padding waste "
                f"{100 * (sum(padded) - real) / real:.1f} %")
