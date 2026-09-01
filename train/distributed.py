"""Rank plumbing for the ``torchrun`` path, in one place.

Every run in this experiment is single-node.  ``sweep``'s
``execution.sbatch.gpus_per_config > 1`` launches the module under
``torchrun --standalone --nproc_per_node N``; with it at 1 (or with the module
run directly) none of the environment variables below are set and every helper
here degrades to the single-process answer, so the same code path serves both.

Two invariants this module exists to protect, both of them scientific rather
than mechanical:

  * **the effective batch is world-size invariant.**  ``batch_size x
    accumulation_steps`` is the binding quantity (see ``RunConfig``), and under
    DDP the *true* effective batch is that product times the world size.  So the
    product is re-factorised per rank -- see ``ddp_factorisation`` -- rather than
    handed to ``TrainingArguments`` unchanged, which would train a 4-GPU run at
    four times the intended batch AND derive four times too many optimizer steps
    from it.

  * **evaluation is world-size invariant, item for item.**  The evaluator
    batches by token budget, and the batching is a deterministic function of the
    split's lengths alone.  So every rank computes the SAME batch list and each
    runs a strided slice of it: the grouping, and therefore the padding, is
    identical to what one GPU would have produced.  That matters because
    ``evaluate.py``'s own OOM note is explicit that a re-grouping can flip a bf16
    near-tie -- sharding *items* and letting each rank batch its own shard would
    have changed the padding on nearly every batch.

Training is only statistically invariant, not bitwise: HF's distributed sampler
splits each global batch across ranks, so the same items land in a different
micro-batch composition and float addition is not associative.  The optimizer
step count, the effective batch, the learning-rate schedule and the data order
are all preserved exactly.  ``train/checks/check_ddp.py`` measures what is left.
"""
import os

import torch
import torch.distributed as dist


def world_size():
    """Ranks in this job -- 1 when not launched under ``torchrun``."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main():
    return rank() == 0


def is_distributed():
    return world_size() > 1


def init_distributed():
    """Join the process group and pin this rank's GPU, if there is one.

    Called before the model is built.  HF's ``TrainingArguments`` would do both,
    but only when it is constructed -- and ``run_train_mode`` loads the model and
    moves it to a device well before that point.  Without this, every rank would
    call ``.to("cuda")`` and land on ``cuda:0``.

    Idempotent, and a no-op in the single-process case.
    """
    if not is_distributed():
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return False
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
    if dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return True


def device():
    """This rank's device."""
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank())
    return torch.device("cpu")


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def shard(items):
    """This rank's strided slice of `items`.

    Strided rather than contiguous so the ranks interleave: the evaluator's
    batches are length-sorted, so a contiguous split would give rank 0 every
    short batch and the last rank every long one, and the whole eval would run at
    the slowest rank's pace.  Striding spreads the cost evenly.

    Every element is owned by exactly one rank and the union is the whole list,
    which is what makes the gathers below a disjoint merge rather than a reduce.
    """
    if not is_distributed():
        return list(items)
    return list(items)[rank()::world_size()]


def gather_objects(obj):
    """`[obj from rank 0, obj from rank 1, ...]` on EVERY rank.

    All-gather, not gather-to-zero: the metrics computed downstream feed
    ``metric_for_best_model``, and HF's Trainer expects every rank to hold the
    same value -- a rank-0-only metric would have the other ranks select a
    different checkpoint (or none).
    """
    if not is_distributed():
        return [obj]
    out = [None] * world_size()
    dist.all_gather_object(out, obj)
    return out


def merge_flags(local, n):
    """Reduce per-rank boolean lists of length `n` by OR.

    Each index is written by exactly one rank (``shard`` is a partition) and the
    others leave it False, so OR recovers the owner's value whether that value
    was True or False.
    """
    if not is_distributed():
        return list(local)
    merged = [False] * n
    for part in gather_objects(list(local)):
        for i, v in enumerate(part):
            if v:
                merged[i] = True
    return merged


def merge_dicts(local):
    """Reduce per-rank dicts with disjoint keys into one."""
    if not is_distributed():
        return dict(local)
    out = {}
    for part in gather_objects(dict(local)):
        out.update(part)
    return out


def sum_numbers(*values):
    """Element-wise sum of the given scalars across ranks."""
    if not is_distributed():
        return tuple(values) if len(values) > 1 else values[0]
    parts = gather_objects(tuple(float(v) for v in values))
    totals = tuple(sum(p[i] for p in parts) for i in range(len(values)))
    return totals if len(values) > 1 else totals[0]


def ddp_factorisation(effective_batch, batch_size, n_ranks):
    """`(per_device_batch, accumulation_steps)` that PRESERVES `effective_batch`.

    HF multiplies by the world size on its own -- the true effective batch is
    ``per_device_batch x accumulation_steps x n_ranks`` -- so the declared
    product is divided by the rank count here and only the remainder is handed
    over.  ``batch_size`` (the micro-batch, a pure memory knob) is kept as
    declared where it still fits, and lowered when the per-rank share is smaller
    than it.

        16 = 4 x 4 on 1 rank  ->  4 x 4      (unchanged)
        16 = 4 x 4 on 2 ranks ->  4 x 2 x 2
        16 = 4 x 4 on 4 ranks ->  4 x 1 x 4
        16 = 4 x 4 on 8 ranks ->  2 x 1 x 8

    Raises rather than round: an effective batch that does not survive the split
    is a silently different optimisation, which is the exact failure this whole
    function exists to prevent.
    """
    if n_ranks <= 1:
        return batch_size, effective_batch // batch_size
    if effective_batch % n_ranks:
        raise ValueError(
            f"effective batch {effective_batch} is not divisible by the world "
            f"size {n_ranks}, so it cannot be preserved across ranks.  Use a "
            f"rank count that divides it, or change batch_size x "
            f"accumulation_steps.")
    per_rank = effective_batch // n_ranks
    micro = min(batch_size, per_rank)
    if per_rank % micro:
        raise ValueError(
            f"per-rank share {per_rank} (= {effective_batch}/{n_ranks}) is not "
            f"divisible by the micro-batch {micro}.")
    return micro, per_rank // micro
