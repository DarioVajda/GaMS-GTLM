#!/usr/bin/env python3
"""`--precompile`: walk every (L, N) shape once so flex is warm (D17).

The collator pads both axes onto a ladder -- L to multiples of 512 with 1.5x
midpoints, N to powers of two floored at 32 -- so that torch.compile sees a
bounded set of shapes.  Warming them means compiling each once, up front, and
persisting inductor's output where the next invocation on the same GPU model
will find it.

Without a warm cache the tool serves eager and says so, rather than making the
first question of every session pay for a compile.

**A warm cache is cheaper, not free (V5).**  `fx_graph_cache` skips codegen and
autotuning; dynamo still traces and re-checks its guards on the first call of
each shape.  Expect the sweep to buy most of the cost back, not all of it.
"""
import os
import re

CACHE_ROOT = os.path.join(os.path.dirname(__file__), "cache")

try:    # the real path, when graph_model is importable
    from src.models.flex_kernel import default_len_buckets, default_node_buckets
    LEN_LADDER = NODE_LADDER = None
except ImportError:
    default_len_buckets = default_node_buckets = None
    # Mirrors those two functions for the demo, which runs on stdlib alone.
    LEN_LADDER = [512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384]
    NODE_LADDER = [32, 64, 128, 256, 512, 1024]

# Tokens per node, from the corpus: p50 is 77 nodes / 1,314 tokens, max 705 /
# 14,055.  Used only to skip (L, N) pairs no real ball can produce -- a full
# cross product would compile shapes that never occur.
TOKENS_PER_NODE = (6, 40)


def ladders():
    if LEN_LADDER is not None:
        return LEN_LADDER, NODE_LADDER
    seen_l, seen_n = [], []
    for x in (1, 513, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289):
        seen_l.append(default_len_buckets(x))
    for x in (1, 33, 65, 129, 257, 513):
        seen_n.append(default_node_buckets(x))
    return seen_l, seen_n


def shapes():
    """The (L, N) pairs a real ball can land on, coarse to fine."""
    lens, nodes = ladders()
    lo, hi = TOKENS_PER_NODE
    out = []
    for n in nodes:
        for length in lens:
            if lo * (n // 2) <= length <= hi * n:
                out.append((length, n))
    return out


def gpu_name():
    """A filesystem-safe name for this GPU model, e.g. `nvidia-h100-80gb-hbm3`.

    The model, not the device: a compiled kernel is valid for any card of the
    same model, and pinning the cache to a device index would throw it away on
    every new allocation.
    """
    try:
        import torch
        name = torch.cuda.get_device_name(0)
    except Exception:
        name = "unknown-gpu"
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def cache_dir(gpu=None):
    return os.path.join(CACHE_ROOT, gpu or gpu_name())


def warm(model, collator, ui, shapes_=None):
    """Compile every shape, printing progress.  Sets TORCHINDUCTOR_CACHE_DIR to
    `cache_dir()` first, since inductor's default lives in a node-local /tmp
    that is wiped between jobs (V5)."""
    raise NotImplementedError(
        "ask.precompile.warm is a stub; run with --demo, or see build step 6 "
        "in ask/PLAN.md")
