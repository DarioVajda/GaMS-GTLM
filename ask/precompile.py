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
# Written last by `warm()`, read by `is_warm()`: the sweep completed here.
MARKER = "shapes.json"
# The balls the shapes are enumerated from -- `train/config.py`'s own default,
# read lazily so importing this module needs nothing installed.
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "data", "datasets", "balls")

try:    # the collator's own ladders, so this cannot drift from them
    from gtlm.models.flex_kernel import default_len_buckets, default_node_buckets
except ImportError:                     # --demo runs on the standard library
    default_len_buckets = default_node_buckets = None

# What the imports above produce, for the demo, and the ceilings the sweep stops
# at either way: the largest ball in the corpus is 705 nodes / 14,055 tokens.
LEN_LADDER = [512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384]
NODE_LADDER = [32, 64, 128, 256, 512, 1024]

# Tokens per node, from the corpus: p50 is 77 nodes / 1,314 tokens, max 705 /
# 14,055.  Used only to skip (L, N) pairs no real ball can produce -- a full
# cross product would compile shapes that never occur.
TOKENS_PER_NODE = (6, 40)


def climb(bucket_of, ceiling):
    """The ladder `bucket_of` defines, read off it rather than restated.

    Ask for the bucket of 1, then of one past that bucket, and so on: each
    answer is the next rung, and the walk ends at `ceiling`.
    """
    out, x = [], 1
    while x <= ceiling:
        b = int(bucket_of(x))
        out.append(b)
        x = b + 1
    return out


def ladders():
    if default_len_buckets is None:
        return LEN_LADDER, NODE_LADDER
    return (climb(default_len_buckets, LEN_LADDER[-1]),
            climb(default_node_buckets, NODE_LADDER[-1]))


def bucket(value, ladder):
    """Round up onto a ladder; the largest rung if nothing is big enough."""
    return next((x for x in ladder if x >= value), ladder[-1])


def attested_shapes(data_root=None, prompt_len=64):
    """The (L, N) buckets the REAL corpus lands on, or None if it is not here.

    12,490 balls, each carrying its own node count and token count, bucketised
    the way the collator will bucketise them.  Enumerating from the corpus
    rather than crossing the two ladders is what keeps the sweep off the
    impossible corners -- a 1,024-node ball of 16 k tokens is a shape the ladder
    admits and the language does not, and on a 40 GB card compiling it asks for
    43 GiB.

    `prompt_len` is what the question adds to the packed length; the default is
    a little over the corpus's typical chat-wrapped prompt, so a ball sitting
    just under a rung is counted on the rung it will really land on.
    """
    import json
    import glob

    root = data_root or DATA_ROOT
    files = sorted(glob.glob(os.path.join(root, "*.jsonl")))
    if not files:
        return None
    lens, nodes = ladders()
    seen = set()
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                seen.add((bucket(int(r["n_tokens"]) + prompt_len, lens),
                          bucket(int(r["n_nodes"]) + 1, nodes)))
    return sorted(seen, key=lambda s: (s[1], s[0]))


def ladder_shapes():
    """Every (L, N) pair the ladders admit that a ball could plausibly produce.

    The fallback for when the corpus is not on disk: the token band is the
    corpus's own (6-40 tokens per node), so this is the same restriction
    reached without the corpus itself.
    """
    lens, nodes = ladders()
    lo, hi = TOKENS_PER_NODE
    return [(length, n) for n in nodes for length in lens
            if lo * (n // 2) <= length <= hi * n]


def shapes(data_root=None):
    """The (L, N) pairs to warm, smallest first.

    Smallest first because the sweep can be interrupted, and because the big
    shapes are where the time goes: the corpus median is 77 nodes / 1,314
    tokens, so a sweep that got through the low rungs has warmed the shapes
    almost every question actually lands on.
    """
    return attested_shapes(data_root) or ladder_shapes()


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


def is_warm(gpu=None):
    """Has this GPU model been through the sweep, and did the sweep reach anything?

    The marker, not the directory's contents: inductor writes into the cache on
    any compile, so a directory can exist holding one shape out of the ladder.
    And the marker's own `hit` count, not merely its existence: a sweep where
    every shape OOMed or landed in the wrong bucket has warmed nothing, and
    reporting that GPU as ready for flex would make the first question of every
    session pay for a compile -- the exact cost D17 exists to avoid.
    """
    import json

    try:
        with open(os.path.join(cache_dir(gpu), MARKER), encoding="utf-8") as f:
            return int(json.load(f).get("hit") or 0) > 0
    except (OSError, ValueError):
        return False


WORD = "beseda"


def previous_bucket(length, lens):
    """The bucket below `length`, or 0 -- the floor a ball must clear."""
    below = [x for x in lens if x < length]
    return max(below) if below else 0


def tokens_per_word(tok):
    """How many tokens one repetition of the filler word costs, measured once."""
    n = len(tok(" ".join([WORD] * 64), add_special_tokens=False)["input_ids"])
    return max(1.0, n / 64.0)


def prompt_tokens(gtlm, question):
    """What the prompt node costs, so the ball can be sized around it.

    The prompt node is packed with the ball and counts toward `L`, so a sweep
    that ignored it would overshoot every bucket by the length of the question.
    """
    from train.chat import chat_prompt
    from ask.answer import PREFIX

    tok = gtlm["tokenizer"]
    text = chat_prompt(tok, question, PREFIX, with_answer=False)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def filler_ball(gtlm, n_nodes, length, question):
    """A synthetic ball that lands in the `(length, n_nodes)` bucket, or None.

    Nothing about the content matters -- the kernel being compiled is keyed on
    the padded shape, and it does not know what it is multiplying.  The SIZE
    matters exactly: warming `L=1536` does nothing for a question that lands on
    `L=512`, so the ball is aimed at the middle of the bucket's own span, from
    a measured tokens-per-word ratio, rather than estimated and hoped for.

    `None` when the shape cannot be reached: `N` nodes need at least `N` tokens,
    so a large node count under a small length is not a shape any real ball can
    produce either.
    """
    from ask.retrieve import Ball

    lens, _nodes = ladders()
    ball_nodes = max(1, n_nodes - 1)          # the prompt node is the last one
    floor = previous_bucket(length, lens)
    # The middle of (floor, length], so neither tokenizer noise nor the
    # off-by-one of an integer division can push it into a neighbouring bucket.
    aim = (floor + length) // 2
    room = aim - prompt_tokens(gtlm, question)
    if room < ball_nodes:
        return None

    per_word = tokens_per_word(gtlm["tokenizer"])
    per_node = max(1, int(room / ball_nodes / per_word))
    # The shortfall from rounding DOWN is per node, so it multiplies: at 511
    # nodes it is up to 511 tokens, which is a whole rung of the ladder and is
    # what made the first version land a bucket low.  One node absorbs it.
    short = max(0, int((room - per_node * per_word * ball_nodes) / per_word))
    texts = [" ".join([WORD] * per_node)] * ball_nodes
    texts[0] = " ".join([WORD] * (per_node + short))
    return Ball(texts=texts,
                edges=[[i, i + 1] for i in range(ball_nodes - 1)],
                anchors=[0], codes=[], strings=[],
                n_tokens=int((per_node * ball_nodes + short) * per_word))


def padded_n(inputs):
    """The N the kernel is compiled against: the bias tensors' node axis.

    Not the real node count -- that is what `num_nodes` holds, and the whole
    point of the ladder is that it is rounded up.  Read off a bias tensor
    because that is the tensor whose shape the compiled kernel guards on.
    """
    import torch

    spd = inputs.get("shortest_path_dists")
    if torch.is_tensor(spd):
        return int(spd.shape[-1])
    mag = inputs.get("magnetic_V")
    if torch.is_tensor(mag):
        return int(mag.shape[1])
    return int(inputs["num_nodes"].max())


def warm(gtlm, ui=None, question="Kaj pomeni ta beseda?"):
    """Compile every (L, N) shape once, and record which ones were reached.

    The sweep runs one PREFILL per shape -- a single forward, not a generation:
    the decode is eager (D16) and has nothing to compile, and a full answer per
    shape would multiply the sweep by the length of an answer.

    Shapes are *measured*, not assumed.  A synthetic ball is built to land in
    each bucket and the collated batch is then read for the `(L, N)` it actually
    produced, because the route from "a ball with these node texts" to "a padded
    shape" runs through the tokenizer and two ladders, and predicting it exactly
    would be reimplementing the collator.  `shapes.json` records what was
    reached; `is_warm()` reads it.
    """
    import json
    import time

    import torch

    from ask.answer import to_batch

    if gtlm.get("impl") != "flex":
        # Belt and braces behind `--precompile`'s implied `--impl flex`: an
        # eager model compiles nothing, so the sweep would run in a second, warm
        # nothing, and write a marker claiming this GPU is ready for flex.
        raise ValueError(
            f"nothing to precompile: the model is on {gtlm.get('impl')!r}, not "
            f"flex.  Load it with graph_attn_impl='flex' first.")

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir()
    model, device = gtlm["model"], gtlm["device"]
    targets = shapes()
    done, failed, missed = {}, [], []
    t0 = time.perf_counter()

    for k, (length, n) in enumerate(targets, 1):
        label = f"prevajam L={length}, N={n}  ({k}/{len(targets)})"
        ctx = ui.stage(label) if ui is not None else _null()
        try:
            with ctx:
                ball = filler_ball(gtlm, n, length, question)
                if ball is None:
                    missed.append((length, n, "unreachable: N nodes need N tokens"))
                    continue
                batch = to_batch(question, ball, gtlm)
                inputs = {kk: (v.to(device) if torch.is_tensor(v) else v)
                          for kk, v in batch.items() if v is not None}
                got = (int(inputs["input_ids"].shape[1]), padded_n(inputs))
                with torch.no_grad():
                    model(**inputs)
                # Keyed by what was ASKED for, not by what was reached: two
                # targets can land on one shape, and keying by the landing site
                # would drop one of them from the record entirely.
                done[f"{length}x{n}"] = {"L": got[0], "N": got[1],
                                         "asked": [length, n]}
                if got != (length, n):
                    # Aimed at one bucket, landed in another: the kernel that
                    # got compiled is not the one this shape needs.  Recorded
                    # rather than retried -- a sweep that silently warmed the
                    # wrong shapes is what this exists to make visible.
                    missed.append((length, n, f"landed on {got[0]}x{got[1]}"))
        except Exception as e:                          # noqa: BLE001
            failed.append((length, n, f"{type(e).__name__}: {e}"))
            if ui is not None:
                ui.fail(f"L={length}, N={n}: {type(e).__name__}: {e}")

    d = cache_dir()
    os.makedirs(d, exist_ok=True)
    hit = sum(1 for s in done.values() if [s["L"], s["N"]] == s["asked"])
    with open(os.path.join(d, MARKER), "w", encoding="utf-8") as f:
        json.dump({"gpu": gpu_name(),
                   "asked": len(targets), "hit": hit,
                   "shapes": sorted(done.values(),
                                    key=lambda s: (s["N"], s["L"])),
                   "missed": [[ln, n, why] for ln, n, why in missed],
                   "failed": [[ln, n, why] for ln, n, why in failed],
                   "seconds": round(time.perf_counter() - t0, 1)},
                  f, ensure_ascii=False, indent=1)
    if ui is not None:
        ui.field("predpomnilnik", f"{hit}/{len(targets)} oblik · "
                                  f"{time.perf_counter() - t0:.0f}s · {d}")
        if missed:
            ui.warn(f"{len(missed)} oblik ni bilo doseženih; glej {MARKER}")
        if failed:
            ui.warn(f"{len(failed)} oblik se ni prevedlo; glej {MARKER}")
    return done, failed


class _null:
    """`warm` without a UI -- for a caller that only wants the cache filled."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
