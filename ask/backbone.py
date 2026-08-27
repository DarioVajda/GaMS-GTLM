#!/usr/bin/env python3
"""Loading: the extractor, the GTLM checkpoint, and the store (D3, D4, D5, D17).

The GTLM side is one call.  `GTLMGemma3ForCausalLM.from_pretrained(ckpt,
graph_attn_impl=...)` (graph_model, `src/models/causal_lm.py:490`) reads the
checkpoint's `base_model_name_or_path`, loads that base, wraps it in the LoRA
adapters and calls `load_bias_parameters` -- base, adapters and graph-bias in a
single line, with the attention implementation overridable, which is what D17's
eager fallback needs.

What the checkpoint does *not* carry is `magnetic_m` (V1): it self-describes the
model, not the features.  Every `arms_v3` arm used the `RunConfig` default of 0,
and a future mismatch fails loudly as a shape error rather than degrading, so
the default is read from `RunConfig` and printed at startup.

**Why two models, when one would do.**  The extractor is `GaMS3-12B-Instruct`
and today's GTLM is `gemma-3-1b-it`, so they are simply different weights.  Once
a GaMS3-12B-based GTLM exists, the two collapse: load it once, run the extraction
turn with `disable_adapter()` (plain backbone, no graph bias) and the answering
turn with the adapters on, saving ~24 GB and one load.  The check is on the
GTLM's `base_model_name_or_path`; the dispatch point is `load()` below, and this
note is deliberately duplicated there so it is found by whoever next reads that
code and not only by whoever reads this docstring.

Everything heavy is imported inside the function that needs it: `--demo` runs on
the standard library alone, and importing this module must not be what breaks
that.
"""
import os
import collections

# Resolved from this file, the way `lib/paths.py` resolves everything: a default
# that depends on the working directory is a default that works from the repo
# root and nowhere else.  `bin/ask` cds there, but `run_ask.sbatch` runs python
# inside a container that does not, and the checkpoint below then does not exist
# -- which is exactly how this was found.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The checkpoint answered from unless --checkpoint says otherwise: the best test
# accuracy of the arms_v3 sweep (0.7917, 1729/2184).  Seed 0 ties on test and
# loses on dev (0.8683 vs 0.8769); the 0.0028 spread is within seed noise.
# A path to a real checkpoint directory, loaded directly -- `ask` never searches
# a run directory for a best-of.
DEFAULT_CHECKPOINT = os.path.join(
    REPO_ROOT, "checkpoints/sl_qa/"
    "arms_v3_0002_data_rootdata-datasets-balls-v2_c_plain_llmFalse_spdTrue_"
    "magneticTrue_max_length2048_batch_size4_accumulation_steps4_seed2/"
    "checkpoint-4400")

DEFAULT_EXTRACTOR = "cjvt/GaMS3-12B-Instruct"


def quiet_libraries():
    """Turn off the log traffic that has no place in an answer.

    `ui.stage` already swallows whatever a stage prints, so this is about the
    lines that escape one -- and about the checks, which have no stages at all
    and would otherwise write tens of thousands of progress-bar frames into a
    job log.  The sliding-window warning is the load-bearing case: it is emitted
    from inside `generate`, on the worker thread, while the answer is streaming.
    `train/_log.py` keeps one copy of it by default because a job log wants to
    be told once; here the same fact is stated at startup on the GTLM line, and
    one copy landing mid-answer is simply corruption.
    """
    from transformers import logging as hf_logging
    from train._log import quiet_repeated_sliding_window_warning

    hf_logging.set_verbosity_error()
    quiet_repeated_sliding_window_warning(first=False)
    try:
        # The shared HF cache belongs to whoever downloaded each model, and its
        # `.no_exist/` marker directories are not group-writable.  So hub cannot
        # record "this repo has no adapter_config.json", says so once per missing
        # file per load, and re-asks next time.  Nothing is wrong -- the weights
        # read fine -- and the cure is a chmod nobody in this repo can perform,
        # or HF_HUB_OFFLINE for a cache known to be complete.
        from huggingface_hub.utils import logging as hub_logging
        hub_logging.set_verbosity_error()
    except Exception:                       # noqa: BLE001 -- cosmetic only
        pass
    try:
        import datasets
        datasets.disable_progress_bars()
    except Exception:                       # noqa: BLE001 -- cosmetic only
        pass


def index_agrees(store, index, n=256, seed=0):
    """Does the cached surface index still describe THIS store?

    The index maps a casefolded surface to anchor *positions* in one store build
    (D9), and `qa/store.py` writes it beside the arrays with nothing tying the
    two together.  A rebuild that reordered nodes would leave every lookup
    pointing at the wrong entry, silently.  So ask the question itself rather
    than a proxy for it: does the anchor an entry names still carry that
    surface, as its lemma or as one of its forms?

    A sample, not a proof -- 256 surfaces cost ~0.13 s against 911k entries,
    where checking all of them *is* the index build.  That is the right trade
    because reordering is not a subtle corruption: it misplaces everything at
    once, so a sample this size either passes cleanly or fails on its first few.
    """
    import random

    keys = list(index)
    if not keys:
        return False
    for key in random.Random(seed).sample(keys, min(n, len(keys))):
        if not any(store.lemma(a).casefold() == key
                   or any(s.casefold() == key for _v, s, _f in store.forms(a))
                   for a in map(int, index[key])):
            return False
    return True


def load_store(path, ui=None):
    """Open the processed graph store, its surface index and the ball cache.

    Returns the bundle `retrieve()` reads, plus the provenance D10 prints.  The
    arrays are memory-mapped, so this is fast (~1 s for 37M nodes); the surface
    index is a 22 MB JSON the store carries next to them.

    **The index cache can outlive the store it was built from**, with nothing in
    the file tying it to one (D9).  The mtimes are the cheap first question --
    is the index older than the arrays? -- and only that.  The store is rebuilt
    reproducibly, so an older index is the ordinary case, and warning on the
    timestamp alone cries wolf at every startup for a file that is very probably
    correct.  So when the mtimes are inverted this *checks*, on a sample
    (`index_agrees`), and says something only about what it found.
    """
    import os
    import numpy as np
    from qa.store import open_store

    store = open_store(path or None, verbose=False)
    index = store.surface_index()
    man = store.manifest

    out = {
        "store": store,
        "index": index,
        "token_len": np.asarray(store.G["token_len"]),
        "cache": {},                       # anchor -> its hop-2 node set
        "stats": collections.Counter(),    # what build() records as it goes
        "path": store.path,
        "graph_version": man.get("graph_version"),
        "created": (man.get("created") or "")[:10],
        "n_nodes": int(man.get("n") or store.n),
        "n_surfaces": len(index),
    }

    idx_file = os.path.join(store.path, "qa_surface_index.json")
    man_file = os.path.join(store.path, "manifest.json")
    try:
        older = os.path.getmtime(idx_file) < os.path.getmtime(man_file)
    except OSError:
        older = False
    if older:
        n = 256
        if index_agrees(store, index, n):
            out["note"] = (f"indeks besednih oblik je starejši od baze, a se "
                           f"vzorec {n} oblik ujema z njo")
        else:
            out["warn"] = ("indeks besednih oblik ne opisuje te baze: oblike v "
                           "vzorcu kažejo na napačna vozlišča, iztočnice bodo "
                           "napačne -- izbriši qa_surface_index.json, da se "
                           "zgradi znova")
    return out


def load_extractor(name=None, ui=None, prompt_path=None):
    """Load the extraction model and the prompt it is asked with.

    Both halves come from `data/analysis/measure_extraction.py` (D15): the load
    was split out of `generate()` there so a session can load once and ask many
    times, and the prompt is the same file the corpus's own extraction run used
    (`data/prompts/extractor_prompt.txt`).  Serving must not paraphrase either --
    the 95.5 % resolution rate the dataset was built at is a measurement of
    *this* model under *this* prompt.
    """
    from analysis.measure_extraction import load_extractor as _load
    from lib.paths import EXTRACTOR_PROMPT

    path = prompt_path or EXTRACTOR_PROMPT
    with open(path, encoding="utf-8") as f:
        template = f.read()
    if "{question}" not in template:
        raise ValueError(f"{path}: no {{question}} placeholder")

    ex = _load(name or DEFAULT_EXTRACTOR)
    ex["template"] = template
    ex["prompt_path"] = path
    return ex


def read_checkpoint(checkpoint):
    """`(config dict, base model name)`, refusing anything that is not a GTLM.

    D3: a `plain_llm` checkpoint has no graph parameters at all, so a ball would
    be retrieved, tokenised, padded -- and then ignored.  The answer would look
    grounded and would not be, which is the one failure this tool must not have.
    Refused at load, by name, rather than diagnosed later from a bad answer.
    """
    import os
    import json

    ck = checkpoint.rstrip("/")
    path = os.path.join(ck, "config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{ck} has no config.json -- --checkpoint takes a path to a real "
            f"checkpoint directory (…/checkpoint-4400), not a run directory")
    with open(path, encoding="utf-8") as f:
        conf = json.load(f)

    if not str(conf.get("model_type", "")).startswith("gtlm"):
        raise ValueError(
            f"{ck} is not a GTLM checkpoint (model_type="
            f"{conf.get('model_type')!r}, architectures="
            f"{conf.get('architectures')}).  `ask` refuses it: a stock LLM has "
            f"no graph parameters, so the ball would be built and then ignored, "
            f"and the answer would look grounded without being it.")

    base = None
    for name in ("adapter_config.json", "graph_bias_config.json"):
        p = os.path.join(ck, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                base = json.load(f).get("base_model_name_or_path")
            if base:
                break
    if not base:
        raise ValueError(
            f"{ck} names no base model: neither adapter_config.json nor "
            f"graph_bias_config.json carries `base_model_name_or_path`, and the "
            f"checkpoint holds only the adapters and the bias tensors.")
    return conf, base


def run_config_for(conf, base, impl, magnetic_m=None, max_length=None):
    """The `RunConfig` this checkpoint was trained under, as far as it records it.

    The checkpoint is the authority for everything it stored -- which bias
    features are on, `max_spd`, `magnetic_q`, `magnetic_dim`, `k_hop` -- and
    `RunConfig`'s defaults fill the two it does not: `magnetic_m`, which the
    collator needs (V1), and `max_length`, the per-node tokenisation cap, which
    is a property of the data build rather than of the model.

    Both are visible at startup and overridable, because a silent mismatch in
    either is a shape error at the first question rather than a wrong answer:
    `magnetic_m` changes the width of the magnetic feature the bias expects, and
    every `arms_v3` arm trained at the default of 0.
    """
    from train.config import RunConfig

    d = RunConfig()
    return RunConfig(
        model_name=base,
        impl=f"v2-{impl}",
        spd=bool(conf.get("spd", d.spd)),
        max_spd=int(conf.get("max_spd", d.max_spd)),
        rrwp=bool(conf.get("rrwp", d.rrwp)),
        max_rw_steps=int(conf.get("max_rw_steps", d.max_rw_steps)),
        magnetic=bool(conf.get("magnetic", d.magnetic)),
        magnetic_dim=int(conf.get("magnetic_dim", d.magnetic_dim)),
        magnetic_q=float(conf.get("magnetic_q", d.magnetic_q)),
        magnetic_m=d.magnetic_m if magnetic_m is None else int(magnetic_m),
        k_hop=int(conf.get("k_hop", d.k_hop)),
        max_length=d.max_length if max_length is None else int(max_length),
        flex_cache_size_limit=int(conf.get("flex_cache_size_limit",
                                           d.flex_cache_size_limit)),
        flex_compile_mode=str(conf.get("flex_compile_mode",
                                       d.flex_compile_mode)),
    ).validate()


def load_gtlm(checkpoint, ui=None, graph_attn_impl=None, magnetic_m=None,
              max_length=None):
    """Load a GTLM checkpoint: base + LoRA + graph bias, in one call.

    `GTLMGemma3ForCausalLM.from_pretrained` (graph_model, causal_lm.py:490)
    reads `base_model_name_or_path` out of the adapter config, loads that base
    under the checkpoint's own config, wraps it in the LoRA adapters and calls
    `load_bias_parameters`.  One call, and `graph_attn_impl` overrides the
    backend the config recorded -- which is what the eager fallback needs.

    The backend is chosen here rather than by the config alone (D17): flex only
    when a compiled cache for this GPU model already exists, since otherwise the
    first question of every session pays for a compile per shape.

    SHARED-BACKBONE NOTE (D4): when `base_model_name_or_path` names the same
    model as the extractor, this and `load_extractor` should return ONE set of
    weights, with extraction running under `disable_adapter()` -- no graph bias,
    no ball, just the plain backbone.  Deferred until a GaMS3-12B-based GTLM
    exists; see this module's docstring.
    """
    import os
    import torch
    from transformers import AutoTokenizer
    from gtlm.utils import GraphCollatorV2

    from train.batching import LeftPadCollator
    from train.evaluate import GradeEvaluator
    from ask import precompile as pc

    conf, base = read_checkpoint(checkpoint)

    impl = graph_attn_impl
    hint = None
    if impl is None:
        warm = pc.is_warm()
        impl = "flex" if warm else "eager"
        if not warm:
            hint = ("za hitrejši prefill prevedi oblike vnaprej: "
                    "`ask --precompile`")
    if impl == "flex":
        # Inductor's default cache lives in a node-local /tmp that is wiped
        # between jobs (V5).  Point it at the per-GPU directory BEFORE anything
        # compiles; `cache_dir()` reads the variable on every call.
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = pc.cache_dir()

    cfg = run_config_for(conf, base, impl, magnetic_m, max_length)
    _config_cls, model_cls = cfg.gtlm_classes()

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = model_cls.from_pretrained(checkpoint.rstrip("/"),
                                      graph_attn_impl=impl,
                                      torch_dtype=cfg.torch_dtype())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    # Serving decodes; training ran with the cache off under gradient
    # checkpointing, and the flag travels on the config.
    model.config.use_cache = True

    collator = LeftPadCollator(GraphCollatorV2(
        tokenizer=tokenizer, k_hop=cfg.k_hop,
        magnetic_m=cfg.magnetic_m if cfg.magnetic else 0,
        pad_to_block=(impl == "flex"),
        max_spd=cfg.max_spd))

    # V3: the generation settings are instance state on the evaluator, and
    # `gen_cfg_for` exists to carry the model's own `cache_implementation` over
    # while suppressing gemma-3-1b-it's `do_sample=True`.  Borrowed, never
    # rebuilt -- a hand-rolled GenerationConfig gets both wrong silently.  It
    # constructs happily with no splits, which is all serving needs of it.
    evaluator = GradeEvaluator(tokenizer, collator, splits=[])

    return {
        "model": model,
        "tokenizer": tokenizer,
        "collator": collator,
        "cfg": cfg,
        "evaluator": evaluator,
        "device": device,
        "base": base,
        # The backend, and nothing else: a startup line is read by whoever is
        # about to ask a question, and "the adapter drops Gemma-3's sliding-window
        # band so every layer attends over the whole ball" is not a fact they can
        # act on.  It stays true, it stays suppressed (the warning that would say
        # it fires from inside `generate`, mid-answer), and it is printed under
        # --debug next to `magnetic_m` for whoever is diagnosing rather than
        # asking.
        # English, against this tool's Slovene: the reason is a fact about
        # torch's compiler, and "flex attention" and "kernel" are the names it
        # is discussed under everywhere else -- in `precompile.py`, in the
        # inductor cache directory, and by anyone who would act on it.  The flex
        # branch beside it is English for the same reason.
        "attn": ("flex (prefill) + eager (decode)" if impl == "flex"
                 else "eager (flex attention kernels not compiled)"),
        "impl": impl,
        "magnetic_m": cfg.magnetic_m,
        "max_length": cfg.max_length,
        "hint": hint,
    }
