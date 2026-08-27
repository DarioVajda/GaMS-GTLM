#!/usr/bin/env python3
"""R4 -- measure the D3 entity-extraction model against our own questions.

D3 hands "which token is the target" to an external extractor.  QA_DATASET_DESIGN
R4 records the risk: *"if it does not return the target word, the right ball is
never extracted and no amount of GTLM training recovers it. Pick the model, then
measure its recall against our own templates (we author them, so the gold target
word is known for free)."*  This is that measurement.

    sbatch analysis/run_measure_extraction.sbatch                # -> results/
    python -m analysis.measure_extraction --n 40

The gold target is `slots.F` (the form a question names) or `slots.L` (the lemma)
-- the string the template put in the question, verified present verbatim in
every one of the 12,493 items.

Three numbers, in increasing order of what they actually license:

  parsed        the output is a Python list at all
  recall        the gold target string is in that list, case-blind
  resolved      some returned string resolves through the surface index (D3
                step 2) to the item's OWN anchor -- the only criterion that
                predicts whether the right ball gets built

`resolved` can beat `recall`: returning an inflected form that still indexes to
the same lexical unit is a win for the pipeline even though the string differs.
It can also lose to it: `unlisted` negatives name a real word the lookup cannot
reach, where returning it faithfully is correct and resolving is impossible.
"""
import os
import re
import ast
import json
import time
import hashlib
import argparse
import collections

import numpy as np

from qa.store import open_store
from lib.errors import StageError
from lib.paths import GENERATED_DIR, EXTRACTOR_PROMPT

DEFAULT_MODEL = "cjvt/GaMS3-12B-Instruct"
LIST_RE = re.compile(r"\[.*?\]", re.S)
STR_RE = re.compile(r'"([^"]*)"' + r"|'([^']*)'")


def target(r):
    """The string the question names -- what the extractor must return."""
    sl = r.get("slots") or {}
    return sl.get("F") or sl.get("O") or sl.get("L") or r["lemma"]


def parse(out):
    """(list_of_strings, ok).  Tolerant: the first [...] wins, then bare quotes.

    Deliberately more forgiving than `ast.literal_eval` on the whole output --
    a model that writes a sentence and then the list has still done the job, and
    scoring it as a parse failure would blame the prompt for the wrong thing.
    """
    m = LIST_RE.search(out)
    if m:
        try:
            v = ast.literal_eval(m.group(0))
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return [x.strip() for x in v], True
        except (ValueError, SyntaxError):
            pass
        got = [a or b for a, b in STR_RE.findall(m.group(0))]
        if got:
            return [g.strip() for g in got], True
        if m.group(0).strip() == "[]":
            return [], True
    return [], False


def load_items(n_per_type, dataset, balance_flavours=False):
    """Up to `n_per_type` items of each type, 0 for all of them.

    Stratifying on the negative FLAVOUR as well (`--balance-flavours`) is useful
    when the question is "how does the extractor handle each frame", but it
    inflates the negative share far past the corpus's 8.5 % and makes the
    aggregate row meaningless -- the `unlisted` slice can never resolve, by
    construction.  Type-only is the default because the headline number should
    describe the corpus.
    """
    per = collections.defaultdict(list)
    for split in ("train", "dev", "test"):
        path = os.path.join(dataset, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            key = ((r["type"], r["negative_flavour"]) if balance_flavours
                   else r["type"])
            per[key].append(r)
    # Shuffle before capping: the files are written positives-first, so a
    # head-of-file sample of any type contains no negatives at all.  The seed
    # comes from a stable digest of the key -- `hash()` on a str is salted per
    # process, so it would pick a different sample on every run.
    cap = max(1, n_per_type // 4) if balance_flavours else n_per_type
    out = []
    for key in sorted(per, key=str):
        rs = per[key]
        if n_per_type > 0 and len(rs) > cap:
            seed = int.from_bytes(
                hashlib.blake2b(str(key).encode(), digest_size=4).digest(), "big")
            rng = np.random.default_rng(seed)
            rs = [rs[i] for i in sorted(rng.permutation(len(rs))[:cap])]
        out.extend(rs)
    return out


def generate(model_name, prompts, batch_size, max_new_tokens):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    chats = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True)
             for p in prompts]
    outs = []
    t0 = time.time()
    for i in range(0, len(chats), batch_size):
        batch = chats[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for j in range(len(batch)):
            outs.append(tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True))
        done = min(i + batch_size, len(chats))
        print(f"  [{done}/{len(chats)}]  {time.time() - t0:.0f}s", flush=True)
    return outs



def score(dump, model=None, prompt=None):
    """Print the per-type table for a list of dump rows and return the summary.

    Split out of `main` so a sharded run can be scored once, off the GPU, from
    the concatenated dumps -- otherwise each shard prints a table over its own
    slice and the aggregate has to be eyeballed.
    """
    rows = collections.defaultdict(collections.Counter)
    n_ret = []
    for d in dump:
        low = {g.casefold() for g in d["parsed"]}
        gold = d["gold"].casefold()
        c = rows[d["type"]]
        c["n"] += 1
        c["parsed"] += bool(d["ok"])
        c["recall"] += bool(d["recall"])
        c["resolved"] += bool(d["resolved"])
        c["empty"] += bool(d["ok"]) and not d["parsed"]
        c["extra"] += len(low - {gold})
        if d.get("negative_flavour"):
            c["neg_n"] += 1
            c["neg_recall"] += bool(d["recall"])
        n_ret.append(len(d["parsed"]))

    print(f"\n{'type':>5} {'n':>6} {'parsed':>8} {'recall':>8} {'resolved':>9} "
          f"{'empty []':>9} {'extra/item':>11}")
    tot = collections.Counter()
    for t in sorted(rows, key=lambda x: (len(x), x)):
        c = rows[t]
        tot.update(c)
        print(f"{t:>5} {c['n']:>6} {100*c['parsed']/c['n']:7.1f}% "
              f"{100*c['recall']/c['n']:7.1f}% {100*c['resolved']/c['n']:8.1f}% "
              f"{100*c['empty']/c['n']:8.1f}% {c['extra']/c['n']:11.2f}")
    n = tot["n"]
    print(f"{'ALL':>5} {n:>6} {100*tot['parsed']/n:7.1f}% "
          f"{100*tot['recall']/n:7.1f}% {100*tot['resolved']/n:8.1f}% "
          f"{100*tot['empty']/n:8.1f}% {tot['extra']/n:11.2f}")
    print(f"\nreturned strings per item: mean {np.mean(n_ret):.2f} "
          f"p90 {np.percentile(n_ret, 90):.0f} max {max(n_ret)}")
    if tot["neg_n"]:
        print(f"negatives: recall {100*tot['neg_recall']/tot['neg_n']:.1f} % "
              f"over {tot['neg_n']} items")
    return {"model": model, "prompt": prompt, "n": n, "overall": dict(tot),
            "by_type": {t: dict(c) for t, c in rows.items()}}


def run(model=DEFAULT_MODEL, prompt=EXTRACTOR_PROMPT,
        dataset=GENERATED_DIR, n=40, batch_size=32,
        max_new_tokens=64, out=None, dump=None, num_shards=1, shard=0,
        score_only=None, balance_flavours=False, store=None):
    """Run the extractor over a dataset (or re-score dumps).  The summary dict.

    Called both by `main()` below and by `data/pipeline/`, which runs one of
    these per GPU as a subprocess and then re-enters here with `score_only` to
    score the merged result off the GPU.
    """
    args = argparse.Namespace(
        model=model, prompt=prompt, dataset=dataset, n=n,
        batch_size=batch_size, max_new_tokens=max_new_tokens, out=out,
        dump=dump, num_shards=num_shards, shard=shard, score_only=score_only,
        balance_flavours=balance_flavours, store=store)

    if args.score_only:
        dump = [json.loads(l) for p in args.score_only
                for l in open(p, encoding="utf-8")]
        print(f"scoring {len(dump):,} rows from {len(args.score_only)} dump(s)")
        # No prompt is read on this path, so none is recorded: the dumps were
        # produced by an earlier run and only that run knows which prompt it used.
        summary = score(dump, args.model, prompt=None)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\n[wrote] {args.out}")
        return summary

    template = open(args.prompt, encoding="utf-8").read()
    if "{question}" not in template:
        raise StageError(f"{args.prompt}: no {{question}} placeholder")

    items = load_items(args.n, args.dataset, args.balance_flavours)
    if args.num_shards > 1:
        # Stride, not a contiguous block: the item list is grouped by type, so
        # contiguous shards would give one GPU all of T20's long sentences and
        # another all of T8's five-word questions.
        items = items[args.shard::args.num_shards]
    print(f"model:  {args.model}")
    print(f"prompt: {args.prompt}")
    print(f"items:  {len(items):,}\n", flush=True)

    outs = generate(args.model, [template.replace("{question}", r["question"])
                                 for r in items],
                    args.batch_size, args.max_new_tokens)

    store = open_store(args.store, verbose=False)
    codes = np.asarray(store.codes)
    idx = store.surface_index()

    rows = collections.defaultdict(lambda: collections.Counter())
    n_ret, dump = [], []
    for r, out in zip(items, outs):
        got, ok = parse(out)
        gold = target(r)
        low = {g.casefold() for g in got}
        i = int(np.searchsorted(codes, r["node_code"]))
        anchored = (i < len(codes) and int(codes[i]) == r["node_code"])
        resolved = anchored and any(
            i in [int(a) for a in idx.get(g, ())] for g in low)
        c = rows[r["type"]]
        c["n"] += 1
        c["parsed"] += ok
        c["recall"] += gold.casefold() in low
        c["resolved"] += resolved
        c["empty"] += ok and not got
        c["extra"] += max(0, len(low - {gold.casefold()}))
        c["has_extra"] += bool(low - {gold.casefold()})
        if r["negative"]:
            c["neg_n"] += 1
            c["neg_recall"] += gold.casefold() in low
        n_ret.append(len(got))
        dump.append({"id": r["id"], "type": r["type"], "band": r["band"],
                     "negative_flavour": r["negative_flavour"],
                     "question": r["question"], "gold": gold,
                     "raw": out.strip(), "parsed": got, "ok": ok,
                     "recall": gold.casefold() in low, "resolved": resolved})

    summary = score(dump, args.model, args.prompt)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n[wrote] {args.out}")
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for d in dump:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"[wrote] {args.dump}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default=EXTRACTOR_PROMPT)
    ap.add_argument("--dataset", default=GENERATED_DIR)
    ap.add_argument("--store", default=None,
                    help="the graph store the extracted strings are resolved "
                         "against; omit to find the one in data/stores")
    ap.add_argument("--n", type=int, default=40, help="items per type (0 = all)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="per-item JSONL of raw outputs")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the item list N ways for data-parallel runs")
    ap.add_argument("--shard", type=int, default=0, help="which shard, 0-based")
    ap.add_argument("--score-only", nargs="+", default=None,
                    help="re-score existing dump JSONLs; no model is loaded")
    ap.add_argument("--balance-flavours", action="store_true",
                    help="stratify on negative flavour too (inflates the "
                         "negative share; use for per-frame diagnosis only)")
    args = ap.parse_args()

    try:
        run(model=args.model, prompt=args.prompt, dataset=args.dataset,
            n=args.n, batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens, out=args.out, dump=args.dump,
            num_shards=args.num_shards, shard=args.shard,
            score_only=args.score_only,
            balance_flavours=args.balance_flavours, store=args.store)
    except StageError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
