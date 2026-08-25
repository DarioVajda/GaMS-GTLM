"""Decode what the label mask actually supervises, for a human to read.

The masker cuts the prompt node at `ODGOVOR:` using character offsets.  If it cut
one token early or late, nothing would fail -- the run would train on a slightly
wrong span and report a slightly wrong number, which is the worst kind of bug to
have in a benchmark.  So the span is printed, decoded, for a person to check.

    .venv/bin/python -m train.checks.check_labels --n 10
"""
import os
import json
import argparse

import networkx as nx
from transformers import AutoTokenizer

from gtlm.utils import TextGraphDataset

from ..config import RunConfig
from ..data import _graph, OffsetLabelMasker


def main(argv=None):
    d = RunConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=d.data_root)
    ap.add_argument("--model-name", default=d.model_name)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-length", type=int, default=d.max_length)
    a = ap.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(a.model_name)
    with open(os.path.join(a.data_root, f"{a.split}.jsonl"), encoding="utf-8") as f:
        all_rows = [json.loads(l) for l in f if l.strip()]
    step = max(1, len(all_rows) // max(a.n, 1))
    rows = all_rows[::step][:a.n]              # spread the sample over the file

    ds = TextGraphDataset([_graph(r, tok) for r in rows])
    ds.tokenize(tok, max_length=a.max_length, add_eos=False)
    ds.compute_labels(OffsetLabelMasker(tok, a.max_length), num_proc=1)

    bad = 0
    for i, r in enumerate(rows):
        ex = ds[i]
        p = ex["prompt_node"]
        ids = list(ex["input_ids"][p])
        labels = list(ex["labels"])
        sup = [t for t, l in zip(ids, labels) if l != -100]
        masked = [t for t, l in zip(ids, labels) if l == -100]
        got = tok.decode(sup)
        # what SHOULD be supervised: everything after "ODGOVOR:" plus the
        # `<end_of_turn>` the chat template closes the model turn with.  The stop
        # token is dropped with skip_special_tokens rather than rstrip -- rstrip
        # strips a CHARACTER SET, so `rstrip("<eos>")` also ate the final `e` of
        # `svetnice` and reported a mismatch that was not one.
        want = r["answer"].split("ODGOVOR:", 1)[1]
        ok = tok.decode(sup, skip_special_tokens=True).strip() == want.strip()
        bad += not ok
        print(f"\n[{r['type']}  {r['id']}]  {'OK' if ok else '*** MISMATCH ***'}")
        print(f"  masked   …{tok.decode(masked)[-60:]!r}")
        print(f"  supervised {got!r}")
        if not ok:
            print(f"  expected   {want!r}")
    print(f"\n{len(rows) - bad}/{len(rows)} spans supervise exactly the answer")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
