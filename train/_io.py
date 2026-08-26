"""JSONL read and append for the experiment's records.

Package-private on purpose: experiments stay independent, so each carries its own
copy rather than sharing a utility (the same pattern `graph_model`'s experiments
follow).  One record per run, hyperparameters and metrics together, is what makes
a results table reproducible from the results directory alone.
"""
import os
import json
import time


def append_jsonl(path, record):
    """Append ``record`` (a dict) as one timestamped JSON line to ``path``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def read_jsonl(path):
    """Every record in ``path``, in file order.  Blank lines are skipped."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_items(root, split):
    """``{id: item}`` for one split of a generated dataset directory."""
    return {r["id"]: r for r in read_jsonl(os.path.join(root, f"{split}.jsonl"))}
