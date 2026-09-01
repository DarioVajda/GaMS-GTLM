#!/usr/bin/env python3
"""Short names for checkpoints (D23).

    ask --alias gams checkpoints/sl_qa/scale_12b_gams_blr_gtlm_0000/checkpoint-9264
    ask --checkpoint gams "Kaj pomeni beseda brahialen?"
    ask --aliases

A checkpoint path in this repo runs to ~190 characters -- the run directory
encodes the whole arm, and `ui.field_path` exists to fold one over four rows
just to print it.  Typing that to choose between four backbones is not something
a session should ask of anyone twice.

The table is a flat `{name: path}` map in `ask/aliases.json`, sorted and
two-space indented: small enough to edit by hand, and diffable when it changes.
Paths are stored relative to the repo when they live under it, so the file
survives being cloned somewhere else.  `checkpoints/` is gitignored and this
file is not, deliberately -- the map from a name to an arm is the same knowledge
`train/SCALING.md` carries in prose, and a fresh clone should get the names even
though it gets none of the weights.  It then fails at load, naming the path,
which is legible.

**A name beats a directory.**  `resolve` consults the table before the
filesystem, so a local directory literally named `gams` would go unseen while
that alias exists.  Alias names are bare words and real checkpoint directories
are 190-character generated names, so the collision is theoretical; resolving it
the other way round would let a stray directory silently redirect a session to
different weights, which is the failure actually worth avoiding.

Nothing here imports torch, and `record` validates through
`backbone.read_checkpoint` -- two small JSON reads -- so a bad entry is refused
when it is written rather than at the first question of some later session.
"""
import os
import re
import json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aliases.json")

# Bare words only.  A name carrying a separator could not be told from the path
# it stands for, and `resolve` has to stay idempotent: resolving an already
# resolved path must be a no-op, which holds only if no name can look like one.
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def load(path=None):
    """The table, or `{}` if there is not one yet.

    A missing file is the ordinary state before the first `--alias`, not an
    error.  A corrupt one raises: it was written by this module or edited by
    hand, and silently answering from an empty table would send the session to
    the wrong weights without saying so.
    """
    p = path or PATH
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        table = json.load(f)
    if not isinstance(table, dict):
        raise ValueError(f"{p}: expected a JSON object of name -> path")
    return {str(k): str(v) for k, v in table.items()}


def save(table, path=None):
    """Write the table back, sorted, with a trailing newline."""
    p = path or PATH
    with open(p, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(table.items())), f,
                  ensure_ascii=False, indent=2)
        f.write("\n")


def store_form(checkpoint):
    """Repo-relative if it lives under the repo, absolute otherwise."""
    full = os.path.abspath(os.path.expanduser(checkpoint)).rstrip("/")
    rel = os.path.relpath(full, REPO_ROOT)
    return full if rel.startswith(os.pardir) else rel


def path_form(stored):
    """The stored form, back as a path that can be opened from anywhere."""
    s = os.path.expanduser(stored)
    return s if os.path.isabs(s) else os.path.join(REPO_ROOT, s)


def resolve(name, path=None):
    """A name to its checkpoint path; anything else through unchanged.

    The miss is not an error here -- `--checkpoint` still takes a path, and
    telling a typo from a path is `read_checkpoint`'s job, where the file that
    is missing can be named.
    """
    if not name:
        return name
    hit = load(path).get(name)
    return path_form(hit) if hit else name


def known(path=None):
    """The recorded names, sorted -- for the message a typo earns."""
    try:
        return sorted(load(path))
    except Exception:                   # noqa: BLE001 -- an error message only
        return []


def base_of(checkpoint):
    """The base model a checkpoint names, or `None` if it cannot be read."""
    from ask import backbone

    try:
        return backbone.read_checkpoint(checkpoint)[1]
    except Exception:                   # noqa: BLE001 -- reported as "manjka"
        return None


def record(name, checkpoint, path=None):
    """Validate, then record. Returns `(stored path, base model)`.

    Validation is `read_checkpoint`, which is also what a session runs: it
    refuses a run directory with no `config.json` and refuses anything whose
    `model_type` is not a GTLM (D3).  Paying that here means the table cannot
    hold an entry that fails only once a 12B load is already 30 seconds in.
    """
    from ask import backbone

    if not NAME.match(name or ""):
        raise ValueError(
            f"{name!r} is not a usable alias: a name is a bare word "
            f"([A-Za-z0-9][A-Za-z0-9._-]*), so that it cannot be mistaken for "
            f"the path it stands for")
    full = path_form(store_form(checkpoint))
    _conf, base = backbone.read_checkpoint(full)
    table = load(path)
    table[name] = store_form(checkpoint)
    save(table, path)
    return table[name], base


def forget(name, path=None):
    """Drop a name. Returns what it pointed at, or `None` if it was not there."""
    table = load(path)
    gone = table.pop(name, None)
    if gone is not None:
        save(table, path)
    return gone


def listing(path=None):
    """`(name, stored path, base model or None)` per alias, sorted by name."""
    table = load(path)
    return [(name, table[name], base_of(path_form(table[name])))
            for name in sorted(table)]
