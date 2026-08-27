# `ask/` — one question in, one grounded answer out

Ask a trained GTLM checkpoint a question about a Slovene word, and get its answer
plus the evidence it read. The whole chain runs in one process:

```
question -> extractor -> strings -> surface index -> anchors -> ball
         -> GTLM (+ ball) -> answer
```

```bash
bin/ask "Kaj pomeni beseda brahialen?"     # one question, then exit
bin/ask -i                                 # a warm shell, many questions
bin/ask --demo -i                          # the same, with simulated models

bin/ask --retrieve-only --words plezalo x  # what would the model be shown?
bin/ask --json "…" > answer.json           # machine-readable, stdout only
```

`bin/ask` is a wrapper, and it is the reason there is one: `train` resolves from
the repo root and `qa`/`lib` only from `data/`, so both go on `PYTHONPATH` in one
readable line and no module does `sys.path` surgery. It also picks an interpreter
that can actually import torch and `gtlm` — only this repo's `.venv` carries the
whole stack — and points `HF_HOME` at the shared cache the 12B extractor lives in.

The design, decision by decision with the alternatives that were rejected, is in
[PLAN.md](PLAN.md). This file is how to run it.

## What a session looks like

Loading is the slow part — the extractor is 12B — so a session loads once and
answers many times. There is no server, nothing to leave running, and nothing on
disk afterwards except the question log and the compile cache. Each stage prints
its result the moment it has one, under a live line naming what is running:

```
  baza       /shared/workspace/povejmo/gams_gtlm/data/stores/kg_graph_gemma3
             graph_version 3 · 37,498,126 vozlišč · zgrajena 2026-08-27
  ekstraktor cjvt/GaMS3-12B-Instruct
  GTLM       /shared/workspace/povejmo/gams_gtlm/checkpoints/sl_qa/
             arms_v3_0002_data_rootdata-datasets-balls-v2_c_plain_llmFalse_spdTrue_magneticTrue_max
             _length2048_batch_size4_accumulation_steps4_seed2/checkpoint-4400
             google/gemma-3-1b-it · flex (prefill) + eager (decode)
  pripravljeno 9.9s

> Navedi različne pomene besede brahialen.

  iztočnice  brahialen
  okolica    73 vozlišč · 73 povezav · ~1,484 tokenov
             ▸ iztočnica: brahialen (pridevnik, imenovalnik, ednina, moški spol, …)
ODGOVOR: glede na roko ali z roko povezane
```

Both paths are printed in full, folded at `/` to the terminal's width: every arm
of a sweep ends in a `checkpoint-NNNN`, and which arm this is — the seed, the
features, the data root — is written in the run directory's name and nowhere
else. The rows concatenate back into the path.

Measured on an A100: startup is ~10 s in total — the store opens in ~1.5 s, the
12B extractor loads in ~7 s, the checkpoint in ~3 s. Per question, extraction is
~1 s, retrieval 20 ms at the median (5 s on a hub word, once — the session
caches it), and the answer a few seconds.

**Progress goes to stderr, the answer goes to stdout.** `bin/ask "…" > out.txt`
gets exactly one line. Not a terminal, no ANSI: the spinner degrades to one plain
line per stage, so a Slurm log stays readable.

## Where to run it

The tool assumes a GPU in the terminal it runs in. Allocate one, then use it
normally. For running the same command on a compute node from the login node —
testing, mostly — there is a wrapper:

```bash
sbatch ask/slurm/run_ask.sbatch --words brahialen "Kaj pomeni brahialen?"
sbatch ask/slurm/run_checks.sbatch          # the acceptance stages, all of them
sbatch ask/slurm/run_checks.sbatch 2,3      # just the parity checks
```

`--demo` needs no GPU, no weights and no store: it runs the real CLI, the real
printing and the real `--json` shape against simulated stages, on the standard
library alone. `bin/ask --list-demos` says what it can be asked.

## Flags

| | |
|---|---|
| `--checkpoint PATH` | a real checkpoint directory; defaults to the best `arms_v3` arm |
| `--store PATH` | the processed graph store; defaults to the one in `data/stores` |
| `--extractor NAME` | the model that names the words a question is about |
| `--words a,b` | supply the headwords; skips extraction *and* its 12B load |
| `--retrieve-only` | stop at the ball; skips the GTLM load |
| `--impl flex\|eager` | force the attention backend |
| `--precompile` | compile every `(L, N)` shape first, and cache it per GPU model |
| `--json` / `--show-ball` | machine-readable output / every node of the ball |
| `--quiet` / `--debug` | answers only / tracebacks, captured library output, the checkpoint's feature settings |
| `--demo` / `--fast` | simulate every stage / run the simulation 8× faster |

## Why flex is opt-in

Serving is the shape training ran: flex for the prefill, eager for the decode,
both axes padded onto the bucket ladder the collator uses. But flex compiles per
shape, and the first question of a cold session would pay for it. So `ask` serves
**eager unless this GPU model has a warm cache**, and says so:

```
  GTLM       …/checkpoint-4400
             google/gemma-3-1b-it · eager (flex attention kernels not compiled)
  compile the shapes first for a faster prefill: `ask --precompile`
```

`--precompile` walks every shape the corpus actually lands on — 21 of them,
enumerated from the 12,490 real balls rather than from the open-ended ladders —
writes inductor's output under `ask/cache/<gpu-model>/` (gitignored, shared by
every session on that GPU model), and records in `shapes.json` which shapes were
asked for, which were reached, and which failed. The sweep took 28 s on an A100
with a warm inductor cache; the next process picks flex on its own.

## The one check that matters

```bash
python -m ask.checks.check_parity --n 32 --impl flex --drift
```

Every accuracy number in `train/results` describes the *evaluated* path. This
asserts that `ask` answers the same string `train/evaluate.py` does, item for
item, from the same on-disk ball — with the generation budget pinned to the
evaluator's, so only the decode is being compared. `--drift` additionally answers
the same items on the other backend, which is what the eager fallback costs.

Retrieval needs no such check: `ask` calls `build_balls.build()` and
`train/data._graph()` directly, so a served ball and a trained ball are the same
object. That it holds was measured anyway — 600 test items, 600 exact matches on
node count, edge count, token count and anchor positions.

## Layout

```
ask/
  __init__.py     Pipeline: the chain, warm; and the JSON report
  backbone.py     loading — store, extractor, checkpoint
  extract.py      question -> strings, with the one repair turn
  retrieve.py     strings -> anchors -> ball
  answer.py       question + ball -> answer, streamed
  precompile.py   the (L, N) sweep and the per-GPU cache
  ui.py           the progress line, the stage results, the stream
  demo.py         simulated stages, real fixtures, estimated delays
  __main__.py     the CLI, and which implementation each stage gets
  checks/         check_parity
  slurm/          run_ask, run_checks (+ check_stages.sh)
  logs/           one JSONL line per question asked (gitignored)
  cache/          inductor's output, per GPU model (gitignored)
```
