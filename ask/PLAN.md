# `ask/` — one question in, one grounded answer out

**Status: design in progress.** Nothing is built yet. This file records what has
been decided and why; it becomes `ask/README.md` once the package exists.

The pipeline's three built halves — the graph store, the QA dataset, the trained
GTLM — have only ever been driven as batch jobs over a corpus. `ask/` is the
fourth thing: the same chain, run once, on a question nobody wrote a template
for.

```
question ──▶ extract ──▶ resolve ──▶ ball ──▶ GTLM ──▶ "ODGOVOR: …"
             (LLM)       (surface   (hop-2   (graph
                          index)     policy)  forward)
```

**The governing principle, which most decisions below are an application of:**
`ask` *imports* the code that built the training input rather than rebuilding
it. A second implementation of the ball policy, the graph assembly or the
generation config would drift silently, and every `arms_v3` number would stop
applying with nothing to show for it.

---

## Decisions

Numbered in the order they were settled. Each records the alternative that was
rejected, because the reason is the part that goes stale silently.

### D1 — v1 is the full chain in one warm process

`question → extract → resolve → ball → GTLM answer`, with a one-shot face and an
`--interactive` face over the same loaded state. No HTTP server. No writer LLM.

*Rejected:* shipping the retrieval half first. The model load path is the part
with unknown unknowns; deferring it defers the risk rather than retiring it.

*Rejected:* an HTTP endpoint in v1. A long-lived server is an operations
problem, not a code problem, and nothing is plugged into it yet.

### D2 — the writer LLM is out of scope

`ask` returns the GTLM's answer. It does not call a second model to write prose
around it. When that lands it will be GaMS3-12B or the GTLM itself, decided
then; no placeholder field is reserved for it now.

### D3 — GTLM checkpoints only

A `plain_llm` checkpoint is refused at load. One input path (ball → graph), one
model class.

*Rejected:* also serving the serialised/plain-stack arm, which scored higher in
`arms_v3` (0.8168 vs 0.7908). That is a study result about a training
condition, not a decision about what to serve, and supporting it would double
the input-encoding path for a baseline that exists to be compared against.

### D4 — the extractor and the GTLM are loaded independently

Two model objects, always, even when their backbone ids match.

**Deferred optimisation, for when a GaMS3-12B-based GTLM exists.** Then the
extractor and the GTLM's backbone are the same weights and loading them twice
costs ~48 GB instead of ~24 GB. Sharing is not free: `GTLMGemma3ForCausalLM`
**drops Gemma-3's sliding window** (its own docstring: *"The sliding window is
dropped… past `sliding_window` packed tokens the adapter stops matching the
pretrained attention pattern on the sliding layers"*). The extractor prompt is
~700–900 tokens, well past the 512-token window, so running extraction through
the GTLM module with `disable_adapter()` would attend fully where stock Gemma
attends within a window — same weights, different attention, possibly different
extracted words than the 96.8 % that was measured.

Two ways to take it when the time comes:

* **cheap** — `disable_adapter()` on the one module, then re-run
  `data/analysis/measure_extraction.py` through that path and check `resolved`
  against the stock number. Within noise, done.
* **correct** — load the weights once and expose a stock `Gemma3ForCausalLM`
  view over the same tensors for extraction, so the window applies. Costs
  parameter rebinding (`load_state_dict(assign=True)`), a test asserting
  `data_ptr()` equality, and a standing prohibition on `merge_and_unload()`.

Take the cheap one first; the measurement decides whether the correct one is
needed. **This note is duplicated as a comment at the dispatch point in the
code**, since that is where someone loading a 12B GTLM will be standing.

### D5 — the checkpoint is a flag with a default constant

`--checkpoint` takes **the path to a real checkpoint directory** and loads it
directly. No searching a run directory for the best step by some metric — that
is overkill for a path that is written down once and updated by hand.

The default is a module-level constant, updated as better models are produced,
matching `DEFAULT_MODEL = "cjvt/GaMS3-12B-Instruct"` at the top of
`measure_extraction.py`. It points at the arm-1 run with the highest **test**
accuracy:

```
checkpoints/sl_qa/arms_v3_0002_data_rootdata-datasets-balls-v2_c_plain_llmFalse_
spdTrue_magneticTrue_max_length2048_batch_size4_accumulation_steps4_seed2/checkpoint-4400
```

Seeds 2 and 0 tied exactly at 0.791667 (1729/2184); seed 2 wins on dev accuracy
(0.8769 vs 0.8683). The spread across the three seeds is 0.0028, within seed
noise (sd 0.0016), so this is a choice of *which* model to serve and not a
claim that it is better than its siblings.

The resolved real path is printed at startup and carried in the JSON output
(D10).

*Rejected:* a `checkpoints/served` symlink. `checkpoints/` is gitignored and
this repo lives on a shared filesystem, so the symlink is mutable global state
another user can repoint under a running colleague.

### D6 — a retrieval miss is answered, not refused

`parse()` plus resolution has four outcomes, and only one of them is an error.
Measured over the corpus's 12,490 extraction rows:

| outcome | count | share | what `ask` does |
|---|--:|--:|---|
| resolved | 11,925 | 95.48 % | the normal ball |
| parsed, did not resolve | 564 | 4.52 % | the `(ni v bazi)` ball, model answers |
| empty list `[]` | 0 | 0 % | the `(ni v bazi)` ball, model answers |
| unparseable | 1 | 0.01 % | **D7** — repair turn, then an error |

**The repair turn fires only on unparseable output.** Never on a resolution
failure: there the extractor did its job correctly, and re-asking would push it
toward returning some *other* word that happens to be in the graph — an
incentive to fabricate a target to satisfy the lookup, converting an honest
`ni podatka v bazi` into a confident answer about the wrong word.

An empty list is a successful parse, not a malfunction, and takes the same path
as a non-resolving word: the model answers, as it was trained to. The only
difference is cosmetic — there is no word to name, so the node is written with
the slot empty (`iztočnica:  (ni v bazi)`) rather than with the question text,
keeping the marker where it sits in all 445 training examples.


When the extractor's words resolve to nothing, the ball is **not** empty. It is
the single node the dataset builder already writes for this case
(`build_balls.py:368-373`):

```python
texts = [f"iztočnica: {lemma} (ni v bazi)"]
```

**445 of 12,490 dataset items are exactly this**, with gold answer
`ODGOVOR: ni podatka v bazi` (`spec.SENTINEL`). The model was trained to say so.
There are zero zero-node balls in the corpus, so serving one would be off
distribution; serving the one-node ball is on it.

At serve time the word in that node comes from the extractor's returned string,
where the dataset builder used the item's gold `lemma`.

*Rejected:* refusing to answer, with or without near-match suggestions. The
model already handles this case; refusing would replace a trained behaviour with
an invented one.

### D7 — unparseable extractor output is an error, after one repair turn

Distinct from every case in D6: the output was not a list at all. This is a
malfunction, not a fact about the knowledge graph, and must not be laundered
into `ni podatka v bazi`.

The one real instance in the corpus shows the failure mode:

```
'["naganja", "naganja", "naganja", … "naganja"'
```

— a degenerate repetition loop truncated by `max_new_tokens=64`, so the list is
never closed. `LIST_RE = \[.*?\]` needs a closing bracket, so `parse()` finds
nothing to salvage even though the strings are right there. **Handle this in
`ask`, not by widening `parse()`**, whose behaviour defines the historical
`parsed` metric. Note also that a greedy repair turn may simply reproduce the
loop; if it does, D7's error path is the outcome and that is correct.

Extraction is greedy (`do_sample=False`), so a plain retry reproduces the same
output. The retry is a **repair turn**: the bad output fed back with a short
corrective instruction, still greedy, so the tool stays deterministic.

The instruction lives in `data/prompts/extractor_repair_prompt.txt`, versioned
alongside `extractor_prompt.txt` — prompts here are files, not string literals.
It takes two placeholders: `{output}` (what came back) and `{question}`, the
latter matching the existing prompt's convention.

The prompt:

```
Tvoj prejšnji odgovor ni bil v zahtevani obliki:
{output}

Poskusi znova. Vrni samo seznam nizov v Python sintaksi, brez razlage in brez
dodatnega besedila. Uporabi dvojne narekovaje, ne enojnih. Vsako besedo navedi
samo enkrat in seznam zaključi z oglatim zaklepajem.

Primeri pravilne oblike:
["avto"]
["avto", "pes"]
["črni pes"]

VPRAŠANJE:
{question}
```

"Vsako besedo navedi samo enkrat" is there for the one observed failure — a
repetition loop — and the closing-bracket instruction for the truncation it
caused.

If the repair turn also fails to parse, `ask` reports the failure and produces no
ball and no answer.

### D8 — the tool assumes a GPU is present, and says nothing about the cluster

`ask` runs where it is started and expects a visible GPU. No partition names, no
`srun` lines, no allocation logic — not in the code, not in the docs.

A thin sbatch wrapper runs the same command on a compute node, for testing.
Internal tooling, not part of the tool's story.

### D9 — the result is the answer plus reconstructable evidence

On screen: the answer, plus a one-line ball summary. Under `--json`: the
question, the extracted strings, the target **node codes**, the anchor node
texts, ball counts, the answer, and the resolved checkpoint path. The full ball
only under an explicit flag.

Carrying codes rather than the whole ball is sound because `build()` sorts by
node id, making the ball a pure function of `(store, targets)`.

**Two node identities, and mixing them is a silent corruption.**

| | what it is | stable? | who uses it |
|---|---|---|---|
| **anchor index** | position in this store's arrays | **no** — changes on rebuild | `surface_index()` returns it; `build()` consumes it |
| **node code** | the node's durable identity | yes | what `relabel.py` persists; translated back at `build_balls.py:358` via `searchsorted` |

Inside `ask` the chain is strings → `surface_index` → **indices** → `build()`,
with no code round-trip; the dataset only round-trips because it persists
targets to disk. But the JSON reports **codes**, or the reconstruction promise
breaks the first time the store is rebuilt.

### D10 — store and checkpoint provenance is printed, not enforced

Startup names the store (path, `graph_version`, `created`) and the resolved
checkpoint; both go into the JSON. A rebuilt store under an old checkpoint would
shift node texts and degrade answers, and this makes it visible without
requiring changes to `train/` or invalidating existing checkpoints.

*Rejected:* writing a store fingerprint into the run record and refusing on
mismatch. Real safety, but it means touching `train/` and no existing checkpoint
carries the field.

### D11 — one parity check ships, in `ask/checks/`

`python -m ask.checks.check_parity --n 200`: take N test-split items, feed each
**on-disk** ball through `ask`'s answer path, assert the answer string equals
what `evaluate.py` produces.

This is the one thing shared imports do not prove by construction, and it is the
claim every `arms_v3` number rests on: *the serving path is the evaluated path*.
It fails if the generation config drifts, the prompt node is attached
differently, chat formatting changes, or the token budget truncates.

**The check must pin its budget to whatever `evaluate.py` used for that batch**
(`min(320, 2·max(gold)+16)`), or D12's larger serving default would show up as
mismatches on long answers when only the cap differed.

*Rejected:* ball parity (redundant once `build()` is shared) and an end-to-end
extraction-drift sample (its number cannot be asserted on — the corpus targets
came from one past batched run, and bf16 batching numerics mean a batch-of-1
greedy re-run is not guaranteed to reproduce it token for token).

### D12 — the serving token budget is 1024

`evaluate.py` sizes generation per batch from the **gold** length
(`min(320, 2·max(gold)+16)`, ceiling `MAX_NEW_TOKENS = 320`). At serve time
there is no gold, so only a fixed budget is available, and it is set to 1024 so
nothing truncates.

Read against the corpus: only **5 of 12,490 items (0.04 %)** have gold answers
past ~320 tokens, all of them T12 (*razlaga pomena* — every sense of a
polysemous word, `|`-joined; *čist* has ~20). The long tail is T12 and T5
(*celotno spreganje* — every form of a verb). So serving above the evaluated
ceiling affects a very small slice, and on that slice a complete answer is worth
more than comparability with a number that was itself truncated.

### D13 — `bin/ask` owns the import seam

`train` is pip-installed and resolves from the repo root; `qa` / `lib` resolve
only with cwd at `data/`, and `pyproject.toml` deliberately refuses to install
them ("installing them would put names as generic as `build` and `lib` on the
global import path"). `ask` is the first thing needing both.

One shell wrapper puts both roots on `PYTHONPATH` and runs `python -m ask` from
the repo root, mirroring `bin/lookup`'s `cd data`. No module does `sys.path`
surgery; the seam is one readable line in one file.

### D14 — questions are logged by default

One JSONL record per question — timestamp, question, extracted strings, target
codes, ball counts, answer, resolved checkpoint — appended under `ask/logs/`
(gitignored). `--no-log` disables.

The model has only ever seen 19 templates. Real questions from colleagues are
the one distribution the project does not have, and capturing them costs
nothing.

### D15 — `measure_extraction.generate()` is split in place

It currently loads the tokenizer and model *inside* the function, which is fine
for a batch job and useless for a warm process. Refactor into
`load_extractor(model_name)` + `generate_with(model, tok, prompts, …)`, keeping
the existing `generate(model_name, …)` signature working for current callers.
`ask/extract.py` imports both, plus the already-clean `parse()` and the prompt
path.

*Rejected:* a small local implementation in `ask/`. It would duplicate five
settings that must match — `torch_dtype=bfloat16`, `padding_side="left"`,
`apply_chat_template(…, add_generation_prompt=True)`, `add_special_tokens=False`,
`do_sample=False` — which is exactly the kind of list that drifts.

*Rejected:* promoting extraction out of `analysis/` into a shared stage module.
Cleaner domain-wise (it *is* a pipeline stage; `data/pipeline/` already calls it
as one), but it moves `measure_extraction`, `data/pipeline/` and the sbatch
scripts together. Worth doing later, not as part of this.

### D16 — serve the way training ran: flex prefill, eager decode, bucketed shapes

**Not** forced eager. The stack already does the right thing:

* `dispatch.py:34-35` — "The decode (q_len < kv_len) fallback of `flex`
  therefore routes here too." Decode is eager automatically.
* `GraphCollatorV2(pad_to_block=True)` buckets **both** `L` and `N` — "both L
  and N trigger recompiles, so without N bucketing every distinct max-node-count
  would force a fresh autotune." Defaults: `L` on a 512-multiple + midpoint
  ladder, `N` on a power-of-two ladder floored at 32.
* Padding must end on the **left** for generation (`evaluate.py`: "with right
  padding a short row would grow its continuation on the far side of its pads"),
  which `to_left_padding()` does by rolling, idempotently.

So `ask` uses the same collator with the same settings, and both the shape
explosion and the padding side are already handled.

### D17 — flex is opt-in via a persisted, per-GPU-model compile cache

Each `(L, N)` bucket pays a one-off compile on first hit, which is unacceptable
interactively and pointless to repeat across sessions.

* `ask --precompile` compiles every reachable shape up front, then exits ready.
  "Reachable" is enumerated from the `(L, N)` bucket pairs **actually attested
  in `data/datasets/balls`** — 12,490 real balls — rather than from the open
  ended ladders, most of whose pairs are impossible (a 32-node ball cannot be
  14 k tokens).
* The compiled cache is written to `ask/cache/<gpu-model>/` — **in the repo, and
  gitignored**. Keyed by GPU model, not device or session, which is how
  Inductor's on-disk cache keys anyway, so one person's `--precompile` warms it
  for everyone on that GPU model.
* With no cache for the current GPU model and no `--precompile`, `ask` **falls
  back to eager** and prints the exact command that would warm flex for this
  device.

The fallback is sound: `dispatch.py:24-27` — "Both paths share the same
numerical core: the `gtlm_eager` dense function… dense (eager) math." Flex is an
acceleration, not different maths. At batch 1 the dense bias is affordable even
at the corpus maximum (L = 14,055 → ~6 GB transient).

`.gitignore` gains `ask/cache/` and `ask/logs/`.

### D18 — one module per stage of the chain

```
ask/
  __init__.py     Pipeline — warm state, answer()
  backbone.py     loads the extractor and the GTLM; home of the D4 note
  extract.py      question -> strings, with the repair turn (D7)
  retrieve.py     strings -> anchors -> ball
  answer.py       question + ball -> answer
  precompile.py   the --precompile shape sweep (D17)
  ui.py           the progress line and staged printing (D20)
  demo.py         the simulated stages behind --demo (D21)
  __main__.py     CLI, and the composition root: which stage gets which
                  implementation
  checks/
    check_parity.py
bin/ask           the PYTHONPATH wrapper (D13)
```

File names match the stages in the diagram at the top of this file, and the
split follows `train/`'s stated rule — by direction of dependency, so a module's
place says whether breaking it can break an answer. `backbone.py` exists as its
own module partly so the deferred shared-backbone work of D4 has one obvious
home.

### D19 — a per-question error never ends the session

A session costs ~2 minutes of loading plus a warm compile cache; losing that to
one bad question is the wrong trade. Every per-question failure — extractor
malfunction (D7), CUDA OOM on a very large ball, malformed input — is caught,
printed with its type, message and the ball size that provoked it, and the
session continues. `Ctrl-D` exits. `--debug` prints the full traceback.

### D20 — the interactive shell reports each stage as it finishes

A question takes a first extractor pass, a store lookup, a ball build, feature
computation and a decode. That is long enough that a silent wait followed by one
dump reads as a hang. Each stage instead prints its result the moment it has one,
under a live one-line progress indicator naming what is running now:

```
> Kaj pomeni beseda "oškropljen"?

  ⠹ luščim iztočnice…
  iztočnice   oškropljen
  ⠴ gradim okolico…
  okolica     84 vozlišč, 141 povezav, ~1.3k tokenov
              iztočnica: oškropljen · glagol: oškropiti · zveza: …
  ⠧ odgovarjam…

ODGOVOR: …
```

Three rules keep it from becoming a liability:

* **Progress goes to stderr, results to stdout.** `bin/ask "…" > out.txt` and
  `--json` stay byte-clean, and the spinner cannot corrupt a redirect.
* **Not a TTY, no ANSI.** When stderr is a file or a Slurm log, the spinner
  degrades to one plain line per stage, no cursor motion and no repaint.
* **One-shot mode prints stages too** (same stderr), because a 2-minute wait
  behind a single question needs the same reassurance; `--quiet` suppresses them.

The answer itself streams token by token, via HF's `TextIteratorStreamer` with
`generate` on a worker thread. Decode is eager and single-sequence (D16), so
tokens are already produced one at a time — streaming only stops the process from
sitting on them. The `--json` path collects the same stream and prints once.

Hand-rolled, in `ui.py`, ~40 lines of ANSI: a braille frame cycle on a daemon
thread plus `\r\033[K`. No `rich`, no `tqdm`. Every write goes through one
locked `_emit` that clears the spinner line first, so a message printed from
*inside* a running stage — the repair turn is the one case — cannot interleave
with a frame; `relabel()` lets that stage change what the live line says while
it is still running, rather than explaining itself afterwards.

*Rejected:* `rich` — it renders well, but it is a new dependency on the serving
path for a spinner and three lines of text, and it takes over stdout in ways that
fight rule one. *Rejected:* printing everything at the end with timings — simpler,
and it is what `--quiet` gives anyone who wants it, but it makes the slowest
stage indistinguishable from a crash.

### D21 — the stages are injected, and a simulated set ships

`Pipeline` does not import the stages it runs; it is handed an object with
`load_store / load_extractor / load_gtlm / extract / retrieve / answer /
precompile`. `__main__` picks the implementation: the real one, or `demo.py`,
which sleeps where a model would run and returns fixtures.

`ask --demo` therefore exercises the real CLI, the real printing, the real error
handling and the real `--json` shape on the standard library alone — no GPU, no
weights, no store. That is what makes D20 arguable before it is expensive to
change, and it stays useful afterwards: the UI can be worked on from a laptop,
and `--demo` is the regression test for output that has no other test.

Two rules keep it honest:

* **The delays are estimates, and say so** — in the ballpark of production so
  the pacing is real, capped at 20 s so watching it is bearable, each replaced
  by a measurement as its stage lands. `--fast` divides them by 8.
* **The fixtures are real rows.** Questions, extracted strings, ball sizes,
  anchor texts and answers come from `datasets/balls/test.jsonl` — the split
  `arms_v3` was scored on — and the node texts are read from that file when it
  is present. A demo that invents plausible-looking Slovene would be teaching
  the wrong expectations.

Beyond the three answering scenarios, two trigger words reach paths that are
otherwise hard to provoke on demand: `pokvarjeno` for the repair turn (D7) and
`napaka` for a mid-question failure (D19).

*Rejected:* a `--demo` flag threaded through each real stage. It puts fake
branches in the code that has to be trusted, and they never get deleted.

---

## Build order

Roughly dependency order, each step leaving something runnable:

1. **Done.** The whole tool against simulated stages (D21): `ui.py`, `demo.py`,
   `__main__.py`, `bin/ask`, `Pipeline`, and the four stage modules as stubs
   carrying their real signatures. `ask --demo -i` walks every path — answer,
   multi-anchor, retrieval miss, repair turn, per-question failure, `--json`,
   `--show-ball`, `--precompile`, `--quiet` — with nothing loaded. The output is
   settled before the expensive parts exist, and the module boundaries get their
   first real use.
2. Move `LeftPadCollator` from `train/run.py` to `train/batching.py` (V4).
3. `retrieve.py` + the real import seam — no model, prints the ball. Proves the
   store, the surface index and the index/code distinction (D9) against the
   printing that already exists.
4. `backbone.py` + `answer.py` — load a checkpoint, answer from a ball supplied
   on disk. Smaller than expected: one `from_pretrained` call restores base +
   LoRA + bias. Nothing else depends on extraction working.
5. `checks/check_parity.py` — before extraction, because it needs only steps 3–4
   and it is what licenses trusting anything downstream.
6. `extract.py` + the `measure_extraction` split (D15) — closes the chain.
7. `precompile.py`'s `warm()` — the only piece the demo cannot stand in for,
   since what it proves is that inductor's cache survives a new process.

---

## Vocabulary

Terms used above with a specific meaning, several of which are easy to conflate:

* **ball** — the hop-2 induced subgraph around the matched units, capped by
  `K_MWE` / `K_COLLOC_CAP`. The model's retrieved context.
* **anchor** — a matched lexical unit; a ball has one per resolved string.
* **anchor index** vs **node code** — see the table in D9. Indices are positions
  in one store build; codes are durable identity.
* **target** — what the extraction step resolved to. In the dataset it is
  persisted as codes; inside `ask` it is indices, never written out as such.
* **prompt node** — the extra node carrying the chat-templated question,
  attached to every anchor and packed last.
* **surface** — a casefolded lemma or inflected form; the key of the surface
  index.
* **the sentinel** — `ni podatka v bazi`, the trained answer for "no evidence"
  (`spec.SENTINEL`); distinct from the `(ni v bazi)` node text that provokes it.

---

## Facts established while designing (not decisions)

* **`max_length` is a per-node cap**, not a sequence cap (`train/config.py:130`).
  A large ball is never truncated; it costs attention.
* **Real ball sizes:** p50 77 nodes / 1,314 tokens, p90 156 / 2,655,
  p99 299 / 5,729, max 705 / 14,055.
* **The surface index holds 911,404 surfaces**; 3.8 % map to more than one
  anchor. Multi-match is settled by the dataset's D3 — union of every match,
  which is what training saw, and `_graph()` attaches the prompt node to *every*
  anchor for the same reason.
* **The ball policy is already a reusable function**:
  `build_balls.build(store, targets, k_mwe, k_colloc, cache, stats)` →
  `(node texts, edges, anchor positions)`, sorted by node id.
* **`parse()` in `measure_extraction` is already pure and reusable** — tolerant,
  first `[...]` wins, then bare quotes.
* **One call loads a checkpoint.** `GTLMGemma3ForCausalLM.from_pretrained(ckpt,
  graph_attn_impl=…)` reads `graph_bias_config.json` for the backbone id and
  `config.json` for the bias settings, loads the base, wraps it with
  `PeftModel.from_pretrained`, and calls `load_bias_parameters`
  (`graph_model/src/models/causal_lm.py:490-500`). Base + LoRA + graph bias, and
  `graph_attn_impl` is an accepted override — which is what D17's eager fallback
  needs.
* **The checkpoint self-describes the MODEL but not the FEATURES.**
  `config.json` carries `spd`, `magnetic`, `max_spd=8`, `magnetic_dim=32`,
  `magnetic_q=0.25`, `rrwp`, `k_hop`, `graph_attn_impl` — but **not
  `magnetic_m`**, which `_features` passes to `compute_magnetic_lap` and which
  `GraphCollatorV2` takes as a truncation. See V1 below.
* **`generation_config.json` for gemma-3-1b-it ships `do_sample=True`,
  `top_p=0.95`, `top_k=64`** and a `GenerationConfig` that does not explicitly
  override them brings sampling back, with only a warning as evidence
  (`evaluate.py:285-292`). This is why the evaluator's greedy path is imported
  rather than re-expressed.

---

## Verified against the code

Every decision above was checked against this repo and the `graph_model`
checkout before any of it was built. What the check changed:

### V1 — `magnetic_m` is not recorded in the checkpoint

`_features` calls `ds.compute_magnetic_lap(q=cfg.magnetic_q, m=cfg.magnetic_m)`
and `GraphCollatorV2` takes `magnetic_m` as an eigenvector truncation, but
`config.json` does not carry it. `RunConfig.magnetic_m` defaults to `0` and
every `arms_v3` arm used `0`, so the current checkpoints are safe.

`ask` uses `0` and states the assumption at the point of use. A future
checkpoint trained with a different value would produce magnetic features of a
different width than its bias module expects, which surfaces as a shape error at
load rather than as quietly wrong answers — so this fails loudly, but it should
still be recorded in the checkpoint eventually.

### V2 — the generation copy needs an answer it will not use

`_graph(row, tokenizer, with_answer=False)` still reads `row["answer"]`:
`chat_prompt` templates the full turn and then *truncates* at `ANSWER_PREFIX`,
which is what keeps the two dataset copies byte-identical up to the marker. A
served row therefore carries a dummy answer of `spec.PREFIX` (`"ODGOVOR: "`),
along with the `id` and `type` keys `_graph` writes into the graph attributes.

### V3 — the generation settings must be borrowed, not rebuilt

They are instance state on `GradeEvaluator`, not module constants. But
`GradeEvaluator(tokenizer, collator, splits=[])` constructs cleanly with no
splits, so `ask` instantiates one solely to take `gen_cfg_for(model)`,
`greedy_kwargs`, `stop_ids` and `use_cache` from it.

This is not fastidiousness. `gen_cfg_for` exists to override
`cache_implementation` from the model config, because the hub's
`generation_config.json` says `"hybrid"` for both stacks and
`GTLMGemma3ForCausalLM` must not have it — "a HybridCache's sliding layers would
truncate the KV span the adapter's attention still expects to see, and would fix
the cache length at the prefill size, which its decode path cannot work
against." A hand-built `GenerationConfig` gets this wrong silently.

### V4 — `LeftPadCollator` sits in the training path

It is defined at `train/run.py:71`, so importing it drags in `gtlm.train`,
`GraphTrainerV2` and the wandb setup, and couples serving to training —
against `train/README.md`'s own rule that a module's directory says whether
breaking it can break a run. **Move `LeftPadCollator` to `train/batching.py`**,
already described there as "shared by it". A small, contained change, made
before `ask` imports it.

### V5 — the compile cache is cheaper, not free

Verified: torch 2.11.0, `torch._inductor.config.fx_graph_cache = True`, and
`TORCHINDUCTOR_CACHE_DIR` is honoured — currently defaulting to
`/tmp/torchinductor_<user>`, which is node-local, per-user and wiped, exactly
the problem D17 solves.

But a warm cache skips *codegen*, not dynamo tracing and guard evaluation, which
are process-local. So a shared cache makes a first hit on a shape much cheaper —
not free, and never as cheap as a shape already compiled in this process. D17's
wording is "warms it for everyone", and that is the honest reading: warms, not
eliminates.

`torch.compiler.save_cache_artifacts()` (Mega-Cache) also exists in 2.11 and is
the more portable mechanism than sharing a raw cache directory; try it first,
fall back to `TORCHINDUCTOR_CACHE_DIR`.

### V6 — per-question feature cost

`TextGraphDataset` runs each structural feature through an HF `.map()`, which
carries fixed per-call overhead regardless of the one item in it. Not a blocker
at a few hundred milliseconds against a multi-second answer, but it is
per-question overhead that is not model time, and it is where to look first if
answers feel slower than the GPU work explains.

### V7 — the miss node names a different word at serve time

`build_balls.py:368` builds the miss node from the *item's* lemma:
`iztočnica: {r['lemma']} (ni v bazi)`. At serve time there is no item, so the
node has to be built from what the extractor returned. Two things follow.

It is not a divergence to fix. Of the 91 `extract_miss` items in the test split,
**59 already name a word unrelated to the question** — `T1-000687` asks about
*trikomponenten* and carries `lemma: institut`, because the out-of-base
questions draw their word from a slot and the row's lemma is left over. The
model was therefore trained on a node whose *word* is unreliable and whose
`(ni v bazi)` marker is not, which is exactly the signal serving needs.

And it is visible in the demo: `folklora` asks about *folklora*, the extractor
returns *folklór*, and the served node reads `iztočnica: folklór (ni v bazi)` —
the extractor's string, which is the only word serving actually has.

---

## Still open

* Whether extraction should eventually be promoted out of `analysis/` into a
  shared stage module (noted under D15; deliberately not part of this work).
* The shared-backbone optimisation (D4), when a GaMS3-12B-based GTLM exists.
