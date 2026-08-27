# `ask/` — one question in, one grounded answer out

**Status: built, and verified on the hardware.** This file is the record of what
was decided and why, and of what the code said back once it existed — the
decisions (D1–D22) as they were settled, then the findings (V1–V13) from
checking each against the two repos and then against a GPU. `ask/README.md` is
how to run the result; this is why it is shaped the way it is.

Where a finding contradicts the decision above it, both are kept: the decision
records what was intended and the finding records what turned out to be true,
and deleting the first would hide a reason that is still load-bearing.

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
at the corpus maximum (L = 14,055 → ~6 GB transient). Measured, it is not even
approximately different: 16/16 answers identical between the two backends
(V12), and the same sentinel on the corpus's largest ball.

**Three things building it changed.** *The sweep is enumerated from the ladders,
not from the attested pairs* — with a `TOKENS_PER_NODE` band (6–40, from the
corpus) filtering out the pairs no ball can produce, which is the same
restriction reached without having to carry a table of observed shapes.
*`--precompile` implies `--impl flex`*, because the default backend choice is
"eager unless warm", and a sweep on an eager model compiles nothing while still
writing the marker that says it worked — a flag that silently did nothing. And
*the sweep aims at each bucket and then measures where it landed*: the first
version sized its filler ball by guessing tokens-per-word, missed every bucket
upward, and warmed shapes no question will ask for. `shapes.json` now records
`asked`, `hit`, `missed` and `failed`, and `is_warm()` believes it only when
`hit > 0`.

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

### D22 — three flags cut the chain short, and skip what they do not need

The chain is extractor → store → GTLM, and each of the three flags below drops
a piece of it *including its load*, which is the point: a model nobody will call
costs 20–60 seconds of startup for nothing.

| flag | what it does | who it is for |
|---|---|---|
| `--words a,b` | supplies the headwords; no extraction, no 12B load | someone who already knows the lemma, and every test of the two stages after it |
| `--retrieve-only` | stops at the ball; no GTLM load | inspecting what the model would be shown, and the fastest check that a word is in the base |
| `--impl flex\|eager` | forces the attention backend | reproducing a number, and the parity check, which must pin both sides |

They were added while building the real stages, for a reason worth recording:
without `--words` there is no way to exercise retrieval and answering before
extraction exists, and a build order that leaves each step runnable (below) has
to have one. They earn their place afterwards — `--retrieve-only` answers "is
this word in the base, and what would the model see?" in about three seconds,
which is a lexicographer's question and not a debugging one.

*Rejected:* a single `--stop-after STAGE`. Tidier as an enumeration, but the two
flags do different things — one supplies an input, the other withholds an output
— and folding them together would hide that `--words` still answers.

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
2. **Done.** `LeftPadCollator`, `PlainCollator` and `to_left_padding` moved from
   `train/run.py` and `train/evaluate.py` into `train/batching.py` (V4), with
   every caller and cross-reference repointed. `to_left_padding` came along
   because `LeftPadCollator` is a one-line wrapper around it, and leaving it
   behind would have made `batching` import the evaluator to do its own padding.
3. **Done.** `retrieve.py` + `backbone.load_store` — verified against 600 corpus
   balls, exactly (V8).
4. **Done.** `backbone.load_gtlm` + `answer.py`. Smaller than expected: one
   `from_pretrained` call restores base + LoRA + bias.
5. **Done.** `checks/check_parity.py` — before extraction, because it needs only
   steps 3–4 and it is what licenses trusting anything downstream.
6. **Done.** `extract.py` + the `measure_extraction` split (D15) — closes the
   chain.
7. **Done.** `precompile.py`'s `warm()` — the piece the demo cannot stand in
   for, since what it proves is that inductor's cache survives a new process.
8. `ask/README.md`, and the acceptance stages under `ask/slurm/` that run all of
   the above on the hardware.

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

## Verified

V1–V7 are what reading the two repos said before anything was built. V8–V13 are
what the code said once it ran: against the corpus, then on a GPU through
`ask/slurm/run_checks.sbatch`. Both are kept, in the order they were found.

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

### V8 — retrieval rebuilds the corpus ball exactly

600 test items whose extraction resolved, run through `ask`'s own
`resolve()` → `build()`: **600 of 600** match the ball
`datasets/balls/test.jsonl` records for them — node count, edge count, token
count and anchor positions, every one. The strongest available evidence that a
served ball and a trained ball are the same object.

Timing is the finding underneath it. Per question, cold: **p50 20 ms, p90
222 ms, max 5.1 s.** The tail is hub anchors — `build_balls._mwe_sort_key` reads
one node text per MWE candidate and a common verb has tens of thousands, capped
at `MWE_SCAN_CAP = 20,000`. The dataset build pays this once per anchor across
12,490 items; a session pays it per new word, and the session cache (`store`
bundle) makes the second question about the same word free. Left alone: it is
dataset policy, shared code, and 5 seconds on the tail of an answer that takes
seconds anyway.

### V9 — the surface index can outlive the store it indexes

`qa/store.py` caches `qa_surface_index.json` beside the store's arrays with no
staleness test, and it maps surfaces to **anchor indices** — the identity that
does not survive a rebuild (D9). The current store was rebuilt on 2026-08-27 and
the index file is from 2026-08-22; every lookup checked still resolves, because
the build is reproducible and node order was preserved.

The first version of this warned whenever the index was older than the manifest.
That is a proxy for the question, and it answers it wrong in the ordinary case:
the build is reproducible, so an older index is *expected*, and the warning fired
at every startup about a file that was correct — the kind of alarm a user learns
to read past, which is worse than none.

So `ask` asks the question itself. When the mtimes are inverted it samples 256
surfaces and checks that the anchor each entry names still carries that surface,
as its lemma or one of its forms (`backbone.index_agrees`). Measured: 0.13 s
against 911k entries, 3000/3000 agreeing on the current pair, and it returns
False on an index whose anchors are shifted by one or whose surfaces have been
reassigned. Reordering misplaces everything at once, so a sample this size is
not a weak test of it — it either passes cleanly or fails on its first few.

A passing probe now prints a dim note, not a warning; a failing one warns and
names the file to delete. A fingerprint written *into* the index would still be
better, since it would let `qa/store.py` invalidate rather than check — but it
costs the pipeline a 2-minute rebuild on every store change and belongs there,
not here.

### V10 — only this repo's venv can run the real chain

`ask` needs numpy *and* torch, transformers, peft and `gtlm`. A node's
`/usr/bin/python3` has numpy alone; `graph_model/.venv` has everything except
`gtlm`; `gams_gtlm/.venv` has all of it. So `bin/ask` prefers the repo venv,
falls back to `pick_python.sh`, and refuses with a message naming
`ask/slurm/run_ask.sbatch` — on a Blackwell host (python3.12) the container is
the only way in. `--demo` skips the question entirely.

It also defaults `HF_HOME` to `/shared/workspace/povejmo/huggingface_cache`,
where `data/analysis/extract_in_container.sh` points: the 12B extractor is
there and not in `~/.cache`.

### V11 — library output has to be captured, not just quieted

The first real answer came out with `datasets`' four progress bars, the
loader's own `print`, and the sliding-window warning *inside the answer string*
— it is logged from `generate`, on the worker thread, while tokens are
streaming.

Two mechanisms, because one is not enough. `ui.stage` redirects `sys.stdout`
and `sys.stderr` into a buffer for the duration of a stage (`--debug` prints
what it caught), which handles anything a stage provokes; and the sliding-window
warning is suppressed outright via a new `first=False` on
`train/_log.quiet_repeated_sliding_window_warning`, with the same fact stated on
the GTLM startup line instead — it is the one message that escapes a stage.

### V12 — the serving path is the evaluated path, on both backends

`check_parity --n 16`, test split, `checkpoint-4400`, A100:

| | result |
|---|---|
| eager: `ask` vs `evaluate.pass2` | **16/16 identical** |
| flex: `ask` vs `evaluate.pass2` | **16/16 identical** |
| flex vs eager, same items through `ask` | **16/16 identical** |

The first two are what D11 exists for: the batch, the generation config and the
decode that `ask/answer.py` assembles produce the same string, token for token,
as the code every `arms_v3` number came out of.

The third is the D17 fallback's price, and on these items it is zero. Not
guaranteed in general — the two backends are the same arithmetic in a different
order, and bf16 plus greedy decoding can turn a last-bit difference into a
different token — but it does not happen here, which is what makes serving eager
on a cold GPU an acceptable default rather than a quiet degradation.

### V13 — the whole chain runs, and the memory ceiling is the biggest ball

`ask/slurm/run_checks.sbatch`, A100-40GB, every stage exit 0.

*The chain, nothing supplied by hand.* `ask "Navedi različne pomene besede
brahialen."` → the 12B extractor returns `["brahialen"]` in ~1 s → 73 nodes →
`ODGOVOR: glede na roko ali z roko povezane`, which is gold. Startup is 11.6 s
in total: store 1.5 s, extractor 6.9 s, checkpoint 3.2 s. The 20 s estimate the
demo carried for the extractor was three times too pessimistic; `demo.py`'s
timings are now the measured ones.

*The extractor is not deterministic across store builds.* Asked
`Katere sklanjatvene oblike ima folklora?` it returned `folklora`, which
resolves — where the corpus's own extraction run returned `folklór`, which does
not, and the item became an `extract_miss`. Same model, same prompt, different
run. Nothing is wrong; it is a reminder that the corpus's 4.5 % miss rate is a
measurement of one run and not a property of the question.

*The corpus's largest ball* (T21-000682, 529 nodes / 11,421 tokens) answers on
both backends and agrees — both say `ODGOVOR: ni podatka v bazi`. Eager takes
3.3 s. Flex took **339 s the first time and 6.6 s once the cache held its
shape**, which is D17's entire argument in two numbers: the compile is real,
it is paid once, and `--precompile` is where it should be paid.

*The sweep, once it was aimed properly:* **21 shapes asked, 21 hit, 0 missed, 0
failed, 27.7 s.** A second process then loaded the same checkpoint, found the
cache, reported `flex (prefill) + eager (decode)` without being told to, and
answered — which is the whole of D17 demonstrated end to end.

*Getting there took three corrections, and the first sweep found all of them.*
It ran on an eager model (so it compiled nothing), aimed its filler balls by
guesswork (so it landed a rung low on the wide shapes — the per-node rounding
loss multiplies, up to 511 tokens at `N = 512`), and keyed its record by the
shape it landed on rather than the one it asked for (so two targets colliding on
one landing site erased a row). It also OOMed on five shapes, asking for 9 to
43 GiB on a 40 GB card, because a session that has just built balls from 1 k to
14 k tokens leaves the caching allocator holding large unusable blocks;
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which is what torch's own
OOM message recommends, is now set by `bin/ask`. After all four, nothing fails.

---

## Still open

* Whether extraction should eventually be promoted out of `analysis/` into a
  shared stage module (noted under D15; deliberately not part of this work).
  `measure_extraction.load_extractor` / `generate_with` are now the seam it
  would move on.
* The shared-backbone optimisation (D4), when a GaMS3-12B-based GTLM exists.
* A fingerprint in `qa_surface_index.json`, so a rebuilt store *invalidates* it
  rather than being sampled on load (V9). Belongs in `qa/store.py`, where it also
  costs the pipeline a rebuild on every store change. The sampled probe is the
  cheap standing answer, not a placeholder for a missing one.
* The retrieval tail (V8): 5 s on hub anchors, from ranking tens of thousands of
  MWE candidates by a text read each. Only worth touching if a session's first
  question about a common verb starts feeling slow — the cache makes the second
  one free, and the policy is the dataset's, not serving's.
