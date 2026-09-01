# Does GTLM beat a plain LLM reading the same graph as text?

**Yes, at 12B. No, at 1b.** That crossover is the study's result.

The question: GTLM injects graph structure as attention bias. The cheap
alternative is to flatten the same subgraph into the prompt and let a plain LLM
read it. Both arms receive **identical information** (verified item by item —
`data/qa/check_variants.py`) and differ only in how it is encoded.

| backbone | GTLM test | serialised test | winner |
|---|--:|--:|---|
| `gemma-3-1b-it` | 0.7953 | **0.8205** | serialised, +0.025 |
| `GaMS3-12B-Instruct` | **0.9217** | 0.8924 | **GTLM, +0.029** |

At 1b the structural bias is a freshly-initialised 48-layer stack the backbone is
too weak to exploit, and flat text wins. At 12B it inverts. A study that had only
run the small cell would have reached the opposite conclusion.

⚠️ The 1b legs are 8 epochs against the 12B's 16, and the 1b serialised cell was
never re-run at 16. The 1b GTLM arm *was* (0.8091), still below serialised's
8-epoch 0.8205, so the direction holds — but the 1b margin is not epoch-matched.

## Final results

All 12B cells: 16 epochs, 9,264 steps, effective batch 16, 4 × B200, one seed.
Dev is n=1040 (full split; the in-training curve uses a fixed 527-item subsample —
different denominator, do not mix them). Tier C is n=115, one item ≈ 0.009.

| cell | arm | dev | test | core | **A** | C | cost |
|---|---|--:|--:|--:|--:|--:|--:|
| **GaMS-12B** | **GTLM** | **0.9788** | **0.9217** | 0.9719 | **0.9362** | 0.1652 | 19.4 h |
| GaMS-12B | serialised | 0.9615 | 0.8924 | 0.9600 | 0.8511 | 0.1217 | ~45 h |
| GaMS-12B | no retrieval | 0.6413 | 0.5975 | 0.6329 | 0.5745 | 0.2000 | 1.4 h |
| gemma-12B | GTLM | 0.9731 | 0.9103 | 0.9675 | 0.8787 | 0.2435 | 19.4 h |
| gemma-12B | no retrieval | 0.6308 | 0.5755 | 0.6098 | 0.5617 | 0.1565 | 1.3 h |
| gemma-4b | GTLM | 0.9404 | 0.8658 | 0.9318 | 0.8277 | 0.1043 | 9.1 h |
| gemma-1b | GTLM (16 ep) | 0.8846 | 0.8091 | 0.8837 | 0.7383 | 0.0609 | 6.3 h |

Tiers: `core` = seen phrasing, **`A` = unseen phrasing**, `C` = unseen relation
(antonyms, test-only).

## The advantage is generalisation, not fit

This is the finding that matters, and the one easiest to overstate. Decomposing
GTLM's +0.0293 test lead over serialised at 12B:

| tier | tests | Δ | items | significance |
|---|---|--:|--:|---|
| core | seen phrasing | +0.0119 | 19/1599 | 1.9σ — **not significant** |
| **A** | **unseen phrasing** | **+0.0851** | 40/470 | **4.3σ** |
| C | unseen relation | +0.0435 | 5/115 | noise |

**The two arms fit the training distribution equally well.** Seven times more of
the gap sits in Tier A than in core, and only Tier A survives a significance test
(unpaired, so conservative). Claim generalisation; do not claim raw accuracy.

The behavioural signature agrees: false `ni podatka v bazi` over Tier A positives
is **0.0173 for GTLM against 0.0741 for serialised** — 4.3×. Shown a rephrasing,
the serialised arm fails to find content sitting in its own context window.

Two things make the measured gap a floor rather than a ceiling: GTLM's best
checkpoint was its **last** (step 9264, still improving), while serialised peaked
at step 7600 and flatlined for its final 1,664 steps; and one dev item plus 8 train
items were **truncated** in the serialised corpus, never in GTLM's.

## Other results

* **Scale, in-family** (gemma GTLM): 1b **0.8091** → 4b **0.8658** → 12b **0.9103**.
  Concave, not saturated. 4b → 12b is the clean leg; 1b is text-native where 4b/12b
  are vision-extracted towers.
* **Slovene pretraining is worth most on Tier A.** GaMS-12B vs gemma-12B, both
  GTLM: test +0.0114, but Tier A **+0.0575**. It helps on the same axis GTLM does,
  and stacks with it.
* **`bias_lr` was the single largest lever in the study.** The GaMS GTLM cell at
  `bias_lr` 1e-2 scored 0.7459; at 4e-3 it scores 0.9217. The 1b ladder shows
  8 → 16 epochs is worth only +0.006, so the LR is doing essentially all of
  +0.176. A 48-layer freshly-initialised bias stack at 20× the LoRA rate diverges.
* **Retrieval is doing the work.** No-retrieval controls sit at 0.5755–0.5975
  test against 0.9217. Neither arm answers from parametric memory.
* **Tier C is unsolved by everything.** Best is 0.2435, and the *no-retrieval
  control* scores 0.2000 — above the GaMS GTLM arm. Nothing here measures unseen
  relations; report it as a negative result, never as a ranking. The failure modes
  differ (GTLM abstains, false-sentinel 0.356; serialised confabulates, 0.087) but
  neither answers.

## Cost, and why the serialised arm is slower

| arm | mean tokens/item | s/step | 9,264 steps |
|---|--:|--:|--:|
| GTLM (graph ball) | 1,556 | 7.53 | 19.4 h |
| serialised (flat text) | 2,499 | 17.51 | ~45 h |
| no retrieval | 56.5 | 0.51 | 1.3 h |

Measured with the GaMS tokenizer over the train split (n=9,266); dev agrees to
1 %. GTLM's sequence is the sum over its ~88 nodes; serialised is one flat prompt.

**Flattening a graph into text costs 1.61× more tokens.** You must spend tokens
writing the structure down — numbering every node so an edge list can refer to it,
then the edge list (`build_variants.py:68-84`). GTLM carries the same structure as
attention bias at zero token cost. The design comment predicted "about 65 %"; the
measurement says 61 %.

The step-time gap is **2.32×**, larger than the length gap. Decomposed:

| | ratio, serialised / GTLM |
|---|--:|
| mean tokens | 1.61× |
| padded tokens (E[max of 4]) | 1.57× |
| **residual, per padded token** | **1.44×** |
| measured step time | 2.32× |

Padding explains none of it (waste is 1.85× for GTLM against 1.82× for serialised;
neither groups by length in training). **The residual is attention cost** — part
length, since attention is quadratic, and part kernel: Gemma-3 realises its
sliding window as a *mask* handed to SDPA (`modeling_gemma3.py:413-431`), so the
plain arm computes dense L×L, saves nothing from the window, and pays to build and
move a `[B,1,L,L]` mask, while GTLM runs compiled block-sparse FlexAttention.
**This study did not separate the two**, and it would take a fixed-length timing
probe to do so. `flash_attention_2` honours `sliding_window` natively and would
skip the masked work, but `flash_attn` is not installed in this venv.

So: **the serialised arm is slow because it is standard, not because it is wrong.**

### ⚠️ `max_length` is not a context length

`config.py:186` is a **per-node** cap, applied per node by `data.py:316`. GTLM
carries ~88 nodes of ~40 tokens, so its 2,048 never binds — the largest single node
in the train split is **226 tokens**. The serialised arm flattens the ball into one
node, so the same field becomes a whole-sequence cap and must clear the longest
ball (hence 17,408).

**The two numbers are not the same quantity. The real length gap is 1.61×, not the
8.5× the caps suggest.** Quote lengths from the table above, never from
`max_length`. Several cost claims in this study were first written against the caps
and were wrong.

### ⚠️ `train_runtime_s` undercounts the serialised cell by 4.5×

That cell ran out of walltime at step 7343 and was resumed from `checkpoint-7200`
in a second job. Its record reads **10.04 h** — the resumed job only. True cost is
36.0 h + 10.2 h ≈ **45 h**, with 143 steps replayed. Any per-cell cost table built
from that field alone is wrong for this one row.

### Estimating rules learned the hard way

* **A walltime extrapolating across both model size *and* context length gets a
  probe, or ×2 headroom.** The 12B serialised estimate (9.4 s/step, cross-checked
  at 10.4) came in at 17.5 — 1.9× low — because it scaled the two factors
  separately and multiplied. Walltimes cannot be raised after submission.
* **Don't price a step from a 30-step probe.** One overstated the 12B by 32 % by
  amortising `torch.compile` warm-up over 30 steps instead of thousands.
* **The 4b is 47 % of the 12B's step time at 33 % of its parameters.** Cost does
  not scale with parameter count.
* **What the graph-bias modules cost is still unmeasured.** GTLM costs 15× its
  own no-retrieval control but feeds 28× the tokens, so per token it is ~1.9×
  *cheaper* than plain SDPA. The control is too small (56 tokens) to normalise
  against in either direction.

## What the design controls for

* **Effective batch is 16 in every cell**, whatever the GPU count — verified.
* **Instruct throughout** (GaMS3-**Instruct** vs `gemma-3-*-**it**`), or the
  Slovene contrast also swaps instruction tuning.
* **The Slovene contrast is architecturally exact** — verified field by field:
  48 layers, hidden 3840, 16 heads, intermediate 15360, vocab 262208, and the same
  resolved attention pattern (`sliding_window` 1024, 40 sliding / 8 full). The
  cells differ in weights and nothing else.
* **One loader for every cell** — extracted towers are flat `gemma3_text`
  directories, so the Slovene contrast is not also a code-path contrast.
* **Checkpoint selection is on dev core tier in every cell**, and the dev
  subsample seed is constant across arms so every run selects on identical items.

**Not controlled: the sliding window moves with the backbone** (512 at 1b, 1024 at
4b/12b). This confounds the serialised arm's 1b → 12B leg *in the direction that
flatters it* — at 1b, five of six layers see at most a 512-token slice of a
2,499-token sequence. GTLM arms have no window at any size (the adapter drops the
band). Another reason 4b → 12b is the clean scale leg.

## Known gaps

* **Every 12B number is n = 1.** Extra seeds were declined on cost.
* **Tier C measures nothing at 12B.** The control answers one antonym in five from
  weights and the GTLM arm scores below it. A future claim about unseen relations
  needs a different instrument.
* **The 1b crossover is not epoch-matched** (8 vs 16), and no serialised cell
  exists at 4b or 12b other than GaMS, so the crossover is located between 1b and
  12B but not bracketed.
* **Length and kernel are not separated** in the 1.44× per-token residual.
* **The graph-bias stack's own cost is unmeasured** — no cell varies it at fixed
  sequence length.
* **The serialised arm's 1b window handicap is unquantified**: known direction,
  unknown size.

---

The full working record — every intermediate result, the discarded runs, the
pre-registered gates and the 14 decisions taken without intervention — is in
`train/SCALING_LOG.md`. This file is the summary; that one is the evidence.
