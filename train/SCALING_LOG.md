# Working log — the scaling study, as it happened

> **This is the evidence file, not the findings.** `train/SCALING.md` carries the
> summary someone new should read first. This one keeps the intermediate results,
> the discarded runs, the pre-registered gates and the decisions taken without
> intervention. Sections below may describe runs as "running" that have since
> finished; the final numbers live in `SCALING.md` and in each sweep's
> `runs.jsonl`. Status as of 2026-09-01: **the study is complete.**

# Scaling the backbone — what it is meant to answer

`arms_v3` generalises worse than its headline suggests. GTLM (spd + magnetic), 3 seeds:

| tier | what it is | n | accuracy |
|---|---|--:|--:|
| core | seen type, seen phrasing | 1599 | 0.867 |
| **A** | **seen type, unseen phrasing** | 470 | **0.711** |
| C | unseen relation (antonyms) | 115 | 0.061 |

False `ni podatka v bazi` over positives goes **0.2 % → 17.3 % → 41.3 %** across
those tiers and accounts for 54 % of Tier A's failures: the model matches a
question to one of ~10 frames per type and refuses when none fires.

The question is whether that gap is **capacity** or **supervision**. Tier A is
plausibly scale-sensitive — the failing frames are trivial paraphrases and a 1b
backbone has little Slovene prior. Tier C probably is not: nothing in training ever
varied a relation tag, and scale cannot supply supervision that was never there.

## Status

| step | cell | state |
|---|---|---|
| 0 | `gemma-3-1b-it`, six arms | **complete** |
| 1 | `GaMS3-12B-Instruct` GTLM + control | **complete** — gate open on a lower bound |
| 1b | `GaMS3-12B-Instruct` serialised plain stack | **complete** — timed out at 12.7 epochs (134406), resumed to 16 (135172), see decisions 13–14 |
| 2 | `gemma-3-12b-it` control | **complete** (134407_0) — GTLM arm **complete** (134407_1) |
| 3 | `gemma-3-4b-it` GTLM | **complete** (134408_0) |
| 4 | seeds 1 and 2 | **declined on cost** (decision 11) |

**Budget: 16 epochs / 9,264 steps in every cell**, doubled from 8 after the first
12B run failed the convergence rule. Holding optimizer steps fixed across sizes
*sounded* conservative and was not: a schedule that converges the 1b but not the
12B is a handicap landing on one cell, not one treatment applied twice. Overfitting
is not a new risk — `load_best_model_at_end` selects on dev.

What each cell isolates: step 2 is **Slovene adaptation at fixed size** (same
architecture, weights-only contrast); step 3 is **scale within one family**; step 1b
is the **baseline GTLM must beat** (decision 12). Note 1b is text-native where 4b
and 12b are vision-extracted, so **4b → 12b is the clean scale leg** and 1b → 12b
confounds scale with pretraining regime.

## The pre-registered gates

Set before the numbers existed. Tier A is 470 items, Tier C is 115 (one item ≈ 0.009).

* **Gate 1 → 2/3 opens** if Tier A at 12B ≥ **0.80**; below **0.75** the gap is
  supervision, stop and spend the time on frames; between, report the ambiguity.
  → **OPEN**, on a lower bound of 0.9362.
* **Tier C under 0.15 at 12B counts as unchanged.** ⚠️ **Dead as written** — the
  12B control reaches 0.20 alone, so a GTLM arm at 0.15 would be *below* its own
  control. Read against **0.20**. Left visible rather than restated, because it is
  what was pre-registered.
* **The no-retrieval arm must stay at chance on Tier C.** ⚠️ **It did not** — 0.20
  against ~0.05. **Tier C at 12B does not measure retrieval**, and no Tier C number
  from this cell may be reported as a retrieval result. The condition was written
  to make this detectable before reading the gate, and it did its job.
* **Gate → 4** requires gate 1 open *and* one contrast exceeding its threshold.
* **Convergence rule.** If the best checkpoint is the last one in more than a third
  of runs, the study measures learning speed, not endpoints. `arms_v2` was
  discarded under it.

## Results

### Step 0 — 1b, six arms (COMPLETE)

The corpus change did **not** move the 1b result; everything lands within
`arms_v3`'s seed spread.

| | `arms_v3` (3 seeds) | `scale_1b` (1 seed) |
|---|--:|--:|
| GTLM (spd + magnetic) | 0.7908 ± 0.0016 | 0.7953 |
| — tier core | 0.867 | 0.867 |
| — **tier A** | 0.711 ± 0.019 | **0.732** |
| — tier C | 0.061 ± 0.009 | 0.052 |
| Graph encoding (headline) | +0.0252 ± 0.0131 | +0.0334 |
| Retrieval | +0.3294 | +0.3356 |
| Structural bias | +0.2166 | +0.2280 |

The re-run was *justified* (161 items had different gold) and *uneventful*. T21,
holding 123 of those, moved 0.623 → 0.600 — inside single-seed noise at n = 115.

**Convergence: the rule FAILS here — 3 of 6 runs (50 %) vs a 33 % limit.** Reported
rather than explained away, but bounded: 3/6 has a 12.8 % chance under `arms_v3`'s
own 4/18 rate, and its Wilson interval is 0.19–0.81, which straddles the limit.
`arms_v3`'s 4/18 gives 0.09–0.45, straddling it too. The rule has never been
resolved at these sample sizes; `report_arms` now prints the interval every time.

The full six-arm table matters for decision 12 and is reproduced there.

### Step 1 — the 12B GTLM arm blew up twice (both runs DISCARDED)

Neither run under-trained. Both **diverged**.

| run | collapse | dev loss | final Tier A |
|---|---|--:|--:|
| 8 epochs | epoch 1.38 | 3.96 | 0.645 |
| 16 epochs | epoch 2.07 | 6.79 | — (cancelled at 13 %) |

```
epoch 1.46  loss 0.7533  gnorm     12.9   <- climbing
epoch 1.67  loss 0.5652  gnorm    276.5   <- first excursion
epoch 1.74  loss 9.1662  gnorm     11.2   <- model destroyed
epoch 2.05  loss 7.6322  gnorm 132294.5   <- second excursion
```

It happened **with gradient clipping already active** (`max_grad_norm` at HF's
default 1.0), so not one bad batch but sustained oversized updates. This reframes
the 8-epoch run: epochs 1.4–3.5 were **recovery, not learning**, which is why its
Tier A came in at 0.645. That number is not a capacity result and the gate was
never read from it.

**Which parameter — from a control already in hand.** At 12B, same `lr = 5e-4`, the
no-retrieval arm (no graph-bias modules) was **stable** while the GTLM arm
(`bias_lr = 1e-2`) collapsed twice; the same `bias_lr` is stable at 1b. So the
graph-bias pathway is responsible, at width and depth.

**Why it was mis-set.** `graph_bias` is not a bias vector. Each of the 48 layers
carries an MLP stack — `deep_set`, `proj`, `lambda_lin` — of freshly initialised
**weight matrices** over width-dependent dimensions, running at 20× the LoRA rate.
The 1e-2 was set at 1b and carried across unchanged.

### Step 1 — no-retrieval control (COMPLETE) — the antonyms trap FIRED

At 8 epochs it did not fire. Under the final 16-epoch / width-schedule
configuration it fires hard, and the earlier reading is **withdrawn**.

| no-retrieval, plain stack | 1b, 16 ep | 12B, 8 ep, `lr` 5e-4 | 12B, 16 ep, `lr` 2e-4 |
|---|--:|--:|--:|
| overall | 0.4895 | 0.4762 | **0.5975** |
| tier core | 0.543 | 0.537 | 0.633 |
| tier A | 0.415 | 0.375 | 0.575 |
| **tier C** (n=115) | 0.052 | 0.0435 | **0.2000** |
| best ckpt / final? | 7600, False | — | 7600, False |

**Tier C went from 5 items to 23 of 115 with no graph attached.** Every other cell
in this study sits between 0.026 and 0.070 — chance. A Slovene-native 12B answers
roughly one antonym question in five from its weights.

Two earlier claims fall with it. Tier C is no longer a retrieval measurement at
12B. And a 12B with no graph is **not** "no better than a 1b with no graph": it is
+0.108 overall, +0.160 Tier A, +0.148 Tier C. Slovene prior knowledge carries part
of this task, and the retrieval effect must be measured net of it.

**Cause unresolved** — budget and `lr` moved together. Budget is unlikely (doubling
it at 1b moved that control's Tier C 0.070 → 0.052, i.e. nothing), leaving the
lower LR, an interaction with scale, or both. See decision 10.

### Step 1 — THE GATE RESULT (COMPLETE, disqualified as an endpoint)

`scale_12b_gams_blr_gtlm`, job 134093, 19 h 32 m on 4 × B200. Clean throughout.

| cell | overall | core | **Tier A** | Tier C | best ckpt | final? |
|---|--:|--:|--:|--:|--:|:--:|
| 1b no-retrieval | 0.4895 | 0.5428 | 0.4149 | 0.0522 | 7600/9264 | False |
| 1b GTLM | 0.8091 | 0.8837 | 0.7383 | 0.0609 | 9200/9264 | False |
| 12B no-retrieval | 0.5975 | 0.6329 | 0.5745 | 0.2000 | 7600/9264 | False |
| **12B GTLM** | **0.9217** | **0.9719** | **0.9362** | 0.1652 | **9264/9264** | **True** |

Tier A **0.9362** against a 0.80 threshold: +0.198 over the 1b, +0.362 over its own
control. **But `best_is_final` is True.**

**The width schedule fixed the divergence** — the run cleared both previous
collapse points and the stability criteria pass by two orders of magnitude:

| | the two collapsed runs | this run |
|---|--:|--:|
| dev loss @ epoch 2.07 | 6.79 (dev acc 0.000) | **0.0448** (dev acc 0.846) |
| max grad norm, post-warmup | 132,294 | **13.61** |
| max train loss, post-warmup | 9.17 | **0.148** |

**Why the convergence rule fired.** Dev from epoch 12.44 on: 0.9791, 0.9753,
0.9772, 0.9791, 0.9753, 0.9791, 0.9810 — seven evals inside a **3-item band** on
527 dev items, mean 0.9780, with the final eval on top of the band. That is a noisy
plateau, not a curve still climbing.

**That reading is not acted on.** It was formed *after* seeing a welcome result —
the exact circumstance pre-registration defends against — and "the rule fired on
noise" is what one would say either way. The pre-commitment was made **before the
number was visible**: the cell-level rule in the config governs over the
study-level fraction, because `lr` was cut 2.5× while the budget was held, making
under-training the predictable failure mode here.

What holds without judgement: the run was flat-or-improving at the end, so
**0.9362 is a lower bound**, and nothing pushes it below 0.80. So the gate
*decision* is robust, while the *number* is quoted as **≥ 0.9362, from a run that
had not converged**. The 32-epoch confirmation was declined on cost (decision 11),
so that caveat is **permanent**.

**Tier C is a negative result.** 0.1652 against its own control's 0.2000 — 4 items
*worse* than answering from weights alone, and refusing far more often (false
sentinel 0.356 vs 0.202). This study has found **no retrieval effect on Tier C**.

### Step 1b — the 12B serialised baseline (RUNNING, will be cut short)

Job 134406_0, started 2026-08-30 07:43 on 4 × B200, 17.5 s/step. Killed by the
wall at ~step 7,400 (12.8 epochs) on 2026-08-31 ~19:43; see decision 13 for what
is done with it. Warmup ends at step 926 = **epoch 1.60**, so most of what is
readable so far is pre-warmup.

| epoch | eval loss | dev core acc | |
|--:|--:|--:|---|
| 0.69 | 0.03234 | 0.8748 | pre-warmup |
| 1.38 | 0.03796 | 0.8254 | pre-warmup |
| 2.07 | 0.05104 | **0.8918** | first post-warmup eval |

**A stability criterion fired: one post-warmup grad norm of 163.0 at epoch 1.79,
against a limit of 100.** Recorded rather than explained away, and then bounded —
it is a single excursion, not the pattern the criterion is written to catch. Of
the 57 post-warmup points through epoch 2.08 the mean is 5.56, two exceed 10, one
exceeds 100, and the run recovered on the next step with loss unbroken
(post-warmup max 0.132 against a limit of 2.0). The collapsed GTLM runs looked nothing like this: sustained
climbing, 276 then 132,294, loss to 9.17. Clipping is active at `max_grad_norm`
1.0 and the logged norm is pre-clip, so the update itself was clipped to 1.0.

**The other criterion — eval loss rising across two consecutive evals — is close,
and the "after warmup" qualifier is what keeps it from firing.** Loss has risen
twice in a row (0.03234 → 0.03796 → 0.05104), but warmup ends at epoch 1.60, so
**only the third eval is post-warmup and no post-warmup *pair* exists yet.** The
epoch-2.76 eval is the first that can complete one. The qualifier is doing
exactly the work it was written for — it is the same clause that keeps a
criterion applied from step 0 from condemning the healthy 1b anchor.

**Against that, accuracy is at its highest — 0.8918 — while loss climbs.** Rising
loss with rising accuracy is calibration drift, not divergence: the model is
getting more confident on what it already gets right and more wrong where it
errs. It is also the harmless direction here, because `METRIC` is
`eval_accuracy` with `greater_is_better=True` (`run.py:92`), so checkpoint
selection never reads the loss. **Both readings are recorded before the next eval
is visible**, so whichever way epoch 2.76 lands, it cannot be chosen after the
fact.

### Step 2 — gemma-3-12b-it no-retrieval control (COMPLETE)

Job 134407_0, 1.29 h on 4 × B200. Best checkpoint 5600/9264, `best_is_final`
False, zero OOM splits — converged well inside the budget.

**The antonyms trap is a property of 12B scale, not of Slovene pretraining.**

| no-retrieval control | overall | core | Tier A | **Tier C** |
|---|--:|--:|--:|--:|
| gemma-3-1b-it | 0.4895 | 0.5428 | 0.4149 | **0.0522** |
| GaMS3-12B (Slovene-native) | 0.5975 | 0.6329 | 0.5745 | **0.2000** |
| gemma-3-12b-it | 0.5755 | 0.6098 | 0.5617 | **0.1565** |

Both 12B controls sit at 0.16–0.20 where every 1b cell sits at 0.05. The 0.0435
gap between them is 5 items of 115, inside single-seed noise. Decision 10 left the
cause of the Tier C jump open between budget and `lr`; this narrows it differently
and more usefully — **whatever drives it, it is not the Slovene weights**, and a
second independent 12B reproduces it.

**This makes the Tier C negative result stronger, not weaker.** The 12B GTLM arm
scored 0.1652, which now sits *between* two independent weights-only baselines
(0.1565 and 0.2000). Retrieval adds nothing on unseen relations against either.

**Slovene pretraining buys almost nothing without retrieval.** GaMS3-12B over
gemma-3-12b-it is +0.022 overall and **+0.013 on Tier A — inside the ±0.019 seed
spread**, so not a difference by the study's own yardstick (decision 11). Whether
it matters *with* the graph present is what step 2's GTLM arm answers; this control
establishes that the contrast starts from near-parity.

**The GTLM arm (134407_1) is running and healthy.** 7.8 s/step on 4 × B200,
~21 h against a 25 h wall — ~5 h of margin. Post-warmup max train loss 0.191 and
max grad norm 11.5, against limits of 2.0 and 100, so **the width schedule
replicates on a second, independent 12B**: it has cleared both epochs where the
GaMS arm collapsed (1.38 and 2.07) without an excursion. Dev core by epoch —
0.5047, 0.7932, 0.8330, 0.8748, 0.8843, 0.8824 at 0.69 / 1.38 / 2.07 / 2.76 /
3.45 / 4.15 — climbing normally. At the matched epoch 2.07 the GaMS arm was at
0.846 against this arm's 0.833, near-parity *with* the graph, the same way their
controls were near-parity without it. Read as a progress note, not a result: dev
core is not Tier A, and the endpoint is what the study reports.

### Step 3 — gemma-3-4b-it GTLM (COMPLETE)

Job 134408_0, 9.09 h `train_runtime` on 4 × B200 (9:13 h of job, against a 10 h
wall — 46 min of margin). Best checkpoint 8800/9264, `best_is_final` False, zero
OOM splits. One arm; the no-retrieval control was pre-registered as skippable here.

| GTLM, 16 epochs | params | overall | core | Tier A | Tier C |
|---|--:|--:|--:|--:|--:|
| `gemma-3-1b-it` | 1.0 B | 0.8091 | 0.8837 | 0.7383 | 0.0609 |
| `gemma-3-4b-it` | 3.9 B | 0.8658 | 0.9318 | **0.8277** | 0.1043 |
| `GaMS3-12B` | 11.8 B | 0.9217 | 0.9719 | **0.9362** | 0.1652 |

**The Tier A curve is not saturating.** This cell exists to give the Tier A
recovery a *shape*: a saturating curve would mean the residual gap is data, a
threshold would mean it is scale. The middle point lands **0.013 below** the
1b–12B chord — convex, i.e. Tier A is still accelerating in log-parameters at
12B. The 12B point is a lower bound (decision 11), so the convexity is if
anything understated. Over this range the residual Tier A gap is **capacity, not
supervision**.

Read as description, not as a law: three single-seed points with two free
parameters leave one degree of freedom, so "log-linear" here means only *where
the middle point sits relative to the chord* (decision 11 forbids more). Under
that limit, core sits on the chord to ±0.0002 and is nearly exhausted at 12B
(0.972) — **whatever headroom scale still buys is in Tier A, not in core.**

Per type, the movement is concentrated: T17 +0.513 and T15 +0.478 from 1b to 12B
carry most of it, while T6 is flat at 0.861 across all three sizes and T10/T11
drift slightly *down*. Scale is not lifting the task uniformly; it is fixing
specific types.

**Tier C tracks size, not retrieval — the 4b interpolates.** 0.1043 sits between
the 1b cells (~0.05) and the two 12B controls (0.157, 0.200), matching the
no-retrieval controls at each size rather than anything retrieval does. Its Tier C
false-sentinel rate is the highest in the study, 0.510: on unseen relations this
cell abstains on half the answerable items. No 4b control was run, so this cell
cannot decompose the two — it is consistent with, not proof of, the Step 2 finding.

Caveat on the leg: 1b is text-native, 4b and 12b are vision-extracted text towers,
and the 12B is Slovene-pretrained. **The clean in-family leg is 4b → 12b**, which
needs 134407_1.

### The 16-epoch series — both 1b arms (COMPLETE)

8-epoch column is the 4-seed pool (`scale_1b` + `arms_v3`); 16 epochs is one seed.

| | no-retr 8 ep | no-retr 16 ep | GTLM 8 ep | GTLM 16 ep |
|---|--:|--:|--:|--:|
| overall | 0.4868 ± 0.0069 | 0.4895 | 0.7919 ± 0.0026 | **0.8091** |
| core | 0.538 ± 0.006 | 0.543 | 0.867 ± 0.006 | 0.884 |
| A | 0.419 ± 0.018 | 0.415 | 0.716 ± 0.019 | 0.738 |
| best ckpt / final? | — | 7600, False | — | 9200, False |

**The doubled budget is worth ~+0.017 at 1b on GTLM and nothing on the control.**
Both cleared the convergence rule, so 16 epochs is *sufficient* at 1b. Two
consequences: the budget change is **not a confound that manufactures a 12B
result**, and the honest same-budget baseline the 12B must beat is **Tier A 0.738,
not 0.716**.

## What the arms actually feed the model

Measured on the current corpus, all three splits pooled (n = 12,490), job 134480,
`train/results/arms_v3/length_stats.json`. **Packed** tokens: the sum over a
graph's nodes of their `input_ids`, which is the sequence the model attends over.

| arm | nodes/item p50 | tokens/item mean | p50 | p90 | p99 | max |
|---|--:|--:|--:|--:|--:|--:|
| GTLM (graph ball) | 78 | 1,605 | 1,369 | 2,704 | 5,768 | 14,131 |
| serialised | 1 | 2,555 | 2,172 | 4,474 | 9,540 | 16,390 |
| no retrieval | 1 | 56.5 | 44 | 96 | 197 | 841 |

**`max_length` is a PER-NODE cap, not a context length** (`config.py:186`,
applied by `data.py:316` to each node's text separately). GTLM carries ~89 nodes
of ~18 tokens each, so its 2,048 is a truncation guard that never binds; the
serialised arm flattens the ball into **one** node, so the same field becomes a
whole-sequence cap and must clear the longest ball in the corpus — hence 17,408.
The two numbers are not the same quantity, and **the real length gap between the
arms is 1.59×, not the 8.5× the caps suggest.** Two claims below were written
against the caps and are corrected there.

Sequences are the missing term in almost every cost statement in this study;
quote lengths from this table, never from `max_length`.

## The LR width schedule

Declared before it was run, anchored at 1b, applied to **both optimizer groups**.
Round values, one significant digit, each at or **below** the `sqrt(1152 / width)`
curve:

| cell | width | `lr` rule | `lr` | `bias_lr` rule | `bias_lr` | margin |
|---|--:|--:|--:|--:|--:|--:|
| 1b | 1152 | 5e-4 | **5e-4** | 1.00e-2 | **1.00e-2** | anchor |
| 4b | 2560 | 3.35e-4 | **3e-4** | 6.71e-3 | **6e-3** | −11 % |
| 12b | 3840 | 2.74e-4 | **2e-4** | 5.48e-3 | **4e-3** | −27 % |

Lower learning rates at larger width is the standard result and what μP formalises;
having adopted a width schedule at all, there is no principled reason to scale one
group and pin the other. The curve is an upper bound, not a fit. It applies to the
4b too, which has never diverged — a rule applied only where it is needed is a
per-cell fit.

**What it costs.** `lr` is no longer shared across cells, so 1b → 12B moves scale
*and* both learning rates together. Unavoidable once a width schedule exists — the
trade μP makes — but the honest claim becomes *"a 12B trained under the declared
width schedule reaches X"*. **The schedule is part of the treatment.** `lr` is
still shared and untuned within `arms_v3` and `scale_1b`, where every cell is one
width.

**Selection is on training stability only**, never on dev or test accuracy, so it
cannot bias the headline. A cell is unstable if, **after warmup ends** (step
`max_steps // 10`), eval loss rises across two consecutive evals, or training loss
exceeds 2.0, or grad norm exceeds 100. The "after warmup" qualifier is load-bearing:
the healthy 1b anchor peaks at 5.9 train loss and gnorm 19.8 early, so a criterion
applied from step 0 would condemn the reference run.

## What the design controls for

* **Instruct throughout** — GaMS3-12B-**Instruct** against `gemma-3-*-**it**`, or
  the Slovene contrast also swaps instruction tuning.
* **The Slovene contrast is architecturally exact.** Checked, not assumed:
  `GaMS3-12B-Instruct` and the extracted `gemma-3-12b-it` tower agree on every
  structural field — 48 layers, hidden 3840, 16 heads, intermediate 15360, vocab
  262208. The cells differ in *weights* and nothing else. **Also on the attention
  pattern**, which the structural check did not cover: both resolve to
  `sliding_window` 1024 with 40 sliding / 8 full layers and
  `use_bidirectional_attention` False. GaMS3 writes it in the newer config format
  (`_sliding_window_pattern` + an explicit `layer_types` list) where the towers
  write `sliding_window_pattern`; transformers 4.50.3 resolves both identically,
  verified layer by layer.
* **One loader for every cell** — extracted towers are flat `gemma3_text`
  directories, so the Slovene contrast carries no code-path difference.
* **Effective batch is 16 in every cell**, whatever the GPU count — verified.

**NOT controlled: the sliding window moves with the backbone.** Gemma-3 ships a
5-local : 1-global interleave whose window is **512 at 1b** and **1024 at 4b and
12b** — a pretraining choice baked into the weights, unchangeable without editing
the model. Three consequences, in descending severity:

* **The serialised arm's 1b → 12B leg is confounded.** That arm is one flat
  sequence averaging 2,555 tokens, so at 1b five of every six layers see at most
  a 512-token slice — about a fifth of a typical ball — and at 12B twice that.
  Decision 12's baseline therefore gains window width along with scale, in the
  direction that *flatters* it. Weigh it when 134406 lands.
* **The retrieval contrast is unaffected.** The no-retrieval control maxes at 841
  tokens over all 12,490 items, under the 12B's 1024 window entirely and over the
  1b's 512 for well under 1 % of items. GTLM-vs-control is not a
  full-attention-vs-windowed contrast in any material way.
* **GTLM arms have no window at any size.** The adapter drops the band — it is
  defined over packed serialization order and would hide most of the graph from
  the sliding layers — so GTLM is uniformly full-attention while the plain arms
  are windowed. Another reason **4b → 12b is the clean leg**: the window is
  constant at 1024 across it, where 1b → 12b moves it.

---

# Decisions taken without intervention

Numbering is stable; config headers cross-reference it.

**1. DDP shards *batches*, not items — verified bitwise.** The evaluator groups by
token budget, a pure function of the split's lengths, so every rank derives the
same batch list and runs a strided slice. Sharding *items* was rejected: re-grouping
changes padding and a bf16 near-tie can flip an argmax, so a disagreement could not
be told from a bug. Verified (job 133521, `train/checks/check_ddp.py`): 96/96 items
agreed on `success`, `pass1`, `reason`, decoded `prediction`; pass-1 loss identical
to six decimals. **Training is statistically invariant, not bitwise** — float
addition is not associative across a different micro-batch split — but step count,
effective batch, LR schedule and data order are preserved exactly.

**2. Step 0 runs all six arms, not the two the gate needs.** The extra arms cost
~7 h *concurrently* and buy the whole contrast table on the current corpus. Without
them the headline contrast stays stranded on a corpus that no longer exists.

**3. The corpus change is real, and not what it looked like.** Ids and splits are
identical to `archive/2026-08-23_arms_v3` (9,266 / 1,040 / 2,184). Of 8,728
differing training balls, **8,712 differ by edge order alone** — invisible to a
permutation-equivariant model. Two parts are not cosmetic: the **serialised** input
is the flattened ball, so order is content there, and **128 train + 33 test items
have different gold**, 123 of them T21.

**4. Multimodal checkpoints converted offline.** `train/extract_text_tower.py`
writes the towers as flat `gemma3_text` directories rather than patching an unwrap
branch into a `graph_model` shared with other experiments. The stronger reason is
scientific: one code path, so the Slovene contrast is not also a loader contrast.
4b = 3.88 B of 4.30 B (34 layers), 12b = 11.77 B of 12.19 B (48).
**A bug this surfaced:** `backbone()` resolved checkpoints by substring-matching
the *name* against `("llama", "gemma-3")` — `cjvt/GaMS3-12B-Instruct` contains
neither, so the study's target model was unloadable. It now reads `model_type`.

**5. Deferred: the dev Tier A slice and the de-confounded T16 slice.** Both are
dataset changes needing a rebuild that would invalidate step 0. Neither confounds
the scaling contrast — checkpoint selection is core-only in *every* cell and Tier
C's confound is identical in every cell. **What it costs:** Tier C is part
relation-effect and part Tier A effect, inseparable here, so no claim may be made
about the *size* of the antonym failure, only whether scale moves it.

**6. Resource allocation.** Four GPUs per 12B cell, not eight: with every node
allocated, an 8-GPU request costs more in queue than the ~7 h it saves.
`gpus_per_config: 8` is a one-line change if the partition drains.

* **Arms of one contrast may run on different GPU models.** The 16-epoch 1b control
  ran on A100-40GB while its GTLM partner ran on B200. **Hardware may vary between
  arms; effective batch, schedule and seed may not.**
* **Declare walltime honestly** — an over-declared limit cannot be backfilled and
  *cannot be raised later* (`scontrol` denies non-admin increases). Cutting 24 h to
  8 h on a measured 5.5 h job started it ~1 h 45 m sooner.
* **A timeout produces no result at all**, not a shortened one: `trainer.train()`
  runs without `resume_from_checkpoint` and there is no resume path, so
  `load_best_model_at_end` and the test eval never run. Measure the steady-state
  rate early; partition `MaxTime` is 7 days if a longer limit is needed.
* **Pin the gres type, not the brand.** `"gpus": ["A100"]` renders as
  `--constraint GPU_BRD:A100`, matching `axa`'s 40 GB cards as well as `ana`'s
  80 GB — enough to land a 12B on a card that cannot hold it. `"A100_80GB:4"`
  renders as `--gres gpu:A100_80GB:4` and cannot be mis-placed.
* **Reason from the post-DDP micro-batch, never `batch_size`.** The real constraint
  is the **logits** at vocab 262,208, and `ddp_factorisation` shrinks the
  micro-batch as ranks are added, so adding GPUs buys headroom as well as speed:

| arm | ranks | micro | logits (+fp32) | weights | total | 40 GB | 80 GB |
|---|--:|--:|--:|--:|--:|:--:|:--:|
| 1b no-retrieval | 2 | 8 | 8.6 + 17.2 GB | 2 GB | ~30 GB | yes | yes |
| 12B no-retrieval | 4 | 4 | 4.3 + 8.6 GB | 23.5 GB | ~42 GB | no | yes |

  The first estimate used 16, put the 12B at ~70 GB, and would have wrongly ruled
  out every 80 GB card in the partition.

**7. The 16-epoch budget is *two* changes.** `train/run.py:418` sets
`warmup_steps = max_steps // 10`, so doubling the budget **doubles warmup in
absolute steps**, 463 → 926. Not intended, and load-bearing: it moved the 12B's
collapse from epoch 1.38 to 2.07 without preventing it. Both cells got both changes
identically, so the contrast holds — but the claim is "longer schedule *and* gentler
ramp". **Matched-epoch comparisons across the two budgets are confounded; only
endpoints are readable** (cosine position differs). Matched-epoch comparison
*within* one budget is fine.

**8. `report_arms` takes several `runs.jsonl` paths.** Joining sweeps at read time
beats merging them on disk, which would be a second copy of the evidence to keep
honest. Verified against `scale_1b` + `arms_v3` — single-path output unchanged, the
join pools 1 and 3 seeds into 4 with the expected mean and spread. The 12B pair:

```
report_arms.py train/results/scale_12b_gams_blr/runs.jsonl \
               train/results/scale_12b_gams_blr_gtlm/runs.jsonl
```

**9. An array element was cancelled on a wrong arm mapping, and the gate was lost.**
A mistake, not a judgement call; it cost ~1 h 20 m and a resubmission. `resolved/
0000...` is the GTLM arm and `resolved/0001...` the control, so array task `_0` was
assumed to be GTLM. It is not — the sweep bakes its own ordering into the `--wrap`:

```
LABELS=(..._0001_noretrieval  ..._0000_gtlm);  i="$SLURM_ARRAY_TASK_ID"
```

Task 0 is the **control**. Cancelling `_1` cancelled the gate.
**The rule: map array elements from `<results_dir>/sbatch_commands.sh`, never from
`resolved/` filenames or the arm order in the config.** Where it matters, submit one
arm per config. What caught it was reading the surviving job's throughput and dev
accuracy against expectation — 0.52 s/step and dev ≈ 0.59 is the control's profile.
The guards applied beforehand were all aimed at the *replacement* and none at the
*identity of what was cancelled*. Net effect on the science: none, and better — the
surviving element ran the control on B200, so both arms of the retrieval contrast
sit on one GPU generation.

**10. The Tier C cause is left unresolved — but narrowed.** The jump
(0.0435 → 0.20) is confounded between budget and `lr`; the 2 × 2 completes with a
12B control at 8 epochs, `lr` 2e-4 (~1.6 h), **not scheduled** on resource
priority. Step 2's control has since done better than that cell would have:
gemma-3-12b-it reproduces the elevated Tier C at 0.1565, so **it is not the Slovene
weights**, and it replicates on a second independent 12B. Whether the remaining
cause is scale, the lower `lr`, or their interaction is still open, and no claim
here depends on it — the *level* is measured against two baselines now.

**11. The 32-epoch confirmation and the extra seeds were declined, on cost.** Both
were **the user's calls**, and both set permanent limits on what may be claimed.

* **No 32-epoch re-run** of the 12B GTLM cell. Standing consequence: **Tier A at
  12B is reported as ≥ 0.9362 with non-convergence attached, permanently** — not as
  a converged endpoint, and not quietly upgraded later.
* **No seeds 1 and 2.** Every 12B number stands at **n = 1**. The only error bars
  the study owns are the 1b four-seed spreads — GTLM ± 0.0026 overall / ± 0.019
  Tier A, serialised ± 0.0040 / ± 0.016. **Nothing inside ± 0.02 on Tier A is a
  difference**, and no scaling law may be drawn through three single-seed points.

**12. The missing baseline: GTLM has never been compared to the cheap option.** At
1b, over 4 seeds, the simplest retrieval baseline **beats GTLM**:

| arm | overall | Tier A |
|---|--:|--:|
| **serialised — plain stack** | **0.8178 ± 0.0040** | **0.751 ± 0.016** |
| GTLM (spd + magnetic) | 0.7919 ± 0.0026 | 0.716 ± 0.019 |
| serialised — GTLM stack | 0.7647 ± 0.0102 | 0.657 ± 0.045 |
| GTLM, no bias | 0.5725 ± 0.0079 | 0.495 ± 0.025 |
| no retrieval — plain stack | 0.4868 ± 0.0069 | 0.419 ± 0.018 |

Step 0's headline "+0.0252 ± 0.0131" is GTLM against the *serialised GTLM stack*,
holding the stack fixed. Legitimate, but **not** the contrast that decides whether
the graph-bias machinery earns its complexity — and on that one GTLM loses by 0.026
overall and 0.035 Tier A, both outside the seed spread.

Surfaced only after the gate returned 0.9217, at which point that number had no
baseline. `scale_12b_gams_serialised.jsonc` (job 134406) supplies it. **Until it
lands, the honest reading of the 12B result is "a 12B with retrieval does well on
this task", not "GTLM works at 12B".**

**13. The 12B serialised run cannot finish inside its walltime, and it was left
running anyway.** Measured at 17.5 s/step against the 9.4 estimated (see *Cost*),
so 16 epochs needs ~48 h against a 36 h wall that cannot be raised (decision 6).
It will be killed at roughly **step 7,400 — 12.8 epochs** — and a killed run does
its final eval never: `trainer.train()` returns nothing, no best model is loaded,
no record is written.

Three options were priced. **Cancelling and resubmitting at 60 h** buys the clean
16-epoch number but re-enters a queue on a cluster where *every* B200/B300 GPU is
allocated and the last wait was 11.5 h; the slot is worth more than the four
epochs. **Cutting to 8 epochs** fits comfortably but is not comparable to a
16-epoch GTLM arm. **Letting it run and scoring the surviving checkpoint** costs no
extra GPU-hours and is what was chosen.

⚠️ **A fourth option was missed, and it is cheaper than the one rejected on
cost.** The checkpoints are fully resumable — `optimizer.pt`, `scheduler.pt`, all
four `rng_state_*.pth`, `trainer_state.json` — so a second job can **resume from
the last save and run only the remaining ~1,900 steps: ~9 h, not the ~48 h a
restart costs.** The option priced above as "cancelling and resubmitting" was a
restart from scratch; resuming is roughly a fifth of that for the same clean
16-epoch number. It still re-enters the queue, and it still needs a
`resume_from_checkpoint` path, which does not exist. **Open, for the user's
call** — the pre-registered asymmetry below is what makes scoring-first the
sensible default: read the 12.8-epoch number first, and spend the resume only if
the baseline *loses* by less than the ~+0.017 the 1b gained from 8 → 16 epochs.

What makes 12 epochs acceptable *here specifically*: the 12B GTLM arm gained ~0.006
on dev between epoch 12.44 and 16, and the 1b gained +0.017 overall from 8 → 16
epochs. The effect this baseline has to resolve is 0.026 overall / 0.035 Tier A —
larger than the budget shortfall. And the asymmetry is favourable: a 12-epoch
baseline **beating** a 16-epoch GTLM arm is a strong result, while it losing proves
little. That direction is stated before the number exists.

Enabling change: `--mode eval`, which loads a written checkpoint and runs the
ordinary graded dev/test pass. ⚠️ **Still unwritten as of 2026-08-30** —
`__main__.py:29` declares `choices=("train",)` and `run.py` exposes only
`run_train_mode`. **Until it is written this cell yields nothing at all**, and it
is the single blocker on the whole plan. No deadline, though: the checkpoint
survives the kill on disk, and `save_total_limit=1` with `load_best_model_at_end`
keeps the dev-selected one. Every 12B number from a timed-out run carries
**"scored from checkpoint N, ~E epochs"** attached, permanently.

**Interim, 5 dev evals in (epoch 3.6) — the truncation risk has largely
evaporated, because this arm peaks EARLY.** Dev accuracy on the shared 527-item
subsample (`sha256 7eaf28c0262db09f`, identical across every cell):

| epoch | serialised — plain, 12B | GTLM, GaMS-12B |
|--:|--:|--:|
| 0.69 | **0.8748** | 0.5294 |
| 1.38 | 0.8254 | 0.7970 |
| 2.07 | **0.8918** ← best so far | 0.8463 |
| 2.76 | 0.8159 | 0.8729 |
| 3.45 | 0.7173 | 0.8994 |
| 12–16 | — | 0.975 → 0.981 |

**The 12B serialised arm overfits; it does not diverge.** Train loss is 0.02–0.10
and stable, with two isolated grad spikes (163 at epoch 1.79, 108 at 3.46) that
recover in one step — nothing like the 9.17 / 132,294 collapse of the discarded
GTLM runs. What is rising is *dev* loss (0.032 → 0.116) while dev accuracy falls.
A 12B reading the whole serialised graph in a 17k-token prompt fits the training
set inside one epoch and then memorises.

**This inverts the 1b behaviour rather than reproducing it.** At 1b the same arm
rose monotonically to epoch 8 and led GTLM at *every* eval (0.9013 vs 0.8748 dev),
which is how it won. At 12B it is ahead only until epoch ~2.4 and then loses
ground while GTLM keeps climbing. Whatever decides this contrast at 12B, it is not
the same mechanism that decided it at 1b.

Consequence for the truncation: `checkpoint-1200` (epoch 2.07) is already on disk
next to the running one, so the dev-selected checkpoint is safe no matter where
the kill lands. The run continues only to cover the LR-decay phase, where the 1b
analogue did most of its climbing — cutting it now would score a 4-epoch baseline
against a 16-epoch GTLM arm, which is the unfair comparison this cell exists to
avoid.

⚠️ **RESOLVED, and the interim reading above was wrong.** "The 12B serialised arm
overfits" was written from a five-eval window and did not survive the sixth. The
epoch 2.76–3.45 collapse to 0.7173 (dev loss 0.116) was a **transient**: the arm
recovered to 0.8918 by epoch 4.15, climbed to a 0.951–0.953 plateau from epoch
10.36, and ended the resumed run at 0.9639 on the subsample with dev loss 0.018 —
its lowest. Nothing overfitted. The lesson is about the instrument, not the arm:
this cell's eval-to-eval noise is ±0.015 (roughly twice GTLM's), so five points is
not enough to call a trend, and a transient trough read as a trajectory produced a
confident diagnosis that was the opposite of the truth. **Do not diagnose this
arm's shape on fewer than ~8 evals.**

**14. The run was resumed rather than scored from its checkpoint** — the fourth
option decision 13 flagged as open. Implemented rather than escalated, because the
cost had changed: the walltime kill happened while the queue was empty enough for a
dependency-chained job to start within hours, making the ~10 h resume clearly
cheaper than the ~48 h restart it had been priced against.

* **`--resume-from` was written, not `--mode eval`.** Decision 13 named `--mode
  eval` the single blocker; resuming makes it unnecessary, since the cell then
  produces its own graded record the ordinary way. ~39 lines across
  `train/run.py` (`_resolve_resume`) and `train/__main__.py`.
* **`auto` resolves the LATEST checkpoint, never the best.** With
  `save_total_limit=1` and `load_best_model_at_end` the directory holds both;
  resuming from the best would silently replay the steps between it and the latest
  under a stale optimizer state — a different training run, not a continuation.
* **Every hyperparameter was held identical** (the job script is byte-identical
  plus the one flag), including `--sweep-id`/`--run-name`, which name the
  checkpoint directory. The cosine schedule, warmup and step count are all derived
  from them.
* **`--dependency=afterany`, not `afterok`** — the parent is killed by its wall, so
  it never exits ok. Submitted ~7 h before the parent died so the child accrued age
  priority meanwhile; it started 10 h after the kill.
* **World size held at 4** so per-rank sharding and the four saved RNG states line
  up with what the checkpoint recorded.
* **Cost of the replay:** `auto` resumed from `checkpoint-7200` while the parent
  had reached step 7343, so **143 steps were trained twice** (~0.7 h). Unavoidable
  at `save_steps: 400` and harmless — the optimizer state matches the checkpoint,
  not the kill point.
* **Result:** COMPLETED 2026-09-01T16:08:55, 10 h 14 m, the full 16-epoch cell.
  This is what makes the headline contrast epoch-matched at 12B.
* **The flag was removed once the cell landed**, at the user's request — it was
  one-off machinery, not part of the study. Reconstructing it is ~39 lines: resolve
  `auto` to the highest-numbered `checkpoint-*` under `output_dir` (never the
  dev-best one), pass it to `trainer.train(resume_from_checkpoint=...)`, and record
  it in the run record so a cell spanning two jobs says so. **The provenance it
  wrote is preserved here rather than in the record**: this cell resumed from
  `checkpoint-7200`.

**Fairshare finding, for future queueing.** `sprio` weights FAIRSHARE at 100000
against AGE 5000 and JOBSIZE 1000, so account choice dominates everything else a
submitter controls. `dario.vajda` holds 0.202279 on `lkm` against 0.011396 on
`povejmo` — an 18× difference. Submit 12B jobs with `-A lkm`.

## Cost — measured

| cell | GPUs | s/step | 9,264 steps | + eval | total |
|---|--:|--:|--:|--:|--:|
| 1b GTLM | 1 × B200 | 3.74 | 9.6 h | 0.1 h | **9.7 h** |
| 1b GTLM | 2 × B200 | 2.59 | 6.7 h | 0.1 h | **6.8 h** |
| 1b no-retrieval | 2 × A100-40G | 0.40 | 1.0 h | 0.4 h | **1.4 h** |
| 1b serialised (2,555 tok) | 1 × B200 | 5.67 | 14.6 h | — | — |
| 12B no-retrieval | 4 × B200 | 0.51 | 1.3 h | 0.3 h | **1.6 h** |
| 12B GTLM | 4 × B200 | 7.65 | 19.7 h | 1.0 h | **20.7 h** |
| 4b GTLM | 4 × B200 | 3.45 | 8.9 h | 0.2 h | **9.1 h** |
| 12B gemma GTLM | 4 × B200 | 7.53 | 19.4 h | 0.4 h | **19.8 h** |
| 12B serialised (2,499 tok) | 4 × B200 | **17.51** | 45.1 h | 0.5 h | **~45.6 h** |

Both 12B figures are `train_runtime_s / 9,264` from `runs.jsonl`. ⚠️ **The
serialised cell's own `train_runtime_s` reads 36,140 s (10.04 h) — the resumed job
only.** Its s/step is derived from that job's 2,064 steps; the cell's true cost is
36.0 h (killed parent) + 10.2 h (child) ≈ 45 h. Any cost table built from that
field alone is wrong for this row (decision 14).

Cell labels in parentheses are **measured mean packed tokens**, not `max_length`.

All measured figures are completed-run `train_runtime`, not probes — a 30-step
probe overstated the 12B by 32 % by amortising torch.compile warm-up over 30 steps
instead of thousands.

**The GTLM arm costs 15× its own control at the same size** — ⚠️ **and the
attribution first written here was wrong.** It read that 15× as the price of the
graph-bias stack. Measured, the GTLM arm feeds **28.4×** the control's tokens
(1,605 against 56.5) for **15.0×** the time, so per token the GTLM stack is
~1.9× *cheaper* than plain SDPA, not dearer. **The cost is sequence length, not
machinery.** The control is also too small to normalise against — at 56 tokens a
step is dominated by fixed overheads, the logits over vocab 262,208 among them —
so it cannot price the bias stack in either direction. **This study has not
measured what the graph-bias modules cost.** Per GPU the 12B costs 8.19× the 1b
against a 12× parameter ratio, so nothing pathological in the flex kernel at 48
layers.

**Cost does not scale with parameter count, and step-cost estimates do not compose
across two axes.** Two measured failures of the obvious arithmetic:

* The 4b is **47 %** of the 12B's step time at **33 %** of its parameters. A
  walltime cut from 16 h to 10 h on the parameter ratio left only 46 min of margin.
* The 12B serialised estimate — 9.4 s/step, cross-checked at 10.4 — came in at
  **17.5 s/step, 1.9× low** (decision 13). The estimate scaled the model-size and
  context-length factors *separately* and multiplied them. ⚠️ **The explanation
  first written here labelled those factors "at 2,048 tokens" and "at 17,408" —
  the `max_length` caps, not lengths.** The measured means are 1,605 and 2,555,
  a 1.59× gap, so the multipliers are right but the quadratic story under them
  was not. What the gap actually decomposes into:

| | ratio, serialised over GTLM |
|---|--:|
| mean tokens per item | 1.61× |
| padded tokens, E[max of 4] | 1.57× |
| **residual, per padded token** | **1.44×** |
| measured step time | 2.32× |

Re-measured 2026-09-01 with the **GaMS tokenizer** over train and dev separately
(the 1.59× above was pooled over all three splits with the gemma tokenizer):
GTLM 1,556 mean tokens against serialised 2,499 on train, 1,555 / 2,501 on dev —
**1.61× on both**. Step time 17.51 / 7.53 = **2.32×**. The largest single node in
train is 226 tokens, against GTLM's 2,048 per-node cap: it never binds.

  Padding explains none of it — waste is 1.85× for GTLM against 1.82× for
  serialised, since neither arm groups by length in training
  (`train_token_budget: 0` on both). **The residual is the attention kernel.**
  Gemma-3's sliding window is realised as a *mask*
  (`modeling_gemma3.py:413-431` builds it with `torch.tril` and `torch.where`,
  then hands it to SDPA), so the plain arm computes dense L×L attention, saves
  no compute from the window, and pays to build and move a `[B,1,L,L]` mask —
  at padded L ≈ 4,600 and micro-batch 4, hundreds of MB. The GTLM arm runs
  FlexAttention, compiled and block-sparse. **The serialised arm is slow because
  it is standard, not because it is wrong**, and its step cost is a kernel
  choice as much as a length one. `flash_attention_2` honours `sliding_window`
  natively and would skip the masked work, but `flash_attn` is not installed in
  this venv.

Rule going forward: **a walltime that extrapolates across both model size and
context length gets a probe, or ×2 headroom.** Walltimes cannot be raised later
(decision 6), so the asymmetry is not close.

**Left on the table:** `gradient_checkpointing` is on everywhere. Turning it off at
12B would plausibly buy ~30 %, but that is unmeasured and an OOM at hour ten costs
more than it saves. Mathematically a no-op on the result, so it stays available.

## Known gaps

* **Tier C measures nothing at 12B.** The control answers one antonym in five from
  weights, and the GTLM arm scores *below* it. Any future claim about unseen
  relations needs a different instrument.
* **The serialised — GTLM stack arm is unfunded at 4b and 12b.** Extending the
  step-0 contrast table to those cells needs it; steps 1b–3 do not fund it.
* **Every 12B number is n = 1**, and the headline Tier A is a lower bound from a
  run that had not converged.
* **What the graph-bias modules cost is unmeasured.** The 15× against the
  no-retrieval control prices sequence length, not machinery (see *Cost*), and no
  cell in this study varies the bias stack at fixed length. The `no bias` arm at
  1b is the only lever, and it was never timed.
* **The serialised arm's window handicap at 1b is unquantified.** It is a
  confound in decision 12's 1b → 12B leg with a known direction and an unknown
  size; separating it needs a 1b serialised cell run with the window disabled,
  which nothing funds.
