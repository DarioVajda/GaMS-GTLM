#!/usr/bin/env python3
"""A dev split that can SEE generalisation: Tier A and Tier C items for dev.

    cd data && python -m qa.build_dev_generalisation
    sbatch data/qa/run_dev_generalisation.sbatch

By design (D12) train and dev are 100 % core: unseen phrasings (Tier A) and the
held-out relations (Tier C: T16, T36) exist only in test.  So the in-training dev
curve says nothing about WHEN a run starts to generalise -- only the final test
number does, once.  This builds the two missing slices for dev, keeps the
existing dev rows byte-identical, and publishes the union as a separate corpus
under `datasets/dev_generalisation/`, beside the live one rather than over it.

Checkpoint selection must stay on the core slice (`train/run.py`'s METRIC) --
the new slices are for watching, not choosing -- and `train/data.py` keeps the
selection subsample identical to the one every earlier run selected on.

**Tier A: the selection subsample, re-asked in a held-out frame.**  Every raw
dev item in the in-training subsample (`train.data.stratified_subset`, whose
digest is asserted below) is rendered again with a tier-A frame and nothing else
changed: same seed, same slots, same answer.  The pair differs only in
phrasing, so a gap between the core and A curves is a phrasing effect and not
a difference in which words were asked about.  Items whose frame pool holds no
tier-A frame for their word class and axes are skipped, and counted.

**Tier C: T16 and T36 on DEV lemmas.**  Generated exactly as the test split's
are (`build_dataset.generate`'s per-split body, including negatives inside the
cap and tier-A frames allowed, as for test), over entries on the dev side of
the lemma split -- so no Tier C dev item names a test lemma, and C11 still holds
over the published union.  Two guards in `build_dataset` refuse Tier C outside
test (`_tier_c_safe`, and `allow_tier_a=(split == "test")`); both are lifted for
the duration of this one generation and for these two types only.

**Then stages 3-6 of the real pipeline**, so the new items get what test items
got: the extractor names the word, relabel decides keep / wrong_ball /
extract_miss against that, balls, variants.  Relabel runs over raw train + test
+ the NEW dev items, so its corpus-derived boilerplate list is the one the live
build used; only the new dev rows are kept from its output.

Selftest over the published union is EXPECTED to fail C6 ("every Tier C item is
a test item", and the train/dev tag-word checks) -- that is this corpus's
purpose, not a defect.  Every other check must pass.
"""
import os
import json
import random
import shutil
import hashlib
import argparse
import collections
import contextlib

from qa import seeds, gen, spec, templates
from qa import build_dataset as bd
from qa.store import open_store
from lib import paths
from lib.errors import StageError

WORK = os.path.join(paths.DATASETS_DIR, "work", "dev_generalisation")
OUT = os.path.join(paths.DATASETS_DIR, "dev_generalisation")
LIVE = paths.DATASETS_DIR
LIVE_WORK = os.path.join(paths.DATASETS_DIR, "work")

#: The in-training dev subsample every final40k run selected on
#: (`dev_subsample` in runs.jsonl).  Rebuilt here and asserted, so the Tier A
#: slice is re-asked from exactly those items or the build stops.
SUBSAMPLE_FRAC, SUBSAMPLE_SEED, SUBSAMPLE_SHA = 0.5, 20260823, "0e498eb3723b7dc0"

#: Id offsets.  The live corpus's per-type counters stop below 2,000, so these
#: cannot collide, and the id keeps its `T<n>-<6 digits>` shape.
ID_OFFSET_A, ID_OFFSET_C = 500_000, 900_000

#: Per Tier C type.  Above dev's 65-per-type cap on purpose: this slice exists
#: to draw a curve, and at 65 one item moves a type's score by 1.5 points.
TIER_C_PER_TYPE = 100


def _subsample_ids(rows):
    """`train.data.stratified_subset`, restated so this runs without the GTLM
    stack.  The digest check in `tier_a_items` is what keeps the two in step."""
    by_type = collections.defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r["id"])
    rng = random.Random(SUBSAMPLE_SEED)
    keep = []
    for t in sorted(by_type):
        ids = sorted(by_type[t])
        k = max(1, round(len(ids) * SUBSAMPLE_FRAC))
        keep.extend(rng.sample(ids, min(k, len(ids))))
    keep = sorted(keep)
    return keep, hashlib.sha256("\n".join(keep).encode("utf-8")).hexdigest()[:16]


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Tier A
# --------------------------------------------------------------------------
def tier_a_items(ctx, seed, report):
    ball_rows = _read(os.path.join(LIVE, "balls", "dev.jsonl"))
    keep, sha = _subsample_ids(ball_rows)
    if sha != SUBSAMPLE_SHA:
        raise StageError(f"dev subsample digest {sha} != {SUBSAMPLE_SHA}: the "
                         f"live dev split is not the one the runs selected on")
    keep = set(keep)
    raw = [r for r in _read(os.path.join(LIVE_WORK, "generated_raw", "dev.jsonl"))
           if r["id"] in keep]
    if len(raw) != len(keep):
        raise StageError(f"{len(keep) - len(raw)} subsample ids have no raw item")

    by_code = {int(ctx.store.codes[e.a]): e for e in ctx.pool.entries}
    skipped = collections.Counter()
    out = []
    for r in raw:
        t, slots = r["type"], r["slots"]
        key = spec.template_key(t, slots)
        axes = frozenset(a for a in ("gender", "definiteness") if slots.get(a))
        if r["negative"]:
            pool = templates.neutral_frames(key, axes)
        else:
            # A phrase-seeded entry is not in the word pool; it names no word
            # class, so only class-neutral frames are true of it.
            e = by_code.get(r["node_code"]) if t not in spec.PHRASE_SEEDED else None
            pool = templates.frames_for_pos(key, e.pos if e else None, axes)
        pool = [f for f in pool if templates.tier_of(key, f) == "A"]
        if not pool:
            skipped[t] += 1
            continue
        frame = pool[bd._rng(seed, t, r["lemma"], "tmpl-A").randrange(len(pool))]
        try:
            q = frame.format(**slots)
        except KeyError:
            skipped[t] += 1
            continue
        n = int(r["id"].split("-")[1])
        out.append(dict(r, id=f"{t}-{ID_OFFSET_A + n:06d}", question=q,
                        template_id=templates.frame_id(key, frame), tier="A",
                        rephrased_from=r["id"]))
    report["tier_A"] = {"subsample": len(raw), "made": len(out),
                        "skipped_no_frame": dict(skipped),
                        "per_type": dict(collections.Counter(r["type"] for r in out))}
    print(f"[tier A] {len(out):,} of {len(raw):,} subsample items re-asked; "
          f"skipped {dict(skipped)}", flush=True)
    return out


# --------------------------------------------------------------------------
# Tier C
# --------------------------------------------------------------------------
@contextlib.contextmanager
def _tier_c_on_dev():
    """Lift build_dataset's two test-only guards, for Tier C types only."""
    safe, make = bd._tier_c_safe, bd.make_item

    def make_item(ctx, type_key, *a, **kw):
        if type_key in spec.TIER_C:
            kw["allow_tier_a"] = True
        return make(ctx, type_key, *a, **kw)

    bd._tier_c_safe = lambda it: it["type"] in spec.TIER_C or safe(it)
    bd.make_item = make_item
    try:
        yield
    finally:
        bd._tier_c_safe, bd.make_item = safe, make


def tier_c_items(ctx, seed, per_type, report):
    """`build_dataset.generate`'s per-split body, for split="dev" and Tier C."""
    split = "dev"
    quota = dict(zip(seeds.BAND_NAMES, seeds.BAND_QUOTA))
    items = {split: []}
    counter = collections.Counter({t: ID_OFFSET_C for t in spec.TIER_C})
    rep = {"gen_errors": collections.Counter()}
    with _tier_c_on_dev():
        for t in spec.TIER_C:
            entries = [e for e in bd.seed_entries(ctx, t) if e.split == split]
            bd._rng(seed, t, "avail").shuffle(entries)
            cap_entries = collections.defaultdict(list)
            for e in entries:
                try:
                    ok = gen.GENERATORS[t](ctx, e, bd._rng(seed, t, e.lemma, "gen"))
                except Exception as exc:                # noqa: BLE001
                    rep["gen_errors"][f"{t}:{type(exc).__name__}"] += 1
                    ok = None
                if ok is not None:
                    cap_entries[e.band].append(e)
            cap = {b: len(cap_entries[b]) for b in seeds.BAND_NAMES}
            supply = sum(bd.allocate(per_type, quota, cap).values())
            n_neg = int(round(min(per_type, supply) * bd.NEGATIVE_RATE))
            take = bd.allocate(per_type - n_neg, quota, cap)
            made_pos, spare = 0, []
            for b, k in take.items():
                pool_b = list(cap_entries[b])
                bd._rng(seed, t, split, b, "order").shuffle(pool_b)
                got = used = 0
                for e in pool_b:
                    if got >= k:
                        break
                    used += 1
                    got += bd._emit_positive(ctx, t, e, split, seed, counter,
                                             items, rep)
                made_pos += got
                spare += pool_b[used:]
            made_neg = bd.emit_negatives(ctx, t, split, take, n_neg,
                                         random.Random(seed), counter, items, seed)
            short = per_type - made_pos - sum(
                v for k, v in made_neg.items() if not k.startswith("_"))
            bd._rng(seed, t, split, "topup").shuffle(spare)
            for e in spare:
                if short <= 0:
                    break
                short -= bd._emit_positive(ctx, t, e, split, seed, counter,
                                           items, rep)
            report.setdefault("tier_C", {})[t] = {
                "eligible_dev_entries": sum(cap.values()), "positives": made_pos,
                "negatives": made_neg, "short": max(short, 0)}
            print(f"[tier C] {t}: {sum(cap.values())} eligible dev entries, "
                  f"{made_pos} positives, negatives {made_neg}, short "
                  f"{max(short, 0)}", flush=True)
    out = items[split]
    bad = [r["id"] for r in out if r["tier"] != "C" or r["split"] != split]
    if bad:
        raise StageError(f"Tier C dev items with the wrong tier/split: {bad[:5]}")
    if rep["gen_errors"]:
        report["tier_C_gen_errors"] = dict(rep["gen_errors"])
    return out


# --------------------------------------------------------------------------
# stages 3-6, and publishing
# --------------------------------------------------------------------------
def _link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def run(store=None, seed=20260821, per_type=TIER_C_PER_TYPE, gpus=None,
        prompt=None, skip_generate=False):
    from pipeline import (Run, gpu_ids, stage_extract, stage_relabel, stage_balls,
                          stage_variants, check_grade, check_balls_,
                          check_variants_, check_selftest, DEFAULT_STORE)
    store_dir = os.path.abspath(store or DEFAULT_STORE)
    prompt = os.path.abspath(prompt or paths.EXTRACTOR_PROMPT)
    rec = Run({"store": store_dir, "seed": seed, "per_type": per_type,
               "prompt": prompt})
    raw = os.path.join(WORK, "generated_raw")
    report_path = os.path.join(raw, "report.json")
    try:
        # ---- the new items ------------------------------------------------
        if not skip_generate:
            st = open_store(store_dir)
            pool = seeds.Pool(st, [seeds.Entry(**d) for d in
                                   _read(os.path.join(LIVE_WORK, "generated_raw",
                                                      "pool.jsonl"))])
            ctx = gen.Ctx(st, pool)
            report = {}
            new = tier_a_items(ctx, seed, report) + tier_c_items(ctx, seed,
                                                                 per_type, report)
            ids = [r["id"] for r in new]
            if len(ids) != len(set(ids)):
                raise StageError("duplicate ids among the new dev items")
            live_ids = {r["id"] for s in ("train", "dev", "test") for r in
                        _read(os.path.join(LIVE_WORK, "generated_raw",
                                           f"{s}.jsonl"))}
            if live_ids & set(ids):
                raise StageError("a new dev id collides with a live item id")
            new.sort(key=lambda r: (r["type"], r["id"]))
            _write(os.path.join(raw, "dev.jsonl"), new)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=dict)
        new_ids = {r["id"] for r in _read(os.path.join(raw, "dev.jsonl"))}
        check_grade(rec, "grade-gold/generated", raw, pre_ball=True)

        # ---- stage 3: the extractor, on the new items only -----------------
        extraction = os.path.join(WORK, "extraction")
        dump = stage_extract(rec, raw, extraction, gpu_ids(cap=gpus), prompt,
                             store_dir)

        # ---- stage 4: relabel, with the live corpus around the new items ---
        rl_in = os.path.join(WORK, "relabel_input")
        os.makedirs(rl_in, exist_ok=True)
        for s in ("train", "test"):
            _link(os.path.join(LIVE_WORK, "generated_raw", f"{s}.jsonl"),
                  os.path.join(rl_in, f"{s}.jsonl"))
        _link(os.path.join(raw, "dev.jsonl"), os.path.join(rl_in, "dev.jsonl"))
        ext_all = os.path.join(WORK, "extraction_all.jsonl")
        with open(ext_all, "w", encoding="utf-8") as g:
            for p in (os.path.join(LIVE_WORK, "extraction", "items.jsonl"), dump):
                with open(p, encoding="utf-8") as f:
                    shutil.copyfileobj(f, g)
        rl_out = os.path.join(WORK, "relabel_output")
        stage_relabel(rec, rl_in, ext_all, rl_out, store_dir)
        relabelled = os.path.join(WORK, "relabelled")
        rows = [r for r in _read(os.path.join(rl_out, "dev.jsonl"))
                if r["id"] in new_ids]
        if len(rows) != len(new_ids):
            raise StageError(f"relabel returned {len(rows)} of {len(new_ids)} "
                             f"new dev items")
        _write(os.path.join(relabelled, "dev.jsonl"), rows)

        # ---- stages 5-6 ------------------------------------------------------
        staging = os.path.join(WORK, "staging")
        if os.path.isdir(staging):
            shutil.rmtree(staging)
        st_balls = os.path.join(staging, "balls")
        st_gen = os.path.join(staging, "generated")
        stage_balls(rec, relabelled, st_balls, st_gen, store_dir, "")
        check_balls_(rec, "check_balls/new", st_gen, st_balls)
        check_grade(rec, "grade-gold/new", st_gen)
        stage_variants(rec, st_balls, staging)
        check_variants_(rec, "check_variants/new", st_balls,
                        [os.path.join(staging, "balls_noretrieval"),
                         os.path.join(staging, "balls_serialised")])

        # ---- publish: live train/test by link, dev = live dev + new ----------
        if os.path.isdir(OUT):
            shutil.rmtree(OUT)
        for name in ("generated", "balls", "balls_serialised", "balls_noretrieval"):
            d = os.path.join(OUT, name)
            os.makedirs(d)
            for s in ("train", "test"):
                _link(os.path.join(LIVE, name, f"{s}.jsonl"),
                      os.path.join(d, f"{s}.jsonl"))
            with open(os.path.join(d, "dev.jsonl"), "w", encoding="utf-8") as g:
                for p in (os.path.join(LIVE, name, "dev.jsonl"),
                          os.path.join(staging, name, "dev.jsonl")):
                    with open(p, encoding="utf-8") as f:
                        shutil.copyfileobj(f, g)
            print(f"[publish] {d}", flush=True)
        shutil.copy(report_path, os.path.join(OUT, "report.json"))

        # The union, whole.  C6 is expected to FAIL here (see the docstring).
        check_selftest(rec, "selftest/union (C6 expected to fail)",
                       os.path.join(OUT, "generated"), store_dir,
                       balls=os.path.join(OUT, "balls"))
        check_grade(rec, "grade-gold/union", os.path.join(OUT, "generated"))
        check_balls_(rec, "check_balls/union", os.path.join(OUT, "generated"),
                     os.path.join(OUT, "balls"))
    finally:
        rec.write(os.path.join(WORK, "run.json"))
    print(f"[done] failed checks: {rec.failed_checks or 'none'}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=None)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--per-type", type=int, default=TIER_C_PER_TYPE)
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--skip-generate", action="store_true",
                    help="reuse work/dev_generalisation/generated_raw")
    args = ap.parse_args()
    try:
        run(store=args.store, seed=args.seed, per_type=args.per_type,
            gpus=args.gpus, prompt=args.prompt, skip_generate=args.skip_generate)
    except StageError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
