"""The whole data build, start to finish, in one process.

    sbatch data/run_pipeline.sbatch

Six stages -- the graph store, then the five that make the QA dataset -- run in
order, in one Slurm job, on one node.  `data/README.md` documents both this path
and the stage-by-stage one; the per-stage `run_*.sbatch` scripts stay the way to
rebuild a single stage while iterating on it.

Why one job rather than a dependency chain.  The five QA stages take ~21 minutes
end to end (measured, job 133080) and the extraction stage's shards already
landed on one node in practice, so chaining six jobs with `--dependency=afterok`
would buy back about twenty GPU-minutes in exchange for six queue waits and a
failure mode -- a half-finished chain -- that this design does not have.
Everything runs in the pyxis container, which is a py3.10 base; that is also
what frees the CPU stages from the `aga,ana,apl` pin they carry today, since
those are only the last three nodes where the venv resolves bare.

Stages call the modules directly (`build_dataset.run(...)`, not a subprocess), so
arguments are values rather than strings and a stage can hand back its own
numbers.  Extraction is the exception: `CUDA_VISIBLE_DEVICES` has to be set
before the child's first torch import, so the shards are subprocesses, one per
GPU, over however many GPUs the job was actually given.

Failure policy, which differs by kind on purpose:

  * a **stage** failure stops the run -- there is nothing downstream to build;
  * a **check** failure (selftest, grade, check_balls, check_variants) is
    recorded and the run continues, with a banner at the end naming every check
    that failed.  The build finished; what to do about a failing check is a
    judgement call, and this path exists to hand you the artefacts plus the
    truth about them rather than to decide for you.

Nothing lands in place until the whole run succeeds: stages write under
`datasets/work/`, and the four finished artefacts are moved into
`datasets/` at the end, the previous build stepping aside into
`datasets/previous/`.  A run that dies in stage 4 therefore cannot leave
`generated/` and `balls/` disagreeing about what a T17 item is supervised on,
which is the failure `qa/build_balls.py` exists to prevent and which would
otherwise land on the directory `train/` reads by default.
"""
import os
import sys
import json
import time
import shutil
import subprocess

from lib import paths
from lib.errors import StageError

DATASETS = paths.DATASETS_DIR
WORK = os.path.join(DATASETS, "work")
PREVIOUS = os.path.join(DATASETS, "previous")
DEFAULT_STORE = os.path.join(paths.STORES_DIR, "kg_graph_gemma3")

# The four directories a finished run publishes, staging name -> final name.
# `build_variants` derives `<basename>_<variant>` from the ball directory it is
# given, so staging them beside each other yields these names for free.
ARTEFACTS = ("generated", "balls", "balls_serialised", "balls_noretrieval")

STAGES = ("store", "generate", "extract", "relabel", "balls", "variants")

# The store meta fields that decide whether an existing store is the one asked
# for.  `builder_sha256` is deliberately NOT among them: it moves when a comment
# in the builder changes, and the current store records the builder's pre-rename
# filename, so keying the skip on it would rebuild 4 GB on every run.  A
# difference there is reported instead.
STORE_IDENTITY = ("tokenizer", "kg_dir", "colloc_text", "sense_snippet",
                  "sense_index", "text_convention")

TOKENIZERS = {"gemma3": "cjvt/GaMS3-12B-Instruct", "gams2b": "cjvt/GaMS-2B"}


# --------------------------------------------------------------------------
# the run record
# --------------------------------------------------------------------------
class Run:
    """What happened, accumulated as it happens and written as `run.json`.

    Written in a `finally`, so a run that dies in stage 4 still leaves a record
    saying which stage died and what the checks had said up to then.
    """

    def __init__(self, args):
        self.started = time.time()
        self.args = args
        self.stages = []
        self.checks = []
        self.store = {}
        self.extractor = {}

    def stage(self, name, seconds, status, detail=None):
        self.stages.append({"name": name, "seconds": round(seconds, 1),
                            "status": status,
                            **({"detail": detail} if detail else {})})

    def check(self, name, ok, detail=None):
        self.checks.append({"name": name, "status": "ok" if ok else "FAIL",
                            **({"detail": detail} if detail else {})})
        print(f"[check] {name}: {'ok' if ok else 'FAIL'}"
              f"{'  ' + detail if detail else ''}", flush=True)
        return ok

    @property
    def failed_checks(self):
        return [c["name"] for c in self.checks if c["status"] == "FAIL"]

    def write(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rec = {
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime(self.started)),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seconds": round(time.time() - self.started, 1),
            "git": _git_state(),
            "slurm": {k: v for k, v in os.environ.items()
                      if k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST",
                               "SLURM_CPUS_PER_TASK", "SLURM_JOB_GPUS")},
            "args": self.args,
            "store": self.store,
            "extractor": self.extractor,
            "stages": self.stages,
            "checks": self.checks,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=2)
        print(f"[wrote] {path}", flush=True)


def _git_state():
    def git(*a):
        try:
            return subprocess.run(("git", "-C", paths.REPO_ROOT) + a,
                                  capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None
    return {"sha": git("rev-parse", "HEAD"),
            "short": git("rev-parse", "--short", "HEAD"),
            "dirty": bool(git("status", "--porcelain"))}


def _sha256(path):
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _banner(text):
    print("\n" + "=" * 78, flush=True)
    print(text, flush=True)
    print("=" * 78, flush=True)


# --------------------------------------------------------------------------
# GPUs
# --------------------------------------------------------------------------
def gpu_ids(cap=None):
    """The GPUs this process may use, as CUDA_VISIBLE_DEVICES-style ids.

    Read from the environment first and `nvidia-smi` only as a fallback, so that
    this never imports torch: the parent process does nothing but CPU work and
    initialising a CUDA context in it would be a waste at best, and at worst
    would interfere with the children that do the real GPU work.
    """
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if vis:
        ids = [x for x in vis.split(",") if x != ""]
    elif os.environ.get("SLURM_GPUS_ON_NODE", "").strip().isdigit():
        ids = [str(i) for i in range(int(os.environ["SLURM_GPUS_ON_NODE"]))]
    else:
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                 text=True, check=True).stdout
            ids = [str(i) for i, ln in enumerate(out.splitlines()) if ln.strip()]
        except Exception:
            ids = []
    if cap is not None and cap > 0 and len(ids) > cap:
        print(f"[gpu] {len(ids)} visible, capped to {cap} by --gpus "
              f"({len(ids) - cap} left idle)", flush=True)
        ids = ids[:cap]
    return ids


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def stage_store(rec, store_dir, variant, rebuild):
    """Build the graph store, or accept the one already on disk.

    Skipped when the existing store's manifest says it was built from the same
    source with the same text convention and tokenizer -- see STORE_IDENTITY.
    """
    from build import build_graph

    # Every field taken from the builder itself rather than restated here, so a
    # change to the text convention -- dropping entry gender, say, or switching
    # collocation nodes to lemma pairs -- makes the store on disk stop matching
    # and get rebuilt, which is the entire point of the comparison.
    colloc_text = "phrase"
    want = {"tokenizer": TOKENIZERS[variant],
            "kg_dir": paths.KG_RAW_DIR,
            "colloc_text": colloc_text,
            "sense_snippet": build_graph.SENSE_SNIPPET_CHARS,
            "sense_index": True,
            "text_convention": "+".join(
                ["collocation-phrases" if colloc_text == "phrase"
                 else "collocation-pairs"]
                + (["entry-gender"] if "gender" in build_graph.UNIT_PROPS
                   else []))}

    manifest_path = os.path.join(store_dir, "manifest.json")
    if os.path.exists(manifest_path) and not rebuild:
        with open(manifest_path, encoding="utf-8") as fh:
            meta = (json.load(fh) or {}).get("meta", {})
        differs = [f"{k}: store {meta.get(k)!r} != wanted {want[k]!r}"
                   for k in STORE_IDENTITY if meta.get(k) != want[k]]
        if not differs:
            rec.store = {"path": store_dir, "built": False, **{
                k: meta.get(k) for k in STORE_IDENTITY}}
            rec.store["builder"] = meta.get("builder")
            rec.store["builder_sha256"] = meta.get("builder_sha256")
            live = _sha256(build_graph.__file__)
            if meta.get("builder_sha256") != live:
                print(f"[store] note: this store was written by "
                      f"{meta.get('builder')} @ {str(meta.get('builder_sha256'))[:12]}, "
                      f"the builder here is @ {live[:12]}. Reusing it anyway; "
                      f"pass --rebuild-store to rebuild.", flush=True)
            print(f"[store] reusing {store_dir} "
                  f"({meta.get('tokenizer')}, {meta.get('text_convention')})",
                  flush=True)
            return False
        print(f"[store] rebuilding: manifest differs on {differs}", flush=True)

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    print(f"[store] building {store_dir} with {workers} workers", flush=True)
    manifest = build_graph.build_store(out_dir=store_dir,
                                       tokenizer=TOKENIZERS[variant],
                                       workers=workers)
    meta = manifest.get("meta", {})
    rec.store = {"path": store_dir, "built": True,
                 **{k: meta.get(k) for k in STORE_IDENTITY},
                 "builder": meta.get("builder"),
                 "builder_sha256": meta.get("builder_sha256")}

    from lib import graph_store
    graph_store.verify_graph(store_dir)
    return True


def stage_generate(rec, store_dir, out, seed, types, scale):
    from qa import build_dataset
    build_dataset.run(store=store_dir, out=out, seed=seed, types=types,
                      scale=scale)


def stage_extract(rec, dataset, out_dir, gpus, prompt, store_dir, model=None):
    """The extractor, data-parallel over `gpus`, one subprocess each.

    Subprocesses rather than threads or an import: `CUDA_VISIBLE_DEVICES` is read
    by torch at initialisation, so a child has to be told which GPU it owns
    before it imports anything.  Sharding is a stride inside
    `measure_extraction`, so the shards are disjoint and the merge is a
    concatenation; `qa/relabel.py` keys the result by item id and does not care
    about order.
    """
    from analysis import measure_extraction

    if not gpus:
        raise StageError(
            "extraction needs at least one GPU and none are visible -- "
            "submit with --gres=gpu:N, or run the stage separately with "
            "data/analysis/run_extract_sharded.sbatch")

    # Stale shards from an earlier run with a different GPU count would be
    # merged in silently by a glob.  Start from an empty directory instead.
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    n = len(gpus)
    parts = [os.path.join(out_dir, f"items.part{k}.jsonl") for k in range(n)]
    procs = []
    for k, gid in enumerate(gpus):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gid)
        env.setdefault("HF_HOME", paths.HF_CACHE)
        env["HF_HUB_OFFLINE"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        # --store explicitly, never by discovery: the child resolves the
        # extractor's strings to node ids, and it has to do that against the
        # same graph the rest of the run is using.
        cmd = [sys.executable, "-m", "analysis.measure_extraction",
               "--dataset", dataset, "--prompt", prompt, "--n", "0",
               "--store", store_dir,
               "--num-shards", str(n), "--shard", str(k), "--dump", parts[k]]
        if model:
            cmd += ["--model", model]
        print(f"[extract] shard {k}/{n} on GPU {gid}: {' '.join(cmd)}",
              flush=True)
        procs.append(subprocess.Popen(cmd, cwd=paths.DATA_DIR, env=env))

    bad = [k for k, p in enumerate(procs) if p.wait() != 0]
    if bad:
        raise StageError(
            f"extraction shard(s) {bad} of {n} failed -- refusing to merge a "
            f"partial run. Their output is above; rerun once the cause is fixed.")

    merged = os.path.join(out_dir, "items.jsonl")
    n_rows = 0
    with open(merged, "w", encoding="utf-8") as g:
        for p in parts:
            if not os.path.exists(p):
                raise StageError(f"extraction shard wrote no dump: {p}")
            with open(p, encoding="utf-8") as f:
                for line in f:
                    g.write(line)
                    n_rows += 1
    print(f"[extract] merged {n} shards -> {merged} ({n_rows:,} rows)",
          flush=True)

    summary = measure_extraction.run(score_only=[merged],
                                     out=os.path.join(out_dir, "score.json"))
    # `resolved` is the number that matters downstream: the share of items whose
    # extracted strings reach the item's own anchor, which is what decides
    # whether relabel keeps an item or turns it into an extract_miss negative.
    tot = (summary or {}).get("overall") or {}
    resolved = (100.0 * tot.get("resolved", 0) / tot["n"]) if tot.get("n") else None
    rec.extractor = {"model": model or measure_extraction.DEFAULT_MODEL,
                     "prompt": prompt, "prompt_sha256": _sha256(prompt),
                     "gpus": list(gpus), "shards": n, "rows": n_rows,
                     "resolved_pct": (round(resolved, 2) if resolved is not None
                                      else None),
                     "score_file": os.path.join(out_dir, "score.json")}
    return merged


def stage_relabel(rec, dataset, extraction, out, store_dir):
    from qa import relabel
    relabel.run(dataset=dataset, extraction=extraction, out=out,
                store=store_dir)


def stage_balls(rec, dataset, out, dataset_out, store_dir, types):
    from qa import build_balls
    build_balls.run(dataset=dataset, out=out, store=store_dir,
                    dataset_out=dataset_out, types=types)


def stage_variants(rec, balls, out_root):
    from qa import build_variants
    build_variants.run(balls=balls, out_root=out_root)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
def check_selftest(rec, name, dataset, store_dir, pre_ball=False):
    from qa import selftest
    try:
        rc = selftest.run(dataset=dataset, store=store_dir, pre_ball=pre_ball)
    except Exception as e:                    # a check must not stop the build
        return rec.check(name, False, f"{type(e).__name__}: {e}")
    return rec.check(name, rc == 0, None if rc == 0 else "see the log above")


def check_grade(rec, name, dataset, pre_ball=False):
    """The grader over the gold itself -- C9.  100 % on every split, or say so.

    `pre_ball=True` before stage 5, where the membership positives have no
    allow-list yet and are therefore not gradeable.  Skipping them is the point:
    grading them here would fail every run on a dataset that is exactly as it
    should be at that stage, and a check that always fails is a check nobody
    reads.  They are graded for real by `grade-gold/final`.
    """
    from qa import grade
    ok, detail = True, []
    for split in ("train", "dev", "test"):
        p = os.path.join(dataset, f"{split}.jsonl")
        if not os.path.exists(p):
            continue
        try:
            # verbose=False: the per-type and per-band tables would be six of
            # them per run, and the only thing this check asserts is 100 %.
            s = grade.run(items=p, verbose=False, pre_ball=pre_ball)
        except Exception as e:
            ok = False
            detail.append(f"{split}: {type(e).__name__}: {e}")
            continue
        skipped = s.get("skipped_pre_ball") or 0
        print(f"[grade] {split}: {s['success']:.2f} % over {s['n']:,} items"
              + (f"  ({skipped:,} membership positives await stage 5)"
                 if skipped else ""), flush=True)
        if abs(s["success"] - 100.0) > 1e-9:
            ok = False
            detail.append(f"{split}: {s['success']:.2f} %")
    return rec.check(name, ok, "; ".join(detail) if detail else None)


def check_balls_(rec, name, dataset, balls):
    from qa import check_balls
    try:
        rc = check_balls.run(dataset=dataset, balls=balls)
    except Exception as e:
        return rec.check(name, False, f"{type(e).__name__}: {e}")
    return rec.check(name, rc == 0, None if rc == 0 else "see the log above")


def check_variants_(rec, name, reference, variants):
    from qa import check_variants
    try:
        rc = check_variants.run(reference=reference, variants=variants)
    except Exception as e:
        return rec.check(name, False, f"{type(e).__name__}: {e}")
    return rec.check(name, rc == 0, None if rc == 0 else "see the log above")


# --------------------------------------------------------------------------
# publish
# --------------------------------------------------------------------------
def publish(staging, datasets, previous):
    """Move the finished artefacts into place, the old build stepping aside.

    One slot for the previous build, overwritten each run.  Not a version
    scheme: it is the undo for a run you did not mean to make.
    """
    os.makedirs(previous, exist_ok=True)
    moved = []
    for name in ARTEFACTS:
        src = os.path.join(staging, name)
        if not os.path.isdir(src):
            raise StageError(f"nothing staged at {src} -- refusing to publish "
                             f"a partial build")
    for name in ARTEFACTS:
        dst = os.path.join(datasets, name)
        keep = os.path.join(previous, name)
        if os.path.isdir(dst):
            if os.path.isdir(keep):
                shutil.rmtree(keep)
            shutil.move(dst, keep)
        shutil.move(os.path.join(staging, name), dst)
        moved.append(dst)
        print(f"[publish] {dst}", flush=True)
    return moved


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
def run(store=None, variant="gemma3", rebuild_store=False, seed=20260821,
        types="", scale=1.0, gpus=None, model=None, prompt=None,
        keep_work=False):
    """Build the store (if needed) and the QA dataset, and publish both."""
    store_dir = os.path.abspath(store or DEFAULT_STORE)
    prompt = os.path.abspath(prompt or paths.EXTRACTOR_PROMPT)
    ids = gpu_ids(cap=gpus)
    # A scaled-down build is a test of the machinery, not a corpus, and is kept
    # out of datasets/ for the same reason a --types subset is.
    subset = bool(types) or scale != 1.0

    args = {"store": store_dir, "variant": variant,
            "rebuild_store": rebuild_store, "seed": seed, "types": types,
            "scale": scale, "gpus": gpus, "gpus_used": ids, "model": model,
            "prompt": prompt}
    rec = Run(args)

    # A subset build works in its own corner of work/.  The intermediates at the
    # top of work/ describe the corpus currently in datasets/ -- which extraction
    # dump produced it, what the relabelled set looked like -- and a two-type
    # test run has no business overwriting that record.
    work = os.path.join(WORK, "subset") if subset else WORK
    staging = os.path.join(work, "staging")
    raw = os.path.join(work, "generated_raw")
    relabelled = os.path.join(work, "relabelled")
    extraction = os.path.join(work, "extraction")
    st_balls = os.path.join(staging, "balls")
    st_generated = os.path.join(staging, "generated")

    _banner(f"data build   store={store_dir}\n"
            f"             gpus={ids or '<none>'}  types={types or '<all>'}  "
            f"seed={seed}")

    if os.path.isdir(staging):
        shutil.rmtree(staging)
    os.makedirs(staging, exist_ok=True)

    failed_stage = None
    try:
        t = time.time()
        _banner("stage 1/6  graph store")
        built = stage_store(rec, store_dir, variant, rebuild_store)
        rec.stage("store", time.time() - t, "built" if built else "reused")

        t = time.time()
        _banner("stage 2/6  generate the items")
        stage_generate(rec, store_dir, raw, seed, types, scale)
        rec.stage("generate", time.time() - t, "ok")
        check_selftest(rec, "selftest/generated", raw, store_dir, pre_ball=True)
        check_grade(rec, "grade-gold/generated", raw, pre_ball=True)

        t = time.time()
        _banner(f"stage 3/6  entity linking on {len(ids)} GPU(s)")
        dump = stage_extract(rec, raw, extraction, ids, prompt, store_dir, model)
        rec.stage("extract", time.time() - t, "ok")

        t = time.time()
        _banner("stage 4/6  relabel against the extraction run")
        stage_relabel(rec, raw, dump, relabelled, store_dir)
        rec.stage("relabel", time.time() - t, "ok")
        check_selftest(rec, "selftest/relabelled", relabelled, store_dir,
                       pre_ball=True)

        t = time.time()
        _banner("stage 5/6  balls, and the dataset they re-verbalise")
        stage_balls(rec, relabelled, st_balls, st_generated, store_dir, types)
        rec.stage("balls", time.time() - t, "ok")
        check_balls_(rec, "check_balls", st_generated, st_balls)
        check_grade(rec, "grade-gold/final", st_generated)

        t = time.time()
        _banner("stage 6/6  the two baseline inputs")
        stage_variants(rec, st_balls, staging)
        rec.stage("variants", time.time() - t, "ok")
        check_variants_(rec, "check_variants", st_balls,
                        [os.path.join(staging, "balls_noretrieval"),
                         os.path.join(staging, "balls_serialised")])

        # A subset build is a test of the pipeline, not a corpus: publishing one
        # would put a two-type or a 2 %-sized dataset where `train/` looks for
        # the real thing.  It stays in staging, and says where.
        if subset:
            published = False
            why = ", ".join(
                ([f"--types {types}"] if types else [])
                + ([f"--scale {scale}"] if scale != 1.0 else []))
            _banner(f"NOT publishing: {why} built a subset, not the corpus.\n"
                    f"The artefacts are in {staging}.")
        else:
            _banner("publishing")
            publish(staging, DATASETS, PREVIOUS)
            published = True
        rec.args["published"] = published

    except BaseException as e:
        failed_stage = f"{type(e).__name__}: {e}"
        rec.stage("FAILED", 0, "failed", failed_stage)
        raise
    finally:
        rec.write(os.path.join(work, "run.json"))
        if (not keep_work and not subset and failed_stage is None
                and os.path.isdir(staging)):
            shutil.rmtree(staging, ignore_errors=True)
        _summarise(rec, failed_stage)

    return rec


def _summarise(rec, failed_stage):
    print()
    print(f"{'stage':<12} {'status':<8} {'seconds':>9}")
    for s in rec.stages:
        print(f"{s['name']:<12} {s['status']:<8} {s['seconds']:>9.1f}")
    total = sum(s["seconds"] for s in rec.stages)
    print(f"{'TOTAL':<12} {'':<8} {total:>9.1f}  ({total/60:.1f} min)")

    if failed_stage:
        _banner(f"BUILD FAILED: {failed_stage}\n"
                f"Nothing was published; datasets/ is untouched.")
        return
    bad = rec.failed_checks
    published = rec.args.get("published")
    where = ("The artefacts ARE published" if published else
             "The artefacts were NOT published (this was a subset build)")
    if bad:
        _banner("BUILD FINISHED, BUT {} CHECK(S) FAILED:\n  {}\n\n"
                "{} -- these checks describe the artefacts, they do not gate "
                "them.\nRead the log above for each, and run.json for the "
                "record.".format(len(bad), "\n  ".join(bad), where))
    else:
        _banner("BUILD FINISHED -- every stage ran and every check passed.")
