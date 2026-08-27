"""Where the store build's memory actually goes.

    cd data && python -m analysis.profile_store_build parse
    cd data && python -m analysis.profile_store_build full /tmp/a_store

Runs the real build with an RSS sampler in a background thread, so the curve can
be lined up against the phase markers the builder already prints ([parse],
[tok], [save]), and reports what the parse accumulators hold.

**Read the process figure and the job figure as different things.** Slurm's
`MaxRSS` comes from the cgroup, which charges page cache, and this build streams
41 GB of N-Triples -- so the job number sits well above the process number and
always will.  This harness reports the process number; `sacct` reports the other.

Written to answer whether the builder needed a columnar rewrite.  It did not:
the parent holds 15 GB of accumulators and peaks at 29 GB, and the 105 GB the
job used to report was the parse pool prefetching without bound.  See `_gated`
in build_graph.py, and "What the store build actually costs" in data/README.md.
Kept because the next memory question deserves a measurement rather than an
estimate -- both of mine were wrong.
"""
import os
import gc
import sys
import time
import glob
import threading


def rss_gb():
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30


class Sampler(threading.Thread):
    def __init__(self, every=2.0):
        super().__init__(daemon=True)
        self.every, self.stop_flag, self.curve = every, False, []
        self.t0 = time.time()

    def run(self):
        while not self.stop_flag:
            self.curve.append((time.time() - self.t0, rss_gb()))
            time.sleep(self.every)

    def report(self, label):
        peak = max(g for _t, g in self.curve) if self.curve else 0.0
        print(f"\n[rss] {label}: peak {peak:.1f} GB over "
              f"{self.curve[-1][0]:.0f}s", flush=True)
        # One line per 30s bucket, so the curve is readable in a log.
        bucket, last = {}, None
        for t, g in self.curve:
            k = int(t // 30)
            bucket[k] = max(bucket.get(k, 0.0), g)
        for k in sorted(bucket):
            bar = "#" * int(bucket[k] / 2)
            if bucket[k] != last:
                print(f"  {k*30:5d}s {bucket[k]:6.1f} GB {bar}", flush=True)
                last = bucket[k]


def measure_parse(kg_dir, workers):
    """Parse only, then report what the accumulators cost and hold."""
    from build import build_graph

    files = sorted(glob.glob(os.path.join(kg_dir, "*.nt")))
    print(f"[parse-only] {len(files)} files x {workers} workers", flush=True)
    gc.collect()
    before = rss_gb()
    t0 = time.time()
    p = build_graph.parse_all(files, workers, t0)
    gc.collect()
    after = rss_gb()
    print(f"\n[parse-only] RSS {before:.1f} -> {after:.1f} GB "
          f"(+{after - before:.1f} GB) in {time.time() - t0:.0f}s", flush=True)

    rows = []
    for name in ("wr", "wrf", "dfn", "val", "pos"):
        d = getattr(p, name)
        rows.append((name, len(d), "dict[int]->str"))
    for name in ("feat", "unit"):
        d = getattr(p, name)
        inner = sum(len(v) for v in d.values())
        rows.append((name, len(d), f"dict[int]->dict, {inner:,} inner pairs"))
    edges = sum(v.shape[0] for v in p.edges.values())
    rows.append(("edges", edges, "numpy (m,2) int64 -- already columnar"))

    print(f"\n{'accumulator':<10} {'entries':>12}   shape")
    for name, n, what in rows:
        print(f"{name:<10} {n:>12,}   {what}")

    # The distinct vocabularies a columnar form would intern against.
    props = set()
    vals = set()
    for d in (p.feat, p.unit):
        for inner in d.values():
            for k, v in inner.items():
                props.add(k)
                vals.add(v)
    print(f"\nfeat/unit vocabulary: {len(props)} distinct properties, "
          f"{len(vals)} distinct values")
    print(f"  -> a (code, prop_id, value_id) row is 8+1+1 = 10 bytes; "
          f"the inner dicts above are ~{len(p.feat) * 184 / 2**30:.1f} GB "
          f"of dict headers alone")
    return after - before


def main():
    from lib.paths import KG_RAW_DIR
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    mode = sys.argv[1] if len(sys.argv) > 1 else "parse"

    s = Sampler()
    s.start()
    try:
        if mode == "parse":
            measure_parse(KG_RAW_DIR, workers)
        else:
            from build import build_graph
            out = sys.argv[2]
            build_graph.build_store(out_dir=out, workers=workers)
    finally:
        s.stop_flag = True
        s.join(timeout=5)
        s.report(mode)


if __name__ == "__main__":
    main()
