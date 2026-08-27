"""Command line for the whole data build.

    cd data && python -m pipeline [options]

Normally submitted rather than run by hand:

    sbatch data/run_pipeline.sbatch [options]

Every option has a default that does the right thing unattended, so the bare
command is the one to reach for; the flags are for the cases the defaults do not
cover -- a smaller GPU allocation, a subset of question types while testing, a
store rebuild after the builder changed.
"""
import argparse

from lib.errors import StageError
from pipeline import run, DEFAULT_STORE, TOKENIZERS


def main():
    ap = argparse.ArgumentParser(
        prog="python -m pipeline", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=None,
                    help=f"the graph store to read, and to build if it is "
                         f"absent (default: {DEFAULT_STORE})")
    ap.add_argument("--variant", default="gemma3", choices=sorted(TOKENIZERS),
                    help="which tokenizer fills the store's token_len when the "
                         "store has to be built (default: gemma3)")
    ap.add_argument("--rebuild-store", action="store_true",
                    help="rebuild the store even if the one on disk matches. "
                         "~15 min and ~48 GB.")
    ap.add_argument("--seed", type=int, default=20260821,
                    help="the generator's seed (default: 20260821)")
    ap.add_argument("--types", default="",
                    help="comma-separated question types, e.g. T1,T3 -- narrows "
                         "the type MIX, not the size: each split's budget is "
                         "divided among the types asked for, so a two-type build "
                         "is no smaller (default: every type)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="shrink every split proportionally, e.g. 0.02 for a "
                         "~250-item end-to-end run. THIS is the fast test; "
                         "combine with --types to also narrow the mix. Any "
                         "value but 1.0 makes the build a subset, which is "
                         "never published (default: 1.0)")
    ap.add_argument("--gpus", type=int, default=None,
                    help="use at most N of the visible GPUs (default: all of "
                         "them; the allocation is what --gres asked for)")
    ap.add_argument("--model", default=None,
                    help="override the extractor model")
    ap.add_argument("--prompt", default=None,
                    help="override the extractor prompt; the corpus is built "
                         "from whatever anchors this prompt resolves to, so "
                         "changing it changes the dataset")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep datasets/work/staging after a successful run")
    args = ap.parse_args()

    try:
        run(store=args.store, variant=args.variant,
            rebuild_store=args.rebuild_store, seed=args.seed, types=args.types,
            scale=args.scale, gpus=args.gpus, model=args.model,
            prompt=args.prompt, keep_work=args.keep_work)
    except StageError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
