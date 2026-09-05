#!/usr/bin/env python3
"""The command line, and the composition root: which implementation each stage
gets.

    ask "Kaj pomeni beseda brahialen?"        one question, then exit
    ask -i                                    a warm shell, many questions
    ask --demo -i                             the same, with simulated models

A session is one process: load once, answer many.  There is no server, nothing
to leave running, and no state on disk beyond the question log and the compile
cache.
"""
import os
import sys
import json
import argparse

import ask
from ask import ui as ui_mod
from ask import backbone, extract, retrieve, answer as answer_mod

BANNER = "ask · GTLM"


class RealStages:
    """The real chain: the store, the two models, and the three stages.

    Thin on purpose.  Everything it does lives in the stage modules; what it
    owns is the session state they are called with, so that `Pipeline` can hold
    the order of the chain without also holding a model.
    """

    def __init__(self, impl=None):
        self.impl = impl        # None lets backbone pick flex or eager (D17)
        self.store = self.extractor = self.gtlm = None
        backbone.quiet_libraries()

    def load_store(self, path):
        self.store = backbone.load_store(path)
        return self.store

    def load_extractor(self, name):
        self.extractor = backbone.load_extractor(name)
        return self.extractor

    def load_gtlm(self, checkpoint):
        self.gtlm = backbone.load_gtlm(checkpoint, graph_attn_impl=self.impl)
        return self.gtlm

    def extract(self, question, ui):
        return extract.extract(question, self.extractor, ui)

    def retrieve(self, strings, ui):
        return retrieve.retrieve(strings, self.store, ui)

    def answer(self, question, ball, ui):
        return answer_mod.answer(question, ball, self.gtlm, ui)

    def precompile(self, ui):
        from ask import precompile as pc
        return pc.warm(self.gtlm, ui)


def parser():
    p = argparse.ArgumentParser(
        prog="ask", description="Ask a GTLM checkpoint a question about "
                                "Slovene, grounded in the lexical graph.")
    p.add_argument("question", nargs="?", help="the question; omit with -i")
    p.add_argument("-i", "--interactive", action="store_true",
                   help="stay warm and keep asking")
    p.add_argument("--checkpoint", default=backbone.DEFAULT_CHECKPOINT,
                   help="an alias (see --aliases) or a path to a checkpoint "
                        f"directory (default: {backbone.DEFAULT_CHECKPOINT})")
    p.add_argument("--alias", nargs=2, metavar=("NAME", "CHECKPOINT"),
                   help="record NAME as a shortcut for CHECKPOINT, then exit")
    p.add_argument("--unalias", metavar="NAME",
                   help="forget a recorded shortcut, then exit")
    p.add_argument("--aliases", action="store_true",
                   help="list the recorded shortcuts, then exit")
    p.add_argument("--store", default="", help="path to the processed graph store")
    p.add_argument("--extractor", default=backbone.DEFAULT_EXTRACTOR,
                   help="model that names the words a question is about")
    p.add_argument("--words", default="",
                   help="comma-separated headwords to retrieve on, skipping "
                        "the extractor (and its load) entirely")
    p.add_argument("--json", action="store_true",
                   help="print one JSON object per question on stdout")
    p.add_argument("--show-ball", action="store_true",
                   help="print every node of the retrieved ball")
    p.add_argument("--retrieve-only", action="store_true",
                   help="stop at the ball; do not load or run the GTLM")
    p.add_argument("--precompile", action="store_true",
                   help="compile every (L, N) shape before asking, and cache it")
    p.add_argument("--impl", default="", choices=["", "flex", "eager"],
                   help="force the attention backend; default picks flex only "
                        "when this GPU model has a warm compile cache")
    p.add_argument("--quiet", action="store_true",
                   help="answers only; no progress on stderr")
    p.add_argument("--no-log", action="store_true",
                   help="do not append to ask/logs/")
    p.add_argument("--debug", action="store_true",
                   help="print a full traceback when a question fails")
    p.add_argument("--demo", action="store_true",
                   help="simulate every stage; no GPU, no weights")
    p.add_argument("--fast", action="store_true",
                   help="with --demo: run the simulated delays 8x faster")
    p.add_argument("--list-demos", action="store_true",
                   help="what --demo can be asked")
    return p


def words_of(args):
    return [w.strip() for w in args.words.split(",") if w.strip()]


def manage_aliases(args):
    """`--alias`, `--unalias`, `--aliases`: edit the table, then exit (D23).

    On stdout as plain text, like `--list-demos` and unlike everything the UI
    prints: this is administration rather than a session, its output is worth
    grepping, and `--quiet` swallowing a listing that was explicitly asked for
    would be simply wrong.
    """
    from ask import aliases as al

    try:
        if args.alias:
            name, checkpoint = args.alias
            stored, base = al.record(name, checkpoint)
            print(f"zapisano   {name} → {stored}")
            print(f"           {base}")
            return 0
        if args.unalias:
            gone = al.forget(args.unalias)
            if gone is None:
                known = ", ".join(al.known()) or "nobene"
                print(f"ni take bližnjice: {args.unalias} (zapisane: {known})")
                return 1
            print(f"pozabljeno {args.unalias} → {gone}")
            return 0
    except Exception as e:              # noqa: BLE001 -- a CLI, not a session
        if args.debug:
            raise
        print(f"{type(e).__name__}: {e}")
        return 1

    rows = al.listing()
    if not rows:
        print("nobene bližnjice; dodaj jo z: ask --alias IME POT")
        return 0
    w = max(len(name) for name, _p, _b in rows)
    for name, path, base in rows:
        print(f"{name:<{w}}  {path}")
        # The base model, because that is what one alias is picked over another
        # for; and its absence, because a gitignored `checkpoints/` means a
        # recorded name can outlive the weights it was recorded for.
        print(f"{'':<{w}}  {base or '(te kontrolne točke ni)'}")
    return 0


def ask_one(pipe, question, args, words=None):
    """One question, and never let it end the session (D19)."""
    u = pipe.ui
    try:
        res = pipe.run(question, words=words)
    except Exception as e:                       # noqa: BLE001 -- D19 is the point
        if args.debug:
            import traceback
            traceback.print_exc()
        b = pipe.last.ball if pipe.last else None
        where = (f" (okolica: {len(b.texts)} {ask.nodes_word(len(b.texts))}, "
                 f"{b.n_tokens:,} {ask.tokens_word(b.n_tokens)})" if b else "")
        u.fail(f"{type(e).__name__}: {e}{where}")
        return None
    if args.json:
        print(json.dumps(res.to_json(with_ball=args.show_ball),
                         ensure_ascii=False, indent=2))
    if not args.no_log:
        # The log is a record, never a reason to lose an answer already given.
        # D19 keeps a failed *question* from ending the session; this keeps a
        # read-only or full `ask/logs/` from ending it either, which is what a
        # checkout shared with someone outside the owning group runs into.
        try:
            ask.log(res)
        except OSError as e:                     # noqa: BLE001 -- noted, not raised
            u.note(f"vprašanja ni bilo mogoče zabeležiti: {e.strerror}")
    return res


def interactive(pipe, args):
    u = pipe.ui
    try:
        import readline           # noqa: F401 -- history and line editing
    except ImportError:
        pass
    u.blank()
    u.note("vprašanje in Enter · Ctrl-D za izhod")
    while True:
        u.blank()
        try:
            question = u.prompt().strip()
        except (EOFError, KeyboardInterrupt):
            u.blank()
            return 0
        if not question:
            continue
        if question in (":q", ":quit", "exit"):
            return 0
        if question in (":?", ":help"):
            u.note(ask_help(args))
            continue
        u.blank()
        ask_one(pipe, question, args)


def ask_help(args):
    from ask import demo
    return demo.HELP if args.demo else "vprašaj karkoli o slovenski besedi"


def main(argv=None):
    args = parser().parse_args(argv)
    if args.list_demos:
        from ask import demo
        print(demo.HELP)
        return 0
    if args.alias or args.unalias or args.aliases:
        return manage_aliases(args)

    # Resolved here, before the Pipeline is built, so the startup line, the
    # `--json` payload and `ask/logs/` all record the path that answered rather
    # than the nickname it was reached by (D23).  The alias is kept only to be
    # printed back, as the confirmation that it went where it was meant to.
    from ask import aliases
    resolved = aliases.resolve(args.checkpoint)
    alias = args.checkpoint if resolved != args.checkpoint else ""
    args.checkpoint = resolved

    u = ui_mod.UI(quiet=args.quiet, echo=not args.json, debug=args.debug)

    if args.demo:
        from ask import demo, precompile as pc
        warm = os.path.exists(os.path.join(pc.cache_dir(), "demo.marker"))
        stages = demo.DemoStages(speed=demo.FAST if args.fast else 1.0,
                                 precompiled=warm)
    else:
        # D8 says this tool assumes a GPU in the terminal it runs in, and until
        # now that was an assumption rather than a check: `backbone` falls back
        # to `cpu` when CUDA is absent, so `ask` on a login node quietly starts
        # pulling a 12B extractor and a 12B backbone into shared RAM.  That is
        # not a slow answer, it is a login node someone else is also using.  The
        # ordinary way to meet this is to forget the allocation, so say so and
        # stop before anything loads.  `--demo` never gets here; it is the
        # answer for a machine without a GPU.
        import torch
        if not torch.cuda.is_available():
            u.fail("ni vidne GPU kartice -- `ask` nalozi dva 12B modela in "
                   "tega na prijavnem vozliscu ne sme poceti.")
            u.note("dodeli si GPU, npr.:  salloc -p dev -w ana "
                   "--gres=gpu:A100_80GB:1 --cpus-per-task=8 --mem=128G "
                   "-t 2:00:00")
            u.note("brez GPU deluje simulacija:  ask --demo -i")
            return 2
        # `--precompile` implies flex: the sweep's whole purpose is to fill
        # inductor's cache, and an eager model compiles nothing to fill it with.
        # Without this the default backend choice (no warm cache -> eager) makes
        # the flag a no-op that still writes the marker saying it worked.
        stages = RealStages(impl=args.impl or ("flex" if args.precompile
                                               else None))

    pipe = ask.Pipeline(stages, ui=u, checkpoint=args.checkpoint,
                        store=args.store, extractor=args.extractor,
                        show_ball=args.show_ball,
                        retrieve_only=args.retrieve_only,
                        checkpoint_alias=alias)

    u.blank()
    u.line(u.bold(BANNER) + (u.dim("  (demo)") if args.demo else ""))
    # `--words` on a one-shot run means nothing will ever call the extractor.
    needs_extractor = args.interactive or not args.words
    try:
        pipe.load(with_extractor=needs_extractor,
                  with_gtlm=not args.retrieve_only)
        if args.precompile:
            stages.precompile(u)
    except Exception as e:      # noqa: BLE001 -- startup failure, not D19
        if args.debug:
            raise
        u.fail(f"{type(e).__name__}: {e}")
        return 1

    if args.question:
        u.blank()
        ask_one(pipe, args.question, args, words=words_of(args))
    if args.interactive:
        return interactive(pipe, args)
    if not args.question:
        u.blank()
        u.note("nič za vprašati: podaj vprašanje ali -i")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
