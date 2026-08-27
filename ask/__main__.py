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
    """The real chain.  Each method is a stub until its build step lands."""

    def __init__(self, precompiled=False):
        self.precompiled = precompiled
        self.store = self.extractor = self.gtlm = None

    def load_store(self, path):
        self.store = backbone.load_store(path, None)
        return self.store

    def load_extractor(self, name):
        self.extractor = backbone.load_extractor(name, None)
        return self.extractor

    def load_gtlm(self, checkpoint):
        self.gtlm = backbone.load_gtlm(checkpoint, None)
        return self.gtlm

    def extract(self, question, ui):
        return extract.extract(question, self.extractor, ui)

    def retrieve(self, strings, ui):
        return retrieve.retrieve(strings, self.store, ui)

    def answer(self, question, ball, ui):
        return answer_mod.answer(question, ball, self.gtlm, ui)

    def precompile(self, ui):
        from ask import precompile as pc
        return pc.warm(self.gtlm, None, ui)


def parser():
    p = argparse.ArgumentParser(
        prog="ask", description="Ask a GTLM checkpoint a question about "
                                "Slovene, grounded in the lexical graph.")
    p.add_argument("question", nargs="?", help="the question; omit with -i")
    p.add_argument("-i", "--interactive", action="store_true",
                   help="stay warm and keep asking")
    p.add_argument("--checkpoint", default=backbone.DEFAULT_CHECKPOINT,
                   help="path to a checkpoint directory (default: the best "
                        "arms_v3 arm)")
    p.add_argument("--store", default="", help="path to the processed graph store")
    p.add_argument("--extractor", default=backbone.DEFAULT_EXTRACTOR,
                   help="model that names the words a question is about")
    p.add_argument("--json", action="store_true",
                   help="print one JSON object per question on stdout")
    p.add_argument("--show-ball", action="store_true",
                   help="print every node of the retrieved ball")
    p.add_argument("--precompile", action="store_true",
                   help="compile every (L, N) shape before asking, and cache it")
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


def ask_one(pipe, question, args):
    """One question, and never let it end the session (D19)."""
    u = pipe.ui
    try:
        res = pipe.run(question)
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
        ask.log(res)
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
            question = input("> ").strip()
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

    u = ui_mod.UI(quiet=args.quiet, echo=not args.json)

    if args.demo:
        from ask import demo, precompile as pc
        warm = os.path.exists(os.path.join(pc.cache_dir(), "demo.marker"))
        stages = demo.DemoStages(speed=demo.FAST if args.fast else 1.0,
                                 precompiled=warm)
    else:
        stages = RealStages()

    pipe = ask.Pipeline(stages, ui=u, checkpoint=args.checkpoint,
                        store=args.store, extractor=args.extractor,
                        show_ball=args.show_ball)

    u.blank()
    u.line(u.bold(BANNER) + (u.dim("  (demo)") if args.demo else ""))
    try:
        pipe.load()
        if args.precompile:
            stages.precompile(u)
    except Exception as e:      # noqa: BLE001 -- startup failure, not D19
        if args.debug:
            raise
        u.fail(f"{type(e).__name__}: {e}")
        return 1

    if args.question:
        u.blank()
        ask_one(pipe, args.question, args)
    if args.interactive:
        return interactive(pipe, args)
    if not args.question:
        u.blank()
        u.note("nič za vprašati: podaj vprašanje ali -i")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
