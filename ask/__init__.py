#!/usr/bin/env python3
"""`ask` -- one question in, one grounded answer out.

    question -> extractor -> strings -> surface index -> anchors -> ball
             -> GTLM (+ ball) -> answer

`Pipeline` is the whole chain held warm in one process: load once, answer many.
It owns the order of the stages and what gets printed between them; the stages
themselves live one per module (`extract`, `retrieve`, `answer`), and the models
they run on come from `backbone`.

The stages are injected rather than imported so `--demo` can substitute the
simulated ones in `demo.py` -- the CLI, the printing and the JSON shape are then
exercised end to end without a GPU, and filling in a real stage is a one-line
swap.  See `ask/PLAN.md` for the decisions behind all of this.
"""
import os
import json
import time
import datetime
import itertools
from dataclasses import dataclass, field

from ask import ui as ui_mod
from ask import extract as extract_mod
from ask import answer as answer_mod

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


def nodes_word(n):
    return ui_mod.plural(n, "vozlišče", "vozlišči", "vozlišča", "vozlišč")


def edges_word(n):
    return ui_mod.plural(n, "povezava", "povezavi", "povezave", "povezav")


def tokens_word(n):
    return ui_mod.plural(n, "token", "tokena", "tokeni", "tokenov")


@dataclass
class Result:
    """One answered question, and enough to reconstruct how it was answered."""
    question: str
    extraction: object = None      # extract.Extraction
    ball: object = None            # retrieve.Ball
    answer: object = None          # answer.Answer
    timings: dict = field(default_factory=dict)
    checkpoint: str = ""
    store: str = ""
    error: str = ""

    def to_json(self, with_ball=False):
        """The `--json` payload.  Anchors are reported as node **codes**, not
        indices: indices are positions in one store build and mean nothing after
        a rebuild, which would break the reconstruction promise (D9)."""
        b, e, a = self.ball, self.extraction, self.answer
        out = {
            "question": self.question,
            "extraction": {
                "strings": list(e.strings) if e else [],
                "repaired": bool(e.repaired) if e else False,
            },
            "targets": list(b.codes) if b else [],
            "anchors": b.anchor_texts() if b else [],
            "ball": {
                "n_nodes": len(b.texts) if b else 0,
                "n_edges": len(b.edges) if b else 0,
                "n_tokens": b.n_tokens if b else 0,
                "missing": bool(b.missing) if b else False,
            },
            "answer": a.text if a else None,
            "sentinel": bool(a.sentinel) if a else None,
            "checkpoint": self.checkpoint,
            "store": self.store,
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
        }
        if self.error:
            out["error"] = self.error
        if with_ball and b:
            out["ball"]["nodes"] = list(b.texts)
            out["ball"]["edges"] = [list(x) for x in b.edges]
        return out


class Pipeline:
    """The chain, warm.  `load()` once, then `run()` per question."""

    def __init__(self, stages, ui=None, checkpoint="", store="", extractor="",
                 show_ball=False, retrieve_only=False, checkpoint_alias=""):
        self.s = stages
        self.ui = ui or ui_mod.UI()
        self.checkpoint = checkpoint
        # For printing only (D23): `checkpoint` is already the resolved path,
        # and it is the path that gets logged.
        self.checkpoint_alias = checkpoint_alias
        self.store = store
        self.extractor = extractor
        self.show_ball = show_ball
        self.retrieve_only = retrieve_only
        self.last = None            # partial Result, for error reporting (D19)

    def _dump_ball(self, ball):
        """Every node, anchors marked -- what the model is about to read."""
        u = self.ui
        for i, text in enumerate(ball.texts):
            mark = "▸" if i in ball.anchors else " "
            u.cont(f"{mark} {i:>3}  {u.clip(text, reserve=8)}")

    # ── startup ─────────────────────────────────────────────────────────────
    def load(self, with_extractor=True, with_gtlm=True):
        """Load the store and the models, announcing provenance (D10).

        Either model can be skipped, because either half of the chain can be
        the part a caller does not need: `--words` supplies the headwords a
        12B extractor would have named, and `--retrieve-only` stops at the ball.
        Loading a model nobody will call is 20-60 seconds spent on nothing.
        """
        u = self.ui
        t0 = time.perf_counter()
        with u.stage("odpiram bazo…"):
            info = self.s.load_store(self.store)
        self.store = os.path.abspath(info["path"])   # named even on the default
        u.field_path("baza", self.store)
        u.cont(f"graph_version {info['graph_version']} · "
               f"{info['n_nodes']:,} vozlišč · zgrajena {info['created']}")
        if info.get("note"):
            u.cont(info["note"])
        if info.get("warn"):
            u.warn(info["warn"])

        if with_extractor:
            with u.stage("nalagam ekstraktor…"):
                self.s.load_extractor(self.extractor)
            u.field("ekstraktor", self.extractor)
        else:
            u.field("ekstraktor", u.dim("preskočen (--words)"))

        if with_gtlm:
            with u.stage("nalagam GTLM…"):
                gt = self.s.load_gtlm(self.checkpoint)
            # The full path, not the basename: every arm of a sweep ends in a
            # `checkpoint-NNNN`, and which arm produced this one -- the seed, the
            # features, the data root -- is written in the run directory's name
            # and nowhere else.  `checkpoint-4400` alone does not say what was
            # loaded, and this line is the record of what answered.
            self.checkpoint = os.path.abspath(self.checkpoint.rstrip("/"))
            u.field_path("GTLM", self.checkpoint)
            if self.checkpoint_alias:
                # The path above is four folded rows of generated run name; this
                # is the one row that says which arm was asked for, and whether
                # the alias went where it was meant to.
                u.cont(f"bližnjica {self.checkpoint_alias}")
            u.cont(f"{gt['base']} · {gt['attn']}")
            if u.debug:
                # Feature-level provenance: worth having when an answer looks
                # wrong, worth nothing to someone about to ask a question.
                u.cont(f"magnetic_m={gt['magnetic_m']} · "
                       f"max_length={gt['max_length']} · brez drsečega okna")
            if gt.get("hint"):
                u.note(gt["hint"])
        else:
            u.field("GTLM", u.dim("preskočen (--retrieve-only)"))
        u.field("pripravljeno", f"{time.perf_counter() - t0:.1f}s")

    # ── one question ────────────────────────────────────────────────────────
    def run(self, question, words=None):
        """One question.  `words` skips extraction and uses those strings."""
        u, t = self.ui, {}
        res = Result(question=question, checkpoint=self.checkpoint,
                     store=self.store)
        self.last = res

        t0 = time.perf_counter()
        if words:
            res.extraction = extract_mod.given(words)
        else:
            with u.stage("luščim iztočnice…"):
                res.extraction = self.s.extract(question, u)
        t["extract"] = time.perf_counter() - t0
        # A repair turn announces itself from inside the stage, while it is
        # happening -- only the stage knows the moment (D7, D20).
        u.field("iztočnice", ", ".join(res.extraction.strings) or u.dim("—"))

        t0 = time.perf_counter()
        with u.stage("gradim okolico…"):
            res.ball = self.s.retrieve(res.extraction.strings, u)
        t["retrieve"] = time.perf_counter() - t0
        b = res.ball
        if b.missing:
            u.field("okolica", "iztočnice ni v bazi")
            u.cont(b.texts[0])
        else:
            u.field("okolica", f"{len(b.texts)} {nodes_word(len(b.texts))} · "
                               f"{len(b.edges)} {edges_word(len(b.edges))} · "
                               f"~{b.n_tokens:,} {tokens_word(b.n_tokens)}")
            if self.show_ball:
                self._dump_ball(b)     # the anchors are marked in the dump
            else:
                for text in b.anchor_texts():
                    u.cont("▸ " + u.clip(text, reserve=2))

        if self.retrieve_only:
            res.timings = t
            u.note(f"({t['extract']:.1f}s + {t['retrieve']:.1f}s)")
            return res

        t0 = time.perf_counter()
        # The prefill happens inside the stage; the spinner stops the moment the
        # first token arrives, and the rest streams under it (D20).
        with u.stage("odgovarjam…"):
            stream = iter(self.s.answer(question, b, u))
            first = next(stream, "")     # the prefill; the spinner ends here
        chunks = u.stream(itertools.chain([first], stream) if first else stream)
        t["answer"] = time.perf_counter() - t0
        res.answer = answer_mod.Answer("".join(chunks), len(chunks))
        res.timings = t

        u.note(f"({t['extract']:.1f}s + {t['retrieve']:.1f}s + "
               f"{t['answer']:.1f}s)")
        return res


def log(result, path=None):
    """Append one line per question (D14).  `ask/logs/` is gitignored."""
    d = path or LOG_DIR
    os.makedirs(d, exist_ok=True)
    day = datetime.date.today().isoformat()
    rec = result.to_json()
    rec["at"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(d, f"{day}.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
