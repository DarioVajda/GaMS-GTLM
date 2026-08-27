#!/usr/bin/env python3
"""Simulated stages: the whole CLI, with sleeps where the models would be.

`ask --demo` runs the real `Pipeline`, the real printing and the real `--json`
shape against these, so the terminal behaviour can be judged, and argued about,
before a single weight is loaded.  Every stage here has the signature its real
counterpart will have; filling one in is deleting a method from `DemoStages` and
pointing `stages.py`'s wiring at the real module.

**The delays are measured**, on an A100-40GB with a warm page cache, from the
acceptance run in `ask/slurm/`.  They were estimates while the real stages were
being built and are not any more -- the estimates were badly wrong in one
direction (the 12B extractor loads in 7 s, not the minute-plus assumed) and
roughly right in the others.  `--fast` divides them all by 8 for when you are
iterating on the output and not the pacing.

**The fixtures are real.**  Questions, extracted strings, ball sizes, anchor
texts and answers are lifted from `datasets/balls/test.jsonl` -- the same test
split `arms_v3` was scored on.  Node texts are read from that file when it is
present and fall back to a ten-node excerpt when it is not.
"""
import os
import re
import time
import json

from ask.extract import Extraction
from ask.retrieve import Ball
from ask.answer import PREFIX, SENTINEL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BALLS = os.path.join(REPO, "data", "datasets", "balls", "test.jsonl")

# ── how long each step takes, in seconds (A100-40GB, warm page cache) ───────
TIMINGS = {
    # 24 GB of bf16 weights off the shared filesystem.  The estimate here was
    # 20 s and the guess behind it was "a minute or more, capped so the demo
    # stays watchable"; it is 7.
    "load_extractor": 7.0,
    # gemma-3-1b plus LoRA plus the bias tensors, one `from_pretrained`.
    "load_gtlm": 3.2,
    # mmap of the store arrays, then the surface index (911,404 surfaces).
    "load_store": 1.5,
    # 64 new tokens, batch of one, on the 12B extractor.
    "extract": 1.0,
    "repair": 1.5,
    # Dictionary lookup; nothing but hashing.
    "lookup": 0.02,
    # hop-2 induced subgraph with the K_MWE / K_COLLOC caps applied.  p50 is
    # 20 ms and p90 222 ms; the tail is hub anchors and runs to 5 s (V8).
    "build": 0.2,
    # SPD, RRWP and the magnetic Laplacian, plus the per-question HF `.map()`
    # overhead V6 flagged.
    "features": 0.4,
    # Prefill, i.e. time to the first token.
    "prefill": 0.5,
    # Eager decode, single sequence: ~35 tokens/s for the 1b backbone.
    "token": 0.03,
    # One (L, N) shape through inductor.
    "compile": 1.2,
}

FAST = 8.0

# ── the store the demo claims to have opened ────────────────────────────────
STORE_INFO = {
    "path": "data/kg_processed/store",
    "graph_version": "v2",
    "n_nodes": 1_284_102,
    "created": "2026-07-30",
}

# ── scenarios, keyed by the word that triggers them ─────────────────────────
# `id` names the row in datasets/balls/test.jsonl the numbers came from.
SCENARIOS = [
    dict(
        id="T12-000592", trigger="brahialen",
        question="Navedi različne pomene besede brahialen.",
        strings=["brahialen"],
        n_nodes=73, n_edges=73, n_tokens=1484,
        anchors=[0], codes=[72057594037942593],
        nodes=[
            "iztočnica: brahialen (pridevnik, imenovalnik, ednina, moški spol, "
            "osnovnik, nedoločna oblika)",
            "iztočnica: arterija (samostalnik, ženski spol, imenovalnik, ednina)",
            "iztočnica: vena (samostalnik, ženski spol, imenovalnik, ednina)",
            "iztočnica: brahialne arterije",
            "iztočnica: brahialna arterija",
            "iztočnica: brahialna vena",
        ],
        answer="ODGOVOR: glede na roko ali z roko povezane",
    ),
    dict(
        # Two anchors from one string: `plezalo` resolves to both the noun and
        # the verb it is derived from, and training saw the union of both (D3).
        id="T1-000597", trigger="plezalo",
        question="Sklanjatev besede plezalo, prosim.",
        strings=["plezalo"],
        n_nodes=169, n_edges=170, n_tokens=2855,
        anchors=[2, 13], codes=[72057594037932683, 72057594038006416],
        nodes=[
            "iztočnica: komu (predlog)",
            "iztočnica: po (predlog)",
            "iztočnica: plezati (glagol, nedovršni, nedoločnik)",
            "iztočnica: otroški (pridevnik, imenovalnik, ednina, moški spol, "
            "osnovnik, določna oblika)",
        ],
        answer=("ODGOVOR: plezalo, plezala, plezalu, plezalo, plezalu, "
                "plezalom; plezali, plezal, plezaloma, plezali, plezalih, "
                "plezaloma; plezala, plezal, plezalom, plezala, plezalih, "
                "plezali"),
    ),
    dict(
        # A retrieval miss is answered, not refused (D6): the ball is one node
        # saying the word is absent, and the sentinel is the trained answer.
        id="T1-000665", trigger="folklora",
        question="Katere sklanjatvene oblike ima folklora?",
        strings=["folklór"],
        n_nodes=1, n_edges=0, n_tokens=7,
        anchors=[0], codes=[],
        nodes=["iztočnica: folklór (ni v bazi)"],
        answer=PREFIX + SENTINEL,
    ),
]

# Words that steer the demo into a path that is otherwise hard to provoke.
REPAIR_TRIGGER = "pokvarjeno"    # D7: unparseable output, one repair turn
ERROR_TRIGGER = "napaka"         # D19: a per-question failure, session survives

HELP = f"""Demo scenarios -- the word in the question picks the path:

  brahialen     a normal answer: 73-node ball, one anchor
  plezalo       one string, two anchors: 169 nodes, the union of both (D3)
  folklora      nothing resolves: the "(ni v bazi)" ball and the sentinel (D6)
  {REPAIR_TRIGGER}    the extractor returns junk, one repair turn fixes it (D7)
  {ERROR_TRIGGER}        the answer stage raises; the session survives (D19)

Anything else is treated as a word the base does not hold."""

WORD = re.compile(r"\w+", re.UNICODE)
# Question boilerplate: never what the extractor would name.
STOP = {"navedi", "kaj", "pomeni", "beseda", "besede", "besedo", "katere",
        "kako", "se", "ima", "za", "in", "je", "sklanjatev", "prosim",
        "razloži", "označuje", "oblike", "vse", "sklone", "vem", "lahko",
        "pomagaš", "kakšen", "kakšna", "kateri", "ki", "so", "pri", "na"}


def pick(question):
    q = question.casefold()
    for s in SCENARIOS:
        if s["trigger"] in q:
            return s
    return None


def guess_word(question):
    """What a real extractor would plausibly have named in a question we have
    no fixture for -- the last non-boilerplate word."""
    words = [w for w in WORD.findall(question) if w.casefold() not in STOP]
    return words[-1].casefold() if words else "beseda"


def load_nodes(scenario):
    """Real node texts from the test split, or the embedded excerpt padded out.

    Padding rather than a short list because everything downstream reads the
    ball's length, and a demo that lies about its own node count is worse than
    one that shows an ellipsis.
    """
    if os.path.exists(BALLS):
        want = scenario["id"]
        with open(BALLS, encoding="utf-8") as f:
            for line in f:
                if want not in line:
                    continue
                row = json.loads(line)
                if row["id"] == want:
                    return row["nodes"]
    nodes = list(scenario["nodes"])
    return nodes + ["…"] * (scenario["n_nodes"] - len(nodes))


class DemoStages:
    """Every stage of the chain, simulated.  Same signatures as the real ones."""

    def __init__(self, speed=1.0, precompiled=False):
        self.speed = speed
        self.precompiled = precompiled
        self.nodes = {}      # scenario id -> node texts, read once per session

    def wait(self, key, n=1.0):
        time.sleep(TIMINGS[key] * n / self.speed)

    # ── startup ─────────────────────────────────────────────────────────────
    def load_store(self, path):
        self.wait("load_store")
        return dict(STORE_INFO, path=path or STORE_INFO["path"])

    def load_extractor(self, name):
        self.wait("load_extractor")
        return {"name": name}

    def load_gtlm(self, checkpoint):
        self.wait("load_gtlm")
        # The same keys `backbone.load_gtlm` returns, so `--demo --debug` prints
        # the same lines: a simulated stage that carries fewer keys than the real
        # one only pretends to exercise the printing.
        out = {"base": "google/gemma-3-1b-it", "magnetic_m": 0,
               "max_length": 2048,
               "attn": "flex (prefill) + eager (decode)"}
        if not self.precompiled:
            # D17: no compiled cache for this GPU, so flex would recompile per
            # shape.  Fall back rather than make the first question pay for it.
            out["attn"] = "eager (flex attention kernels not compiled)"
            out["hint"] = ("compile the shapes first for a faster prefill: "
                           "`ask --precompile`")
        return out

    # ── the chain ───────────────────────────────────────────────────────────
    def extract(self, question, ui):
        self.wait("extract")
        if REPAIR_TRIGGER in question.casefold():
            # An unterminated list -- the one real failure in 12,490 items was
            # this shape, a repetition loop cut off by max_new_tokens (D7).
            junk = '["' + '", "'.join([REPAIR_TRIGGER] * 6) + '", "'
            ui.warn("ekstraktor ni vrnil veljavnega seznama; poskušam znova")
            ui.relabel("luščim iztočnice (2/2)…")
            self.wait("repair")
            return Extraction(strings=[REPAIR_TRIGGER], raw=f'["{REPAIR_TRIGGER}"]',
                              repaired=True, attempts=[junk])
        s = pick(question)
        strings = s["strings"] if s else [guess_word(question)]
        return Extraction(strings=strings, raw=json.dumps(strings,
                                                          ensure_ascii=False))

    def retrieve(self, strings, ui):
        self.wait("lookup")
        s = pick(" ".join(strings))
        if s is None:
            # Nothing resolved: the single truthful node, per build_balls.py:368.
            self.wait("build", 0.2)
            word = strings[0] if strings else "beseda"
            return Ball(texts=[f"iztočnica: {word} (ni v bazi)"], edges=[],
                        anchors=[0], codes=[], strings=list(strings),
                        n_tokens=7)
        self.wait("build")
        self.wait("features")
        if s["id"] not in self.nodes:
            self.nodes[s["id"]] = load_nodes(s)
        texts = self.nodes[s["id"]]
        edges = [(i, i + 1) for i in range(s["n_edges"])]
        return Ball(texts=texts, edges=edges, anchors=list(s["anchors"]),
                    codes=list(s["codes"]), strings=list(strings),
                    n_tokens=s["n_tokens"])

    def precompile(self, ui):
        """The D17 shape sweep, one stage line per (L, N)."""
        from ask import precompile as pc
        shapes = pc.shapes()
        d = pc.cache_dir()
        ui.field("prevajam", f"{len(shapes)} oblik → {os.path.relpath(d, REPO)}")
        for i, (length, n) in enumerate(shapes, 1):
            with ui.stage(f"L={length}, N={n}  ({i}/{len(shapes)})"):
                self.wait("compile")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "demo.marker"), "w") as f:
            f.write("written by ask --demo --precompile\n")
        self.precompiled = True
        ui.field("predpomnilnik", "pripravljen · flex za prefill")

    def answer(self, question, ball, ui):
        """Yield the answer one token at a time, prefill delay included."""
        self.wait("prefill")
        if ERROR_TRIGGER in question.casefold():
            raise RuntimeError("CUDA out of memory: tried to allocate 2.41 GiB")
        s = pick(question)
        text = s["answer"] if s else PREFIX + SENTINEL
        for tok in tokenish(text):
            self.wait("token")
            yield tok


def tokenish(text):
    """Split roughly the way a subword tokenizer would, so the stream paces
    like a real decode rather than word by word."""
    out = []
    for word in re.findall(r"\s*\S+", text):
        if len(word) <= 6:
            out.append(word)
        else:
            cut = len(word) // 2
            out += [word[:cut], word[cut:]]
    return out
