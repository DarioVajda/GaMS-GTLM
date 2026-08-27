#!/usr/bin/env python3
"""Terminal output for `ask`: one live line for what is running now, one block
per stage as it finishes (D20).

Three rules the rest of the package depends on:

  * **Progress and stage results go to stderr, the answer goes to stdout.**
    `ask "..." > out.txt` and `--json` therefore stay byte-clean, and the
    spinner can never corrupt a redirect.
  * **Not a TTY, no ANSI.** Under a file or a Slurm log the spinner degrades to
    one plain line per stage -- no cursor motion, no repaint.
  * **Every write goes through `_emit`**, which holds the lock and clears the
    spinner line first, so a message printed from inside a running stage (the
    D7 repair warning) does not interleave with a frame.

User-facing strings are Slovene; flags, help and exceptions stay English, which
is what the rest of the repo speaks.
"""
import os
import sys
import time
import shutil
import itertools
import threading
from contextlib import contextmanager

FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
TICK = 0.08
FIELD = 11          # width of the field-name column, e.g. "iztočnice  "
INDENT = "  "

DIM = "\033[2m"
BOLD = "\033[1m"
RED = "\033[31m"
YELLOW = "\033[33m"
OFF = "\033[0m"


def plural(n, one, two, few, many):
    """Slovene number agreement: 1 vozlišče, 2 vozlišči, 3 vozlišča, 5 vozlišč.

    Worth the four arguments -- the counts in this tool's output are read by
    lexicographers, and `1 vozlišč` is exactly the kind of thing they notice.
    """
    r = abs(n) % 100
    return one if r == 1 else two if r == 2 else few if r in (3, 4) else many


class UI:
    """Progress on stderr, results on stdout.

    `echo=False` keeps the answer out of stdout (--json builds its own output
    from the same string); `quiet=True` silences stderr entirely.
    """

    def __init__(self, err=None, out=None, quiet=False, echo=True):
        self.err = err or sys.stderr
        self.out = out or sys.stdout
        self.quiet = quiet
        self.echo = echo
        self.tty = self.err.isatty() and os.environ.get("TERM") != "dumb"
        self._lock = threading.RLock()
        self._stop = None
        self._spin = None
        self._label = ""

    # ── colour ──────────────────────────────────────────────────────────────
    def _c(self, text, code):
        return f"{code}{text}{OFF}" if self.tty else text

    def dim(self, t):
        return self._c(t, DIM)

    def bold(self, t):
        return self._c(t, BOLD)

    # ── primitives ──────────────────────────────────────────────────────────
    def _emit(self, text=""):
        if self.quiet:
            return
        with self._lock:
            if self.tty:
                self.err.write("\r\033[K")
            self.err.write(text + "\n")
            self.err.flush()

    def blank(self):
        self._emit()

    def line(self, text):
        self._emit(INDENT + text)

    def field(self, name, value, colour=None):
        """`  iztočnice   brahialen` -- the name column is fixed width."""
        pad = " " * max(1, FIELD - len(name))
        label = self._c(name, colour) if colour else self.dim(name)
        self._emit(f"{INDENT}{label}{pad}{value}")

    def cont(self, value):
        """A continuation line under the last field, aligned to its value."""
        self._emit(f"{INDENT}{' ' * FIELD}{self.dim(value)}")

    def warn(self, text):
        self.field("opozorilo", text, colour=YELLOW)

    def fail(self, text):
        self.field("napaka", text, colour=RED)

    def note(self, text):
        for line in text.splitlines() or [""]:
            self._emit(self.dim(INDENT + line))

    def width(self):
        # A pty can report 0 columns (no window size set, e.g. under `script`),
        # which would clip every line to nothing.  Treat anything implausible
        # as an ordinary terminal.
        cols = shutil.get_terminal_size((100, 24)).columns
        return cols if cols >= 40 else 100

    def clip(self, text, reserve=0):
        """Trim to the terminal width so a long node text cannot wrap."""
        room = max(20, self.width() - len(INDENT) - FIELD - reserve - 1)
        return text if len(text) <= room else text[:room - 1] + "…"

    # ── the live line ───────────────────────────────────────────────────────
    @contextmanager
    def stage(self, label):
        """Spin on `label` until the block exits; yields nothing.

        On a TTY the line is erased on exit and the caller prints the result.
        Off a TTY the label is printed on entry (so a hang is visible in a log)
        and again with its elapsed time on exit.
        """
        t0 = time.perf_counter()
        if self.quiet:
            yield
            return
        self._label = label
        if self.tty:
            self._spin_start(t0)
        else:
            self._emit(f"{INDENT}... {label}")
        try:
            yield
        finally:
            if self.tty:
                self._spin_stop()
            else:
                self._emit(f"{INDENT}... {self._label} "
                           f"{time.perf_counter() - t0:.1f}s")

    def relabel(self, label):
        """Change what the running stage says it is doing.

        For a stage that visibly changes course mid-flight -- the D7 repair turn
        is the one case -- so the line reads `luščim iztočnice (2/2)` while the
        second attempt is actually running, rather than explaining itself
        afterwards.
        """
        with self._lock:
            self._label = label
        if not self.tty:
            self._emit(f"{INDENT}... {label}")

    def _spin_start(self, t0):
        self._stop = threading.Event()

        def run():
            for frame in itertools.cycle(FRAMES):
                if self._stop.is_set():
                    return
                with self._lock:
                    el = time.perf_counter() - t0
                    body = f"{INDENT}{frame} {self._label}  {el:.1f}s"
                    self.err.write("\r\033[K" + self._c(body, DIM))
                    self.err.flush()
                self._stop.wait(TICK)

        self._spin = threading.Thread(target=run, daemon=True)
        self._spin.start()

    def _spin_stop(self):
        if not self._spin:
            return
        self._stop.set()
        self._spin.join()
        self._spin = None
        with self._lock:
            self.err.write("\r\033[K")
            self.err.flush()

    # ── the answer, on stdout ───────────────────────────────────────────────
    def say(self, text):
        if self.echo:
            self.out.write(text + "\n")
            self.out.flush()

    def stream(self, chunks):
        """Write chunks as they arrive; return them as a list.

        A list rather than the joined text because one chunk is one decoded
        token, which is the only place the token count is free.  Collects even
        when `echo` is off, so --json gets what the terminal would have shown.
        """
        buf = []
        for c in chunks:
            buf.append(c)
            if self.echo:
                self.out.write(c)
                self.out.flush()
        if self.echo and buf:
            self.out.write("\n")
            self.out.flush()
        return buf
