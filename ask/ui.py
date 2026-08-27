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
    D7 repair warning) does not interleave with a frame.  A running stage also
    CAPTURES `sys.stdout` and `sys.stderr`, because the libraries under it are
    not so disciplined -- `datasets` draws four progress bars per question and
    the loader prints a line of its own, and either would be shredded by the
    spinner repainting over it.  `--debug` prints what was captured, dimmed,
    once the stage is done.

User-facing strings are Slovene; flags, help and exceptions stay English, which
is what the rest of the repo speaks.  One deliberate exception: the attention
backend is named in English (`flex attention kernels not compiled`), because it
is a fact about torch's compiler and those are the words it is discussed under
everywhere the reader would go next.
"""
import io
import os
import sys
import time
import shutil
import itertools
import threading
import contextlib
from contextlib import contextmanager

FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
CR = "\r"
TICK = 0.08
FIELD = 11          # width of the field-name column, e.g. "iztočnice  "
INDENT = "  "

DIM = "\033[2m"
BOLD = "\033[1m"
RED = "\033[31m"
YELLOW = "\033[33m"
OFF = "\033[0m"

# The tint the question sits on.  256-colour rather than a named background:
# the named eight are whatever the user's theme says they are, and half of them
# are loud.  236/252 is a grey one step off black under a dark theme, and a
# deliberate dark bar under a light one -- both legible, because the pair sets
# the foreground too rather than leaving it to the theme.
INPUT_BG = "\033[48;5;236m"
INPUT_FG = "\033[38;5;252m"


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

    def __init__(self, err=None, out=None, quiet=False, echo=True, debug=False):
        # Captured once, and written to directly ever after: a stage redirects
        # `sys.stdout`/`sys.stderr` to swallow library noise, and the progress
        # line must not be swallowed with it.
        self.err = err or sys.stderr
        self.out = out or sys.stdout
        self.quiet = quiet
        self.echo = echo
        self.debug = debug
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

    def cont(self, value, dim=True):
        """A continuation line under the last field, aligned to its value."""
        self._emit(f"{INDENT}{' ' * FIELD}{self.dim(value) if dim else value}")

    def field_path(self, name, path, colour=None):
        """A path as a field, folded at `/` when it will not fit one row.

        A checkpoint path is ~190 characters -- the run directory alone encodes
        the whole arm -- and the terminal's own wrap would run it back under the
        label column and turn the block into a smear.  Folding at the separators
        keeps every row aligned under the value, and the pieces keep their `/`,
        so the rows concatenate back into the path that was printed.
        """
        room = max(20, self.width() - len(INDENT) - FIELD - 1)
        parts = path.split("/")
        rows, cur = [], ""
        for i, part in enumerate(parts):
            piece = part + ("/" if i < len(parts) - 1 else "")
            while len(piece) > room:        # a component longer than the row
                if cur:
                    rows.append(cur)
                    cur = ""
                rows.append(piece[:room])
                piece = piece[room:]
            if cur and len(cur) + len(piece) > room:
                rows.append(cur)
                cur = piece
            else:
                cur += piece
        rows.append(cur)
        self.field(name, rows[0], colour=colour)
        for row in rows[1:]:
            self.cont(row, dim=False)

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
        self._label = label
        if not self.quiet:
            if self.tty:
                self._spin_start(t0)
            else:
                self._emit(f"{INDENT}... {label}")
        try:
            with self._captured() as buf:
                yield
        finally:
            if not self.quiet:
                if self.tty:
                    self._spin_stop()
                else:
                    self._emit(f"{INDENT}... {self._label} "
                               f"{time.perf_counter() - t0:.1f}s")
            self._replay(buf)

    @contextmanager
    def _captured(self):
        """Swallow whatever the libraries write while a stage runs.

        Both streams, because the offenders are split across them: `datasets`
        draws four progress bars per question on stderr, and
        `causal_lm.from_pretrained` announces itself on stdout.  Neither can be
        let through -- stdout is the answer channel, and stderr is where the
        spinner is repainting.  This UI's own writes go to the stream objects
        captured in `__init__`, so they are unaffected.
        """
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), \
                self._logging_to(buf):
            yield buf

    @contextmanager
    def _logging_to(self, buf):
        """Point live `logging` handlers at the capture buffer as well.

        `redirect_stderr` rebinds the *name* `sys.stderr`.  A `StreamHandler`
        built at import time holds the stream *object*, so every record it
        formats goes straight to the terminal past the redirect -- which is how
        `huggingface_hub`'s "could not cache non-existence of file" warnings came
        to be painted over the spinner, one per missing file, mid-load.

        Only handlers pointing at the two streams this UI captured are moved,
        and each is put back in a `finally`: a handler on some other file is not
        this class's business.
        """
        import logging

        loggers = [lg for lg in logging.Logger.manager.loggerDict.values()
                   if isinstance(lg, logging.Logger)]
        moved = []
        for h in [*logging.root.handlers, *(h for lg in loggers
                                            for h in lg.handlers)]:
            if isinstance(h, logging.StreamHandler) and \
                    getattr(h, "stream", None) in (self.err, self.out):
                moved.append((h, h.stream))
                h.stream = buf
        try:
            yield
        finally:
            for h, stream in moved:
                h.stream = stream

    def _replay(self, buf):
        """Print what a stage swallowed, under `--debug`."""
        if not self.debug or buf is None:
            return
        for line in buf.getvalue().strip().splitlines():
            # A progress bar rewrites one line with \r; keep its final state.
            self._emit(self.dim(f"{INDENT}| {line.rsplit(CR, 1)[-1]}"))

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

    # ── the question, from stdin ────────────────────────────────────────────
    def prompt(self, mark="> "):
        """Read one question on a tinted bar; return it as typed.

        The tint is painted three times, because a terminal colours a cell only
        when something is written into it and the row starts out empty:

          * **Before the prompt**, the background is set and the row erased.
            `\\033[K` paints with the *current* background rather than the
            default one -- "background colour erase", which every terminal in
            use here has -- so the whole row is tinted while the cursor has not
            moved.  Where it is missing the row simply stays untinted ahead of
            the cursor, which is the next case on its own.
          * **As the prompt itself**, so the characters being typed land on the
            tint: the background the prompt set is still in effect for the echo,
            and for readline's own redraws, whose `\\033[K` re-paints by the
            same rule.
          * **Over the finished line**, padded out to the terminal width, so the
            question stays a full-width block as it scrolls up.

        Two details that are not optional.  Readline counts the prompt's
        printing width to know where the cursor is, so the escapes must be
        bracketed in \\001/\\002 or every line edit lands one column off per
        byte.  And the background is still set when Enter echoes its newline, so
        it is reset in a `finally` -- otherwise a Ctrl-C leaves the rest of the
        session tinted.

        Both streams must be the terminal: `input` writes the prompt to stdout,
        the repaint goes to stderr, and under `--json > file` the first is a
        pipe.  Then this is plain `input`, which is what it always was.
        """
        if self.quiet or not (self.tty and self.out.isatty()):
            return input(mark)
        skin = INPUT_BG + INPUT_FG
        head = " " + mark
        try:
            # Tint the row the cursor is already on, then hand it to readline.
            self.err.write(skin + "\033[K")
            self.err.flush()
            text = input(f"\001{skin}\002{head}")
        finally:
            self.err.write(OFF)
            self.err.flush()
        shown = head + text
        room = self.width()
        if len(shown) < room:
            # One row only: the cursor is at the start of the row below, so
            # `1A` is where the line is.  A question long enough to wrap sits on
            # rows this cannot count reliably (readline may have scrolled), and
            # keeps the prompt-painted tint it already has.
            self.err.write(f"\033[1A\r\033[K{skin}{shown}"
                           f"{' ' * (room - len(shown))}{OFF}\n")
            self.err.flush()
        return text

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
