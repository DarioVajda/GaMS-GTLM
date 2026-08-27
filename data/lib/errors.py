"""The one exception a stage raises when its inputs are wrong.

Every stage in `data/` is both a CLI (`python -m qa.build_balls ...`) and a
library call the pipeline makes (`build_balls.run(...)`).  Those two want
different things from a bad input: the CLI wants the message on stderr and a
non-zero exit, the caller wants something it can catch and record.

`SystemExit` cannot serve both.  It derives from `BaseException`, so a caller
writing the obvious `except Exception` does not catch it and the whole pipeline
exits mid-stage with no summary written.  So `run()` raises `StageError` and
`main()` converts it back:

    def main():
        ...
        try:
            run(**vars(args))
        except StageError as e:
            raise SystemExit(str(e))

which leaves the command-line behaviour byte-identical to what it was before --
same message, same exit code -- while giving the pipeline something ordinary to
catch.
"""


class StageError(RuntimeError):
    """A stage cannot run: its inputs are missing, malformed or inconsistent.

    Distinct from an unexpected `RuntimeError`, which is a bug and should keep
    its traceback rather than be flattened into a tidy message.
    """
