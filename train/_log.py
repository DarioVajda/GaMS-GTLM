"""Stop one upstream warning from drowning the job log.

`GTLMGemma3TextModel._update_causal_mask` warns that the adapter drops Gemma-3's
sliding-window band once the packed sequence passes 512 tokens.  It is a real and
useful warning -- these balls run to 14,055 packed tokens, so it fires on almost
every batch -- and it is emitted through `logger.warning_once`.

But `warning_once` is keyed on the *message text*, and this message embeds the
sequence length.  During autoregressive decoding that length increments by one
per token, so every step is a "new" message: a 300-token generation over a few
hundred eval items writes tens of megabytes of the same sentence, and the log a
person actually has to read scrolls past unseen.

So the first one is kept and the rest are dropped.  Nothing is suppressed that
the reader has not already been told.
"""
import logging

PREFIX = "GTLM-Gemma3: packed sequence length"


class _OnlyOnce(logging.Filter):
    def __init__(self, prefix=PREFIX):
        super().__init__()
        self.prefix = prefix
        self.seen = False

    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:                                    # noqa: BLE001
            return True
        if not msg.startswith(self.prefix):
            return True
        if self.seen:
            return False
        self.seen = True
        return True


_INSTALLED = False


def quiet_repeated_sliding_window_warning():
    """Idempotent; safe to call from every entry point."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    f = _OnlyOnce()
    try:
        from gtlm.models import modeling_gtlm_gemma3 as m
        m.logger.addFilter(f)
    except Exception:                                        # noqa: BLE001
        pass
    # Belt and braces: a filter on the logger only sees records logged through
    # that logger object, and the module name the adapter registers under depends
    # on how `gtlm` was installed.  Handler-level filters catch it either way.
    for name in ("transformers", ""):
        for h in logging.getLogger(name).handlers:
            h.addFilter(f)
