#!/bin/bash
# Resolve a Python that can import the graph_model venv's packages ON THIS NODE.
#
# The venv is on the shared filesystem but is NOT node-independent:
#   .venv/bin/python -> python3 -> /usr/bin/python3
# is an absolute symlink, so on a node whose /usr/bin/python3 is not 3.10 the
# interpreter starts up fine and then cannot see .venv/lib/python3.10/site-packages
# (job 128295 died this way on `axa`: ModuleNotFoundError: No module named 'numpy').
#
# Strategy: use the venv launcher when the node's /usr/bin/python3 really is 3.10;
# otherwise drive a 3.10 interpreter directly and inject the venv's site-packages.
# Prints the interpreter path on stdout; exports PYTHONPATH when it had to fall back.
VENV=${VENV:-/shared/workspace/povejmo/graph_model/.venv}
SITE="$VENV/lib/python3.10/site-packages"

_ok() { "$1" -c 'import numpy, transformers' >/dev/null 2>&1; }

if _ok "$VENV/bin/python"; then
    echo "$VENV/bin/python"; exit 0
fi

for cand in /usr/bin/python3.10 /usr/local/bin/python3.10 "$(command -v python3.10 || true)"; do
    [ -x "$cand" ] || continue
    "$cand" -V 2>&1 | grep -q '^Python 3\.10\.' || continue
    if PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}" _ok "$cand"; then
        echo "$cand"; exit 0
    fi
done

echo "no usable Python 3.10 on $(hostname)" >&2
echo "  /usr/bin/python3 -> $(readlink -f /usr/bin/python3 2>/dev/null) ($(/usr/bin/python3 -V 2>&1))" >&2
echo "  available: $(ls /usr/bin/python3.* 2>/dev/null | tr '\n' ' ')" >&2
exit 1
