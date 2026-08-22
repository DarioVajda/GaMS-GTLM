"""Fine-tuning GTLM on the generated Slovene lexicographical QA dataset.

A standalone argparse program, invoked from the repo root as

    .venv/bin/python -m train --types T3,T4,T9,T10

which is also the shape the generic `sweep` runner drives:

    .venv/bin/python -m sweep train train/configs/<name>.jsonc

`sweep` and the GTLM stack (`gtlm.models`, `gtlm.train`, `gtlm.utils`) come from
the `graph_model` repo, editable-installed into this repo's `.venv`.  Nothing
lexicography-specific lives over there: the model code is a dependency, the
experiments are ours.

Note `gtlm.train` (the shared training utilities, in graph_model) and this
package are different things despite the name; the import is always spelled in
full, so the two never collide.
"""

from .config import RunConfig

__all__ = ["RunConfig"]
