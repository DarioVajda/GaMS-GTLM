"""One import seam onto the dataset's own grader.

`data/qa/grade.py` IS the contract -- the same module that graded the corpus at
generation time and that `qa.check_balls` reads.  Re-implementing it here would
give the trainer a second opinion about what counts as correct, which is how a
reported number stops meaning what the dataset says it means.  So it is imported,
not copied.

`data/` is not a package (there is no `data/__init__.py`; `data/qa/` is the
package), so the directory goes on `sys.path` -- appended, not inserted, so
nothing in `data/` can shadow a real dependency.
"""
import os
import sys

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data")
if _DATA not in sys.path:
    sys.path.append(_DATA)

from qa.grade import grade, contract, count_ok, parse   # noqa: E402,F401
from qa.spec import SENTINEL, PREFIX                 # noqa: E402,F401

# A pass-1 token mismatch is a definite failure in these modes: `sequence`
# compares the answer position by position, and a negative's sentinel is
# exact-match in every mode.  Everything else can still succeed with a different
# string, so it has to be generated and graded.
EXACT_MODES = frozenset({"sequence"})


def needs_generation(item):
    """Would generating this item's answer ever change a pass-1 miss into a pass?"""
    if item.get("negative"):
        return False
    # Through `contract`, never off the row: a dataset on disk carries whatever
    # the spec said the day it was built, and reading it directly is how T19 was
    # treated as `sequence` for a whole run after the spec said `membership`.
    return contract(item)["mode"] not in EXACT_MODES
