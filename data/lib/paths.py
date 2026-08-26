"""Where the pipeline's inputs and outputs live, resolved from this file.

Everything is derived from the repo root so a checkout in another location, or a
job whose working directory is not `data/`, resolves the same paths.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data")

KG_RAW_DIR = os.path.join(DATA_DIR, "kg_raw", "OntoLex DSB")
STORES_DIR = os.path.join(DATA_DIR, "stores")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
GENERATED_DIR = os.path.join(DATASETS_DIR, "generated")
RESULTS_DIR = os.path.join(DATA_DIR, "analysis", "results")

PROMPTS_DIR = os.path.join(DATA_DIR, "prompts")
# The prompt the extraction run behind the current corpus used, and the one a
# re-run must use to reproduce it.  Superseded revisions are in prompts/archive/.
EXTRACTOR_PROMPT = os.path.join(PROMPTS_DIR, "extractor_prompt.txt")

# Shared model cache on this cluster; outside the repo, so it stays a literal.
HF_CACHE = "/shared/workspace/povejmo/huggingface_cache"


def kg_raw_file(name):
    """Path to one raw N-Triples file, e.g. `kg_raw_file("220-words.nt")`."""
    return os.path.join(KG_RAW_DIR, name)
