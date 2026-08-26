#!/bin/bash
#SBATCH --partition=frida
#SBATCH --account=lkm
# Only aga/ana/apl still run Ubuntu 22.04 with /usr/bin/python3 = 3.10, which is
# what graph_model/.venv was built against; axa and every ix* node are 24.04 and
# ship 3.12 only, where the venv's site-packages are invisible.  Probed
#SBATCH --nodelist=aga,ana,apl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --job-name=kg_sizing
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/sizing_study_%j.out

set -euo pipefail
# Shared-filesystem venv, unlike the per-node /opt/deepops venv that .venv
# symlinks to.  It is NOT node-independent -- see pick_python.sh.
export VENV=/shared/workspace/povejmo/graph_model/.venv
PY=$(/shared/workspace/povejmo/gams_gtlm/data/lib/pick_python.sh)
if [ "$PY" != "$VENV/bin/python" ]; then
    export PYTHONPATH="$VENV/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
    echo "note: venv launcher unusable on $(hostname); using $PY + PYTHONPATH"
fi
export HF_HOME=/shared/workspace/povejmo/huggingface_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true   # no forks after tokenizer load -> safe & faster

echo "host: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}  start: $(date)"
# fail fast with a clear message if the env is wrong on this node
"$PY" -c "import numpy, transformers; print('env OK  numpy', numpy.__version__, 'transformers', transformers.__version__)"

# The k-hop input-sizing study: how big an extracted neighbourhood gets, in
# nodes and in tokens, over 8 build variants x 400 seeds.  It builds the graph
# in memory and throws it away -- it does NOT write a store; that is
# build/run_save.sbatch.
cd /shared/workspace/povejmo/gams_gtlm/data
"$PY" -m build.build_graph \
  --n-seeds 400 --max-hops 3 --workers "${SLURM_CPUS_PER_TASK}" --seed 42
echo "end: $(date)"
