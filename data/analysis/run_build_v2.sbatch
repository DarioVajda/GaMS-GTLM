#!/bin/bash
#SBATCH --partition=frida
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=04:00:00
#SBATCH --job-name=kg_build_v2
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/build_v2_%j.out

set -euo pipefail
# self-contained venv on the shared filesystem (identical on every node),
# unlike the per-node /opt/deepops venv that .venv symlinks to.
PY=/shared/workspace/povejmo/graph_model/.venv/bin/python
export HF_HOME=/shared/workspace/povejmo/huggingface_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true   # no forks after tokenizer load -> safe & faster

echo "host: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}  start: $(date)"
# fail fast with a clear message if the env is wrong on this node
"$PY" -c "import numpy, transformers; print('env OK  numpy', numpy.__version__, 'transformers', transformers.__version__)"

"$PY" /shared/workspace/povejmo/gams_gtlm/data/analysis/build_gtlm_graph.py \
  --kg-dir "/shared/workspace/povejmo/gams_gtlm/data/kg_raw/OntoLex DSB" \
  --out    "/shared/workspace/povejmo/gams_gtlm/data/analysis/results/results_v2.json" \
  --n-seeds 400 --max-hops 3 --workers "${SLURM_CPUS_PER_TASK}" --seed 42
echo "end: $(date)"
