#!/bin/bash
#SBATCH --partition=frida
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --job-name=kg_neigh_sem
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/job_sem_%j.out

set -euo pipefail
cd /shared/workspace/povejmo/gams_gtlm
source .venv/bin/activate
export HF_HOME=/shared/workspace/povejmo/huggingface_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "host: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}  start: $(date)"
python data/analysis/analyze_neighborhoods.py \
  --kg-dir "/shared/workspace/povejmo/gams_gtlm/data/kg_raw/OntoLex DSB" \
  --out    "/shared/workspace/povejmo/gams_gtlm/data/analysis/results/results_semcore.json" \
  --n-seeds 400 --max-hops 3 --workers "${SLURM_CPUS_PER_TASK}" --seed 42 --no-decomp
echo "end: $(date)"
