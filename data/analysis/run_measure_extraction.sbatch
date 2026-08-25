#!/bin/bash
#
# R4 -- measure the D3 entity extractor against the generated questions.
#
#     sbatch analysis/run_measure_extraction.sbatch                     # defaults
#     sbatch analysis/run_measure_extraction.sbatch --prompt ../prompt_v2.txt
#     sbatch analysis/run_measure_extraction.sbatch --model cjvt/GaMS-9B-Instruct
#
# One A100-80GB holds GaMS3-12B-Instruct in bf16 (~24 GB) with room for a batch
# of 32.  ana/aga run Ubuntu 22.04 / python3.10 natively, which is what the venv
# is built against -- see data/README.md "Reproduce".
#
#SBATCH --partition=frida
#SBATCH --account=lkm
#SBATCH --nodelist=ana
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=extract_eval
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/extract_%j.out

set -euo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
export HF_HOME=/shared/workspace/povejmo/huggingface_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY=$("$ROOT/data/lib/pick_python.sh")

echo "host: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$ROOT/data"

"$PY" -m analysis.measure_extraction \
    --out "$ROOT/data/analysis/results/extraction_v1.json" \
    --dump "$ROOT/data/analysis/results/extraction_v1_items.jsonl" \
    "$@"

echo "end: $(date)"
