#!/usr/bin/env bash
#SBATCH --job-name=len_stats
#SBATCH --partition=frida
#SBATCH --account=povejmo
#SBATCH --nodelist=ana
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=data/logs/len_stats_%j.out
# Tokenization only -- no GPU.  Pinned to `ana` for the same reason the arms are:
# it is the node whose python this repo's .venv actually resolves against.
set -uo pipefail
cd /shared/workspace/povejmo/gams_gtlm || exit 1
.venv/bin/python -m train.analysis.length_stats "$@"
echo "exit: $?"
