#!/bin/bash
#
# Does a Blackwell host start the pyxis container and run THIS repo's venv?
#
#     sbatch train/run_container_probe.sbatch [NODELIST]
#
# Two things are being checked, and they fail differently:
#   * enroot starts the container at all -- ixb7's mellanox hook was failing on
#     /dev/infiniband/rdma_cm after its 2026-08-22 09:36 reboot (jobs 130295,
#     130297), which is a CURRENT-STATE fault and has to be re-tested rather than
#     assumed;
#   * the .venv (built against python3.10) imports inside it, and torch sees the
#     GPU -- the whole reason the container is there, since ixb* run 24.04/3.12.
#
#SBATCH --partition=frida
#SBATCH --account=povejmo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=sl_qa_probe
#SBATCH --output=/shared/workspace/povejmo/gams_gtlm/data/logs/sl_qa_probe_%j.out

set -uo pipefail
ROOT=/shared/workspace/povejmo/gams_gtlm
CONTAINER=/shared/workspace/povejmo/containers/transformers_deepspeed_latest.sqsh

echo "host: $(hostname)  start: $(date)"
srun --container-image="$CONTAINER" --container-mounts=/shared:/shared \
     env HOME="$HOME" PYTHONUNBUFFERED=1 \
     "$ROOT/.venv/bin/python" -c "
import platform, torch, transformers, peft
print('python      ', platform.python_version())
print('torch       ', torch.__version__)
print('transformers', transformers.__version__, ' peft', peft.__version__)
print('cuda        ', torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
print('matmul ok   ', float((x @ x).float().abs().mean()))
import gtlm.models as m
print('gtlm        ', m.GTLMGemma3ForCausalLM.__name__)
"
echo "srun exit: $?"
echo "end: $(date)"
