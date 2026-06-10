#!/bin/sh
#PBS -q rt_HF
#PBS -l select=1:ngpus=8
#PBS -l walltime=24:00:00
#PBS -P gag51454
#PBS -j oe
#PBS -k oed

set -eu

cd "${PBS_O_WORKDIR}"

source /etc/profile.d/modules.sh
module load nvhpc/24.9
module load hpcx/2.20
source ~/.bashrc

conda activate airoapi

cd /groups/gag51454/workspace_makihara/openpi

# Choose one:
# - pi05_umi_original_right_h16_bs32_30k
# - pi05_umi_original_bimanual_h16_bs32_30k
# - pi05_umi_original_right_prev_current_third_slot_h16_bs32_30k_10hz
CONFIG_NAME="${CONFIG_NAME:-pi05_umi_original_right_h16_bs32_30k}"
EXP_NAME="${EXP_NAME:-${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)}"

export GIT_LFS_SKIP_SMUDGE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

echo "[INFO] CONFIG_NAME=${CONFIG_NAME}"
echo "[INFO] EXP_NAME=${EXP_NAME}"

# Compute stats with a single GPU to avoid multi-GPU dataloader/video decode instability.
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
uv run scripts/train.py "${CONFIG_NAME}" --exp-name "${EXP_NAME}" --overwrite
