#!/bin/bash
#SBATCH --job-name=openpi-input-diagnose
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:h200:8
#SBATCH --time=12:00:00
#SBATCH --output=/fsx/koshi.makihara/openpi/slurm-%x-%j.out
#SBATCH --error=/fsx/koshi.makihara/openpi/slurm-%x-%j.err
set -euo pipefail
DIAG_ARGS=("$@")
set --
source ~/.bashrc
source /fsx/koshi.makihara/miniforge3/bin/activate
conda activate airoapi
cd /fsx/koshi.makihara/openpi
export GIT_LFS_SKIP_SMUDGE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export RTYPE="${RTYPE:-rt_HF}"
srun .venv/bin/python -u scripts/diagnose_train_input.py \
  --output "outputs/input_diagnose_${SLURM_JOB_ID}" "${DIAG_ARGS[@]}"
