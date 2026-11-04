#!/bin/bash
#SBATCH --job-name=openpi-resize-224
#SBATCH --partition=gpusv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/fsx/koshi.makihara/openpi/slurm-%x-%j.out
#SBATCH --error=/fsx/koshi.makihara/openpi/slurm-%x-%j.err

# Stop the old conversion before submitting: two processes must not write the same output.
set -euo pipefail
RESIZE_EXTRA_ARGS=("$@")
set --
source /fsx/koshi.makihara/miniforge3/bin/activate airoapi
cd /fsx/koshi.makihara/openpi
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun --cpu-bind=cores .venv/bin/python -u scripts/resize_lerobot_videos.py \
  --source /fsx/koshi.makihara/datasets_tasked/lerobot_v21_ph2_PickAndPlace_merged_gripper \
  --output /fsx/koshi.makihara/datasets_tasked/lerobot_v21_ph2_PickAndPlace_merged_gripper_224 \
  --workers "${SLURM_CPUS_PER_TASK:-32}" \
  --exclude-episode-index 2167 \
  --resume \
  "${RESIZE_EXTRA_ARGS[@]}"
