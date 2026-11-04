#!/bin/bash
#SBATCH --job-name=openpi-mm-gop12
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/fsx/koshi.makihara/openpi/slurm-%x-%j.out
#SBATCH --error=/fsx/koshi.makihara/openpi/slurm-%x-%j.err
set -euo pipefail
source ~/.bashrc
source /fsx/koshi.makihara/miniforge3/bin/activate
conda activate airoapi
cd /fsx/koshi.makihara/openpi
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
srun .venv/bin/python -u scripts/reencode_lerobot_gop.py \
  --source /fsx/koshi.makihara/datasets_tasked/lerobot_v21_ph2_PickAndPlace_merged_gripper_224_train_mm_f30_s300_trim1s2s \
  --output /fsx/koshi.makihara/datasets_tasked/lerobot_v21_ph2_PickAndPlace_merged_gripper_224_train_mm_f30_s300_trim1s2s_gop12 \
  --gop 12 --workers 8
