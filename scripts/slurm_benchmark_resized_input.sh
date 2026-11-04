#!/bin/bash
#SBATCH --job-name=openpi-input-224-bench
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:h200:8
#SBATCH --time=192:00:00
#SBATCH --output=/fsx/koshi.makihara/openpi/slurm-%x-%j.out
#SBATCH --error=/fsx/koshi.makihara/openpi/slurm-%x-%j.err

set -euo pipefail
BENCH_ARGS=("$@")
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

# Episode 2167 was already removed and IDs renumbered: do NOT exclude it again.
# State/action values are unchanged; use the same normalization as the original benchmark.
srun .venv/bin/python -u scripts/benchmark_input_variants.py \
  --dataset /fsx/koshi.makihara/datasets_tasked/lerobot_v21_ph2_PickAndPlace_merged_gripper_224 \
  --norm-asset-id lerobot_v21_ph2_PickAndPlace_merged_gripper \
  --batches 128 256 512 1024 \
  --workers 8 16 32 64 \
  --backends pyav \
  --table-query-modes grouped \
  --warmup 128 \
  --steps 256 \
  --timeout 28800 \
  "${BENCH_ARGS[@]}"
