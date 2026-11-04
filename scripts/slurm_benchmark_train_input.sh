#!/bin/bash
#SBATCH --job-name=openpi-input-bench
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:h200:8
#SBATCH --time=48:00:00
#SBATCH --output=/fsx/koshi.makihara/openpi/slurm-%x-%j.out
#SBATCH --error=/fsx/koshi.makihara/openpi/slurm-%x-%j.err

set -euo pipefail
BENCH_ARGS=("$@")
set --

if [[ "${#BENCH_ARGS[@]}" -eq 0 ]]; then
	BENCH_ARGS=(
		--config pi05_umi_original_dual_arm_relative_state_hz10_h50_bs256_100k_PickAndPlace_merged_gripper
		--dataset /fsx/koshi.makihara/datasets_tasked/lerobot_v21_ph2_PickAndPlace_merged_gripper
		--batches 32 64 128 256 512 1024
		--workers 16 32 64
		--warmup 10
		--steps 50
		--timeout 10800
		--output "outputs/input_benchmark_pickandplace_$(date +%Y%m%d_%H%M%S)"
		--downsample-anchor-frames
		--excluded-episode-path /fsx/koshi.makihara/datasets_tasked/lerobot_v21_ph2_PickAndPlace_merged_gripper/data/chunk-002/episode_002167.parquet
	)
fi

source ~/.bashrc
source /fsx/koshi.makihara/miniforge3/bin/activate
conda activate airoapi
cd /fsx/koshi.makihara/openpi
export GIT_LFS_SKIP_SMUDGE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export RTYPE="${RTYPE:-rt_HF}"

srun .venv/bin/python -u scripts/benchmark_train_input.py "${BENCH_ARGS[@]}"
