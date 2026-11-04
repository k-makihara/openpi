#!/bin/bash
#SBATCH --job-name=upload-wasabi
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
export AWS_ACCESS_KEY_ID=LQNGH1R84KO23AD9WVSQ
export AWS_SECRET_ACCESS_KEY=47X1eez2ZEV4qHEIzkpqlJWPfjKJ2KMaXnUCkBdl

aws s3 sync \
  /fsx/koshi.makihara/datasets_tasked/lerobot_v21_ph2_PickAndPlace_merged_gripper \
  s3://yubi-data/tmp/lerobot_v21_ph2_PickAndPlace_merged_gripper \
  --endpoint-url=https://s3.ap-northeast-1.wasabisys.com
