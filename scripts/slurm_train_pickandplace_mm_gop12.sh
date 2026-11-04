#!/bin/bash
#SBATCH --job-name=openpi-pickplace-mm-gop12
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=96
#SBATCH --time=192:00:00
#SBATCH --output=/fsx/koshi.makihara/openpi/slurm-%x-%j.out
#SBATCH --error=/fsx/koshi.makihara/openpi/slurm-%x-%j.err

set -euo pipefail
export CONFIG_NAME=pi05_umi_original_dual_arm_relative_state_hz10_h50_bs512_100k_PickAndPlace_merged_gripper_224_train_mm_f30_s300_trim1s2s
export DATA_REPO_ID=/fsx/koshi.makihara/datasets_tasked/lerobot_v21_ph2_PickAndPlace_merged_gripper_224_train_mm_f30_s300_trim1s2s_gop12
export EXP_NAME="pickandplace_mm_gop12_$(TZ=Asia/Tokyo date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}"
export RESUME=false
export COMPUTE_NORM_STATS=auto
export WANDB_ENABLED=false

# Require completed conversion before making the dataset available to training.
/usr/bin/python3 - "$DATA_REPO_ID" <<'CHECK'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
report = json.loads((root / "recompression.json").read_text())
assert report["state"] == "complete", "GOP conversion is incomplete"
assert report["settings"]["gop"] == 12
assert report["all_timestamps_exact"]
info = json.loads((root / "meta/info.json").read_text())
assert report["videos"] == info["total_videos"], "Video count mismatch"
CHECK

# Enable grouped temporal table reads and forward optional training arguments.
exec bash /fsx/koshi.makihara/openpi/scripts/pbs_train_openpi_umi_rt_HF.sh \
  --data.lerobot-table-query-mode grouped "$@"
