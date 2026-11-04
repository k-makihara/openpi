#!/bin/bash
#SBATCH --job-name=openpi-assembly-room2204
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:h200:8
#SBATCH --time=192:00:00
#SBATCH --output=/fsx/koshi.makihara/openpi/slurm-%x-%j.out
#SBATCH --error=/fsx/koshi.makihara/openpi/slurm-%x-%j.err
set -euo pipefail
export CONFIG_NAME=pi05_umi_original_dual_arm_relative_state_hz30_h32_bs512_100k_assemblyecu_all_tasked_20260917_20260918_20261002_merged
export DATA_REPO_ID=/fsx/koshi.makihara/datasets_tasked/lerobot_v21_assemblyecu_room2204_6days_plus_5sources_20261021_224_gop12
export EXP_NAME="assemblyecu_room2204_6days_plus_5sources_$(TZ=Asia/Tokyo date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
export RESUME=false
export COMPUTE_NORM_STATS=auto
export WANDB_ENABLED=false
/usr/bin/python3 - "$DATA_REPO_ID" <<'CHECK'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
report = json.loads((root / 'meta/pipeline_complete.json').read_text())
info = json.loads((root / 'meta/info.json').read_text())
assert report['state'] == 'complete' and report['all_video_gop_and_timestamps_verified']
assert info['total_frames'] == report['frames'] == 3514631
assert info['total_episodes'] == report['episodes'] == 8498
assert len(report['sources']) == 7
CHECK
exec bash /fsx/koshi.makihara/openpi/scripts/pbs_train_openpi_umi_rt_HF.sh \
  --data.lerobot-table-query-mode grouped --data.lerobot-video-backend pyav \
  --batch-size 512 --num-workers 32 --num-train-steps 100000
