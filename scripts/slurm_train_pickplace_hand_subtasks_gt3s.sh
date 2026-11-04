#!/bin/bash
#SBATCH --job-name=openpi-pickplace-hand-gt3s
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:h200:8
#SBATCH --time=192:00:00
#SBATCH --output=/fsx/koshi.makihara/openpi/slurm-%x-%j.out
#SBATCH --error=/fsx/koshi.makihara/openpi/slurm-%x-%j.err
set -euo pipefail
cd /fsx/koshi.makihara/openpi
export CONFIG_NAME=pi05_umi_original_dual_arm_relative_state_hz30_h50_bs512_100k_PickAndPlace_merged_gripper_224
export DATA_REPO_ID=/fsx/koshi.makihara/datasets/lerobot_v21_ph2_PickAndPlace_merged_gripper_224_yubi_20261022_gop12_hand_subtasks_gt3s
export EXP_NAME="pickplace_hand_subtasks_gt3s_hz30_h50_$(TZ=Asia/Tokyo date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
export RESUME=false
export COMPUTE_NORM_STATS=always
export WANDB_ENABLED=false
# Submit only once the download has finished. Check all expected data files exist.
/usr/bin/python3 - "$DATA_REPO_ID" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
info = json.loads((root / 'meta/info.json').read_text())
assert info['fps'] == 60, '30Hz configuration expects a 60fps dataset'
rows = [json.loads(line) for line in (root / 'meta/episodes.jsonl').open() if line.strip()]
assert len(rows) == info['total_episodes'] and rows
assert sum(row['length'] for row in rows) == info['total_frames']
for row in rows:
    index = row['episode_index']
    args = dict(episode_index=index, episode_chunk=index // info['chunks_size'])
    paths = [root / info['data_path'].format(**args)]
    paths += [root / info['video_path'].format(**args, video_key=key)
              for key, feature in info['features'].items() if feature['dtype'] == 'video']
    for path in paths:
        assert path.is_file() and path.stat().st_size > 0, f'Missing or empty file: {path}'
print(f'Preflight: {len(rows)} episodes, {info["total_frames"]} frames')
PY
exec bash scripts/pbs_train_openpi_umi_rt_HF.sh \
  --data.downsample-anchor-frames \
  --data.lerobot-table-query-mode grouped \
  --data.lerobot-video-backend pyav \
  --batch-size 512 --num-workers 32 --num-train-steps 100000 \
  --save-interval 1000 --keep-period 15000
