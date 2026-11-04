#!/bin/bash
#SBATCH --job-name=openpi-pickplace-hz30-h50
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:h200:8
#SBATCH --time=192:00:00
#SBATCH --output=/fsx/koshi.makihara/openpi/slurm-%x-%j.out
#SBATCH --error=/fsx/koshi.makihara/openpi/slurm-%x-%j.err
set -euo pipefail
KIND="$1"
set --
source ~/.bashrc
source /fsx/koshi.makihara/miniforge3/bin/activate
conda activate airoapi
cd /fsx/koshi.makihara/openpi
export GIT_LFS_SKIP_SMUDGE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CC=gcc CXX=g++ RTYPE=rt_HF
CONFIG=pi05_umi_original_dual_arm_relative_state_hz30_h50_bs512_100k_PickAndPlace_merged_gripper_224
case "$KIND" in
 old) DATA=/fsx/koshi.makihara/datasets_tasked/lerobot_v21_ph2_PickAndPlace_merged_gripper_224_train_mm_f30_s300_trim1s2s_gop12 ;;
 latest) DATA=/fsx/koshi.makihara/datasets_tasked/lerobot_v21_ph2_PickAndPlace_merged_gripper_224_yubi_20261022_gop12 ;;
 *) exit 2 ;;
esac
.venv/bin/python - "$DATA" "$KIND" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
assert json.loads((root/'meta/build_manifest.json').read_text())['state']=='complete'
if sys.argv[2]=='latest':
 assert json.loads((root/'meta/pipeline_complete.json').read_text())['all_video_gop_and_timestamps_verified']
else:
 assert json.loads((root/'recompression.json').read_text())['state']=='complete'
PY
# Different rate/horizon/anchor sampling: compute dedicated statistics, never reuse 10Hz assets.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/compute_norm_stats.py \
 --config-name "$CONFIG" --data-repo-id "$DATA" --downsample-anchor-frames True
mountpoint -q /opt/dlami/nvme
LOCAL_ROOT=$(mktemp -d "/opt/dlami/nvme/koshi-pickplace-${SLURM_JOB_ID}.XXXXXX")
cleanup() {
 case "$LOCAL_ROOT" in /opt/dlami/nvme/koshi-pickplace-"${SLURM_JOB_ID}".*) rm -rf -- "$LOCAL_ROOT" ;; esac
}
trap cleanup EXIT
LOCAL_DATA="$LOCAL_ROOT/$(basename "$DATA")"
mkdir "$LOCAL_DATA"
rsync -aL --stats "$DATA/" "$LOCAL_DATA/"
EXP="pickplace_${KIND}_hz30_h50_anchor_20261022_${SLURM_JOB_ID}"
echo "CONFIG=$CONFIG SOURCE=$DATA LOCAL_DATA=$LOCAL_DATA EXP=$EXP"
srun .venv/bin/python -u scripts/monitor_training_io.py --output "outputs/pickplace_hz30_h50_${SLURM_JOB_ID}" -- \
 .venv/bin/python -u scripts/train.py "$CONFIG" --exp-name "$EXP" \
 --data.repo-id "$LOCAL_DATA" --data.downsample-anchor-frames \
 --data.lerobot-table-query-mode grouped --data.lerobot-video-backend pyav \
 --batch-size 512 --num-workers 32 --num-train-steps 100000 --no-wandb-enabled
