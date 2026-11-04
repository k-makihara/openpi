#!/bin/bash
#SBATCH --job-name=openpi-resume-io
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:h200:8
#SBATCH --time=192:00:00
#SBATCH --output=/fsx/koshi.makihara/openpi/slurm-%x-%j.out
#SBATCH --error=/fsx/koshi.makihara/openpi/slurm-%x-%j.err
set -euo pipefail
RUN_KIND="$1"
set --
source ~/.bashrc
source /fsx/koshi.makihara/miniforge3/bin/activate
conda activate airoapi
cd /fsx/koshi.makihara/openpi
export GIT_LFS_SKIP_SMUDGE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export CC="${CC:-gcc}" CXX="${CXX:-g++}" RTYPE="${RTYPE:-rt_HF}"
case "$RUN_KIND" in
pickplace)
 CONFIG_NAME=pi05_umi_original_dual_arm_relative_state_hz10_h50_bs512_100k_PickAndPlace_merged_gripper_224_train_mm_f30_s300_trim1s2s
 EXP_NAME=pickandplace_mm_gop12_20261017_144059_343
 DATA_REPO_ID=/fsx/koshi.makihara/datasets_tasked/lerobot_v21_ph2_PickAndPlace_merged_gripper_224_train_mm_f30_s300_trim1s2s_gop12
 ;;
assembly)
 CONFIG_NAME=pi05_umi_original_dual_arm_relative_state_hz30_h32_bs512_100k_assemblyecu_all_tasked_20260917_20260918_20261002_merged
 EXP_NAME=assemblyecu_yubi_20261018_050645_345
 DATA_REPO_ID=/fsx/koshi.makihara/datasets_tasked/lerobot_v21_assemblyecu_5sources_yubi_20261017_224_gop12
 ;;
*) echo "Unknown run: $RUN_KIND" >&2; exit 2 ;;
esac
test -d "checkpoints/$CONFIG_NAME/$EXP_NAME"
test -s "assets/$CONFIG_NAME/$(basename "$DATA_REPO_ID")/norm_stats.json"
STORAGE_OUT="$PWD/outputs/storage_resume_${SLURM_JOB_ID}"
cleanup_local() {
 if [[ -f "$STORAGE_OUT/local_root.txt" ]]; then
  local root
  root=$(cat "$STORAGE_OUT/local_root.txt")
  case "$root" in /opt/dlami/nvme/koshi-resume-"${SLURM_JOB_ID}"-*) rm -rf -- "$root" ;; esac
 fi
}
trap cleanup_local EXIT
DATA_REPO_ID=$(.venv/bin/python scripts/select_resume_storage.py \
 --report "$PWD/outputs/fsx_local_348/comparison.json" \
 --source "$DATA_REPO_ID" --checkpoints "$PWD/checkpoints/$CONFIG_NAME/$EXP_NAME" \
 --output "$STORAGE_OUT" --job-id "$SLURM_JOB_ID")
cat "$STORAGE_OUT/decision.json"
echo "Resuming $CONFIG_NAME / $EXP_NAME"
echo "Training log and IO records: outputs/resume_io_${SLURM_JOB_ID}/"
srun .venv/bin/python -u scripts/monitor_training_io.py --output "outputs/resume_io_${SLURM_JOB_ID}" -- \
  .venv/bin/python -u scripts/train.py "$CONFIG_NAME" --exp-name "$EXP_NAME" --resume \
  --data.repo-id "$DATA_REPO_ID" --data.lerobot-table-query-mode grouped --data.lerobot-video-backend pyav \
  --batch-size 512 --num-workers 32 --num-train-steps 100000 --no-wandb-enabled
