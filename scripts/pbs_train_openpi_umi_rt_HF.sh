#!/bin/bash
#SBATCH --job-name=openpi-umi-rt-hf
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=96
#SBATCH --time=192:00:00
#SBATCH --output=/fsx/koshi.makihara/openpi/slurm-%x-%j.out
#SBATCH --error=/fsx/koshi.makihara/openpi/slurm-%x-%j.err

set -euo pipefail

TRAIN_EXTRA_ARGS=("$@")
set --

source ~/.bashrc
source /fsx/koshi.makihara/miniforge3/bin/activate

conda activate airoapi

cd /fsx/koshi.makihara/openpi

# Choose one:
# - pi05_umi_original_right_h16_bs32_30k
# - pi05_umi_original_bimanual_h16_bs32_30k
# - pi05_umi_original_right_prev_current_third_slot_h16_bs32_30k_10hz
CONFIG_NAME="${CONFIG_NAME:-pi05_umi_original_right_h16_bs32_30k}"
RESUME="${RESUME:-false}"
if [[ "${RESUME}" == "true" && -z "${EXP_NAME:-}" ]]; then
  echo "[ERROR] RESUME=true requires EXP_NAME to identify the existing run directory." >&2
  exit 2
fi
EXP_NAME="${EXP_NAME:-${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)}"

export GIT_LFS_SKIP_SMUDGE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export RTYPE="${RTYPE:-rt_HF}"

echo "[INFO] CONFIG_NAME=${CONFIG_NAME}"
echo "[INFO] EXP_NAME=${EXP_NAME}"
echo "[INFO] DATA_REPO_ID=${DATA_REPO_ID:-}"
echo "[INFO] RESUME=${RESUME}"
echo "[INFO] WANDB_ENABLED=${WANDB_ENABLED:-false}"
echo "[INFO] COMPUTE_NORM_STATS=${COMPUTE_NORM_STATS:-auto}"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "[INFO] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"

STATS_ARGS=(--config-name "${CONFIG_NAME}")
TRAIN_ARGS=("${CONFIG_NAME}" --exp-name "${EXP_NAME}")
if [[ "${RESUME}" == "true" ]]; then
  CHECKPOINT_RUN_DIR="/fsx/koshi.makihara/openpi/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
  if [[ ! -d "${CHECKPOINT_RUN_DIR}" ]]; then
    echo "[ERROR] Resume checkpoint directory does not exist: ${CHECKPOINT_RUN_DIR}" >&2
    exit 2
  fi
  TRAIN_ARGS+=(--resume)
else
  TRAIN_ARGS+=(--overwrite)
fi
STATS_DOWNSAMPLE_ANCHOR_FRAMES="False"
for arg in "${TRAIN_EXTRA_ARGS[@]}"; do
  case "${arg}" in
    --data.downsample-anchor-frames)
      STATS_DOWNSAMPLE_ANCHOR_FRAMES="True"
      ;;
    --data.no-downsample-anchor-frames)
      STATS_DOWNSAMPLE_ANCHOR_FRAMES="False"
      ;;
  esac
done
STATS_ARGS+=(--downsample-anchor-frames "${STATS_DOWNSAMPLE_ANCHOR_FRAMES}")
if [[ -n "${DATA_REPO_ID:-}" ]]; then
  STATS_ARGS+=(--data-repo-id "${DATA_REPO_ID}")
  TRAIN_ARGS+=(--data.repo-id "${DATA_REPO_ID}")
fi
if [[ "${WANDB_ENABLED:-false}" != "true" ]]; then
  TRAIN_ARGS+=(--no-wandb-enabled)
fi

if [[ "${#TRAIN_EXTRA_ARGS[@]}" -gt 0 ]]; then
  TRAIN_ARGS+=("${TRAIN_EXTRA_ARGS[@]}")
fi

#xTRAIN_ARGS+=(--data..base-config.excluded-episode-paths /fsx/koshi.makihara/datasets_tasked/lerobot_v21_ph2_PickAndPlace_merged_gripper/data/chunk-002/episode_002167.parquet)

ASSET_ID="${DATA_REPO_ID:+$(basename "${DATA_REPO_ID}")}"
NORM_STATS_PATH=""
NORM_STATS_METADATA_PATH=""
if [[ -n "${ASSET_ID}" ]]; then
  NORM_STATS_PATH="/fsx/koshi.makihara/openpi/assets/${CONFIG_NAME}/${ASSET_ID}/norm_stats.json"
  NORM_STATS_METADATA_PATH="/fsx/koshi.makihara/openpi/assets/${CONFIG_NAME}/${ASSET_ID}/norm_stats_metadata.json"
fi

STATS_MODE_MATCHES="true"
if [[ -f "${NORM_STATS_PATH}" ]]; then
  if [[ -f "${NORM_STATS_METADATA_PATH}" ]]; then
    if ! SAVED_DOWNSAMPLE_ANCHOR_FRAMES=$(uv run python -c \
      'import json, sys; print(str(json.load(open(sys.argv[1]))["downsample_anchor_frames"]))' \
      "${NORM_STATS_METADATA_PATH}" 2>/dev/null); then
      STATS_MODE_MATCHES="false"
    elif [[ "${SAVED_DOWNSAMPLE_ANCHOR_FRAMES}" != "${STATS_DOWNSAMPLE_ANCHOR_FRAMES}" ]]; then
      STATS_MODE_MATCHES="false"
    fi
  elif [[ "${STATS_DOWNSAMPLE_ANCHOR_FRAMES}" == "True" ]]; then
    # Legacy stats without metadata were computed from all anchor frames.
    STATS_MODE_MATCHES="false"
  fi
fi

if [[ "${COMPUTE_NORM_STATS:-auto}" == "always" || ! -f "${NORM_STATS_PATH}" || "${STATS_MODE_MATCHES}" != "true" ]]; then
  if [[ -f "${NORM_STATS_PATH}" && "${STATS_MODE_MATCHES}" != "true" ]]; then
    echo "[INFO] Recomputing norm stats because anchor sampling mode changed."
  fi
  # Compute stats with a single GPU to avoid multi-GPU dataloader/video decode instability.
  CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py "${STATS_ARGS[@]}"
else
  echo "[INFO] Reusing existing norm stats: ${NORM_STATS_PATH}"
fi
srun uv run scripts/train.py "${TRAIN_ARGS[@]}"
