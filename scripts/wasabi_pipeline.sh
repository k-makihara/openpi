#!/usr/bin/env bash
#
# Usage:
#   bash wasabi_pipeline.sh [config-file]          # preview only
#   bash wasabi_pipeline.sh [config-file] --run    # execute pipeline
#
# Pipeline:
#   1. Check and download LeRobot datasets from Wasabi.
#   2. Merge datasets and annotate task instructions.
#   3. Transform gripper values for OpenPI training.
#   4. Add a temporary OpenPI config.
#   5. Submit a Slurm job that trains and uploads the checkpoint to Wasabi.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
#SCRIPT_DIR=/fsx/koshi.makihara/openpi/scripts
CONF="$SCRIPT_DIR/wasabi_pipeline.conf"
EXECUTE=false

die() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  echo "Usage: bash $0 [config-file] [--run] (default: preview only)"
}

run() {
  printf '%q ' "$@"
  printf '\n'

  if "$EXECUTE"; then
    "$@"
  fi
}

write_job() {
  printf '#!/usr/bin/env bash\n'
  printf 'set -euo pipefail\n'
  printf '\n'
  printf 'export CONFIG_NAME=%q\n' "$CONFIG_NAME"
  printf 'export EXP_NAME=%q\n' "$EXP_NAME"
  printf 'export DATA_REPO_ID=%q\n' "$TRAIN_DATA"
  printf 'export COMPUTE_NORM_STATS=always\n'
  printf '\n'
  printf 'bash %q\n' "$OPENPI/scripts/pbs_train_openpi_umi_rt_HF.sh"
  printf '\n'
  printf 'test -d %q\n' "$CHECKPOINT/params"
  printf 'test -d %q\n' "$CHECKPOINT/assets"
  printf '\n'
  printf 'aws s3 sync %q %q --endpoint-url %q\n' \
    "$CHECKPOINT/" \
    "$REMOTE_MODEL/$((NUM_TRAIN_STEPS - 1))/" \
    "$ENDPOINT"
  printf 'aws s3 sync %q %q --endpoint-url %q\n' \
    "$RUN_DIR/repro/" \
    "$REMOTE_MODEL/repro/" \
    "$ENDPOINT"
  printf '\n'
  printf 'echo %q\n' "Completed: $REMOTE_MODEL"
}

for arg in "$@"; do
  case "$arg" in
    --run)
      EXECUTE=true
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      die "Unknown option: $arg"
      ;;
    *)
      CONF="$arg"
      ;;
  esac
done

source "$CONF"

# ---- Validate settings ------------------------------------------------------

[[ "$TASK_MODE" == single || "$TASK_MODE" == tasks-txt ]] || die 'TASK_MODE must be single or tasks-txt'

if [[ "$TASK_MODE" == single ]]; then
  [[ -n "${TASK//[[:space:]]/}" ]] || die 'TASK must not be empty'
fi

[[ "$NUM_TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]] || die 'Invalid NUM_TRAIN_STEPS'
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die 'Invalid BATCH_SIZE'
[[ "$CONFIG_PREFIX" =~ ^[a-zA-Z0-9_-]+$ ]] || die 'Invalid CONFIG_PREFIX'
[[ "$REMOTE_ROOT" == s3://* && "$MODEL_ROOT" == s3://* ]] || die 'Remote roots must be s3:// paths'
[[ ${#DATASETS[@]} -gt 0 ]] || die 'DATASETS is empty'

for name in "${DATASETS[@]}"; do
  [[ "$name" =~ ^[a-zA-Z0-9_-]+$ ]] || die "Invalid dataset name: $name"
done

PYTHON="$OPENPI/.venv/bin/python"
[[ -x "$PYTHON" ]] || die "Missing Python: $PYTHON"

# ---- Derived run paths ------------------------------------------------------

RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$_${RANDOM}"
CONFIG_NAME="${CONFIG_PREFIX}_${RUN_ID}"
EXP_NAME="$CONFIG_NAME"
RUN_DIR="$WORK_ROOT/$RUN_ID"
MERGED="$RUN_DIR/merged"
TRAIN_DATA="$RUN_DIR/gripper"
CHECKPOINT="$OPENPI/checkpoints/$CONFIG_NAME/$EXP_NAME/$((NUM_TRAIN_STEPS - 1))"
REMOTE_MODEL="${MODEL_ROOT%/}/$CONFIG_NAME/$EXP_NAME"

echo "Run: $RUN_ID"
echo "Checkpoint: $CHECKPOINT"
echo "Upload: $REMOTE_MODEL/$((NUM_TRAIN_STEPS - 1))/"

if "$EXECUTE"; then
  command -v aws >/dev/null || die 'aws not found'
  command -v sbatch >/dev/null || die 'sbatch not found'
  [[ ! -e "$RUN_DIR" ]] || die "Run directory already exists: $RUN_DIR"
fi

# ---- Download input data ----------------------------------------------------

# Validate all remote datasets before starting downloads.
for name in "${DATASETS[@]}"; do
  run aws s3 ls "${REMOTE_ROOT%/}/$name/meta/info.json" --endpoint-url "$ENDPOINT"
done

DATASET_ARGS=()

for name in "${DATASETS[@]}"; do
  run aws s3 sync "${REMOTE_ROOT%/}/$name/" "$RAW_ROOT/$name/" --endpoint-url "$ENDPOINT"
  DATASET_ARGS+=(--dataset-dir "$RAW_ROOT/$name")

  if "$EXECUTE"; then
    [[ -f "$RAW_ROOT/$name/meta/info.json" ]] || die "Missing info.json: $name"

    if [[ "$TASK_MODE" == tasks-txt ]]; then
      [[ -f "$RAW_ROOT/$name/tasks.txt" ]] || die "Missing tasks.txt: $name"
    fi
  fi
done

# ---- Prepare training dataset ----------------------------------------------

TASK_ARGS=(--task-mode "$TASK_MODE")
[[ "$TASK_MODE" != single ]] || TASK_ARGS+=(--task "$TASK")

run "$PYTHON" "$OPENPI/scripts/merge_lerobot_task_datasets.py" \
  "${DATASET_ARGS[@]}" \
  "${TASK_ARGS[@]}" \
  --out-dir "$MERGED"

run "$PYTHON" "$OPENPI/scripts/transform_lerobot_grippers.py" \
  --dataset-dir "$MERGED" \
  --out-dir "$TRAIN_DATA"

run "$PYTHON" "$OPENPI/scripts/add_openpi_config.py" \
  --base-config "$BASE_CONFIG" \
  --new-config "$CONFIG_NAME" \
  --repo-id "$TRAIN_DATA" \
  --num-train-steps "$NUM_TRAIN_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --no-wandb-enabled \
  --write

# ---- Submit training and upload job ----------------------------------------

JOB_FILE="$RUN_DIR/train_and_upload.sh"
SBATCH_ARGS=(
  --parsable
  --export=ALL
  --job-name=openpi-wasabi
  --partition="$PARTITION"
  --nodes=1
  --ntasks=1
  --gres="$GRES"
  --time="$TIME_LIMIT"
  --output="$RUN_DIR/slurm-%j.out"
  --error="$RUN_DIR/slurm-%j.err"
)

if "$EXECUTE"; then
  mkdir -p "$RUN_DIR/repro"

  cp "$OPENPI/src/openpi/training/config.py" "$RUN_DIR/repro/config.py"
  cp "$OPENPI/src/openpi/policies/umi_original_policy.py" "$RUN_DIR/repro/umi_original_policy.py"

  printf 'config=%s\nexperiment=%s\ntrain_data=%s\ncheckpoint=%s\nremote=%s\n' \
    "$CONFIG_NAME" \
    "$EXP_NAME" \
    "$TRAIN_DATA" \
    "$CHECKPOINT" \
    "$REMOTE_MODEL" \
    > "$RUN_DIR/repro/run.txt"

  write_job > "$JOB_FILE"

  JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" "$JOB_FILE")
  printf '%s\n' "$JOB_ID" > "$RUN_DIR/job_id.txt"

  echo "Submitted job: $JOB_ID"
  echo "Logs: $RUN_DIR/slurm-${JOB_ID}.out / .err"
  echo "Status: sacct -j $JOB_ID --format=JobID,State,ExitCode,Elapsed"
else
  echo '--- Generated Slurm job (preview) ---'
  write_job
  run sbatch "${SBATCH_ARGS[@]}" "$JOB_FILE"
  echo 'Preview only. Add --run to download, prepare data, and submit training.'
fi
