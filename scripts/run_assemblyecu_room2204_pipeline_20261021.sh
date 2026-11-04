#!/bin/bash
# Durable preparation supervisor. Argument: this pipeline's held Slurm training job.
set -euo pipefail
cd /fsx/koshi.makihara/openpi
JOB_ID="$1"
STATUS="$PWD/outputs/assemblyecu_room2204_pipeline_20261021.status"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec 9>"$PWD/outputs/assemblyecu_room2204_pipeline_20261021.lock"
flock -n 9
on_error() {
  echo "FAILED: preparation or release failed; training job $JOB_ID remains held. Check pipeline log." > "$STATUS"
}
trap on_error ERR
echo "PREPARING: training job $JOB_ID held pending verified dataset" > "$STATUS"
.venv/bin/python -u scripts/prepare_assemblyecu_room2204_20261021.py
echo "DATASET_COMPLETE: releasing training job $JOB_ID" > "$STATUS"
scontrol release "$JOB_ID"
echo "TRAINING_QUEUED: job $JOB_ID; dataset fully verified" > "$STATUS"
