#!/bin/bash
#SBATCH --job-name=openpi-fsx-local-bench
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:h200:8
#SBATCH --time=08:00:00
#SBATCH --output=/fsx/koshi.makihara/openpi/slurm-%x-%j.out
#SBATCH --error=/fsx/koshi.makihara/openpi/slurm-%x-%j.err
set -euo pipefail
source ~/.bashrc
source /fsx/koshi.makihara/miniforge3/bin/activate
conda activate airoapi
cd /fsx/koshi.makihara/openpi
export GIT_LFS_SKIP_SMUDGE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export CC=gcc CXX=g++ RTYPE=rt_HF
CONFIG=pi05_umi_original_dual_arm_relative_state_hz10_h50_bs512_100k_PickAndPlace_merged_gripper_224_train_mm_f30_s300_trim1s2s
SOURCE=/fsx/koshi.makihara/datasets_tasked/lerobot_v21_ph2_PickAndPlace_merged_gripper_224_train_mm_f30_s300_trim1s2s_gop12
ASSET=$(basename "$SOURCE")
OUT="$PWD/outputs/fsx_local_${SLURM_JOB_ID}"
mkdir "$OUT"
exec > >(tee -a "$OUT/job.log") 2>&1
hostname
findmnt -T /opt/dlami/nvme
df -h /opt/dlami/nvme
# Require a separate local block-device mount, never silently fall back to /tmp or FSx.
mountpoint -q /opt/dlami/nvme
[[ $(findmnt -n -o SOURCE -T /opt/dlami/nvme) == /dev/* ]]
test -s "assets/$CONFIG/$ASSET/norm_stats.json"
.venv/bin/python - "$SOURCE" <<'PY'
from pathlib import Path
import shutil
import sys
size = sum(p.stat().st_size for p in Path(sys.argv[1]).rglob('*') if p.is_file())
free = shutil.disk_usage('/opt/dlami/nvme').free
assert free > size + 20 * 1024**3, (size, free)
print(f'Dataset bytes={size}; local free bytes={free}', flush=True)
PY
LOCAL_ROOT=$(mktemp -d "/opt/dlami/nvme/koshi-storage-${SLURM_JOB_ID}.XXXXXX")
LOCAL_DATA="$LOCAL_ROOT/$ASSET"
mkdir "$LOCAL_DATA"
# Only remove this job's newly created directory, never an existing data location.
cleanup() {
 case "$LOCAL_ROOT" in /opt/dlami/nvme/koshi-storage-"${SLURM_JOB_ID}".*) rm -rf -- "$LOCAL_ROOT" ;; esac
}
trap cleanup EXIT
run_case() {
 local label="$1" dataset="$2"
 date -Is
 squeue -o '%.18i %.35j %.10T %.20R' > "$OUT/${label}_queue.txt"
 srun .venv/bin/python -u scripts/monitor_training_io.py --output "$OUT/${label}_io" -- \
  .venv/bin/python -u scripts/benchmark_storage_case.py \
  --config "$CONFIG" --dataset "$dataset" --norm-asset-id "$ASSET" \
  --warmup 128 --steps 512 --result "$OUT/${label}.json"
 cat "$OUT/${label}.json"
}
run_case fsx_before "$SOURCE"
.venv/bin/python - "$SOURCE" "$LOCAL_DATA" "$OUT" <<'PY'
import json
from pathlib import Path
import subprocess
import sys
import time
source, target, output = map(Path, sys.argv[1:])
start = time.monotonic()
with (output / 'rsync.log').open('w') as stream:
    subprocess.run(['rsync', '-aL', '--stats', str(source) + '/', str(target) + '/'],
                   stdout=stream, stderr=subprocess.STDOUT, check=True)
elapsed = time.monotonic() - start

def manifest(root):
    return {str(p.relative_to(root)): p.stat().st_size for p in root.rglob('*') if p.is_file()}
original, copied = manifest(source), manifest(target)
assert original == copied, 'Copied file paths/sizes differ'
size = sum(copied.values())
report = dict(seconds=elapsed, bytes=size, files=len(copied), effective_MiB_per_second=size / elapsed / 1024**2,
              source=str(source), destination=str(target),
              note='End-to-end rsync copy rate including metadata; not FSx service throughput. '
                   'rsync verifies transfers; file names and sizes checked afterward.')
(output / 'copy.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
PY
run_case local_nvme "$LOCAL_DATA"
run_case fsx_after "$SOURCE"
.venv/bin/python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys
out = Path(sys.argv[1])
results = {name: json.loads((out / f'{name}.json').read_text())
           for name in ('fsx_before', 'local_nvme', 'fsx_after')}
copy = json.loads((out / 'copy.json').read_text())
local = results['local_nvme']['seconds_per_step']
comparison = {}
for name in ('fsx_before', 'fsx_after'):
    fsx = results[name]['seconds_per_step']
    comparison[name] = dict(speedup=fsx/local,
        copy_break_even_steps=copy['seconds']/(fsx-local) if fsx > local else None)
report = dict(results=results, copy=copy, comparison=comparison,
    note='Synchronized GPU steps; includes page cache effects. Check queue snapshots for concurrent jobs. '
         'FSx repeat follows full copy and can benefit from warmed cache. No global cache flush performed.')
(out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
