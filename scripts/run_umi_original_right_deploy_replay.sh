#!/usr/bin/env bash
set -eo pipefail

source /home/makihara/miniconda3/etc/profile.d/conda.sh
conda activate airoapi

cd /home/makihara/openpi
exec uv run scripts/test_umi_original_right_deploy_replay.py "$@"
