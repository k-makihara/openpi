#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/train_pi05_umi_handover_20hz.sh [EXP_NAME]
#
# Notes:
# - Config: pi05_umi_handover_20hz
# - Action chunk sampling is downsampled from 60Hz to 20Hz via action_sample_stride=3.

EXP_NAME="${1:-pi05_umi_handover_20hz_run}"

export GIT_LFS_SKIP_SMUDGE=1
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

# Compute normalization stats for this config (safe to rerun).
python scripts/compute_norm_stats.py --config-name pi05_umi_handover_20hz

# Start training.
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
python scripts/train.py pi05_umi_handover_20hz --exp-name="${EXP_NAME}" --overwrite
