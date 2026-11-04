#!/bin/bash
#SBATCH --job-name=openpi-assemblyecu-curate
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/fsx/koshi.makihara/openpi/slurm-%x-%j.out
#SBATCH --error=/fsx/koshi.makihara/openpi/slurm-%x-%j.err
set -euo pipefail
source ~/.bashrc
source /fsx/koshi.makihara/miniforge3/bin/activate
conda activate airoapi
cd /fsx/koshi.makihara/openpi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export FFMPEG=/fsx/koshi.makihara/miniforge3/envs/airoapi/bin/ffmpeg
export FFPROBE=/fsx/koshi.makihara/miniforge3/envs/airoapi/bin/ffprobe
srun .venv/bin/python -u scripts/prepare_assemblyecu_20261017.py
