#!/bin/bash
#SBATCH --job-name=g2048
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=g2048_%j.out
#SBATCH --error=g2048_%j.err

# simple-2048 recreation - single-GPU SLURM job.
# One-time setup on the login node:
#   module load python/3.11 cuda   # adjust to your cluster's modules
#   python -m venv ~/venvs/g2048 && source ~/venvs/g2048/bin/activate
#   pip install -r requirements.txt   # torch+cuda per your cluster's docs
# Then: sbatch run_hpc.sbatch
# Requeue-friendly: the run auto-resumes if a latest.pt exists in DATA_DIR.

set -euo pipefail
source ~/venvs/g2048/bin/activate
cd "$SLURM_SUBMIT_DIR"

DATA_DIR=${DATA_DIR:-experiments}

# Original configuration = no flags at all:
#   python train.py
# Below keeps the exact original scale but auto-resumes across requeues.
LATEST=$(ls -t ${DATA_DIR}/*/latest.pt 2>/dev/null | head -1 || true)
RESUME=""
if [[ -n "${LATEST}" ]]; then
    RESUME="--resume ${LATEST}"
    echo "Resuming from ${LATEST}"
fi

srun python train.py \
    --num-envs 16384 \
    --total-timesteps 6767676767 \
    --data-dir "${DATA_DIR}" \
    --tag hpc \
    ${RESUME}

# Afterwards:
#   python eval.py --checkpoint ${DATA_DIR}/<run>/latest.pt --min-episodes 20000
