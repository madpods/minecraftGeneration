#!/usr/bin/env bash
#SBATCH --job-name=mcgen-verify
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=preempt
#SBATCH --constraint=el9&a40
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#
# First run on a GPU node: smoke tests + a 2-epoch training run on CUDA
# (exercises fp16 autocast and the GPU code paths the CPU box never hit).
#
#     sbatch scripts/slurm/verify.sbatch
#     tail -f logs/mcgen-verify-<jobid>.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/slurm/env.sh

python -m ml.tests.smoke_test

DATA="${DATA:-ml/data/houses}"
if ls "$DATA"/*.pt >/dev/null 2>&1; then
    python -m ml.train --data-dir "$DATA" --output-dir "ml/checkpoints/_verify" \
        --epochs 2 --repeat 2 --loss split --split-seed 14 --keep-checkpoints 1
    echo "VERIFY OK: tests passed and CUDA training ran on $DATA"
else
    echo "VERIFY PARTIAL: tests passed, but no data in $DATA -- scp ml/data/ over first (see ml/docs/hpc.md)"
fi
