#!/usr/bin/env bash
# Submit a sweep config as a Slurm job array sized to its line count.
#
#     bash scripts/slurm/submit_sweep.sh scripts/slurm/configs/village_v1.txt [throttle]
#
# throttle = max concurrent tasks (default 12). preempt has no GPU cap; use
# 8 on dgxh, 2 on share/ampere (pass --partition via SBATCH_EXTRA, e.g.
#     SBATCH_EXTRA="--partition=dgxh --constraint=el9" bash scripts/slurm/submit_sweep.sh ...).
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="${1:?usage: $0 <config.txt> [throttle]}"
THROTTLE="${2:-12}"
N="$(grep -cvE '^[[:space:]]*(#|$)' "$CONFIG" || true)"
[ "$N" -gt 0 ] || { echo "no runs in $CONFIG" >&2; exit 1; }

# Slurm opens --output files itself at task launch; a missing logs/ fails
# every task instantly with no explanation.
mkdir -p logs

echo "submitting $N runs from $CONFIG (max $THROTTLE at once)"
# shellcheck disable=SC2086
sbatch --array="0-$((N - 1))%$THROTTLE" --export=ALL,CONFIG="$CONFIG" ${SBATCH_EXTRA:-} scripts/slurm/sweep.sbatch
