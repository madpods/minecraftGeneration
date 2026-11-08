#!/usr/bin/env bash
#SBATCH --job-name=mcgen
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --partition=preempt
#SBATCH --constraint=el9&a40
#SBATCH --requeue
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#
# One training run per array task; task N runs line N of $CONFIG (ignoring
# blank lines and # comments). Each line is:   <run-name>  <ml.train args...>
#
# Submit with the helper, which sizes the array and creates logs/ first:
#
#     bash scripts/slurm/submit_sweep.sh scripts/slurm/configs/village_v1.txt
#
# Preemption is cheap: --requeue restarts the task and `--resume auto` picks up
# from the run's newest checkpoint. Each run ends with ml.evaluate, whose
# report lands in ml/checkpoints/<run>/eval.txt.
#
# python runs directly, not under srun: --ntasks=1 gains nothing from a job
# step, and step launch/teardown hangs were observed on this cluster when
# several array tasks shared a node.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/slurm/env.sh

CONFIG="${CONFIG:?set CONFIG=path/to/configs.txt}"
IDX="${SLURM_ARRAY_TASK_ID:?must run as a job array}"

LINE="$(grep -vE '^[[:space:]]*(#|$)' "$CONFIG" | sed -n "$((IDX + 1))p")"
[ -n "$LINE" ] || { echo "ERROR: no line $IDX in $CONFIG" >&2; exit 1; }
RUN="${LINE%% *}"
ARGS="${LINE#* }"
OUT="ml/checkpoints/$RUN"
echo "task $IDX: $RUN"
echo "args: $ARGS"

# shellcheck disable=SC2086
python -m ml.train $ARGS --output-dir "$OUT" --resume auto

python -m ml.evaluate --checkpoint "$OUT/best.pt" --out "ml/samples/$RUN" --ascii 1 > "$OUT/eval.txt" 2>&1
echo "done: $RUN  (eval report: $OUT/eval.txt)"
