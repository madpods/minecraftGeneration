#!/usr/bin/env bash
# One-time environment setup on the OSU CoE HPC. Run on the LOGIN node
# (submit-a), from the repo root:
#
#     bash scripts/slurm/setup_env.sh
#
# Builds a venv from the cluster's python module and installs CUDA torch from
# PyPI. pip runs here because compute nodes may have no outbound network, and
# building here (EL8) produces a venv that runs on the EL9 GPU nodes -- the
# safe direction for glibc. Then verify on a GPU node with:
#
#     sbatch scripts/slurm/verify.sbatch
set -euo pipefail
cd "$(dirname "$0")/../.."

VENV="${VENV:-.venv-gpu}"

command -v srun >/dev/null 2>&1 || module load slurm 2>/dev/null || true

# `python3` on PATH is 3.6.8, so the module is mandatory. `sort -V`, not a
# lexical sort, or python/3.9 beats python/3.13. Every pipeline ends in
# `|| true`: under `set -e` + pipefail, a grep that matches nothing would
# otherwise kill the script with no message.
if [ -z "${MODULE_PYTHON:-}" ]; then
    MODULE_PYTHON="$(module -t avail python 2>&1 | tr -d '\r' | grep -E '^python/3\.[0-9]+$' | sort -V | tail -1 || true)"
fi
if [ -z "$MODULE_PYTHON" ]; then
    echo "ERROR: no python/3.x module found; set MODULE_PYTHON=python/3.13 and retry" >&2
    exit 1
fi
echo "loading $MODULE_PYTHON"
module load "$MODULE_PYTHON"
python3 --version

if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip
# torch from PyPI: its Linux wheels are CUDA-enabled and bundle the runtime.
"$VENV/bin/python" -m pip install torch numpy tqdm nbtlib flask

# Records replayed by scripts/slurm/env.sh at job time
echo "$MODULE_PYTHON" > "$VENV/modules.env"
( . /etc/os-release; echo "${ID}${VERSION_ID%%.*}" ) > "$VENV/build_os"
mkdir -p logs

"$VENV/bin/python" -c "import torch, nbtlib; print('torch', torch.__version__, '| built for CUDA', torch.version.cuda)"
echo "venv ready: $VENV (modules: $(cat "$VENV/modules.env"), built on $(cat "$VENV/build_os"))"
echo "next: sbatch scripts/slurm/verify.sbatch"
