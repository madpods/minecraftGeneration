# Shared job environment for the OSU CoE HPC. Sourced (not executed) by the
# sbatch scripts after `cd` to the repo root:
#
#     source scripts/slurm/env.sh
#
# Loads the modules setup_env.sh built the venv with, checks the node can run
# that venv, and activates it. See ml/docs/hpc.md for the why.

VENV="${VENV:-.venv-gpu}"

command -v srun >/dev/null 2>&1 || module load slurm 2>/dev/null || true

# Replay the build-time modules (the CoE Lmod docs: modules used to build a
# python environment must be loaded again to run it). Here that is just the
# python module -- this project has no Triton/CUDA-toolkit dependency, torch's
# PyPI wheels bundle their own CUDA runtime.
if [ -s "$VENV/modules.env" ]; then
    # `if`, not `[ ] && ...`: under `set -e` a false test as the loop's last
    # command makes the whole loop return 1 and silently kills the job.
    while read -r m; do
        if [ -n "$m" ]; then
            module load "$m" || echo "WARN: could not load $m"
        fi
    done < "$VENV/modules.env"
else
    echo "ERROR: $VENV/modules.env missing or empty -- run: bash scripts/slurm/setup_env.sh" >&2
    exit 1
fi

# glibc is backward- but not forward-compatible, and the cluster mixes Rocky 8
# and 9: a venv built on EL8 (the submit-a login node) runs fine on EL9 GPU
# nodes, but an EL9-built venv dies on EL8 with "GLIBC_2.34 not found".
node_os="$(. /etc/os-release 2>/dev/null; echo "${ID:-unknown}${VERSION_ID%%.*}")"
build_os="$(cat "$VENV/build_os" 2>/dev/null || echo unknown)"
if [ "${build_os##*[!0-9]}" -gt "${node_os##*[!0-9]}" ] 2>/dev/null; then
    echo "ERROR: venv built on $build_os cannot run on this $node_os node." >&2
    echo "       Rebuild the venv on the EL8 login node (submit-a)." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONIOENCODING=utf-8
mkdir -p logs ml/checkpoints ml/samples

echo "host $(hostname)  os $node_os (venv built on $build_os)  partition ${SLURM_JOB_PARTITION:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
