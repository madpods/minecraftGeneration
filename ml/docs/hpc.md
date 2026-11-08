# Running on the OSU CoE HPC

Training on a laptop CPU takes ~40–120 s per epoch; a single A40 is far faster, and a Slurm job array runs a whole comparison study (many configurations × seeds) at once. Everything here runs from the repo root on the cluster.

The cluster facts below (Rocky 8/9 mix, module behaviour, partitions) were established in the `linearTesting` project; this project is simpler because it has **no Triton / CUDA-toolkit dependency** — torch's PyPI wheels bundle their own CUDA runtime, so the only module needed is `python`.

## One-time setup

**1. Push the code** (local machine) — the cluster clones from GitHub. The repo is public, so the clone needs no credentials.

**2. Clone and build the environment** (cluster login node):

```bash
ssh <onid>@submit-a.hpc.engr.oregonstate.edu
git clone -b claude/npc-building-generation-U3lMs https://github.com/madpods/minecraftGeneration.git
cd minecraftGeneration
bash scripts/slurm/setup_env.sh
```

`setup_env.sh` loads the newest `python/3.x` module (the system `python3` is 3.6), creates `.venv-gpu`, installs torch + deps from PyPI, and records the module and OS generation for job-time checks. Build on the login node (`submit-a`, Rocky 8): an EL8-built venv runs on the EL9 GPU nodes, but not the other way round (glibc).

**3. Copy the training data** (local machine, PowerShell, repo root). The extracted village data is Mojang's assets — it is git-ignored and must never be pushed to the public repo, so it goes over `scp` (~17 MB):

```powershell
scp -r ml/data/houses ml/data/extended <onid>@submit-a.hpc.engr.oregonstate.edu:minecraftGeneration/ml/data/
```

**4. Verify on a GPU node** (cluster):

```bash
sbatch scripts/slurm/verify.sbatch
squeue -u $USER                      # wait for it to finish
tail -n 20 logs/mcgen-verify-*.out   # expect "All 21/21 tests passed" and "VERIFY OK"
```

## Running a sweep

A sweep config is a text file with one run per line: `<run-name> <ml.train args...>`. `scripts/slurm/configs/village_v1.txt` is the first study — loss mode × dataset × style conditioning, 3 seeds each (24 runs), all evaluated on the same 17 held-out houses.

```bash
bash scripts/slurm/submit_sweep.sh scripts/slurm/configs/village_v1.txt
squeue -u $USER
```

Each task trains into `ml/checkpoints/<run>/` and then runs `ml.evaluate`, writing `ml/checkpoints/<run>/eval.txt` and the structure files to `ml/samples/<run>/`.

Jobs use `--partition=preempt --requeue` with `--constraint=el9&a40`. A preempted task restarts and continues from its newest checkpoint (`--resume auto`), so preemption costs little. To use another partition, override both partition and constraint, e.g. H100s on `dgxh` (cap of 8 concurrent GPUs):

```bash
SBATCH_EXTRA="--partition=dgxh --constraint=el9" bash scripts/slurm/submit_sweep.sh scripts/slurm/configs/village_v1.txt 8
```

## Collecting results

On the cluster, summarise (light enough for the login node):

```bash
module load $(cat .venv-gpu/modules.env) && source .venv-gpu/bin/activate
python -m ml.summarize ml/checkpoints
tar czf results_v1.tgz ml/samples ml/checkpoints/*/eval.txt
```

Then on the local machine:

```powershell
scp <onid>@submit-a.hpc.engr.oregonstate.edu:minecraftGeneration/results_v1.tgz .
tar xzf results_v1.tgz
```

## Notes

- **Home is 25 GB.** Each run keeps 2 epoch checkpoints + `best.pt` (~26 MB each, `--keep-checkpoints 2`), so a 24-run sweep is ~2 GB. Delete `ml/checkpoints/<run>/checkpoint_epoch_*.pt` once a study is summarised; `best.pt` is all evaluation needs.
- **`logs/` must exist before `sbatch`** — Slurm opens the output file itself at launch. `logs/.gitkeep` is tracked and `submit_sweep.sh` also creates it.
- **python runs directly, not under `srun`**, inside the job scripts: launch/teardown hangs at the job-step boundary were observed on this cluster when several array tasks shared a node.
- GPU runs use fp16 autocast; the CPU runs used fp32. Compare GPU numbers with GPU numbers.
