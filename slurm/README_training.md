# BDT training — runbook

## Decision rubric

- **Notebook:** debugging, plotting saved outputs, eyeballing ≤ 1M rows.
- **`srun --pty`:** ad hoc full-statistics run; > 10 min or > 8 GB.
- **`sbatch` via `submit_training_bdt.sh`:** the canonical Week-4 run.

---

## Before you start

Run the preflight once.  Address every FAIL line before submitting.

```bash
cd ~/CLAS/SULI/suli2026_pid
./slurm/check_farm_access.sh
```

Each check prints a `→ Fix:` hint on failure.  Exit 0 = all clear.

---

## Tier 1 — JupyterHub (inspection only)

JupyterHub's default kernel is **not** the `suli2026_pid` conda env.  Do
not `pip install` into it.  Use JupyterHub only to:

- Open and inspect output CSVs and PNGs from a completed run.
- Open notebooks that load the parquet files from `/volatile/` for quick
  ad-hoc exploration.

For anything that runs Python code against the training scripts, use Tier 2
or Tier 3.

---

## Tier 2 — `srun --pty` (interactive)

Use for the full-statistics pipeline when you want to watch output scroll in
real time, or for any smoke test with `--max-files 2`.

```bash
# 1. Allocate a compute node
srun --pty --account=clas12 --time=2:00:00 --mem=16G --cpus-per-task=8 bash

# 2. Set up environment (do this every time on a fresh node)
export TMPDIR=/tmp
source /etc/profile.d/modules.sh        # makes `module` available
module use /cvmfs/oasis.opensciencegrid.org/jlab/scicomp/sw/el9/modulefiles
module use /scigroup/cvmfs/hallb/clas12/sw/modulefiles
module use /cvmfs/oasis.opensciencegrid.org/jlab/hallb/clas12/sw/modulefiles
module load clas12

# Source the conda hook (choose the path that exists for your install):
source /work/clas12/$USER/miniconda3/etc/profile.d/conda.sh
# or: source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate suli2026_pid

# 3. Go to repo root — required for script imports
cd ~/CLAS/SULI/suli2026_pid

# 4. Run the three-step pipeline
python scripts/training/build_dataset.py \
    --mc-dir /volatile/clas12/$USER/SULI/mc_v01 \
    --split-dir slurm \
    --outdir /volatile/clas12/$USER/SULI/datasets/v01 \
    --features-file scripts/training/feature_list.txt \
    --p-max 3.0 --overwrite

python scripts/training/train_bdt.py \
    --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \
    --outdir /volatile/clas12/$USER/SULI/models/v01 --overwrite

python scripts/training/evaluate.py \
    --model /volatile/clas12/$USER/SULI/models/v01/model.joblib \
    --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \
    --outdir /volatile/clas12/$USER/SULI/eval/v01 --overwrite
```

For a smoke test (2 files per split, runs in ~1–2 min), add
`--max-files 2` to `build_dataset.py`.

---

## Tier 3 — `sbatch` (canonical run)

Use for the one canonical Week-4 full-statistics BDT training run.
`build_dataset.py` and `evaluate.py` run interactively (Tier 2); only
`train_bdt.py` goes through sbatch.

```bash
cd ~/CLAS/SULI/suli2026_pid

# Submit
./slurm/submit_training_bdt.sh \
    --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \
    --model-dir   /volatile/clas12/$USER/SULI/models/v01

# Monitor
squeue -u $USER
tail -f /farm_out/$USER/suli/training_bdt_<jobid>.out

# Cancel if needed
scancel <jobid>

# After completion — check accounting
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS

# Copy model off volatile before the 2-week purge
cp -r /volatile/clas12/$USER/SULI/models/v01 \
      /work/clas12/$USER/SULI/models/v01
```

The `check_status.sh` and `resubmit_failed.sh` scripts are for the MC
**array** job, not for this single-job training run.

---

## Troubleshooting

**Python traceback in batch log:**
Reproduce on a 2-file subset with Tier 2 (`--max-files 2`); read the full
traceback; fix the code; resubmit.

**Job timed out:**
`sacct -j <jobid>` shows `TIMEOUT`.  Bump `--time` and resubmit.

**Out of memory:**
`sacct` shows `OUT_OF_MEMORY` or MaxRSS near `--mem`.  Bump `--mem`.

**conda activate fails in batch:**
Worker script probes three known hook paths; the error message names them.
Confirm your conda install location and update the hook path in
`_training_bdt_job.sh` if none of the three match.

**`module load clas12` fails:**
Check that `TMPDIR=/tmp` is exported before any module command (already done
in the worker script) and that `/cvmfs/` is mounted.
