# PID ntuple production — slurm batch submission

Scripts for running `processing_mc_pid_training.groovy` and
`processing_data_pid_training.groovy` as slurm array jobs on JLab ifarm.
One slurm task per input HIPO file. Output is a directory of per-file ROOT
files, not a single merged ROOT.

---

## Files

| File | Purpose |
|---|---|
| `submit_mc.sh` | User-facing: submit MC array (full ~318 tasks or a subset) |
| `submit_data.sh` | User-facing: submit data array (default 20 files) |
| `_pid_training_array.sh` | Internal array task script (not invoked directly) |
| `check_status.sh` | Monitor job progress via `sacct` |
| `resubmit_failed.sh` | Re-run only the failed/timed-out tasks |

---

## BDT training (Week 4)

For training the ML classifier after ntuple production is complete, see the
separate runbook at `slurm/README_training.md`.  Short version:

```bash
./slurm/check_farm_access.sh         # preflight
./slurm/submit_training_bdt.sh       # canonical sbatch run
```

Scripts: `submit_training_bdt.sh` (wrapper) · `_training_bdt_job.sh` (worker)
Training scripts: `scripts/training/{build_dataset,train_bdt,evaluate}.py`

---

## Full workflow

```bash
# 1. Log in to ifarm and go to the repo root
ssh <user>@ifarm.jlab.org
cd ~/CLAS/SULI/suli2026_pid
git pull origin main

# 2. Submit MC array (full run, ~318 tasks)
./slurm/submit_mc.sh

# Or submit a 2-task smoke test first:
./slurm/submit_mc.sh 2

# 3. Note the job ID printed by sbatch, then monitor:
squeue -u $USER
./slurm/check_status.sh <jobid>

# 4. After completion, check outputs:
ls -lh /volatile/clas12/$USER/SULI/mc_v01/

# 5. Resubmit any failures:
./slurm/resubmit_failed.sh <jobid> mc
```

Same pattern for data (substitute `submit_data.sh` and `data` for `mc`):

```bash
./slurm/submit_data.sh        # 20 files (default)
./slurm/submit_data.sh 2      # smoke test
./slurm/check_status.sh <jobid>
./slurm/resubmit_failed.sh <jobid> data
```

---

## Output locations

| What | Where |
|---|---|
| MC ROOT files | `/volatile/clas12/$USER/SULI/mc_v01/<stem>.root` |
| Data ROOT files | `/volatile/clas12/$USER/SULI/data_v01/<stem>.root` |
| MC logs | `/farm_out/$USER/suli/pid_train_<jobid>_<idx>.{out,err}` |
| Data logs | same pattern under `/farm_out/$USER/suli/` |
| MC file list | `slurm/_mc_file_list.txt` (written at submit time) |
| Data file list | `slurm/_data_file_list.txt` (written at submit time) |

`<stem>` is the input HIPO filename without the `.hipo` extension.

`/volatile/clas12/` is purged after ~2 weeks. Copy final ROOT files to
`/work/clas12/$USER/SULI/` before they expire.

---

## Version label (`v01`)

Output directories are named `mc_v01/` and `data_v01/`. To reprocess with a
different groovy version or corrected scripts, bump the label:

1. Edit the `OUTPUT_DIR` variable in `submit_mc.sh` and `submit_data.sh`
   (change `mc_v01` → `mc_v02`, etc.).
2. Edit `FINAL_DIR` in `_pid_training_array.sh` to match.
3. Submit fresh — old outputs in `v01/` are preserved until volatile purge.

---

## No automatic merge

Each task writes one ROOT file named after its input HIPO file. There is no
`hadd` merge step. Downstream tools (audit scripts, training code) must iterate
the output directory with a glob. Example:

```python
import glob
files = sorted(glob.glob("/volatile/clas12/$USER/SULI/mc_v01/*.root"))
```

To merge manually after all tasks complete:

```bash
module load clas12
hadd -f mc_pid_training_full.root /volatile/clas12/$USER/SULI/mc_v01/*.root
```

---

## Account / partition — what to do if SLURM rejects

This is your first batch submission on ifarm; the correct account and partition
are not yet confirmed. The scripts submit with no `--account` or `--partition`
directive, which works if SLURM has a user default configured.

If `sbatch` returns an error like `"no default account"`, `"no valid partition"`,
or `"QOS not permitted"`:

1. Find your account:
   ```bash
   sacctmgr show user $USER
   ```

2. Find available partitions:
   ```bash
   sinfo -s
   ```

3. Open `slurm/_pid_training_array.sh` and uncomment these two lines near the
   top, substituting the values from steps 1–2:
   ```bash
   #SBATCH --account=<your_account>
   #SBATCH --partition=<partition_name>
   ```

4. Resubmit.

---

## `/cache/` staging for data

Data files live in `/cache/clas12/`, which is tape-backed. A task that reads
from tape will stall and hit the 45-minute wall-time limit.

Before running `submit_data.sh`:

```bash
# Check disk vs tape status for a file
jstat /cache/clas12/rg-a/production/recon/fall2018/torus-1/pass2/main/train/nSidis/<file>.hipo

# Request staging (repeat for each file you plan to process)
jcache stage /cache/clas12/rg-a/production/recon/fall2018/torus-1/pass2/main/train/nSidis/<file>.hipo
```

Wait until `jstat` reports `DISK` status, then submit.

---

## run-groovy jar-swap race (known limitation)

`coatjava/bin/run-groovy` deletes and re-copies `processing_classes.jar` on
every invocation. With up to 50 concurrent tasks sharing the same checkout,
this can race and cause `ClassNotFoundException` in a small fraction of tasks.
Those tasks fail and are caught by `resubmit_failed.sh`. For a first production
run this is acceptable. If failures are frequent, contact Maria to apply the
rsync-per-submission workaround (design spec §7.5).
