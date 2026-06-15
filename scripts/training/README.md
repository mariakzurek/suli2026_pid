# scripts/training/

Three-script pipeline for training the Week-4 BDT K⁺/π⁺ classifier and
evaluating it against the pass-2 chi2pid baseline.

**Runtime environment:** `conda activate suli2026_pid` (on ifarm).  The
default JupyterHub kernel is not this env — see the three-tier guide in
`slurm/README_training.md` before running anything.

---

## `build_dataset.py`

### WHAT IT DOES

Converts the MC ROOT files in `/volatile/clas12/$USER/SULI/mc_v01/` into
three analysis-ready parquet files (`train.parquet`, `val.parquet`,
`test.parquet`) plus a `manifest.json`.

Reads the file-level split lists produced by Cooper's Phase-0 task
(`slurm/{train,val,test}_files.txt`), applies EB-K+ selection (`pid==321`)
and a momentum cap (`--p-max`), and assigns binary labels:

- `train.parquet` / `val.parquet`: EB-K+ ∩ `mc_matching_pid ∈ {211, 321}`;
  non-nullable `int8` label (1=K, 0=π).
- `test.parquet`: **all** EB-K+ rows; nullable `Int8` label (`NA` for
  protons / unmatched / other).  The test file is intentionally wider so
  `evaluate.py` can compute C^{p→K} from proton rows.

The features loaded are those listed in `--features-file` (one branch name
per line, `#` comments stripped).  That file should contain the audit KEEP
list from `figures/feature_audit/kp/feature_audit_summary.csv`.

### WHEN TO USE

Run once after the Week-3 feature audit is finalised and Cooper has chosen
`--p-max` from his beta-vs-p plots.  Re-run with `--overwrite` if
`--features-file` or `--p-max` changes.

```bash
# Full production build
python scripts/training/build_dataset.py \
    --mc-dir /volatile/clas12/$USER/SULI/mc_v01 \
    --split-dir slurm \
    --outdir /volatile/clas12/$USER/SULI/datasets/v01 \
    --features-file scripts/training/feature_list.txt \
    --p-max 3.0 \
    --overwrite

# Smoke test (2 files per split, runs in ~1 min)
python scripts/training/build_dataset.py \
    --mc-dir /volatile/clas12/$USER/SULI/mc_v01 \
    --split-dir slurm \
    --outdir /tmp/build_smoke \
    --features-file scripts/training/feature_list.txt \
    --p-max 3.0 \
    --max-files 2 \
    --overwrite
```

### PITFALLS

- `--features-file` must be non-empty; the script errors with a pointer to
  the audit CSV rather than silently building an empty feature set.
- Missing ROOT files > 5% of the split cause a hard failure.  Pass
  `--allow-missing-files` to proceed anyway.  Cooper cross-checks overlaps
  with:
  ```bash
  diff <(sort -u slurm/train_files.txt) \
       <(ls /volatile/clas12/$USER/SULI/mc_v01/ | sed 's/\.root$//' | sort -u)
  ```
- ROOT loading uses `library="np"` + `filter_name=` (numpy fast-path,
  ~30× faster than `library="pd"`).  Do not change this.

---

## `train_bdt.py`

### WHAT IT DOES

Fits a LightGBM binary BDT (K vs π) on `train.parquet`, Platt-calibrates
it on a held-out 20% slice of train (stratified, row-level — **never** val
or test), and writes:

- `model.joblib` — fitted calibrated model (compress=3)
- `training_summary.csv` — AUC, Brier, log-loss for train/val pre/post cal
- `reliability_diagram.png` — 2-panel pre/post calibration (on val set)
- `roc_val.png` — ROC curve on val set
- `feature_importance.png` / `.csv` — top-15 features by LightGBM gain
- `README.md` — run provenance

Reads `manifest.json` from `--dataset-dir` for the feature list; no need
to re-specify it on the command line.

Defaults from `cooper_10week_plan.md §300`:
`n_estimators=200`, `learning_rate=0.05`, `max_depth=6`,
`objective='binary'`, `random_state=42`.

### WHEN TO USE

Run after `build_dataset.py`.  Rerun with `--overwrite` when
hyperparameters or the dataset change.  For the canonical Week-4 run use
the `sbatch` path (`slurm/submit_training_bdt.sh`).

```bash
python scripts/training/train_bdt.py \
    --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \
    --outdir /volatile/clas12/$USER/SULI/models/v01 \
    --overwrite
```

### PITFALLS

- The script **never loads test.parquet**.  All test-set evaluation is in
  `evaluate.py`.
- Calibration is on a held-out slice of **train only**.  The reliability
  diagram is evaluated on **val** — never on the calibration slice (that
  would be circular).
- `--reweight-map` is optional; omit it for the v1 unweighted run.  The
  reweight-map generator is a separate Week 4/5 deliverable.

---

## `evaluate.py`

### WHAT IT DOES

Loads `model.joblib` and `test.parquet`, sweeps classification thresholds
in each `(p, θ)` bin, and compares BDT performance against the pass-2
chi2pid baseline.  Produces:

- `per_bin_sweep.csv` — `(eff_K, C_pi, C_p)` at every threshold per bin
- `comparison_summary.csv` — matched-eff and matched-contam comparison
- `contam_vs_ptheta_baseline_vs_bdt.png` — **headline plot**: 1×2 viridis
  heatmap (shared color scale, `n_label<50` bins shown gray with "n<50")
  of C_{π→K} at the threshold that matches baseline eff_K
- `cp_to_K_map.png` — 1×1 viridis heatmap of C^{p→K} at matched-eff
  threshold; Cooper's Phase-4 input for the proton-contamination decision

Default bin edges match the Week-2 audit grid:
`--p-edges 1.0 2.0 3.0 4.0 5.0` (GeV/c),
`--theta-edges 5 15 25 35` (deg).  Both are CLI-configurable.

### WHEN TO USE

Run after `train_bdt.py`.  Safe to re-run with `--overwrite` and different
edge grids to explore different binnings.

```bash
python scripts/training/evaluate.py \
    --model /volatile/clas12/$USER/SULI/models/v01/model.joblib \
    --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \
    --outdir /volatile/clas12/$USER/SULI/eval/v01 \
    --overwrite
```

### PITFALLS

- This script reads **test.parquet only**.  Do not pass a different split.
- Bins with `n_label < 50` labeled rows are masked gray in heatmaps; do
  not quote metrics from these bins.
- The baseline comparison imports
  `scripts.baseline_chi2pid.passes_kplus_chi2pid_cut`; this resolves when
  you run from the repo root (`~/CLAS/SULI/suli2026_pid/`).
- Threshold selection (which threshold to use in production) is a **Week-5**
  task.  This script sweeps thresholds; it does not pick one.

---

## Three-step interactive workflow (Tier 2 on ifarm)

```bash
# Allocate compute node
srun --pty --account=clas12 --time=2:00:00 --mem=16G --cpus-per-task=8 bash

# Activate env
module load clas12
source ~/miniconda3/etc/profile.d/conda.sh   # or wherever conda is installed
conda activate suli2026_pid
cd ~/CLAS/SULI/suli2026_pid

# Step 1: Build dataset (smoke test first with --max-files 2)
python scripts/training/build_dataset.py \
    --mc-dir /volatile/clas12/$USER/SULI/mc_v01 \
    --split-dir slurm \
    --outdir /volatile/clas12/$USER/SULI/datasets/v01 \
    --features-file scripts/training/feature_list.txt \
    --p-max 3.0 --max-files 2 --overwrite

# Step 2: Train
python scripts/training/train_bdt.py \
    --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \
    --outdir /volatile/clas12/$USER/SULI/models/v01 --overwrite

# Step 3: Evaluate
python scripts/training/evaluate.py \
    --model /volatile/clas12/$USER/SULI/models/v01/model.joblib \
    --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \
    --outdir /volatile/clas12/$USER/SULI/eval/v01 --overwrite
```

For the canonical full-statistics run, use
`./slurm/submit_training_bdt.sh` instead.  See `slurm/README_training.md`.
