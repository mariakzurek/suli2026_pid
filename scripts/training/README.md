# scripts/training/

Three-script pipeline for training the Week-4 BDT K⁺/π⁺ classifier and
evaluating it against the pass-2 chi2pid baseline.

**Runtime environment:** `conda activate suli2026_pid` (on ifarm).  The
default JupyterHub kernel is not this env — see the three-tier guide in
`slurm/README_training.md` before running anything.

---

## ⚠️ BEFORE YOU START WEEK 5

Your existing dataset at `/volatile/clas12/$USER/SULI/tier1/dataset_v01/`
was built with only 4 columns (Tier 1). It does NOT contain the columns
Tier 2 and Tier 3 need (chi2pid, ECAL, FTOF 1A, etc.).

**You must rebuild the dataset once with the maximal column set** before
running any tier comparison. Store on `/work/` not `/volatile/` since
`/volatile/` purges after ~2 weeks:

```bash
python scripts/training/build_dataset.py \
    --mc-dir    /volatile/clas12/$USER/SULI/mc_v01 \
    --split-dir slurm \
    --outdir    /work/clas12/$USER/SULI/datasets/v02 \
    --p-max     3.2 --overwrite
```

This uses `columns_maximal.txt` by default (22 audit-relevant columns).
Use a new version suffix (`v02`) so the old `v01` dataset on /volatile/
is not affected.

After the rebuild, all three tier trainings read from the same
`datasets/v02/` directory — no further rebuilds needed.

---

## Workflow

The pipeline has three steps and is designed so that **the dataset is built
once** and **training can be repeated many times with different feature sets**
without rebuilding.

```
Step 1  build_dataset.py   ROOT files → parquet (maximal column set, once)
Step 2  train_bdt.py       parquet + features file → model.joblib (per tier)
Step 3  evaluate.py        model.joblib + test.parquet → plots + CSVs
```

**Decoupling: dataset columns vs. training features.**
The dataset stores the _maximal_ set of audit-relevant columns (defined in
`columns_maximal.txt`).  Training picks a _subset_ of those columns at load
time via `--features-file`.  This means Cooper can try Tier 1, Tier 2, and
Tier 3 feature sets in Week 5 without rebuilding the dataset:

```bash
# Build once (uses columns_maximal.txt by default)
python scripts/training/build_dataset.py \
    --mc-dir /volatile/clas12/$USER/SULI/mc_v01 \
    --split-dir slurm \
    --outdir /volatile/clas12/$USER/SULI/datasets/v01 \
    --p-max 3.0 --overwrite

# Train Tier 1 (minimal: beta + FTOF 1B)
python scripts/training/train_bdt.py \
    --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \
    --features-file scripts/training/features_tier1.txt \
    --outdir /volatile/clas12/$USER/SULI/models/tier1 --overwrite

# Train Tier 2 (adds chi2pid + FTOF 1A)
python scripts/training/train_bdt.py \
    --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \
    --features-file scripts/training/features_tier2.txt \
    --outdir /volatile/clas12/$USER/SULI/models/tier2 --overwrite

# Train Tier 3 (adds ECAL + HTCC)
python scripts/training/train_bdt.py \
    --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \
    --features-file scripts/training/features_tier3.txt \
    --outdir /volatile/clas12/$USER/SULI/models/tier3 --overwrite
```

### Adding a new feature

**(a) Feature is already in `columns_maximal.txt`** (it was loaded into the
parquet when the dataset was built): just add the column name to your tier
file and retrain — no rebuild needed.

**(b) Feature is NOT in `columns_maximal.txt`** (it was not built into the
parquet): edit `columns_maximal.txt` to add it, then rebuild the dataset with
`build_dataset.py --overwrite`, then add the column name to your tier file and
retrain.

---

## FIXING PREFIT ERROR IN TRAIN_BDT.PY
if the error that prefit is not recognized in train_bdt.py make sure sklearn is properly installed
run pip install -U scikit-learn in a jupyter notebook or python file in the suli2026_pid to make sure it is in the enviroment.

The working version of sklearn when this issue was first discovered was 1.9.0, so anything later should work barring a deprication of prefit for cv

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

The columns loaded are those listed in `--columns-file` (default:
`columns_maximal.txt`) — the inclusive set of all audit-relevant features.
Training then picks a subset of those columns via `train_bdt.py --features-file`.

### WHEN TO USE

Run **once** after the Week-3 feature audit is finalised and Cooper has chosen
`--p-max` from his beta-vs-p plots.  Re-run with `--overwrite` **only if**
`--columns-file` or `--p-max` changes (i.e., when the parquet schema itself
changes).  Feature subset changes for training experiments do **not** require
a rebuild.

```bash
# Full production build (maximal column set, once)
python scripts/training/build_dataset.py \
    --mc-dir /volatile/clas12/$USER/SULI/mc_v01 \
    --split-dir slurm \
    --outdir /volatile/clas12/$USER/SULI/datasets/v01 \
    --p-max 3.0 \
    --overwrite

# Smoke test (2 files per split, runs in ~1 min)
python scripts/training/build_dataset.py \
    --mc-dir /volatile/clas12/$USER/SULI/mc_v01 \
    --split-dir slurm \
    --outdir /tmp/build_smoke \
    --p-max 3.0 \
    --max-files 2 \
    --overwrite
```

The old `--features-file` flag is accepted as a deprecated alias for
`--columns-file` and prints a warning.

### PITFALLS

- `--columns-file` must be non-empty; the script errors with a pointer to
  the audit CSV rather than silently building an empty column set.
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

- `model.joblib` — wrapper dict `{"model": calibrated_clf, "features": [...]}`
- `training_summary.csv` — AUC, Brier, log-loss for train/val pre/post cal
- `reliability_diagram.png` — 2-panel pre/post calibration (on val set)
- `roc_val.png` — ROC curve on val set
- `feature_importance.png` / `.csv` — top-15 features by LightGBM gain
- `README.md` — run provenance

**`--features-file` is required.**  Pass one of `features_tier1.txt`,
`features_tier2.txt`, `features_tier3.txt`, or a custom file listing column
names from `columns_maximal.txt`.  The feature list is embedded in
`model.joblib` so `evaluate.py` can recover it without consulting the manifest.

Defaults from `cooper_10week_plan.md §300`:
`n_estimators=200`, `learning_rate=0.05`, `max_depth=6`,
`objective='binary'`, `random_state=42`.

### WHEN TO USE

Run after `build_dataset.py`.  Rerun with `--overwrite` and a different
`--features-file` to try a different feature tier — **no rebuild needed**.
For the canonical Week-4 run use the `sbatch` path (`slurm/submit_training_bdt.sh`).

```bash
# Tier 1
python scripts/training/train_bdt.py \
    --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \
    --features-file scripts/training/features_tier1.txt \
    --outdir /volatile/clas12/$USER/SULI/models/tier1 \
    --overwrite

# Tier 2
python scripts/training/train_bdt.py \
    --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \
    --features-file scripts/training/features_tier2.txt \
    --outdir /volatile/clas12/$USER/SULI/models/tier2 \
    --overwrite

# Tier 3
python scripts/training/train_bdt.py \
    --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \
    --features-file scripts/training/features_tier3.txt \
    --outdir /volatile/clas12/$USER/SULI/models/tier3 \
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
- `model.joblib` is now a wrapper dict; load it with
  `obj = joblib.load(path); model = obj["model"]; features = obj["features"]`.

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

Feature names are read from `model.joblib` (the wrapper dict produced by the
updated `train_bdt.py`).  Old bare-estimator `model.joblib` files fall back
to `manifest.json` with a deprecation warning.

Default bin edges match the Week-2 audit grid:
`--p-edges 1.0 2.0 3.0 4.0 5.0` (GeV/c),
`--theta-edges 5 15 25 35` (deg).  Both are CLI-configurable.

### WHEN TO USE

Run after `train_bdt.py`.  Safe to re-run with `--overwrite` and different
edge grids to explore different binnings.

```bash
python scripts/training/evaluate.py \
    --model /volatile/clas12/$USER/SULI/models/tier1/model.joblib \
    --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \
    --outdir /volatile/clas12/$USER/SULI/eval/tier1 \
    --overwrite
```


NOTE: ALL THESE SCRIPTS HAVE A MLP VARIATION UNDER MLPS/ THESE TAKE THE SAME COMMAND PARAMETERS

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

# Step 1: Build dataset once (smoke test with --max-files 2 first)
python scripts/training/build_dataset.py \
    --mc-dir /volatile/clas12/$USER/SULI/mc_v01 \
    --split-dir slurm \
    --outdir /volatile/clas12/$USER/SULI/datasets/v01 \
    --p-max 3.0 --max-files 2 --overwrite

# Step 2: Train (three tiers, no rebuild needed between them)
python scripts/training/train_bdt.py \
    --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \
    --features-file scripts/training/features_tier1.txt \
    --outdir /volatile/clas12/$USER/SULI/models/tier1 --overwrite

python scripts/training/train_bdt.py \
    --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \
    --features-file scripts/training/features_tier2.txt \
    --outdir /volatile/clas12/$USER/SULI/models/tier2 --overwrite

python scripts/training/train_bdt.py \
    --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \
    --features-file scripts/training/features_tier3.txt \
    --outdir /volatile/clas12/$USER/SULI/models/tier3 --overwrite

# Step 3: Evaluate each tier
python scripts/training/evaluate.py \
    --model /volatile/clas12/$USER/SULI/models/tier1/model.joblib \
    --dataset-dir /volatile/clas12/$USER/SULI/datasets/v01 \
    --outdir /volatile/clas12/$USER/SULI/eval/tier1 --overwrite
```

For the canonical full-statistics run, use
`./slurm/submit_training_bdt.sh` instead.  See `slurm/README_training.md`.
