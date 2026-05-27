# SULI 2026 — ML PID for CLAS12

ML-based particle identification for the `ep → e' p K+ X` SIDIS channel using CLAS12 RGA pass-2 data.

**Student:** Cooper Bell (SULI 2026)
**PI:** Maria Zurek (ANL)
**Succesfully Edited!**
## What's in this repo

- `notebooks/` — Jupyter notebooks for data exploration, model training, evaluation
- `scripts/` — standalone Python scripts (training pipeline, evaluation, plot generation)
- `figures/` — generated plots and figures (not source files — those live in the notebooks/scripts)
- `slurm/` — ifarm batch submission scripts
- `notes/` — project notes, weekly summaries, intermediate writeups

## Companion repo

The CLAS12 framework and data-production scripts (groovy + Java + C++ converter) live in:
https://github.com/mariakzurek/clas12_analysis_software (branch `suli_kaon_pid`, off `rich_studies`)

That repo runs on ifarm to produce the training/analysis ntuples that this repo consumes.

## Quick start

**For interactive notebook work** (recommended for Week 1):
1. Log in to JLab JupyterHub at `https://jupyterhub.jlab.org` with your CUE credentials
2. Navigate to `/work/clas12/<username>/SULI/suli2026_pid/notebooks/`
3. Open or create a notebook with the Python 3 kernel
4. The default kernel has numpy, pandas, matplotlib, scikit-learn, uproot pre-installed
5. For `lightgbm`: run `!pip install --user lightgbm` once in a notebook cell

**For batch jobs** (Week 2+):
```bash
# Set up the conda env (one-time, see notes/cooper_day1_and_week1.md Section 4f for full instructions)
conda env create -f environment.yml
conda activate suli2026_pid
```

The conda env contains numpy, pandas, matplotlib, scikit-learn, uproot, awkward, lightgbm, xgboost — runtime libraries only. No jupyterlab (use JupyterHub for notebooks).

See `notes/cooper_day1_and_week1.md` for the full onboarding doc.

## Project plan and onboarding

See `notes/` directory.
