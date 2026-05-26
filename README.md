# SULI 2026 — ML Kaon PID for CLAS12

ML-based kaon particle identification for the `ep → e' p K+ X` exclusive channel using CLAS12 RGA pass-2 data.

**Student:** Cooper (SULI 2026)
**PI:** Maria Zurek (ANL)
**Supervisor:** Fatiha (JLab)

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

```bash
# On ifarm, after the ntuple is produced:
git clone git@github.com:mariakzurek/suli2026_pid.git
cd suli2026_pid
conda env create -f environment.yml    # or pip install -r requirements.txt
jupyter lab notebooks/
```

## Project plan and onboarding

See `notes/` directory (initially populated from `~/CLAS/SULI/notes/` on the PI's local machine).
