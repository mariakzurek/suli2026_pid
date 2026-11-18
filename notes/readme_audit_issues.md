# README audit: missing documentation and undocumented files

**Audit date:** 2026-07-31  
**Auditor:** research agent (opencode)  
**Scope:** All six README files cross-checked against disk contents for `suli2026_pid/`.

---

## Worst documentation debt

**`figures/README.md`** is the most outdated by a large margin. It documents 3 subdirectories and describes only content from the early project weeks. At least 10 subdirectories and ~30 top-level files in `figures/` are completely undocumented. It reads as a Week-2 artifact that was never updated as the project grew.

**`slurm/README.md`** has the second-largest gap: its formal Files table covers only 5 scripts, while 9 additional files (including `submit_apply_bdt.sh`, `_apply_bdt_job.sh`, `check_farm_access.sh`, all three split files, `_apply_bdt_file_list.txt`, and the entire `multiclass/` subdirectory) are either undocumented or mentioned only in narrative prose without explanation.

---

## Documented but missing from disk

These are things a README explicitly says exist or implies should exist, but the file is not on disk.

- [ ] **`slurm/_mc_file_list.txt`** — `slurm/README.md` (Output locations table, line ~106) says MC file lists are written to `slurm/_mc_file_list.txt` at submit time. This file does **not exist** on disk. Only `slurm/_data_file_list.txt` exists. Either `_mc_file_list.txt` was never written, was deleted, or the table entry has the name wrong and it was never created by `submit_mc.sh`. Needs investigation: does `submit_mc.sh` actually write this file?

- [ ] **`figures/feature_audit/COLUMNS.md`** — `scripts/README.md` (line 164) tells users: *"The full per-column glossary for `feature_audit_summary.csv` lives at `figures/feature_audit/COLUMNS.md`. Start there if you're new to the audit output."* This file does **not exist** at `figures/feature_audit/COLUMNS.md`. `COLUMNS.md` files exist inside `figures/feature_audit_week3/` and `figures/feature_audit_NOCUT/` subdirectories (which are stale audit snapshots), but the canonical path the README advertises is empty. A new or fresh user following the README will find nothing.

---

## Present on disk but undocumented

These files and directories exist on disk but are not mentioned in any README.

### Top level (`suli2026_pid/`)

- [ ] **`environment.yml`** — Present at `suli2026_pid/environment.yml`. The top-level `README.md` references it in a code block (`conda env create -f environment.yml`) but never mentions it in the "What's in this repo" inventory. It should be listed there with a one-line description.

- [ ] **`pushtest.txt`** — Present at `suli2026_pid/pushtest.txt`. Not mentioned in any README. This appears to be a leftover test file from an early git push check. It should either be deleted or, if intentionally kept, noted.

- [ ] **`report/`** — A `report/` directory exists at the top level (containing `README.md`, `references.bib`, `report_template.tex`) and is not listed in the top-level `README.md`'s "What's in this repo" section. The suli-pid skill documents it, but the repo README does not.

### `scripts/` (direct children)

- [ ] **`common_functions.py`** — Present at `scripts/common_functions.py`. Not mentioned in `scripts/README.md` or any other README. This is a non-trivial 18 KB file (last modified Jul 31). Needs at least a one-paragraph entry in `scripts/README.md` explaining what it does and which scripts import it.

- [ ] **`scripts/figures/`** — A `figures/` subdirectory exists inside `scripts/` (containing an `optimized/` subdir with `optimized_bdt_thresholds.csv`). Not mentioned in `scripts/README.md`. It is ambiguous whether this is intentional or whether these outputs should live in the top-level `figures/` instead.

### `scripts/training/` (direct children)

- [ ] **`feature_list2.txt`**, **`feature_list3.txt`**, **`feature_list4.txt`** — All three present in `scripts/training/`. `scripts/training/README.md` documents `feature_list.txt` (deprecated), `features_tier1.txt`, `features_tier2.txt`, and `features_tier3.txt`, but says nothing about `feature_list2.txt` through `feature_list4.txt`. It is unclear whether these are experimental feature sets that Cooper is actively using, superseded drafts, or forgotten artifacts. They need either documentation or deletion.

- [ ] **`manifestCheck.ipynb`** — Present at `scripts/training/manifestCheck.ipynb`. Not mentioned in `scripts/training/README.md`. This is a Jupyter notebook inside a scripts directory, which is unusual placement. Its purpose is not documented.

- [ ] **`multiclass/` subdirectory** — `scripts/training/multiclass/` exists and contains four Python scripts: `build_dataset_multiclass.py`, `train_bdt_multiclass.py`, and their `BACKUP_` prefixed copies. None of these are documented in any README file anywhere. The multiclass pipeline has zero documentation.

- [ ] **`MLPS/` subdirectory** — `scripts/training/MLPS/` exists and contains three Python scripts: `build_dataset_MLP.py`, `train_bdt_MLP.py`, `evaluate_MLP.py`. None are documented in any README. This is an entire parallel training pipeline (MLP variant) with no documentation at all.

### `slurm/` (direct children and subdirectories)

- [ ] **`check_farm_access.sh`** — Present at `slurm/check_farm_access.sh`. It is referenced in code blocks in `slurm/README.md` (BDT training section) and `slurm/README_training.md`, but **not listed in the `slurm/README.md` Files table**. A user scanning that table to understand what's in `slurm/` would not learn this script exists or what it does.

- [ ] **`submit_apply_bdt.sh`** and **`_apply_bdt_job.sh`** — Both present in `slurm/`. The `slurm/README.md` has a "BDT scoring / apply" narrative section that mentions them, but they are **not in the Files table**. The Files table is the at-a-glance inventory for this directory; these scripts need a row there.

- [ ] **`val_files.txt`**, **`train_files.txt`**, **`test_files.txt`** — All three present in `slurm/`. The skill documents them as the Week-3 70/15/15 file-level split lists, and they are referenced in `scripts/training/README.md` (via `--split-dir slurm`), but `slurm/README.md` never mentions them at all — not in the Files table, not in narrative prose. A user reading `slurm/README.md` has no idea these files exist or what they're for.

- [ ] **`_apply_bdt_file_list.txt`** — Present at `slurm/_apply_bdt_file_list.txt`. Not mentioned in any README. Presumably generated at submit time by `submit_apply_bdt.sh` (analogous to `_data_file_list.txt`), but this is not documented anywhere.

- [ ] **`multiclass/` subdirectory** — `slurm/multiclass/` exists and contains: `_training_bdt_job_multiclass.sh`, `submit_training_bdt_multiclass.sh`, `proton_files.txt`, `scan_summary.txt`, `train_files.txt`, `test_files.txt`. None of these are mentioned in `slurm/README.md` or `slurm/README_training.md`. Note that `slurm/multiclass/` has `train_files.txt` and `test_files.txt` but **no `val_files.txt`** — if the multiclass pipeline uses a val split, this is a missing artifact; if it doesn't use one, that design choice should be documented.

- [ ] **`MLPS/` subdirectory** — `slurm/MLPS/` exists and contains: `_training_MLP_job.sh`, `submit_training_MLP.sh`. Neither is documented in any README.

### `figures/` (direct children — directories only)

The `figures/README.md` documents only: `audit`, `BetaVsP`, `ConfusionMatrix`, `Baselines` (and within it: `thetaCutPlots`, `globalThetaCuts`). The following directories are present but completely undocumented:

- [ ] **`figures/audit_NOCUT/`** — Not mentioned in `figures/README.md`. Presumably audit output from runs without SIDIS cuts, but purpose and relationship to `figures/audit/` are undocumented.

- [ ] **`figures/feature_audit/`** — Not mentioned in `figures/README.md`, even though `scripts/README.md` treats it as the canonical audit output location. This is one of the most important output directories in the whole project.

- [ ] **`figures/feature_audit_NOCUT/`** — Not mentioned in `figures/README.md`.

- [ ] **`figures/feature_audit_week3/`** — Not mentioned in `figures/README.md`. This appears to be a snapshot of audit results from Week 3 and is where the `COLUMNS.md` files actually live (see "Stale or incorrect references" below).

- [ ] **`figures/full_range/`** — Not mentioned in `figures/README.md`.

- [ ] **`figures/optimized/`** — Not mentioned in `figures/README.md`.

- [ ] **`figures/otherPlots/`** — Not mentioned in `figures/README.md`.

- [ ] **`figures/Data_Application/`** — Not mentioned in `figures/README.md`.

- [ ] **`figures/OLD/`** and **`figures/efficienciesOLD/`** — Not mentioned in `figures/README.md`. Even a one-line "legacy, do not use" note would be sufficient.

- [ ] **`figures/binary_MLP/`** — Added in the `week8` branch (not yet on `main`). It has its own auto-generated `README.md` (provenance file from `evaluate.py`). The top-level `figures/README.md` should be updated to describe this directory once the branch merges.

---

## Stale or incorrect references

These are paths, file names, or descriptions in READMEs that are factually wrong or misleading given the current state of the repo.

- [ ] **`slurm/README_training.md` references `feature_list.txt` (deprecated) in the Tier 2 workflow** — Line 69 of `slurm/README_training.md` shows:
  ```bash
  python scripts/training/build_dataset.py \
      ...
      --features-file scripts/training/feature_list.txt \
  ```
  `feature_list.txt` is explicitly marked `DEPRECATED` in its own header comment and `scripts/training/README.md` says it is no longer read by the pipeline. The `--features-file` flag in `build_dataset.py` is now a deprecated alias for `--columns-file`; the correct flag is `--columns-file` with `columns_maximal.txt`. This example command in `README_training.md` will cause a confusing deprecation warning and produce wrong behavior. It should be updated to use `--columns-file scripts/training/columns_maximal.txt` (or simply omit the flag since `columns_maximal.txt` is the default).

- [ ] **`scripts/training/README.md` "FIXING PREFIT ERROR" section gives wrong advice** — The section (lines ~97–102) says: *"The working version of sklearn when this issue was first discovered was 1.9.0, so anything later should work."* This is incorrect. Per the project's own `environment.yml` and the standing convention in the suli-pid skill: `scikit-learn` is pinned to `1.5.2` precisely because `CalibratedClassifierCV(cv='prefit')` was **deprecated in 1.6** and **removed in 1.8**. sklearn 1.9.0 is the version that *broke* the code, not one where it works. The section should say: pin sklearn to `<=1.5.x` (the `environment.yml` already does this) and point at the `FrozenEstimator` migration path for anyone upgrading past 1.6.

- [ ] **`figures/feature_audit/COLUMNS.md` is advertised but does not exist at that path** — `scripts/README.md` (line 164) directs users to `figures/feature_audit/COLUMNS.md`. The actual `COLUMNS.md` files live inside `figures/feature_audit_week3/kp/COLUMNS.md`, `figures/feature_audit_week3/kp_sidis/COLUMNS.md`, and `figures/feature_audit_NOCUT/kp/COLUMNS.md`. The canonical path in `scripts/README.md` is wrong. Either move/copy a `COLUMNS.md` to `figures/feature_audit/`, or update the README to point at the correct location.

- [ ] **`slurm/README.md` Output locations table lists `slurm/_mc_file_list.txt` but this file does not exist** — The table (around line 106) lists both `slurm/_mc_file_list.txt` and `slurm/_data_file_list.txt` as files "written at submit time." Only `_data_file_list.txt` exists on disk. Either `submit_mc.sh` never actually writes `_mc_file_list.txt`, or it was deleted. The README and/or `submit_mc.sh` should be corrected to match reality.

- [ ] **`slurm/README.md` Files table omits all post-Week-3 scripts** — The Files table (lines 12–19) lists only the 5 original ntuple-production scripts. It does not mention `check_farm_access.sh`, `submit_apply_bdt.sh`, `_apply_bdt_job.sh`, `submit_training_bdt.sh`, or `_training_bdt_job.sh`, even though all are present and in active use. The Files table should be expanded or split into sections (ntuple production / training / scoring) so the directory is self-documenting.

- [ ] **`figures/README.md` is Week-2 vintage and has no mention of the feature audit outputs** — The README documents early exploratory output directories (`audit`, `BetaVsP`, `ConfusionMatrix`, `Baselines`) but has never been updated to reflect that `feature_audit/`, `feature_audit_week3/`, `optimized/`, `full_range/`, `Data_Application/`, and `binary_MLP/` are now the primary analysis outputs. Any new contributor reading this README would have a profoundly incomplete picture of what is in the directory.

---

## Summary table

| README | Missing from disk | Undocumented on disk | Stale/wrong refs |
|---|---|---|---|
| `README.md` (top-level) | 0 | 3 (`environment.yml` unlisted, `pushtest.txt`, `report/`) | 0 |
| `scripts/README.md` | 1 (`figures/feature_audit/COLUMNS.md`) | 2 (`common_functions.py`, `scripts/figures/`) | 1 (wrong COLUMNS.md path) |
| `slurm/README.md` | 1 (`_mc_file_list.txt`) | 9+ (see above) | 2 (stale Files table, wrong `_mc_file_list.txt`) |
| `slurm/README_training.md` | 0 | 0 | 1 (deprecated `feature_list.txt` in example) |
| `scripts/training/README.md` | 0 | 4 (`feature_list2/3/4.txt`, `manifestCheck.ipynb`, `multiclass/`, `MLPS/`) | 1 (wrong sklearn advice) |
| `figures/README.md` | 0 | 10+ dirs undocumented | 0 (but entire README is stale) |
