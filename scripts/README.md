# scripts/

Standalone Python utilities for the SULI 2026 ML PID project.

---

## `baseline_chi2pid.py`

Python reference implementations of the pass-2 K⁺ and π⁺ chi2pid cuts used
in the CLAS12 RGA analysis (`pid_cuts.java`).

**Functions:**
- `passes_kplus_chi2pid_cut(chi2pid, p)` — pass-2 K⁺ cut, momentum-dependent (3 regimes)
- `passes_piplus_chi2pid_cut(chi2pid, p)` — pass-2 π⁺ cut, momentum-dependent (2 regimes)
- `passes_loose_chi2pid_cut(chi2pid, p, threshold=3)` — older flat |chi2pid| < threshold
- `passes_per_run_chi2pid_cut(chi2pid, p, pid, runnum, detector)` — per-run-period μ ± Nσ windows for all five species

**Run the self-test:**
```bash
python scripts/baseline_chi2pid.py
```
Prints cut-window tables and saves comparison plots to `figures/`.

---

## `compare_mc_data.py`

MC vs data distribution comparison for individual variables.  Used in the
Week 3 Task 3a feature audit to decide KEEP / CANDIDATE / DROP for each ML
training feature.

### What it does

For each variable and each (p, θ) slice in a 3×3 audit grid
(p ∈ {[1,2], [2,3], [3,5]} GeV/c × θ ∈ {[5°,15°], [15°,25°], [25°,35°]}):

1. Strips sentinel values (−9999 and +9999 for chi2pid).
2. Auto-ranges the histogram to the 1st–99th percentile of the combined
   MC+data sample.
3. Produces a **2-panel PNG**: histogram overlay (top) + (data−MC)/MC
   residuals with error bars (bottom).
4. Computes three generic, scale-free drift metrics designed to work
   uniformly across every variable in the audit:
   - **Wasserstein-1 / data IQR** (`wasserstein_normalized`) — earth-mover's
     distance between MC and data normalized by the data IQR, so it is
     comparable across variables.  KEEP < 0.05, CANDIDATE 0.05–0.20,
     DROP ≥ 0.20.
   - **Population Stability Index** (`psi_score`) — standard ML feature-drift
     metric with quantile binning that auto-adapts to any variable.
     KEEP < 0.10, CANDIDATE 0.10–0.25, DROP ≥ 0.25.
   - **Max local quantile residual** (`max_local_residual`) — worst single-bin
     residual on equal-count bins.  Catches localized tail mismatches that the
     integrated metrics average away.  KEEP < 0.30, CANDIDATE 0.30–0.80,
     DROP ≥ 0.80.

A per-cell `drift_decision` (KEEP / CANDIDATE / DROP / UNKNOWN) is computed
automatically via `classify_drift(psi, w_norm, max_resid)`: DROP if at least
two metrics flag DROP, CANDIDATE if any one flags CANDIDATE or DROP, otherwise
KEEP.  NaN metrics are skipped.

KS distance and χ²/ndof are still computed and stored in the output CSV as
sanity-check legacy columns, but they no longer drive the decision.

Every statistical function has a thorough docstring explaining what it tests,
when to use it, and what its pitfalls are.  Read the docstrings.

### Quick start

```bash
# Run the self-test (synthetic data, no ROOT files needed)
python scripts/compare_mc_data.py

# Full audit — canonical invocation matching notes/cooper_10week_plan.md
python scripts/compare_mc_data.py \
    --mc   /volatile/clas12/<username>/SULI/mc_pid_training_full.root \
    --data /volatile/clas12/<username>/SULI/data_pid_training.root \
    --vars all_audit kinematics \
    --outdir figures/feature_audit

# Single-variable iteration
python scripts/compare_mc_data.py \
    --mc mc.root --data data.root --vars beta \
    --outdir figures/feature_audit/
```

### Variable group aliases

| Alias | Variables |
|-------|-----------|
| `ml_features` | beta, chi2pid, ftof\_energy\_1A/1B, ftof\_time\_1A/1B, ftof\_path\_1A/1B, ecin/ecout\_energy, ecin/ecout\_time, ecin/ecout\_path, nphe\_htcc |
| `candidate_features` | pcal\_energy/time/path, ftof\_energy/time/path\_2 |
| `kinematics` | p, theta, phi, vz, sector |
| `all_audit` | ml\_features + candidate\_features |

### Output

- **`figures/feature_audit/<variable>/<variable>_p<lo>-<hi>_theta<lo>-<hi>.png`** — one overlay PNG per (variable, slice).
- **`figures/feature_audit/feature_audit_summary.csv`** — one row per (variable, p-bin, θ-bin) with all metrics.  Key columns:
  - `wasserstein`, `wasserstein_norm` — raw and IQR-normalised Wasserstein-1 drift metrics.
  - `psi` — Population Stability Index.
  - `max_local_residual` — worst single-bin quantile residual.
  - `drift_decision` — KEEP / CANDIDATE / DROP / UNKNOWN.
  - `hit_frac_mc`, `hit_frac_data` — fraction of rows with a valid (non-sentinel) value for this variable, computed on the **parent** (full species-selected) DataFrame; reflects global detector acceptance.
  - `n_total_mc_cell`, `n_total_data_cell` — total MC and data rows in the (p, θ) cell before sentinel stripping.
  - `n_hit_mc_cell`, `n_hit_data_cell` — rows in the cell with a valid (non-sentinel) value.
  - `hit_frac_mc_cell`, `hit_frac_data_cell` — per-cell hit fractions (n_hit / n_total for this kinematic cell); NaN if the cell is empty.
  - `hit_frac_delta` — `hit_frac_data_cell − hit_frac_mc_cell`; |Δ| > 0.05 flags a hit-fraction mismatch that should be reviewed before trusting the shape comparison for this cell.
  - Legacy columns: `ks_distance`, `chi2_per_ndof`, `mean_rel_diff`, `max_abs_rel_diff`, `ks_flag`.
  - `status` — `ok` / `missing_column` / `empty`.
  - After running, the audit operator appends a `decision_notes` free-text column with any visual-cross-check observations or override rationale.

### Using as a library

```python
from scripts.compare_mc_data import compare_distribution, run_feature_audit

stats = compare_distribution(
    df_mc, df_data, variable="beta",
    bins=60, normalize=True,
    selection_mc="pid == 321", selection_data="pid == 321",
    save_path="figures/feature_audit/beta/beta_p1-2_theta5-15.png",
)
print(stats["drift_decision"], stats["psi"], stats["wasserstein_norm"], stats["max_local_residual"])
```

### Decision criteria (Week 3 audit)

| Per-cell condition | Per-cell decision |
|---|---|
| All three drift metrics in their KEEP band | **KEEP** |
| At least one metric in CANDIDATE band; none or one in DROP band | **CANDIDATE** |
| At least two metrics in DROP band | **DROP** |
| All metrics NaN (no usable entries) | **UNKNOWN** |

Per-variable aggregation: variable is **DROP** if any cell is DROP; **CANDIDATE**
if any cell is CANDIDATE and none are DROP; **KEEP** only if all cells are KEEP.
Variables flagged CANDIDATE or DROP require visual cross-check of the per-cell
PNGs before locking the decision.

Record decisions in `figures/feature_audit/feature_audit_summary.csv`
(`decision_notes` column) and write up the per-variable narrative in Section 5
of the Week 1-2 report.  See `notes/cooper_10week_plan.md` Task 3a and Task 3c
for the full audit workflow.

### Missing columns

If a requested variable is not present in the MC or data ntuple, the script
emits a warning, marks the row `status = "missing_column"` in the summary CSV,
and continues to the next variable rather than crashing.  Check the CSV's
`status` column after every run to confirm all requested variables were actually
audited.

---

## `audit_species.py`

Opinionated thin driver that wraps `compare_mc_data.run_feature_audit` with
the SULI-2026-project-specific MC and data selections for a chosen EB particle
species.  The generic engine stays generic; this script encodes the
project-specific workflow: species pid cut, MC truth-match mode, and the
provenance README that makes each audit output directory self-documenting.

The full per-column glossary for `feature_audit_summary.csv` lives at
`figures/feature_audit/COLUMNS.md`. Start there if you're new to the audit
output — it groups the ~25 CSV columns into "what to look at," "what explains
the decision," and "sanity checks" tiers with plain-language definitions.

### Species aliases

| Alias | PID    | Human label |
|-------|--------|-------------|
| `kp`  | 321    | K+          |
| `pip` | 211    | π+          |
| `p`   | 2212   | p           |
| `em`  | 11     | e−          |
| `pim` | −211   | π−          |
| `kn`  | −321   | K−          |

### `--truth-mode` semantics

| Mode      | MC selection | When to use |
|-----------|--------------|-------------|
| `matched` (default) | `(pid == SPEC) & (mc_matching_pid != -9999)` | ML feature drift audits — the classifier sees the EB-labeled sample including mis-IDs, so auditing that population is the correct diagnostic. |
| `pure`    | `(pid == SPEC) & (mc_matching_pid == SPEC)` | Detector-response physics studies where you need truth-pure tracks.  Not appropriate for ML feature audits. |
| `off`     | `pid == SPEC` | When truth matching is suspect, or for apples-to-apples with data without imposing extra MC quality cuts. |

Data selection is always `pid == SPEC` regardless of truth mode (data has no `mc_matching_pid`).

### Vertex-z cut flags

| Flag | Description |
|------|-------------|
| `--vz-cut MIN MAX` | Apply a vertex-z window cut (exclusive bounds, in cm) to both MC and data after the species/truth-match filter. Default: `-8 2` (i.e., `-8 < vz < 2 cm`), matching the standard SULI target-window definition. |
| `--no-vz-cut` | Disable the vertex-z cut entirely. If both `--no-vz-cut` and `--vz-cut` are supplied, `--no-vz-cut` wins and a warning is printed. |

The cut is recorded in the per-run `README.md` written into the output directory.

### Event-level kinematic cut flags

| Flag | Default | Description |
|---|---|---|
| `--q2-cut MIN MAX` | off | Exclusive Q² range in GeV². Use `inf` for an unbounded upper end (e.g., `--q2-cut 2 inf`). |
| `--no-q2-cut` | — | Disable Q² cut even when `--sidis-cuts` is set. |
| `--w-cut MIN MAX` | off | Exclusive W range in GeV. |
| `--no-w-cut` | — | Disable W cut. |
| `--y-cut MIN MAX` | off | Exclusive inelasticity y range (dimensionless). |
| `--no-y-cut` | — | Disable y cut. |
| `--mx-cut MIN MAX` | off | Species-aware missing-mass cut (GeV): kp→`Mx_eKX`, pip→`Mx_epiX`, p→`Mx_epX`. Errors out cleanly for species without a defined Mx column (em, pim, kn). |
| `--no-mx-cut` | — | Disable Mx cut. |

All four cuts apply exclusive bounds: a row is kept when `MIN < value < MAX`.  The `vz` column uses the same convention (see above).

#### `--sidis-cuts` convenience flag

`--sidis-cuts` enables all four event-level cuts simultaneously with SIDIS-standard values: Q² > 2 GeV², W > 2 GeV, 0 < y < 0.75, and the species-appropriate Mx lower bound (1.6 GeV for kp, 1.5 for pip, 1.0 for p).  For species without a defined Mx cut (em, pim, kn), the Mx cut is silently skipped under `--sidis-cuts`.

Flag priority: explicit `--no-X-cut` always wins.  An explicit `--X-cut MIN MAX` overrides the `--sidis-cuts` default for that variable.  Otherwise the cut is off unless `--sidis-cuts` is set.

#### Output directory auto-suffix

When `--sidis-cuts` is active and `--outdir` was not explicitly provided, the output directory is automatically suffixed with `_sidis` (e.g., `figures/feature_audit/kp` becomes `figures/feature_audit/kp_sidis`).  This keeps primary and diagnostic audit outputs visually separable without requiring the user to rename the directory manually.

#### When to use which

The audit defaults to no event-level kinematic cuts because the trained classifier will see the uncut per-track sample at training time.  KEEP/CANDIDATE/DROP decisions should be based on the uncut comparison.  A second audit run with `--sidis-cuts` is a useful diagnostic: variables that disagree in the uncut comparison but agree in the SIDIS regime have their disagreement driven by exclusive contamination in data.  Variables that disagree in both have more fundamental MC mismodeling.  See `notes/cooper_10week_plan.md` Task 3a for the full workflow.

### Column pruning

The audit loads only the ROOT branches it needs (audit variables + species selector + cut columns).  On production-scale files this is dramatically faster than loading every branch.  Pass `--load-all-cols` to override and load everything — useful for debugging or for inspecting columns beyond the audit set.  The preamble prints the column counts so you can confirm pruning is engaging.

### Canonical invocations

```bash
# Primary audit (no event-level cuts; matches what the classifier sees at training):
python scripts/audit_species.py \
    --mc   /volatile/clas12/<username>/SULI/mc_pid_training_full.root \
    --data /volatile/clas12/<username>/SULI/data_pid_training_full.root \
    --species kp \
    --vars all_audit kinematics \
    --outdir figures/feature_audit/kp

# Diagnostic SIDIS-cut audit (event-level Q² > 2, W > 2, y < 0.75, Mx_eKX > 1.6):
python scripts/audit_species.py \
    --mc   /volatile/clas12/<username>/SULI/mc_pid_training_full.root \
    --data /volatile/clas12/<username>/SULI/data_pid_training_full.root \
    --species kp \
    --vars all_audit kinematics \
    --sidis-cuts
# (Output auto-suffixed to figures/feature_audit/kp_sidis/)
```

### What it produces

- All per-cell PNGs and `feature_audit_summary.csv` via the underlying `run_feature_audit` engine.
- A `README.md` in the output directory recording the species, selections, input files, variable list, and run timestamp — a provenance record so the audit is self-documenting weeks later.
- A hit-fraction alert section listing any `(variable, cell)` pairs where `|hit_frac_delta| > 0.05`.

### Further reference

- See `compare_mc_data.py` section above for all metric definitions and CSV column meanings.
- See `notes/cooper_10week_plan.md` Task 3a for the full audit workflow.

---

## `apply_bdt.py`

Apply a trained BDT (from `scripts/training/train_bdt.py`) to a ROOT file and
write an augmented ROOT with all original branches plus `bdt_score` (float32)
and optionally `bdt_pass` (bool).

The model's feature list is authoritative — extracted from the wrapper dict
`{"model": ..., "features": [...]}` in `model.joblib`.  Do not pass a feature
list at the CLI; the model is the source of truth.

**Threshold modes:**
- *(none)* — writes `bdt_score` only.
- `--threshold FLOAT` — also writes `bdt_pass = score > threshold` (strict `>`).
- `--threshold-csv PATH` — per-`(p, theta)` bin lookup; CSV needs columns
  `p_low, p_high, theta_low, theta_high, t_optimal`.  Tracks outside all bins
  get `bdt_pass = False`.

**Single-file usage:**

```bash
python scripts/apply_bdt.py \
    --input  /volatile/clas12/$USER/SULI/data_v01/run_001.root \
    --model  /work/clas12/$USER/SULI/models/tier1_v01/model.joblib \
    --output /volatile/clas12/$USER/SULI/scored_data_v01/run_001.root \
    --threshold-csv eval/v01/per_bin_thresholds.csv
```

For a directory of files, use the SLURM array wrapper `slurm/submit_apply_bdt.sh`.

---

## `plot_all_variables.py`

Diagnostic plots for all 54 columns in the training ntuple.  Produces one
PNG per variable (overall histogram + per-truth-class overlay) and a
missing-fraction summary.  Useful for a first sanity check after ntuple
production.

```bash
python scripts/plot_all_variables.py mc_training.root \
    --output-dir figures/variable_check/ --max-rows 500000
```
