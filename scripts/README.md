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
4. Computes three complementary statistics and returns them in a dict:
   - **KS distance** (`ks_test`) — bin-free CDF comparison; flags D > 0.05.
   - **χ²/ndof** (`chi2_test`) — global shape summary; accounts for
     uncertainty in both histograms.
   - **Bin-by-bin relative difference** (`relative_difference`) — tells you
     *where* the distributions differ.

Every statistical function has a thorough docstring explaining what it tests,
when to use it, and what its pitfalls are.  Read the docstrings.

### Quick start

```bash
# Run the self-test (synthetic data, no ROOT files needed)
python scripts/compare_mc_data.py

# Audit all ML features (requires ROOT files from processing_mc_pid_training.groovy)
python scripts/compare_mc_data.py \
    --mc   /path/to/mc_training.root \
    --data /path/to/data_training.root \
    --vars ml_features \
    --outdir figures/feature_audit/

# Audit candidate features (PCAL + FTOF layer 2)
python scripts/compare_mc_data.py \
    --mc   mc.root --data data.root \
    --vars candidate_features \
    --outdir figures/feature_audit/

# Quick test with only 100k rows
python scripts/compare_mc_data.py \
    --mc mc.root --data data.root --vars beta chi2pid \
    --max-rows 100000 --outdir /tmp/audit_test/
```

### Variable group aliases

| Alias | Variables |
|-------|-----------|
| `ml_features` | beta, chi2pid, ftof\_energy\_1A/1B, ftof\_time\_1A/1B, ftof\_path\_1A/1B, ecin/ecout\_energy, ecin/ecout\_time, ecin/ecout\_path, nphe\_htcc, nphe\_ltcc |
| `candidate_features` | pcal\_energy/time/path, ftof\_energy/time/path\_2 |
| `kinematics` | p, theta, phi, vz, sector |
| `all_audit` | ml\_features + candidate\_features |

### Output

- **`figures/feature_audit/<variable>/<variable>_p<lo>-<hi>_theta<lo>-<hi>.png`** — one overlay plot per (variable, slice).
- **`figures/feature_audit/feature_audit_summary.csv`** — one row per (variable, p-bin, θ-bin) with KS distance, χ²/ndof, hit fractions, and the `ks_flag` column.

### Using as a library

```python
from scripts.compare_mc_data import compare_distribution, run_feature_audit

stats = compare_distribution(
    df_mc, df_data, variable="beta",
    bins=60, normalize=True,
    selection_mc=(df_mc["p"] > 1.0) & (df_mc["p"] < 2.0),
    save_path="figures/feature_audit/beta/beta_p1-2_theta5-15.png",
)
print(stats["ks_distance"], stats["chi2_per_ndof"])
```

### Decision criteria (Week 3 audit)

| Criterion | Decision |
|-----------|----------|
| KS D < 0.05 in all 9 slices, residuals flat | **KEEP** |
| KS D > 0.05 in ≤ 3 slices, shape difference mild | **CANDIDATE** — investigate further |
| KS D > 0.05 in most slices, or residuals show coherent large bias | **DROP** — exclude from training |

Record decisions in `notes/feature_audit.md`.

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
