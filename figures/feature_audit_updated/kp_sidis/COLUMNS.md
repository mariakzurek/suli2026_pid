# feature_audit_summary.csv — Column Glossary

This file is the **canonical** glossary for the `feature_audit_summary.csv`
output produced by `audit_species.py` / `compare_mc_data.run_feature_audit`.
It is maintained alongside the code in `scripts/` and is **copied** into each
new audit output directory at run time.

> **Do not edit copies in `figures/feature_audit_week3/` or other frozen
> output trees.**  Edit only this canonical source; copies are generated
> automatically.

---

## Identification columns

| Column | Type | Description |
|---|---|---|
| `variable` | str | Variable (branch) name from the ROOT ntuple. |
| `p_lo`, `p_hi` | float | Momentum bin edges in GeV/c for this cell. |
| `theta_lo`, `theta_hi` | float | Polar-angle bin edges in degrees for this cell. |

---

## Population counts

| Column | Type | Description |
|---|---|---|
| `n_mc` | int | MC entries used for histogramming and metrics in this cell — after sentinel removal, and after the negative-time filter if `--exclude-negative-time` was active for a timing variable. |
| `n_data` | int | Same for data. |
| `n_total_mc_cell` | int | Total MC rows in this (p, θ) cell before sentinel stripping. |
| `n_total_data_cell` | int | Same for data. |
| `n_hit_mc_cell` | int | MC rows in this cell with a valid (non-sentinel) value. |
| `n_hit_data_cell` | int | Same for data. |

---

## Detector-hit fractions (sentinel-based; always computed before negative-time filter)

These metrics measure **detector-hit availability** — how often the detector
recorded a value at all.  They are based solely on the `-9999` no-hit sentinel
(and `+9999` for `chi2pid`).  They are **not** affected by `--exclude-negative-time`.

| Column | Type | Description |
|---|---|---|
| `hit_frac_mc` | float | Fraction of MC rows in the **full species-selected parent DataFrame** (before any (p, θ) slice) with a valid (non-sentinel) value.  Reflects global detector acceptance. |
| `hit_frac_data` | float | Same for data, on the full parent DataFrame. |
| `hit_frac_mc_cell` | float | `n_hit_mc_cell / n_total_mc_cell` — per-cell hit fraction.  NaN if the cell is empty. |
| `hit_frac_data_cell` | float | Same for data in this cell. |
| `hit_frac_delta` | float | `hit_frac_data_cell − hit_frac_mc_cell`.  Positive means data fires more often than MC.  `\|Δ\| > 0.05` flags a hit-fraction mismatch that should be reviewed before trusting the shape comparison for this cell. |

---

## Negative-time removal fractions (timing filter only)

These metrics are **entirely separate** from the sentinel-based hit fractions
above.  They quantify timing quality **among tracks that did produce a hit**:
what fraction of post-sentinel valid-hit entries had a physically non-meaningful
negative timing value.

> **NaN convention:** both fields are `NaN` when `--exclude-negative-time` was
> not set, or when the variable is not in `{ecin_time, ecout_time, pcal_time}`
> (the three ECAL/PCAL calorimeter timing columns; FTOF scintillator timing
> columns are not affected).  A `NaN` means "filter was not applicable/not
> enabled"; a real `0.0` means the filter ran and found zero negative entries.
> Do **not** confuse a 1–3% removal fraction here with the ~45–55% sentinel
> hit fractions above.

| Column | Type | Description |
|---|---|---|
| `neg_time_frac_removed_mc` | float | Fraction of post-sentinel MC entries with value < 0 that were removed by the negative-time filter.  Computed as `count(arr < 0) / len(arr)` on the post-sentinel array, before the filter is applied.  NaN when flag is off or variable is not in `{ecin_time, ecout_time, pcal_time}`. |
| `neg_time_frac_removed_data` | float | Same metric for data. |
| `time_range_upper_mode` | str | Which upper-bound rule was used for the histogram x-axis when `--exclude-negative-time` is active and the variable is in `{ecin_time, ecout_time, pcal_time}`.  `"max"` — raw maximum of the combined positive-time array was used (the standard path).  `"percentile_99_fallback"` — the max exceeded 3× the 99th percentile (outlier guard triggered); 99th percentile was used instead.  `NaN` for all other variables or when the flag is off.  Inspect this column to know when the safety guard fired. |

---

## Shape-drift metrics (what to look at first)

These three metrics drive the `drift_decision`.  They operate on the
post-sentinel (and post-negative-time-filter, if enabled) event arrays.

| Column | Type | Description |
|---|---|---|
| `wasserstein` | float | Raw Wasserstein-1 distance between MC and data in native variable units.  Scale-dependent; use `wasserstein_norm` for cross-variable comparisons. |
| `wasserstein_norm` | float | `wasserstein / data_IQR` (dimensionless).  Thresholds: < 0.05 → KEEP, 0.05–0.20 → CANDIDATE, ≥ 0.20 → DROP. |
| `psi` | float | Population Stability Index with adaptive quantile binning.  Thresholds: < 0.10 → KEEP, 0.10–0.25 → CANDIDATE, ≥ 0.25 → DROP. |
| `max_local_residual` | float | Worst single-bin `\|data_i − mc_i\| / max(mc_i, 1e-6)` on equal-count bins.  Catches localised tail mismatches.  Thresholds: < 0.30 → KEEP, 0.30–0.80 → CANDIDATE, ≥ 0.80 → DROP. |
| `drift_decision` | str | Per-cell consensus decision: `KEEP` / `CANDIDATE` / `DROP` / `UNKNOWN`.  DROP if ≥ 2 metrics flag DROP; CANDIDATE if any one flags CANDIDATE or DROP; else KEEP.  NaN metrics are skipped. |

---

## Legacy / sanity-check columns

These are retained for backward compatibility and cross-checking.  They no
longer drive the `drift_decision`.

| Column | Type | Description |
|---|---|---|
| `ks_distance` | float | KS statistic D = sup\|CDF_data − CDF_MC\|.  Bin-free, operates on raw event arrays. |
| `ks_pvalue` | float | Two-sided KS p-value under H₀ (same distribution). |
| `ks_flag` | bool | True if `ks_distance > 0.05`. |
| `chi2` | float | χ² statistic from the normalised histogram comparison. |
| `ndof` | int | Degrees of freedom (N_bins_used − 1). |
| `chi2_per_ndof` | float | χ² / ndof.  Values near 1 indicate good agreement. |
| `chi2_pvalue` | float | P(χ²_ν ≥ χ²_obs). |
| `mean_rel_diff` | float | Mean of `\|data_i − mc_i\| / mc_i` over non-NaN bins. |
| `max_abs_rel_diff` | float | Max of same. |

---

## Processing status

| Column | Type | Description |
|---|---|---|
| `status` | str | `ok` — normal run, all metrics computed.  `missing_column` — variable not found in MC or data; all metric fields are NaN.  `empty` — variable present but no valid entries after sentinel (and optional negative-time) removal. |

---

## Operator-added column

| Column | Type | Description |
|---|---|---|
| `decision_notes` | str | **Not generated automatically.**  The audit operator appends this column manually after visual cross-check, recording any override rationale or observation for CANDIDATE/DROP variables. |

---

## Histogram-range convention

- **Non-timing variables (default):** 1st–99th percentile of the combined
  MC+data post-sentinel sample.  Robust against a few extreme outliers.
- **Calorimeter timing variables (`ecin_time`, `ecout_time`, `pcal_time`) with
  `--exclude-negative-time`:**
  - **Lower bound:** **1st percentile** (`np.percentile(combined, 1.0)`) of
    the combined positive-time MC+data distribution, computed *after* both the
    −9999 sentinel and any negative values (`arr < 0`) have been removed —
    i.e. the percentile never includes non-positive values.  Consistent with
    the 1st–99th-percentile convention used for all other variables; avoids a
    single sparse low outlier near the origin anchoring the axis too close to 0
    (the failure mode of the former first-non-empty-bin approach).  If the
    computed lower bound equals or exceeds the upper bound, it resets to `0.0`
    as a degenerate-range guard.
  - **Upper bound:** raw maximum of the combined positive-time array (primary
    path, recorded as `time_range_upper_mode = "max"`).  If `max > 3 × p99`,
    the 99th percentile is used instead (`time_range_upper_mode =
    "percentile_99_fallback"`) as a lightweight guard against a single
    pathological outlier blowing out the entire axis range.
- **Energy variables (`ecin_energy`, `ecout_energy`, `pcal_energy`,
  `ftof_energy_1A`, `ftof_energy_1B`, `ftof_energy_2`):** same 1st–99th
  percentile range as other variables, but with **double the requested bin
  count** (default 50 → 100) for finer resolution in the low-energy peak
  region.

## Normalization convention

Each histogram is independently normalised to unit area:
`density_i = count_i / (total_counts × bin_width_i)`.  The normalization
is applied separately for MC and data so that shape comparisons are
independent of sample size differences.
