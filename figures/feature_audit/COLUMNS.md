# Audit CSV columns — what each one means

`feature_audit_summary.csv` is the main output of the feature audit. It has one row
per (variable, momentum cell, theta cell), so a single ML feature like `beta` produces
nine rows — one for each cell in the 3×3 (p, θ) grid. There are ~25 columns, which is
a lot to take in at once. This file groups them into three tiers: what you look at every
time, what explains the decision, and sanity-check bookkeeping you rarely need.

---

## Tier 1 — what you actually look at

These are the columns worth reading on every audit run.

- **`variable`** — the name of the feature being audited (e.g. `beta`, `chi2pid`,
  `ftof_energy_1A`). Each variable has nine rows in the CSV, one per (p, θ) cell.

- **`p_lo`, `p_hi`, `theta_lo`, `theta_hi`** — the kinematic cell this row covers.
  Momentum bounds in GeV/c, theta bounds in degrees. The audit uses a 3×3 grid:
  p ∈ {[1,2], [2,3], [3,5]} GeV/c and θ ∈ {[5°,15°], [15°,25°], [25°,35°]}.

- **`drift_decision`** — KEEP / CANDIDATE / DROP / UNKNOWN. This is the column you
  trust 80% of the time. KEEP means all three drift metrics agree the MC and data
  distributions are close enough for training. CANDIDATE means at least one metric
  raised a flag — open the plot and eyeball it before deciding. DROP means at least
  two metrics flag serious disagreement — this variable probably should not be used
  as a training feature without correction. UNKNOWN means there were no usable entries
  in the cell (both MC and data are empty here, or sentinel values ate everything).

- **`hit_frac_delta`** — the difference in detector hit rate between data and MC for
  this kinematic cell: `hit_frac_data_cell − hit_frac_mc_cell`. If |delta| > 0.05, the
  detector fired at noticeably different rates in MC vs data, which is a bigger problem
  than the distributions disagreeing on the *value* when it did fire. A variable with
  a large `hit_frac_delta` should be flagged regardless of what `drift_decision` says.
  Example: if `hit_frac_data_cell = 0.85` and `hit_frac_mc_cell = 0.80`, delta = +0.05
  — right at the boundary, worth a look.

- **`status`** — `ok` means the variable was found in both files and the cell had enough
  entries to fill a histogram. `missing_column` means the variable name was not present
  in the MC or data ntuple at all — that row has NaN everywhere else and you should check
  your `--vars` list. `empty` means the column exists but had no valid (non-sentinel)
  entries in this cell — the histogram would have been blank.

- **`decision_notes`** — this column does not come from the script; you fill it in by hand.
  Write a short note for each variable you've reviewed: what you saw in the plot,
  whether you agree with `drift_decision`, any override rationale, or anything Maria
  should know. One or two sentences per row is plenty. Blank is fine for KEEP rows
  you didn't need to look at.

---

## Tier 2 — the columns that explain WHY the decision came out the way it did

`drift_decision` is computed from three drift metrics. If a decision is CANDIDATE or
DROP, these columns tell you which metric triggered it and what kind of disagreement
is responsible.

- **`psi`** — Population Stability Index. This is a standard ML feature-drift metric.
  The script divides both the MC and data distributions into 10 equal-probability bins
  (using the data's quantile edges), then measures how differently the two distributions
  fill those bins. A PSI of 0 means identical distributions; larger means more drift.
  KEEP: PSI < 0.10. CANDIDATE: 0.10 ≤ PSI < 0.25. DROP: PSI ≥ 0.25. What it
  catches: overall distribution shift — if the bulk of the distribution has moved,
  PSI will flag it.

- **`wasserstein_norm`** — Wasserstein distance divided by the data IQR (interquartile
  range). Imagine sorting all the data values from smallest to largest, doing the same
  for MC, and measuring how much "probability mass" you'd have to push around to turn
  one distribution into the other. `wasserstein_norm` puts that distance in units of
  the typical spread of the data, so you can compare it across variables that live on
  completely different scales. KEEP: < 0.05. CANDIDATE: 0.05–0.20. DROP: ≥ 0.20.
  What it catches: both bulk shifts *and* shape changes. If `wasserstein_norm` is high
  but `psi` is low, the distributions have the same shape but different centers.

- **`max_local_residual`** — the worst single quantile bin's residual. The script splits
  both distributions into equal-count quantile bins and computes (data − MC)/MC for each
  bin; `max_local_residual` is the largest absolute value. KEEP: < 0.30. CANDIDATE:
  0.30–0.80. DROP: ≥ 0.80. What it catches: localized disagreement where the bulk
  looks fine but one part of the distribution (often a tail) is badly wrong. This matters
  for PID because the tail of `beta`, for example, is exactly where mis-ID happens — a
  variable can look fine on average but be useless for separating kaons from pions in
  the tail region.

If `drift_decision` is CANDIDATE or DROP, check which of these three metrics is
responsible. That tells you whether the disagreement is a bulk shift (PSI), a center
offset or shape change (Wasserstein), or a localized tail problem (max local residual).
Those three failure modes have different implications for whether the variable is salvageable.

---

## Tier 3 — sanity checks and counting

You rarely need to read these. They're here so the CSV is a complete record of everything
the script computed.

- **`n_total_mc_cell`, `n_total_data_cell`** — total tracks in this (p, θ) cell, before
  sentinel removal. Useful for checking whether a cell is just statistics-limited.

- **`n_hit_mc_cell`, `n_hit_data_cell`** — of those tracks, how many had a non-sentinel
  (i.e., actually reconstructed) value for this variable.

- **`hit_frac_mc_cell`, `hit_frac_data_cell`** — the two ratios that produce
  `hit_frac_delta`: `n_hit / n_total` within this kinematic cell. NaN if the cell is
  empty.

- **`hit_frac_mc`, `hit_frac_data`** — the same hit-fraction idea, but computed globally
  over the entire species-selected sample, not just this cell. A variable where
  `hit_frac_mc` ≈ 0 across the board is probably a dead detector or a column that was
  never filled — no need to audit it further.

- **`n_mc`, `n_data`** — the number of tracks that actually went into the histograms
  for this cell, after sentinel removal.

- **`wasserstein`** — raw Wasserstein distance in the variable's own units (GeV, ns, cm,
  etc.). Don't use this for cross-variable comparison; use `wasserstein_norm` instead.
  Kept here in case you want to know the absolute shift in physical units.

- **`ks_distance`, `ks_pvalue`** — legacy: the largest gap between the two cumulative
  distributions (Kolmogorov-Smirnov test). Kept as a sanity check; no longer used for
  decisions. Note: KS p-values are essentially meaningless at our sample sizes — with
  tens of thousands of tracks, even a trivially small difference is "statistically
  significant." Ignore the p-value; glance at `ks_distance` only if you want a
  second opinion.

- **`chi2`, `ndof`, `chi2_per_ndof`, `chi2_pvalue`** — legacy chi-squared test on the
  histograms. Same caveat as KS: the p-values are unreliable at large N. `chi2_per_ndof`
  can be a quick gut-check (much greater than 1 = obvious disagreement), but don't
  use it to override `drift_decision`.

- **`mean_rel_diff`, `max_abs_rel_diff`** — average and worst bin-by-bin relative
  difference (data − MC)/MC across all histogram bins. Useful for a quick sanity-check
  of the residual panel in the PNG: if `max_abs_rel_diff` is 0.80 but the plot looks
  fine, something is off.

- **`ks_flag`** — legacy boolean: True if `ks_distance` > 0.05. Ignore this; the new
  `drift_decision` column supersedes it.

---

## A 30-second workflow

When you open the CSV after an audit run, do this:

1. Sort by `drift_decision`. Look at everything DROP first, then CANDIDATE. KEEP rows
   with no flags need no further attention.
2. For each flagged row, check `hit_frac_delta`. If |delta| > 0.05, the detector hit
   rate itself disagrees — that's a separate problem from the shape comparison.
3. Open the PNG at
   `figures/feature_audit/<species>/<variable>/<variable>_p<X>-<X>_theta<X>-<X>.png`
   for each flagged cell. Eyeball the histogram overlay and residuals.
4. Write a short note in `decision_notes` for each variable you've reviewed. A sentence
   is enough: what you saw, whether you agree, anything unusual.
5. If a variable is KEEP in all nine cells, you're done with it.

---

## Need the formal definitions?

The metric implementations live in `scripts/compare_mc_data.py`. Each function
(`psi_score`, `wasserstein_normalized`, `max_local_residual`, `classify_drift`) has a
docstring with the exact formula, pitfalls, and references. Read those when you want
the rigor.
