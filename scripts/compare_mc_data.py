"""
compare_mc_data.py
------------------
Compare MC and data distributions for individual variables:
visual overlays (histogram + residuals) and quantitative statistics
(relative difference, chi-squared test, KS test).

This script is simultaneously a tool and a teaching document.  Every
function has a docstring that explains *what* it does, *when* to use
it, and *what can go wrong*.  Read the docstrings before reading the
code — they are the primary documentation for Cooper's Week 3 audit.

Typical usage on a pandas DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    import pandas as pd
    from scripts.compare_mc_data import compare_distribution

    df_mc   = pd.read_parquet("mc_training.parquet")
    df_data = pd.read_parquet("data_training.parquet")

    stats = compare_distribution(
        df_mc, df_data, variable="beta",
        bins=60, normalize=True,
        save_path="figures/feature_audit/beta.png",
    )
    print(stats)

Column map (processing_mc_pid_training.groovy, 54 columns)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Event-level (1-8):   runnum, evnum, helicity, Q2, W, x, y, nu
  Per-track (9-15):    pid, p, theta, phi, vz, sector, status
  ML features (16-31): beta, chi2pid,
                        ftof_energy_1A, ftof_energy_1B,
                        ftof_time_1A,   ftof_time_1B,
                        ftof_path_1A,   ftof_path_1B,
                        ecin_energy,    ecout_energy,
                        ecin_time,      ecout_time,
                        ecin_path,      ecout_path,
                        nphe_htcc,      nphe_ltcc
  PCAL + FTOF 2 (32-37): pcal_energy, pcal_time, pcal_path,
                          ftof_energy_2, ftof_time_2, ftof_path_2
  RICH (38-51):  rich_emilay ... rich_best_ntot  (cross-check only)
  MC truth (52-54): mc_matching_pid, mc_parent_pid, mc_match_quality

Missing-value convention
~~~~~~~~~~~~~~~~~~~~~~~~
  Groovy sentinel:  -9999 (any column) and +9999 (chi2pid only).
  Both sentinels are dropped before any histogram is filled.
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ──────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ──────────────────────────────────────────────────────────────────────────────

# Sentinel values produced by the groovy processing script.
# Any bin with these values is considered "not reconstructed" and is excluded.
SENTINEL_LOW  = -9999   # used for all columns
SENTINEL_HIGH =  9999   # used only for chi2pid (EB couldn't compute a pull)

# KS-distance threshold used in the Week 3 feature audit.
# A distance above this value in any (p, θ) slice flags the variable as
# CANDIDATE or DROP (pending visual inspection).  The choice of 0.05 is a
# pragmatic one: at our typical statistics it corresponds roughly to a
# p-value < 1e-4 on the KS test.
KS_FLAG_THRESHOLD = 0.05

# Default colour palette: MC in blue, Data in orange, matching the physics
# convention established in plot_all_variables.py.
COLOR_MC   = "steelblue"
COLOR_DATA = "darkorange"

# Feature groups, in the order used by the groovy ntuple (columns 16-37).
# These are the variables Cooper audits in Task 3a.
ML_FEATURES = [
    "beta", "chi2pid",
    "ftof_energy_1A", "ftof_energy_1B",
    "ftof_time_1A",   "ftof_time_1B",
    "ftof_path_1A",   "ftof_path_1B",
    "ecin_energy",    "ecout_energy",
    "ecin_time",      "ecout_time",
    "ecin_path",      "ecout_path",
    "nphe_htcc",      "nphe_ltcc",
]

CANDIDATE_FEATURES = [
    "pcal_energy", "pcal_time", "pcal_path",
    "ftof_energy_2", "ftof_time_2", "ftof_path_2",
]

KINEMATIC_FEATURES = ["p", "theta", "phi", "vz", "sector"]

# Default (p, θ) slice grid for the feature audit.
# Three momentum bins × three theta bins = nine cells, as specified in the
# Week 3 task description.
P_BINS_AUDIT     = [(1.0, 2.0), (2.0, 3.0), (3.0, 5.0)]   # GeV/c
THETA_BINS_AUDIT = [(5.0, 15.0), (15.0, 25.0), (25.0, 35.0)]  # degrees


# ──────────────────────────────────────────────────────────────────────────────
# Statistical functions
# ──────────────────────────────────────────────────────────────────────────────

def relative_difference(mc_hist, data_hist, mc_err=None, data_err=None):
    """
    Compute the bin-by-bin relative difference (data - MC) / MC with
    propagated uncertainty.

    WHAT IT DOES
    ------------
    For each bin i, compute:

        r_i = (d_i - m_i) / m_i

    where d_i is the data bin count and m_i is the MC bin count.
    r_i = 0 means perfect agreement.  r_i = +0.10 means data is 10% above MC.
    r_i = -0.25 means data is 25% below MC.

    The uncertainty on r_i is propagated from the uncertainties on d_i and m_i.
    For r = (d - m) / m, treating d and m as independent:

        σ_r² = (∂r/∂d)² σ_d² + (∂r/∂m)² σ_m²
             = (1/m)² σ_d² + (d/m²)² σ_m²
        σ_r  = sqrt( (σ_d/m)² + (d·σ_m/m²)² )

    WHEN TO USE IT
    --------------
    Use this as the residual panel under a histogram overlay.  It answers the
    local question "where does MC disagree with data, and by how much?"
    The residuals are what you *see* in physics papers as the bottom panel of
    a comparison plot.  This is the most intuitive single-number summary of
    local agreement.

    PITFALLS
    --------
    * Bins with few MC counts have huge error bars that look alarming but are
      just statistical fluctuations.  Do not flag a variable DROP solely
      because low-statistics bins at the edge of the range look bad.
    * This test does NOT produce a single summary number.  It tells you *where*
      the distributions differ but does not aggregate that information.  Use
      chi2_test() for a single-number summary.
    * If you normalize both histograms to unit area before comparing, r_i is
      shape-only — the overall rate difference is divided out.  Make sure you
      understand what the denominator is before interpreting the sign of r_i.

    Parameters
    ----------
    mc_hist : array-like, shape (N_bins,)
        MC histogram counts (or densities if normalize=True was applied).
    data_hist : array-like, shape (N_bins,)
        Data histogram counts (or densities).
    mc_err : array-like or None
        Uncertainty on each MC bin.  If None, Poisson errors sqrt(N) are used.
        Pass explicit errors if MC was reweighted or if you are using densities.
    data_err : array-like or None
        Uncertainty on each data bin.  If None, Poisson errors sqrt(N) are used.

    Returns
    -------
    rel_diff : np.ndarray, shape (N_bins,)
        (d - m) / m per bin.  NaN where mc_hist == 0.
    rel_err : np.ndarray, shape (N_bins,)
        Propagated 1σ uncertainty on rel_diff.  NaN where mc_hist == 0.
    """
    mc   = np.asarray(mc_hist,   dtype=float)
    data = np.asarray(data_hist, dtype=float)

    # Default to Poisson errors: σ = sqrt(N).  For normalised histograms the
    # caller must supply explicit errors because sqrt(density) is nonsensical.
    if mc_err is None:
        mc_err = np.sqrt(np.clip(mc, 0, None))
    else:
        mc_err = np.asarray(mc_err, dtype=float)

    if data_err is None:
        data_err = np.sqrt(np.clip(data, 0, None))
    else:
        data_err = np.asarray(data_err, dtype=float)

    # Build output arrays, defaulting to NaN (handles mc == 0 divisions cleanly)
    rel_diff = np.full_like(mc, np.nan)
    rel_err  = np.full_like(mc, np.nan)

    nonzero = mc != 0.0
    m  = mc[nonzero]
    d  = data[nonzero]
    se = mc_err[nonzero]
    sd = data_err[nonzero]

    rel_diff[nonzero] = (d - m) / m
    rel_err[nonzero]  = np.sqrt((sd / m) ** 2 + (d * se / m ** 2) ** 2)

    return rel_diff, rel_err


def chi2_test(mc_hist, data_hist, mc_err=None, data_err=None):
    """
    Chi-squared test comparing two histograms, returning χ², ndof, χ²/ndof,
    and the p-value.

    WHAT IT DOES
    ------------
    Tests whether two histograms are consistent with being drawn from the same
    distribution.  The test statistic is:

        χ² = Σ_i  (d_i - m_i)² / (σ_d_i² + σ_m_i²)

    summed only over bins where both σ_d_i > 0 and σ_m_i > 0 (i.e. at least
    one entry in both histograms in that bin).  Using the *combined* variance
    σ_d² + σ_m² in the denominator correctly accounts for the statistical
    uncertainty in *both* histograms — this is Pearson's chi-squared for two
    independent samples, not the goodness-of-fit formula.

    The number of degrees of freedom is set to N_bins_used - 1 when the
    histograms have been normalised to the same area (one constraint is
    removed because the integrals are forced equal), or N_bins_used when
    comparing unnormalised absolute counts.  The driver function
    compare_distribution() always normalises before calling chi2_test, so
    you will usually see ndof = N_used - 1.

    The p-value is P(χ²_ν ≥ χ²_obs) under the null hypothesis that the two
    histograms are drawn from the same distribution.  Small p-value (< 0.05)
    → reject the null → the distributions are statistically distinguishable.

    WHEN TO USE IT
    --------------
    Use when you want ONE number summarising the overall agreement between
    MC and data across the full distribution.  It is the standard test in
    HEP for "do these two histograms agree?"

    This function does NOT normalise the histograms.  The caller is
    responsible for normalisation.  compare_distribution() normalises before
    calling here, so the quoted ndof is N_used - 1.

    PITFALLS
    --------
    * χ² is very sensitive to bin choice.  With wide bins, real shape
      differences wash out.  With narrow bins, statistical noise inflates χ².
      A reasonable default is 50 bins covering the 1st–99th percentile range,
      which is what compare_distribution() uses.
    * A single bin with a large discrepancy can dominate χ² and make the
      global p-value tiny even when most bins agree.  Always look at the
      residual panel from relative_difference() to see where the problem is.
    * χ² assumes Gaussian bin uncertainties, which is only valid when N ≳ 5
      per bin.  Low-statistics tails violate this.  The chi2_test function
      already guards against this by excluding bins with zero counts, but
      bins with 1-4 counts are still borderline.
    * On very large samples, even tiny physical differences give a huge χ²
      and tiny p-value.  Use χ²/ndof and the residuals to judge practical
      significance, not just the p-value.

    Parameters
    ----------
    mc_hist : array-like, shape (N_bins,)
        MC bin counts or densities.  Do NOT call this function with raw counts
        if one sample is much larger than the other — normalise first.
    data_hist : array-like, shape (N_bins,)
        Data bin counts or densities.
    mc_err : array-like or None
        Uncertainty on each MC bin.  If None, Poisson errors sqrt(N) are used.
    data_err : array-like or None
        Uncertainty on each data bin.  If None, Poisson errors sqrt(N) are used.

    Returns
    -------
    chi2 : float
        The chi-squared test statistic.
    ndof : int
        Degrees of freedom (N_bins_used - 1).
    chi2_per_ndof : float
        chi2 / ndof.  Values near 1 indicate good agreement.
    p_value : float
        P(χ²_ν ≥ χ²_obs).  Values < 0.05 are conventionally called
        "statistically significant" discrepancies.
    """
    mc   = np.asarray(mc_hist,   dtype=float)
    data = np.asarray(data_hist, dtype=float)

    if mc_err is None:
        mc_err = np.sqrt(np.clip(mc, 0, None))
    else:
        mc_err = np.asarray(mc_err, dtype=float)

    if data_err is None:
        data_err = np.sqrt(np.clip(data, 0, None))
    else:
        data_err = np.asarray(data_err, dtype=float)

    # Only include bins where both histograms have nonzero uncertainty.
    # Bins with σ = 0 either have zero counts (uninformative) or represent
    # a known fixed value (not a stochastic measurement).
    var_total = mc_err ** 2 + data_err ** 2
    good_bins = var_total > 0.0

    n_used = int(np.sum(good_bins))

    if n_used == 0:
        # Edge case: no usable bins (e.g. all counts are zero).
        return np.nan, 0, np.nan, np.nan

    residuals_sq = (data[good_bins] - mc[good_bins]) ** 2 / var_total[good_bins]
    chi2_val  = float(np.sum(residuals_sq))

    # ndof = N_used - 1 because the normalisation constraint removes one
    # degree of freedom.  If you call this on un-normalised histograms,
    # manually set ndof = N_used.
    ndof = n_used - 1
    if ndof <= 0:
        return chi2_val, ndof, np.nan, np.nan

    chi2_per_ndof = chi2_val / ndof
    p_value = float(sp_stats.chi2.sf(chi2_val, df=ndof))

    return chi2_val, ndof, chi2_per_ndof, p_value


def ks_test(mc_array, data_array):
    """
    Two-sample Kolmogorov-Smirnov test comparing MC and data event-by-event.

    WHAT IT DOES
    ------------
    The KS test compares the empirical cumulative distribution functions (ECDFs)
    of two samples.  It finds the maximum absolute difference between the two
    CDFs:

        D = sup_x | CDF_data(x) - CDF_MC(x) |

    D is called the KS distance (or KS statistic).  It lives in [0, 1].
    D = 0 means the two ECDFs are identical.  D = 1 means they are completely
    separated.  The function also returns a p-value under the null hypothesis
    H₀: "MC and data are drawn from the same continuous distribution."

    KS works directly on the raw event arrays (not histograms), so it has no
    bin-choice dependence.  Use it as a quick first screening of all features
    before committing to the slower histogram comparison.

    In the Week 3 feature audit, the threshold is KS distance > 0.05 in any
    (p, θ) slice — see KS_FLAG_THRESHOLD at the top of this file.

    WHEN TO USE IT
    --------------
    Use KS as a fast, bin-free screen: if D < 0.05 everywhere, the variable
    is probably fine.  If D > 0.05 in one or more slices, look at the
    histogram overlay from compare_distribution() to understand *where* and
    *how* the shapes differ.

    KS is better than χ² for small samples (it makes no Gaussian-bin
    assumption) and for detecting differences in the *body* of a distribution
    (not just the tails).

    PITFALLS
    --------
    * KS is insensitive to differences in the far tails where both CDFs are
      already near 0 or near 1 — the tails saturate and the differences
      become invisible to the supremum.  If you suspect a tail difference
      (e.g. high-energy ECAL deposits), look at the residuals instead.
    * On very large samples (hundreds of thousands of events), even physically
      tiny differences produce a KS distance that exceeds 0.05 and a p-value
      of essentially zero.  At that point, D is still meaningful as a measure
      of *how different* the distributions are, but the p-value becomes
      useless as a decision criterion.  The Week 3 audit uses D > 0.05 as
      the flag threshold precisely to avoid this problem — D = 0.05 is a
      physically meaningful 5% maximum CDF difference regardless of N.
    * KS does not tell you *where* the distributions differ or what the
      difference looks like.  Always follow up a flagged KS test with a
      histogram overlay.
    * KS requires the data arrays to contain only valid (non-sentinel) values.
      The driver function compare_distribution() strips sentinels before
      calling ks_test.

    Parameters
    ----------
    mc_array : array-like
        1-D array of MC values for the variable (sentinels already removed).
    data_array : array-like
        1-D array of data values for the variable (sentinels already removed).

    Returns
    -------
    ks_distance : float
        The KS statistic D = sup|CDF_data - CDF_MC|.
    p_value : float
        Two-sided p-value under H₀.  Small → distributions are distinguishable.
    """
    mc_arr   = np.asarray(mc_array,   dtype=float)
    data_arr = np.asarray(data_array, dtype=float)

    if len(mc_arr) == 0 or len(data_arr) == 0:
        warnings.warn("ks_test: one or both arrays are empty — returning NaN.",
                      RuntimeWarning, stacklevel=2)
        return np.nan, np.nan

    result = sp_stats.ks_2samp(mc_arr, data_arr)
    return float(result.statistic), float(result.pvalue)


# ──────────────────────────────────────────────────────────────────────────────
# Sentinel stripping
# ──────────────────────────────────────────────────────────────────────────────

def _strip_sentinels(arr, variable):
    """
    Remove sentinel values from a numpy array.

    Uses SENTINEL_LOW (-9999) for all variables.  For chi2pid also removes
    SENTINEL_HIGH (+9999), which the Event Builder writes when it cannot
    compute a beta pull.

    Parameters
    ----------
    arr : np.ndarray
    variable : str

    Returns
    -------
    np.ndarray
        Filtered array with sentinels removed.
    """
    arr = np.asarray(arr, dtype=float)
    mask = arr == SENTINEL_LOW
    if variable == "chi2pid":
        mask = mask | (arr == SENTINEL_HIGH)
    return arr[~mask]


# ──────────────────────────────────────────────────────────────────────────────
# Driver function
# ──────────────────────────────────────────────────────────────────────────────

def compare_distribution(
    df_mc,
    df_data,
    variable,
    bins=50,
    range=None,
    normalize=True,
    selection_mc=None,
    selection_data=None,
    label_mc="MC",
    label_data="Data",
    title=None,
    save_path=None,
    figsize=(10, 6),
):
    """
    Make a 2-panel diagnostic plot comparing MC and data for one variable,
    and return a dictionary of statistics.

    Top panel:    histogram overlay (MC filled, Data points with error bars)
    Bottom panel: (data - MC) / MC residuals with propagated error bars,
                  plus a dashed reference line at 0 (perfect agreement) and
                  at ±20% (a common practical tolerance in HEP).

    HOW TO READ THE PLOT
    --------------------
    * If the histograms in the top panel overlap nearly perfectly and the
      residual panel is flat near zero, the variable agrees well → KEEP.
    * If the histograms are systematically offset in shape (e.g. one is
      broader) and the residuals show a coherent pattern (positive on one
      side, negative on the other), that is a shape difference → CANDIDATE
      or DROP depending on magnitude.
    * If the residuals are large (+/-50% or more) in most bins, the variable
      has a real MC/data discrepancy and should probably be excluded from
      training until understood → DROP.
    * The KS distance and chi²/ndof in the statistics dict give you the
      two complementary one-number summaries.

    DESIGN DECISION: NORMALISATION
    -------------------------------
    By default (normalize=True), both histograms are normalised to unit area
    before comparison.  This compares *shapes*, ignoring any difference in
    the total number of events.  For the feature audit this is almost always
    what you want: we have different numbers of MC and data events, and we
    care about shape agreement, not rate.

    Set normalize=False only if you want to compare absolute counts — for
    example, when studying yield differences or when both samples are drawn
    from the same parent distribution and you want to check overall
    normalisation.

    Parameters
    ----------
    df_mc : pd.DataFrame
        MC DataFrame (e.g. read from the training ntuple ROOT file).
    df_data : pd.DataFrame
        Data DataFrame (e.g. RGA pass-2 EB-K+ tracks).
    variable : str
        Column name to compare.  Must exist in both DataFrames.
    bins : int or array-like
        Number of histogram bins or explicit bin edges.  Default 50.
    range : (float, float) or None
        Histogram range (x_min, x_max).  If None, the 1st–99th percentile
        of the *combined* (MC + data) clean sample is used.  This is more
        robust than [min, max] when a few outliers are present.
    normalize : bool
        If True (default), normalise both histograms to unit area before
        computing chi2 and relative differences.
    selection_mc : array-like of bool or None
        Boolean mask to apply to df_mc before processing.  None = no cut.
        Example: ``(df_mc["p"] > 1.0) & (df_mc["p"] < 2.0)``
    selection_data : array-like of bool or None
        Boolean mask to apply to df_data before processing.  None = no cut.
    label_mc : str
        Legend label for MC.  Default "MC".
    label_data : str
        Legend label for Data.  Default "Data".
    title : str or None
        Figure title.  If None, uses "{variable}  |  MC vs Data".
    save_path : str or None
        If provided, save the figure to this path (parent directory is
        created automatically).
    figsize : (float, float)
        Matplotlib figure size.  Default (10, 6).

    Returns
    -------
    stats : dict
        Keys:
          'variable'       : str   — the variable name
          'n_mc'           : int   — number of MC events after selection + sentinel strip
          'n_data'         : int   — number of data events after selection + sentinel strip
          'hit_frac_mc'    : float — fraction of MC rows with a non-sentinel value
                                     (before the (p,θ) selection, so it reflects
                                     detector acceptance, not the slice)
          'hit_frac_data'  : float — same for data
          'ks_distance'    : float — KS statistic D
          'ks_pvalue'      : float — KS p-value
          'chi2'           : float — chi-squared statistic
          'ndof'           : int   — degrees of freedom
          'chi2_per_ndof'  : float — chi2 / ndof (near 1 = good agreement)
          'chi2_pvalue'    : float — chi-squared p-value
          'mean_rel_diff'  : float — mean of |rel_diff| over non-NaN bins
          'max_abs_rel_diff': float — max of |rel_diff| over non-NaN bins
          'ks_flag'        : bool  — True if ks_distance > KS_FLAG_THRESHOLD (0.05)
    """
    # ── 1. Apply selections ───────────────────────────────────────────────────
    df_mc_sel   = df_mc   if selection_mc   is None else df_mc[selection_mc]
    df_data_sel = df_data if selection_data is None else df_data[selection_data]

    # ── 2. Extract the column ─────────────────────────────────────────────────
    if variable not in df_mc_sel.columns:
        raise KeyError(f"Variable '{variable}' not found in df_mc columns: "
                       f"{list(df_mc_sel.columns)}")
    if variable not in df_data_sel.columns:
        raise KeyError(f"Variable '{variable}' not found in df_data columns: "
                       f"{list(df_data_sel.columns)}")

    raw_mc   = df_mc_sel[variable].to_numpy(dtype=float)
    raw_data = df_data_sel[variable].to_numpy(dtype=float)

    # Hit-fraction computed on the full (unsliced) DataFrame so that it
    # reflects detector acceptance rather than the kinematic slice.
    hit_frac_mc   = float(np.mean(df_mc[variable].to_numpy(dtype=float) != SENTINEL_LOW))
    hit_frac_data = float(np.mean(df_data[variable].to_numpy(dtype=float) != SENTINEL_LOW))

    # ── 3. Strip sentinels ────────────────────────────────────────────────────
    arr_mc   = _strip_sentinels(raw_mc,   variable)
    arr_data = _strip_sentinels(raw_data, variable)

    # ── 4. Determine histogram range ──────────────────────────────────────────
    if range is None:
        combined = np.concatenate([arr_mc, arr_data])
        if len(combined) == 0:
            warnings.warn(f"compare_distribution: no valid data for '{variable}' "
                          "after sentinel removal.  Returning empty stats.",
                          RuntimeWarning, stacklevel=2)
            return _empty_stats(variable, 0, 0)
        plo = float(np.percentile(combined, 1))
        phi = float(np.percentile(combined, 99))
        if plo == phi:
            plo, phi = plo - 1.0, phi + 1.0
        hist_range = (plo, phi)
    else:
        hist_range = tuple(range)

    # ── 5. Build histograms ───────────────────────────────────────────────────
    # Use the raw *counts* (not density) for Poisson error assignment,
    # then separately compute the normalised version if requested.
    counts_mc,   bin_edges = np.histogram(arr_mc,   bins=bins, range=hist_range)
    counts_data, _         = np.histogram(arr_data, bins=bins, range=hist_range)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths  = np.diff(bin_edges)

    # Poisson errors on the raw counts
    err_mc_counts   = np.sqrt(counts_mc.astype(float))
    err_data_counts = np.sqrt(counts_data.astype(float))

    if normalize:
        # Normalise to unit area (density = count / (total * bin_width)).
        # Propagate errors: σ_density = σ_count / (total * bin_width).
        # If total == 0, leave as zeros to avoid division.
        total_mc   = float(counts_mc.sum())
        total_data = float(counts_data.sum())

        denom_mc   = total_mc   * bin_widths if total_mc   > 0 else np.ones_like(bin_widths)
        denom_data = total_data * bin_widths if total_data > 0 else np.ones_like(bin_widths)

        hist_mc   = counts_mc.astype(float)   / denom_mc
        hist_data = counts_data.astype(float) / denom_data
        err_mc    = err_mc_counts   / denom_mc
        err_data  = err_data_counts / denom_data

        ylabel_top = "Probability density (normalised)"
    else:
        hist_mc   = counts_mc.astype(float)
        hist_data = counts_data.astype(float)
        err_mc    = err_mc_counts
        err_data  = err_data_counts
        ylabel_top = "Counts"

    # ── 6. Compute statistics ─────────────────────────────────────────────────
    # KS test uses raw event arrays (no binning)
    ks_dist, ks_pval = ks_test(arr_mc, arr_data)

    # chi2 test on the (possibly normalised) histograms
    chi2_val, ndof, chi2_per_ndof, chi2_pval = chi2_test(
        hist_mc, hist_data, mc_err=err_mc, data_err=err_data
    )

    # Relative difference (for residual panel)
    rel_diff, rel_err = relative_difference(
        hist_mc, hist_data, mc_err=err_mc, data_err=err_data
    )

    finite_rd = rel_diff[np.isfinite(rel_diff)]
    mean_rel_diff     = float(np.mean(np.abs(finite_rd)))   if len(finite_rd) > 0 else np.nan
    max_abs_rel_diff  = float(np.max(np.abs(finite_rd)))    if len(finite_rd) > 0 else np.nan

    stats = {
        "variable"        : variable,
        "n_mc"            : len(arr_mc),
        "n_data"          : len(arr_data),
        "hit_frac_mc"     : hit_frac_mc,
        "hit_frac_data"   : hit_frac_data,
        "ks_distance"     : ks_dist,
        "ks_pvalue"       : ks_pval,
        "chi2"            : chi2_val,
        "ndof"            : ndof,
        "chi2_per_ndof"   : chi2_per_ndof,
        "chi2_pvalue"     : chi2_pval,
        "mean_rel_diff"   : mean_rel_diff,
        "max_abs_rel_diff": max_abs_rel_diff,
        "ks_flag"         : (ks_dist > KS_FLAG_THRESHOLD) if np.isfinite(ks_dist) else False,
    }

    # ── 7. Build figure ───────────────────────────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.05)

    # --- Top panel: histogram overlay ----------------------------------------
    # MC: filled step histogram (like a shaded region)
    ax_top.stairs(
        hist_mc, bin_edges,
        color=COLOR_MC, linewidth=1.5,
        fill=True, alpha=0.35, label=f"{label_mc}  (N={len(arr_mc):,})",
    )
    ax_top.stairs(
        hist_mc, bin_edges,
        color=COLOR_MC, linewidth=1.5,
        fill=False,
    )

    # Data: points with error bars (physics convention: data = points, MC = filled)
    ax_top.errorbar(
        bin_centers, hist_data, yerr=err_data,
        fmt="o", ms=3, color=COLOR_DATA, linewidth=1.0, capsize=2,
        label=f"{label_data}  (N={len(arr_data):,})",
    )

    ax_top.set_ylabel(ylabel_top, fontsize=10)
    ax_top.legend(fontsize=9, framealpha=0.8)
    ax_top.set_xlim(hist_range)
    ax_top.tick_params(labelbottom=False)

    # Statistics annotation in top-right corner
    if np.isfinite(ks_dist):
        ks_str = f"KS D = {ks_dist:.4f}"
        ks_str += " ⚑" if stats["ks_flag"] else ""
    else:
        ks_str = "KS D = N/A"

    if np.isfinite(chi2_per_ndof):
        chi2_str = f"χ²/ndof = {chi2_per_ndof:.2f}  (p = {chi2_pval:.2e})"
    else:
        chi2_str = "χ²/ndof = N/A"

    annotation = f"{ks_str}\n{chi2_str}"
    ax_top.text(
        0.98, 0.97, annotation,
        transform=ax_top.transAxes, ha="right", va="top",
        fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
    )

    plot_title = title if title is not None else f"{variable}  |  {label_mc} vs {label_data}"
    ax_top.set_title(plot_title, fontsize=11, fontweight="bold")

    # --- Bottom panel: residuals ---------------------------------------------
    # Plot (data - MC) / MC with propagated error bars
    ax_bot.axhline(0.0,  color="gray",   linewidth=1.0, linestyle="--", zorder=1)
    ax_bot.axhline(+0.2, color="gray",   linewidth=0.6, linestyle=":",  zorder=1)
    ax_bot.axhline(-0.2, color="gray",   linewidth=0.6, linestyle=":",  zorder=1)

    # Only plot bins where rel_diff is finite (mc > 0)
    good = np.isfinite(rel_diff)
    if good.any():
        ax_bot.errorbar(
            bin_centers[good], rel_diff[good], yerr=rel_err[good],
            fmt="o", ms=3, color=COLOR_DATA, linewidth=1.0, capsize=2,
        )

    ax_bot.set_ylabel("(Data−MC)/MC", fontsize=9)
    ax_bot.set_xlabel(variable, fontsize=10)
    ax_bot.set_xlim(hist_range)

    # Clamp the residual y-axis to ±1.0 so a single outlier bin doesn't
    # swallow the rest of the plot.  Bins outside this range are still
    # computed; they just fall off the visible axes.
    ax_bot.set_ylim(-1.0, 1.0)

    # ── 8. Save ───────────────────────────────────────────────────────────────
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)

    return stats


def _empty_stats(variable, n_mc, n_data):
    """Return a stats dict populated with NaN values (empty-data guard)."""
    return {
        "variable"        : variable,
        "n_mc"            : n_mc,
        "n_data"          : n_data,
        "hit_frac_mc"     : np.nan,
        "hit_frac_data"   : np.nan,
        "ks_distance"     : np.nan,
        "ks_pvalue"       : np.nan,
        "chi2"            : np.nan,
        "ndof"            : 0,
        "chi2_per_ndof"   : np.nan,
        "chi2_pvalue"     : np.nan,
        "mean_rel_diff"   : np.nan,
        "max_abs_rel_diff": np.nan,
        "ks_flag"         : False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Feature-audit batch runner
# ──────────────────────────────────────────────────────────────────────────────

def run_feature_audit(
    df_mc,
    df_data,
    variables,
    output_dir,
    p_bins=None,
    theta_bins=None,
    bins=50,
    normalize=True,
    label_mc="MC",
    label_data="Data",
):
    """
    Run compare_distribution() over a list of variables and (p, θ) slices,
    saving one PNG per (variable, slice) and returning a summary DataFrame.

    This is the primary entry point for the Week 3 Task 3a feature audit.
    It automates the 9-cell (p, θ) grid and saves everything under
    ``output_dir/<variable>/<variable>_p<plo>-<phi>_theta<tlo>-<thi>.png``.

    The returned summary DataFrame has one row per (variable, p-bin, θ-bin)
    and the full statistics dict flattened into columns.  It is intended to
    be written to CSV and pasted into ``notes/feature_audit.md``.

    Parameters
    ----------
    df_mc : pd.DataFrame
    df_data : pd.DataFrame
    variables : list of str
        Variable names to audit.
    output_dir : str
        Root output directory.  Created if absent.
    p_bins : list of (float, float) or None
        Momentum slice edges in GeV/c.  Defaults to P_BINS_AUDIT.
    theta_bins : list of (float, float) or None
        Theta slice edges in degrees.  Defaults to THETA_BINS_AUDIT.
    bins : int
        Histogram bin count passed to compare_distribution().
    normalize : bool
        Normalise histograms to unit area.  Default True.
    label_mc : str
    label_data : str

    Returns
    -------
    pd.DataFrame
        Summary table with columns: variable, p_lo, p_hi, theta_lo, theta_hi,
        n_mc, n_data, hit_frac_mc, hit_frac_data, ks_distance, ks_pvalue,
        chi2, ndof, chi2_per_ndof, chi2_pvalue, mean_rel_diff,
        max_abs_rel_diff, ks_flag.
    """
    if p_bins is None:
        p_bins = P_BINS_AUDIT
    if theta_bins is None:
        theta_bins = THETA_BINS_AUDIT

    # Check that p and theta are available (needed for slicing)
    for col in ("p", "theta"):
        if col not in df_mc.columns:
            raise KeyError(f"run_feature_audit requires column '{col}' in df_mc")
        if col not in df_data.columns:
            raise KeyError(f"run_feature_audit requires column '{col}' in df_data")

    rows = []

    for var in variables:
        var_dir = os.path.join(output_dir, var)
        os.makedirs(var_dir, exist_ok=True)

        for (p_lo, p_hi) in p_bins:
            for (t_lo, t_hi) in theta_bins:
                sel_mc = (
                    (df_mc["p"] > p_lo) & (df_mc["p"] <= p_hi) &
                    (df_mc["theta"] > t_lo) & (df_mc["theta"] <= t_hi)
                )
                sel_data = (
                    (df_data["p"] > p_lo) & (df_data["p"] <= p_hi) &
                    (df_data["theta"] > t_lo) & (df_data["theta"] <= t_hi)
                )

                tag   = f"p{p_lo:.0f}-{p_hi:.0f}_theta{t_lo:.0f}-{t_hi:.0f}"
                fname = f"{var}_{tag}.png"
                save_path = os.path.join(var_dir, fname)
                title = (f"{var}   {label_mc} vs {label_data}\n"
                         f"p ∈ [{p_lo}, {p_hi}] GeV/c,  θ ∈ [{t_lo}°, {t_hi}°]")

                st = compare_distribution(
                    df_mc, df_data, variable=var,
                    bins=bins, normalize=normalize,
                    selection_mc=sel_mc, selection_data=sel_data,
                    label_mc=label_mc, label_data=label_data,
                    title=title,
                    save_path=save_path,
                )

                row = dict(p_lo=p_lo, p_hi=p_hi, theta_lo=t_lo, theta_hi=t_hi)
                row.update(st)
                rows.append(row)

        print(f"  {var:30s}  done  ({len(p_bins) * len(theta_bins)} slices)")

    summary = pd.DataFrame(rows)
    # Reorder columns so variable comes first
    cols = ["variable", "p_lo", "p_hi", "theta_lo", "theta_hi"] + [
        c for c in summary.columns if c not in ("variable", "p_lo", "p_hi", "theta_lo", "theta_hi")
    ]
    summary = summary[cols]
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Command-line interface
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description=(
            "MC vs Data distribution comparison for SULI 2026 ML PID feature audit.\n"
            "Reads two ROOT/Parquet files, runs compare_distribution() over the\n"
            "requested variables in a (p, θ) grid, and saves plots + a CSV summary."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Compare all ML features across the default (p, θ) grid
  python compare_mc_data.py \\
      --mc   mc_training.root \\
      --data data_training.root \\
      --vars ml_features \\
      --outdir figures/feature_audit/

# Compare a single variable with a custom range
  python compare_mc_data.py \\
      --mc   mc.root --data data.root \\
      --vars beta --bins 80 \\
      --outdir figures/feature_audit/

# Compare candidate features (PCAL + FTOF layer 2)
  python compare_mc_data.py \\
      --mc   mc.root --data data.root \\
      --vars candidate_features \\
      --outdir figures/feature_audit/

Feature group aliases
---------------------
  ml_features        : beta, chi2pid, FTOF 1A/1B, ECAL inner/outer, nphe_htcc/ltcc
  candidate_features : pcal_energy/time/path, ftof_energy/time/path_2
  kinematics         : p, theta, phi, vz, sector
  all_audit          : ml_features + candidate_features
""",
    )
    p.add_argument("--mc",   required=True,
                   help="Path to MC ROOT file (PhysicsEvents tree) or Parquet file.")
    p.add_argument("--data", required=True,
                   help="Path to data ROOT file (PhysicsEvents tree) or Parquet file.")
    p.add_argument("--vars", nargs="+", default=["ml_features"],
                   help=("Variable names or group aliases "
                         "(ml_features | candidate_features | kinematics | all_audit). "
                         "Default: ml_features"))
    p.add_argument("--bins", type=int, default=50,
                   help="Number of histogram bins.  Default 50.")
    p.add_argument("--outdir", default="figures/feature_audit",
                   help="Output directory.  Default: figures/feature_audit/")
    p.add_argument("--label-mc",   default="MC",   help="MC label in plots.")
    p.add_argument("--label-data", default="Data", help="Data label in plots.")
    p.add_argument("--no-normalize", action="store_true",
                   help="Skip histogram normalisation (compare raw counts).")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Load at most N rows from each file (for quick tests).")
    return p


def _resolve_variables(var_list):
    """Expand group aliases into concrete variable lists."""
    aliases = {
        "ml_features"       : ML_FEATURES,
        "candidate_features": CANDIDATE_FEATURES,
        "kinematics"        : KINEMATIC_FEATURES,
        "all_audit"         : ML_FEATURES + CANDIDATE_FEATURES,
    }
    result = []
    for v in var_list:
        if v in aliases:
            result.extend(aliases[v])
        else:
            result.append(v)
    # Deduplicate while preserving order
    seen = set()
    out  = []
    for v in result:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _load_file(path, max_rows=None):
    """Load a ROOT or Parquet file into a pandas DataFrame."""
    if path.endswith(".parquet") or path.endswith(".pq"):
        df = pd.read_parquet(path)
        if max_rows is not None:
            df = df.iloc[:max_rows]
        print(f"  Loaded {len(df):,} rows from {path}  (parquet)")
        return df

    # ROOT: requires uproot
    try:
        import uproot
    except ImportError:
        sys.exit("uproot is required to read ROOT files.  "
                 "Install with: conda install -c conda-forge uproot")

    # Support "file.root:TreeName" syntax
    if ":" in path:
        parts = path.rsplit(":", 1)
        if os.path.exists(parts[0]):
            file_path, tree_name = parts[0], parts[1]
        else:
            file_path, tree_name = path, "PhysicsEvents"
    else:
        file_path, tree_name = path, "PhysicsEvents"

    with uproot.open(file_path) as f:
        tree = f[tree_name]
        entry_stop = max_rows
        df = tree.arrays(library="pd", entry_stop=entry_stop)

    print(f"  Loaded {len(df):,} rows from {file_path}  (tree: {tree_name})")
    return df


def main(argv=None):
    parser = _build_parser()
    args   = parser.parse_args(argv)

    print(f"\n{'='*60}")
    print("compare_mc_data.py  —  MC vs Data feature audit")
    print(f"{'='*60}")

    print(f"\nLoading MC:   {args.mc}")
    df_mc   = _load_file(args.mc,   max_rows=args.max_rows)
    print(f"Loading Data: {args.data}")
    df_data = _load_file(args.data, max_rows=args.max_rows)

    variables = _resolve_variables(args.vars)
    print(f"\nVariables to audit ({len(variables)}):  {variables}")
    print(f"Output directory: {args.outdir}")
    print(f"Bins: {args.bins}   Normalize: {not args.no_normalize}")
    print(f"\nRunning (p, θ) audit grid: {len(P_BINS_AUDIT)} × {len(THETA_BINS_AUDIT)} = "
          f"{len(P_BINS_AUDIT)*len(THETA_BINS_AUDIT)} slices per variable\n")

    summary = run_feature_audit(
        df_mc, df_data,
        variables=variables,
        output_dir=args.outdir,
        bins=args.bins,
        normalize=not args.no_normalize,
        label_mc=args.label_mc,
        label_data=args.label_data,
    )

    # Save summary CSV
    csv_path = os.path.join(args.outdir, "feature_audit_summary.csv")
    summary.to_csv(csv_path, index=False, float_format="%.6g")
    print(f"\nSummary CSV saved to: {csv_path}")

    # Print a quick per-variable max-KS table
    print(f"\n{'─'*72}")
    print(f"{'Variable':<22}  {'max KS dist':>11}  {'flag':>5}  "
          f"{'max χ²/ndof':>11}  {'N_MC (all slices)':>18}")
    print(f"{'─'*72}")
    for var in variables:
        sub = summary[summary["variable"] == var]
        if sub.empty:
            continue
        max_ks   = sub["ks_distance"].max()
        flagged  = bool(sub["ks_flag"].any())
        max_chi2 = sub["chi2_per_ndof"].max()
        n_mc_tot = sub["n_mc"].sum()
        flag_str = "⚑" if flagged else " "
        print(f"  {var:<20}  {max_ks:>11.4f}  {flag_str:>5}  "
              f"{max_chi2:>11.2f}  {n_mc_tot:>18,}")
    print(f"{'─'*72}")
    print(f"\n  Threshold: KS D > {KS_FLAG_THRESHOLD} → ⚑ CANDIDATE or DROP")
    print("  Next step: open each flagged variable's overlay plots and decide")
    print("  KEEP / CANDIDATE / DROP in notes/feature_audit.md\n")


# ──────────────────────────────────────────────────────────────────────────────
# __main__ — assertions + worked example with synthetic data
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math

    # If the user supplied CLI flags, go straight to the argument parser.
    # Otherwise, run the assertions + worked example (no ROOT files needed).
    if len(sys.argv) > 1:
        main()
        sys.exit(0)

    print("Running compare_mc_data.py self-test …\n")

    # ── relative_difference assertions ───────────────────────────────────────
    mc   = np.array([10.0, 20.0,  0.0, 5.0], dtype=float)
    data = np.array([12.0, 18.0,  3.0, 5.0], dtype=float)

    rd, re = relative_difference(mc, data)

    # Bin 0: (12-10)/10 = 0.20
    assert abs(rd[0] - 0.20) < 1e-10, f"FAIL rel_diff[0]: expected 0.20, got {rd[0]}"
    # Bin 1: (18-20)/20 = -0.10
    assert abs(rd[1] - (-0.10)) < 1e-10, f"FAIL rel_diff[1]: expected -0.10, got {rd[1]}"
    # Bin 2: MC=0 → NaN
    assert np.isnan(rd[2]), f"FAIL rel_diff[2]: expected NaN, got {rd[2]}"
    # Bin 3: (5-5)/5 = 0.0
    assert abs(rd[3]) < 1e-10, f"FAIL rel_diff[3]: expected 0.0, got {rd[3]}"

    # Propagated uncertainty for bin 0:
    # σ_r = sqrt( (σ_d/m)² + (d·σ_m/m²)² )
    #       σ_d = sqrt(12) = 3.464, σ_m = sqrt(10) = 3.162
    #       = sqrt( (3.464/10)² + (12*3.162/100)² )
    #       = sqrt( 0.11996 + 0.14395 ) = sqrt(0.26391) ≈ 0.5137
    expected_re0 = math.sqrt((math.sqrt(12)/10)**2 + (12*math.sqrt(10)/100)**2)
    assert abs(re[0] - expected_re0) < 1e-8, \
        f"FAIL rel_err[0]: expected {expected_re0:.6f}, got {re[0]:.6f}"

    print("relative_difference: all assertions passed.")

    # ── chi2_test assertions ──────────────────────────────────────────────────
    # Identical histograms → χ² = 0, p = 1
    mc_same   = np.array([10.0, 20.0, 30.0, 20.0, 10.0], dtype=float)
    data_same = np.array([10.0, 20.0, 30.0, 20.0, 10.0], dtype=float)
    chi2_val, ndof, c2ndof, pval = chi2_test(mc_same, data_same)
    assert abs(chi2_val) < 1e-10, f"FAIL chi2 identical histograms: {chi2_val}"
    assert pval > 0.99,           f"FAIL p-value identical histograms: {pval}"

    # Clearly different histograms → large χ²
    mc_diff   = np.array([100.0, 100.0, 100.0], dtype=float)
    data_diff = np.array([  1.0,   1.0,   1.0], dtype=float)
    chi2_val2, ndof2, c2ndof2, pval2 = chi2_test(mc_diff, data_diff)
    assert chi2_val2 > 10.0,  f"FAIL chi2 different histograms too small: {chi2_val2}"
    assert pval2    < 0.001,  f"FAIL p-value different histograms too large: {pval2}"

    # Zero-only bins → should return nan cleanly
    chi2_nan, ndof_nan, c2_nan, p_nan = chi2_test(
        np.array([0.0, 0.0]), np.array([0.0, 0.0])
    )
    assert np.isnan(chi2_nan), f"FAIL chi2 all-zero should be nan: {chi2_nan}"

    print("chi2_test:          all assertions passed.")

    # ── ks_test assertions ────────────────────────────────────────────────────
    rng = np.random.default_rng(seed=42)
    same_a = rng.normal(0, 1, 10_000)
    same_b = rng.normal(0, 1, 10_000)
    diff_a = rng.normal(0, 1, 10_000)
    diff_b = rng.normal(5, 1, 10_000)   # clearly shifted

    ks_same, p_same = ks_test(same_a, same_b)
    ks_diff, p_diff = ks_test(diff_a, diff_b)

    assert ks_same < 0.05,  f"FAIL KS same distribution: D={ks_same:.4f} (expected < 0.05)"
    assert ks_diff > 0.90,  f"FAIL KS different distributions: D={ks_diff:.4f} (expected > 0.90)"
    assert p_same  > 0.05,  f"FAIL KS same p-value: {p_same:.4f} (expected > 0.05)"
    assert p_diff  < 0.001, f"FAIL KS different p-value: {p_diff:.6f} (expected < 0.001)"

    # Empty array guard
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ks_empty, p_empty = ks_test(np.array([]), np.array([1.0, 2.0]))
    assert np.isnan(ks_empty), f"FAIL ks_test empty array: {ks_empty}"
    assert len(caught) == 1 and issubclass(caught[0].category, RuntimeWarning)

    print("ks_test:            all assertions passed.")

    # ── _strip_sentinels assertions ───────────────────────────────────────────
    arr_with_sentinel = np.array([-9999.0, 1.0, 2.0, -9999.0, 3.0])
    stripped = _strip_sentinels(arr_with_sentinel, "beta")
    assert list(stripped) == [1.0, 2.0, 3.0], f"FAIL strip sentinels: {stripped}"

    # chi2pid strips both -9999 and +9999
    arr_chi2 = np.array([-9999.0, -1.0, 0.5, 9999.0, 1.2])
    stripped_chi2 = _strip_sentinels(arr_chi2, "chi2pid")
    assert list(stripped_chi2) == [-1.0, 0.5, 1.2], f"FAIL strip chi2pid: {stripped_chi2}"

    print("_strip_sentinels:   all assertions passed.")

    # ── compare_distribution worked example with synthetic DataFrames ─────────
    print("\nRunning compare_distribution() worked example …")

    rng2 = np.random.default_rng(seed=7)
    n_mc   = 50_000
    n_data = 30_000

    # Synthetic beta: MC from Beta(8,2) re-scaled to [0,1]; data slightly shifted
    beta_mc   = rng2.beta(8, 2, n_mc)
    beta_data = rng2.beta(7, 2, n_data)  # slightly different shape

    # Inject some sentinels (~5% missing)
    beta_mc[rng2.integers(0, n_mc, size=2500)]     = SENTINEL_LOW
    beta_data[rng2.integers(0, n_data, size=1500)] = SENTINEL_LOW

    df_mc_ex   = pd.DataFrame({"beta": beta_mc,
                                "p":     rng2.uniform(1.0, 5.0, n_mc),
                                "theta": rng2.uniform(5.0, 35.0, n_mc)})
    df_data_ex = pd.DataFrame({"beta": beta_data,
                                "p":     rng2.uniform(1.0, 5.0, n_data),
                                "theta": rng2.uniform(5.0, 35.0, n_data)})

    # Check whether matplotlib Agg can save a file
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmpdir:
        save = os.path.join(tmpdir, "beta_test.png")
        st = compare_distribution(
            df_mc_ex, df_data_ex, variable="beta",
            bins=60, normalize=True,
            save_path=save,
        )
        assert os.path.exists(save), "FAIL: compare_distribution did not write PNG"

    # Check returned stats dict has all expected keys
    expected_keys = {
        "variable", "n_mc", "n_data", "hit_frac_mc", "hit_frac_data",
        "ks_distance", "ks_pvalue", "chi2", "ndof", "chi2_per_ndof",
        "chi2_pvalue", "mean_rel_diff", "max_abs_rel_diff", "ks_flag",
    }
    assert expected_keys == set(st.keys()), \
        f"FAIL: stats dict keys mismatch.\nExpected: {expected_keys}\nGot: {set(st.keys())}"

    # n_mc should be less than n_mc after sentinel removal
    assert st["n_mc"]   < n_mc,   f"FAIL: n_mc should be < {n_mc} after sentinel strip"
    assert st["n_data"] < n_data, f"FAIL: n_data should be < {n_data} after sentinel strip"

    # hit_frac should be between 0 and 1
    assert 0 < st["hit_frac_mc"]   <= 1.0
    assert 0 < st["hit_frac_data"] <= 1.0

    # KS distance should be finite and in [0, 1]
    assert np.isfinite(st["ks_distance"]), "FAIL: ks_distance not finite"
    assert 0.0 <= st["ks_distance"] <= 1.0

    # chi2/ndof should be positive
    assert st["chi2_per_ndof"] > 0.0, f"FAIL: chi2_per_ndof = {st['chi2_per_ndof']}"

    print(f"\n  compare_distribution stats for synthetic 'beta':")
    for k, v in st.items():
        if isinstance(v, float):
            print(f"    {k:<22} = {v:.6g}")
        else:
            print(f"    {k:<22} = {v}")

    print("\nAll self-tests passed.\n")
