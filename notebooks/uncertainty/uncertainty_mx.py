#!/usr/bin/env python
# coding: utf-8

# ============================================================
# epiN mis-ID contamination sensitivity to the Mx peak_width
#
# Reuses gaussian_fitter / compute_yield_uncertainty /
# compute_epiN_misID / compute_pion_efficiency /
# compute_epiN_contamination verbatim from the working epiN
# contamination script. Runs the (theta, p) bin loop twice:
#   - current peak_width (0.15)
#   - shifted peak_width (0.20, i.e. +0.05)
# and records contamination_initial, contamination_shifted,
# and delta_c = |shifted - initial| per bin in a CSV, matching
# the format of the other sensitivity CSVs (no _err columns).
#
# NOTE: this fits the Mx neutron peak twice per bin (once per
# width), each requiring 3 Gaussian+polynomial fits (denominator,
# mis-ID numerator, efficiency numerator) -- so total runtime is
# ~2x the original script's binning loop. Plotting/PDF output is
# skipped entirely since only the CSV was requested.
# ============================================================

import glob
import sys
sys.path.append("../../scripts/")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import awkward as ak
import uproot
from scipy.integrate import quad
from scipy.optimize import curve_fit

import common_functions as au

# ------------------------------------------------------------
# Fit functions (copied verbatim from the working script)
# ------------------------------------------------------------

peak_width = 0.15  # mutated at call time by run_grid() below


def gauss_poly(x, A, mu, sigma, m, b):
    gaussian = A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    background = m * x + b
    background = np.clip(background, 0, None)
    return gaussian + background


def gaussian_fitter(df, output_png=False, peak_width=peak_width, title=None):

    if isinstance(df, pd.DataFrame):
        mx = df["Mx_epiX"].dropna().to_numpy()
    else:
        mx = np.asarray(df["Mx_epiX"])
        mx = mx[~pd.isna(mx)]

    mx = mx[mx != -9999]

    neutron_mass = 0.95
    fit_min = neutron_mass - peak_width
    fit_max = neutron_mass + peak_width

    counts, edges = np.histogram(mx, bins=50, range=(fit_min, fit_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]

    fit_counts = counts
    fit_centers = centers

    initial_mass = np.mean(mx[(mx >= fit_min) & (mx <= fit_max)])
    if np.isnan(initial_mass):
        initial_mass = neutron_mass
    if initial_mass > 0.92:
        initial_mass = neutron_mass

    p0 = [
        max(fit_counts.max(), 1),
        initial_mass,
        0.02,
        0.0,
        np.median(fit_counts)
    ]

    bounds = (
        [0, neutron_mass - peak_width, 0.005, -np.inf, -np.inf],
        [np.inf, neutron_mass + peak_width, 0.5, np.inf, np.inf]
    )

    fit_ok = True
    try:
        popt, pcov = curve_fit(
            gauss_poly, fit_centers, fit_counts,
            p0=p0, bounds=bounds, maxfev=10000
        )
    except RuntimeError:
        print("Warning: Gaussian fit failed, using initial guess")
        popt = np.asarray(p0)
        pcov = np.zeros((5, 5))
        fit_ok = False

    A, mu, sigma, m, b = popt

    gaussian = lambda x: A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    integral, _ = quad(gaussian, fit_min, fit_max)
    neutron_yield = integral / bin_width

    neutron_yield_err = compute_yield_uncertainty(
        popt, pcov, fit_min, fit_max, bin_width, fit_ok=fit_ok
    )

    expected = gauss_poly(fit_centers, *popt)
    errors = np.sqrt(np.maximum(fit_counts, 1))
    chi2 = np.sum(((fit_counts - expected) / errors) ** 2)
    ndf = len(fit_counts) - len(popt)
    chi2_ndf = chi2 / max(ndf, 1)

    fig = None
    if output_png:
        xfit = np.linspace(fit_centers.min(), fit_centers.max(), 500)
        yfit = gauss_poly(xfit, *popt)
        y_gauss = gaussian(xfit)
        y_background = m * xfit + b

        fig, ax = plt.subplots(figsize=(7, 5))
        errors_all = np.sqrt(np.maximum(counts, 1))

        ax.plot(xfit, yfit, label="Gaussian + Polynomial")
        ax.plot(xfit, y_gauss, "--", label="Gaussian")
        ax.plot(xfit, y_background, ":", label="Background")
        ax.errorbar(
            centers, counts, yerr=errors_all, fmt="o",
            markersize=3, capsize=2, linestyle="none", label="Mx data"
        )
        ax.set_xlabel(r"$M_X(e\pi)$ [GeV]")
        ax.set_ylabel("Counts")
        ax.set_xlim(fit_min - 0.001, fit_max + 0.001)
        if title is not None:
            ax.set_title(title)
        ax.text(
            0.05, 0.95,
            f"$\\mu$ = {mu:.4f}\n$\\sigma$ = {sigma:.4f}\n"
            f"Yield = {neutron_yield:.0f} $\\pm$ {neutron_yield_err:.0f}\n"
            f"$\\chi^2$ = {chi2:.1f}\n$\\chi^2$/ndf = {chi2_ndf:.2f}",
            transform=ax.transAxes, verticalalignment="top"
        )
        ax.legend()
        fig.savefig("neutron_fit_debug.png", dpi=150, bbox_inches="tight")

    return {
        "mu": mu, "sigma": sigma, "A": A, "m": m, "b": b,
        "chi2": chi2, "ndf": ndf, "chi2_ndf": chi2_ndf,
        "params": popt, "covariance": pcov,
        "yield": neutron_yield, "yield_err": neutron_yield_err
    }, fig


def compute_yield_uncertainty(popt, pcov, fit_min, fit_max, bin_width, fit_ok=True, eps=1e-6):

    if not fit_ok:
        return 0.0

    pcov = np.asarray(pcov, dtype=float)
    if not np.all(np.isfinite(pcov)):
        print("Warning: non-finite covariance matrix, yield uncertainty set to 0")
        return 0.0

    def yield_func(params):
        A, mu, sigma, m, b = params
        gaussian = lambda x: A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        integral, _ = quad(gaussian, fit_min, fit_max)
        return integral / bin_width

    n_params = len(popt)
    jac = np.zeros(n_params)

    for i in range(n_params):
        step = eps * max(abs(popt[i]), 1.0)
        dp_plus = np.array(popt, dtype=float)
        dp_minus = np.array(popt, dtype=float)
        dp_plus[i] += step
        dp_minus[i] -= step
        y_plus = yield_func(dp_plus)
        y_minus = yield_func(dp_minus)
        jac[i] = (y_plus - y_minus) / (2 * step)

    variance = jac @ pcov @ jac.T
    if not np.isfinite(variance) or variance < 0:
        return 0.0

    return float(np.sqrt(variance))


def compute_epiN_misID(df, cutMask):

    denom_fit, denom_fig = gaussian_fitter(df, output_png=False, peak_width=peak_width)
    plt.close(denom_fig) if denom_fig is not None else None

    n_total = denom_fit["yield"]
    n_total_err = denom_fit["yield_err"]

    df_fake = df[cutMask]
    if len(df_fake) == 0:
        return 0, 0

    fake_fit, fake_fig = gaussian_fitter(df_fake, output_png=False, peak_width=peak_width)
    plt.close(fake_fig) if fake_fig is not None else None

    n_fake = fake_fit["yield"]
    n_fake_err = fake_fit["yield_err"]

    if n_total <= 0:
        return 0, 0

    misID = n_fake / n_total

    if n_fake > 0 and n_total > 0:
        error = misID * np.sqrt(
            (n_fake_err / n_fake) ** 2 + (n_total_err / n_total) ** 2
        )
    else:
        error = 0

    return misID, error


def compute_pion_efficiency(df):

    total_fit, total_fig = gaussian_fitter(df, peak_width=peak_width)
    plt.close(total_fig) if total_fig is not None else None

    n_total = total_fit["yield"]
    n_total_err = total_fit["yield_err"]

    df_pi = df[df["pid"] == 211]

    pion_fit, pion_fig = gaussian_fitter(df_pi, peak_width=peak_width)
    plt.close(pion_fig) if pion_fig is not None else None

    n_pi = pion_fit["yield"]
    n_pi_err = pion_fit["yield_err"]

    if n_total <= 0:
        return 0, 0

    efficiency = n_pi / n_total

    if n_pi > 0 and n_total > 0:
        error = efficiency * np.sqrt(
            (n_pi_err / n_pi) ** 2 + (n_total_err / n_total) ** 2
        )
    else:
        error = 0

    return efficiency, error


def compute_epiN_contamination(df_SIDIS, misID, misID_err, efficiency, efficiency_err, cutMask):

    df_pi = df_SIDIS[df_SIDIS["pid"] == 211]

    n_reco_pi = len(df_pi)
    n_reco_pi_err = np.sqrt(n_reco_pi) if n_reco_pi > 0 else 0

    if efficiency <= 0:
        return 0, 0

    n_true_pi = n_reco_pi / efficiency

    if n_reco_pi > 0 and efficiency > 0:
        rel_err_n_true_pi = np.sqrt(
            (n_reco_pi_err / n_reco_pi) ** 2 + (efficiency_err / efficiency) ** 2
        )
    else:
        rel_err_n_true_pi = 0

    n_true_pi_err = n_true_pi * rel_err_n_true_pi

    n_fake = misID * n_true_pi

    if n_fake > 0 and misID > 0:
        rel_err_n_fake = np.sqrt(
            (misID_err / misID) ** 2 + rel_err_n_true_pi ** 2
        )
    else:
        rel_err_n_fake = 0

    n_fake_err = n_fake * rel_err_n_fake

    n_k = len(df_SIDIS[cutMask])
    n_k_err = np.sqrt(n_k) if n_k > 0 else 0

    if n_k <= 0:
        return 0, 0

    contamination = n_fake / n_k

    if n_fake > 0 and n_k > 0:
        err = contamination * np.sqrt(
            rel_err_n_fake ** 2 + (n_k_err / n_k) ** 2
        )
    else:
        err = 0

    return contamination, err


# ------------------------------------------------------------
# Load data (same as the working script)
# ------------------------------------------------------------

print("opening root file")

N_FILES = 5  # <-- change this

cols = [
    "pid", "p", "theta", "beta", "chi2pid", "rich_RQ", "vz",
    "bdt_pass", "rich_best_PID", "rich_best_ntot", "bdt_score"
]
kinematics = ["Mx_eKX", "Mx_epiX", "Mx_epX", "Q2", "W", "y"]
cols.extend(kinematics)

data_dir = "/work/clas12/CooperBe/MLStuff/scored_data_v02/"
files = sorted(glob.glob(data_dir + "*.root"))

if len(files) == 0:
    raise FileNotFoundError("No ROOT files found")

files_to_use = files[:N_FILES]
print(f"Found {len(files)} ROOT files")
print(f"Using {len(files_to_use)} ROOT files")

dfs = []
for f in files_to_use:
    print("Opening:", f)
    df_temp = uproot.open(f + ":PhysicsEvents").arrays(cols)

    if isinstance(df_temp, ak.Array):
        df_temp = ak.to_dataframe(df_temp)
        df_temp = df_temp.reset_index(drop=True)
    elif not isinstance(df_temp, pd.DataFrame):
        raise TypeError(f"Unknown data type: {type(df_temp)}")

    dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)
print("Loaded events:", len(df))

# ------------------------------------------------------------
# Bin edges (same as the working script)
# ------------------------------------------------------------

pEdges = au.makeBinEdges(2.75, 5, 10)
tEdges = au.makeBinEdges(5, 20, 1)

n_theta = len(tEdges) - 1
n_p = len(pEdges) - 1

# ------------------------------------------------------------
# Run the (theta, p) grid at a given peak_width
# ------------------------------------------------------------

def run_grid(df, width):
    global peak_width
    peak_width = width

    tbins = au.makeBins(df, "theta", binEdges=tEdges)

    con_grid = []

    for i, tbin in enumerate(tbins):
        pbins = au.makeBins(tbin, "p", binEdges=pEdges)

        con_row = []

        for j, pbin in enumerate(pbins):

            pbin_SIDIS = au.apply_Sidis_Cuts(pbin)

            passes = (
                (pbin["pid"] == 321) &
                (pbin["bdt_pass"] == True)
            )

            passes_SIDIS = (
                (pbin_SIDIS["pid"] == 321) &
                (pbin_SIDIS["bdt_pass"] == True)
            )

            misID, misID_err = compute_epiN_misID(pbin, passes)
            efficiency, efficiency_err = compute_pion_efficiency(pbin)
            contamination, contamination_err = compute_epiN_contamination(
                pbin_SIDIS, misID, misID_err, efficiency, efficiency_err, passes_SIDIS
            )

            con_row.append(contamination)

            plt.close("all")  # gaussian_fitter always builds a figure
                               # internally even with output_png=False;
                               # close it here so figures don't pile up
                               # across the many fits in this loop

        con_grid.append(con_row)

    return con_grid


print(f"Running pipeline with peak_width = 0.15 (current)")
con_initial = run_grid(df, 0.15)

print(f"Running pipeline with peak_width = 0.20 (+0.05 shifted)")
con_shifted = run_grid(df, 0.20)

# ------------------------------------------------------------
# Assemble output rows
# ------------------------------------------------------------

rows = []

for i in range(n_theta):
    theta_lo, theta_hi = tEdges[i], tEdges[i + 1]

    for j in range(n_p):
        p_lo, p_hi = pEdges[j], pEdges[j + 1]

        c_init = con_initial[i][j]
        c_shift = con_shifted[i][j]
        delta_c = abs(c_shift - c_init)

        rows.append({
            "theta_lo": theta_lo,
            "theta_hi": theta_hi,
            "p_lo": p_lo,
            "p_hi": p_hi,
            "contamination_initial": c_init,
            "contamination_shifted": c_shift,
            "delta_c": delta_c,
        })

out_df = pd.DataFrame(rows)
outCSV = "mx_width_sensitivity.csv"
out_df.to_csv(outCSV, index=False)

print(f"Wrote {len(out_df)} rows to {outCSV}")
print(out_df)