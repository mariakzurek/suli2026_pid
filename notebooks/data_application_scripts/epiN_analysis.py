from matplotlib.backends.backend_pdf import PdfPages
import argparse
import pathlib
import importlib

import joblib
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import awkward as ak
import json
import math
from scipy.integrate import quad
from scipy.optimize import curve_fit
import uproot
import sys
sys.path.append("../../scripts/")

from pathlib import Path
import common_functions as au
from baseline_chi2pid import passes_kplus_chi2pid_cut




##########################################################################



peak_width=0.1


def gauss_poly(x, A, mu, sigma, m, b):
    gaussian = A * np.exp(-0.5 * ((x - mu)/sigma)**2)

    background = m*x + b

    # prevent negative background
    background = np.clip(background, 0, None)

    return gaussian + background

def poisson_chi2(data, model):

    data = np.asarray(data)
    model = np.asarray(model)

    chi2 = 0

    for n, mu in zip(data, model):

        if mu <= 0:
            continue

        if n > 0:
            chi2 += 2*(mu - n + n*np.log(n/mu))

        else:
            chi2 += 2*mu

    return chi2


def gaussian_fitter(df, output_png=False, peak_width=peak_width, title=None, Rlim=0):

    # ----------------------------
    # Get missing mass values
    # ----------------------------
    
    if isinstance(df, pd.DataFrame):
        mx = df["Mx_epiX"].dropna().to_numpy()
    else:
        mx = np.asarray(df["Mx_epiX"])
        mx = mx[~pd.isna(mx)]

    mx = mx[mx != -9999]


    # ----------------------------
    # Define neutron window
    # ----------------------------

    neutron_mass = 0.95

    fit_min = neutron_mass - peak_width
    fit_max = neutron_mass + peak_width
    if ((Rlim!=0)&(fit_max>Rlim)&(fit_min<Rlim)):
        fit_max=Rlim


    # ----------------------------
    # Histogram ONLY in neutron range
    # ----------------------------

    counts, edges = np.histogram(
        mx,
        bins=50,
        range=(fit_min, fit_max)
    )

    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]


    # These are now the actual fit points
    fit_counts = counts
    fit_centers = centers


    # ----------------------------
    # Initial guesses
    # ----------------------------

    initial_mass = np.mean(mx[(mx >= fit_min) & (mx <= fit_max)])

    # Fallback in case the window is empty
    if np.isnan(initial_mass):
        initial_mass = neutron_mass
    if initial_mass >0.92:
        intial_mass=neutron_mass

    p0 = [
        max(fit_counts.max(), 1),   # Amplitude
        initial_mass,               # Mean
        0.02,                       # Sigma
        0.0,                        # Background slope
        np.median(fit_counts)       # Background intercept
    ]


    bounds = (
        [
            0,
            neutron_mass - peak_width,
            0.005,
            -np.inf,
            -np.inf
        ],
        [
            np.inf,
            neutron_mass + peak_width,
            0.5,
            np.inf,
            np.inf
        ]
    )


    # ----------------------------
    # Fit
    # ----------------------------

    fit_ok = True

    try:

        popt, pcov = curve_fit(
            gauss_poly,
            fit_centers,
            fit_counts,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )

    except RuntimeError:

        print("Warning: Gaussian fit failed, using initial guess")

        popt = np.asarray(p0)
        pcov = np.zeros((5,5))
        fit_ok = False


    A, mu, sigma, m, b = popt


    # ----------------------------
    # Gaussian yield
    # ----------------------------

    gaussian = lambda x: A * np.exp(
        -0.5 * ((x - mu) / sigma)**2
    )


    integral, _ = quad(
        gaussian,
        fit_min,
        fit_max
    )


    neutron_yield = integral / bin_width


    # ----------------------------
    # Yield uncertainty (delta method)
    #
    # The yield is a nonlinear function of the fit parameters
    # (A, mu, sigma, m, b). Rather than assuming Poisson counting
    # statistics on the extracted yield (which ignores background
    # subtraction and correlations between the fit parameters),
    # we propagate the full parameter covariance matrix from
    # curve_fit through to the yield using a numerically evaluated
    # Jacobian (first-order Taylor / delta method):
    #
    #   Var(yield) = J^T . Cov(popt) . J
    #
    # where J_i = d(yield)/d(popt_i)
    # ----------------------------

    neutron_yield_err = compute_yield_uncertainty(
        popt,
        pcov,
        fit_min,
        fit_max,
        bin_width,
        fit_ok=fit_ok
    )


    # ----------------------------
    # Chi2
    # ----------------------------

    expected = gauss_poly(
        fit_centers,
        *popt
    )


    errors = np.sqrt(
        np.maximum(fit_counts, 1)
    )


    chi2 = np.sum(
        ((fit_counts - expected) / errors)**2
    )


    ndf = len(fit_counts) - len(popt)

    chi2_ndf = chi2 / max(ndf,1)



    # ----------------------------
    # Curves
    # ----------------------------

    xfit = np.linspace(
        fit_centers.min(),
        fit_centers.max(),
        500
    )


    yfit = gauss_poly(
        xfit,
        *popt
    )


    y_gauss = gaussian(xfit)


    y_background = (
        m*xfit + b
    )


    # ----------------------------
    # Plot
    # ----------------------------

    fig, ax = plt.subplots(
        figsize=(7,5)
    )


    errors_all = np.sqrt(
        np.maximum(counts,1)
    )


    


    ax.plot(
        xfit,
        yfit,
        label="Gaussian + Polynomial"
    )


    ax.plot(
        xfit,
        y_gauss,
        "--",
        label="Gaussian"
    )


    ax.plot(
        xfit,
        y_background,
        ":",
        label="Background"
    )

    ax.errorbar(
        centers,
        counts,
        yerr=errors_all,
        fmt="o",
        markersize=3,
        capsize=2,
        linestyle="none",
        label="Mx data"
    )


    ax.set_xlabel(
        r"$M_X(e\pi)$ [GeV]"
    )

    ax.set_ylabel(
        "Counts"
    )


    plot_margin = 0.001

    ax.set_xlim(
        fit_min - plot_margin,
        fit_max + plot_margin
    )


    if title is not None:
        ax.set_title(title)



    ax.text(
        0.05,
        0.95,
        f"$\\mu$ = {mu:.4f}\n"
        f"$\\sigma$ = {sigma:.4f}\n"
        f"Yield = {neutron_yield:.0f} $\\pm$ {neutron_yield_err:.0f}\n"
        f"$\\chi^2$ = {chi2:.1f}\n"
        f"$\\chi^2$/ndf = {chi2_ndf:.2f}",
        transform=ax.transAxes,
        verticalalignment="top"
    )


    ax.legend()


    if output_png:
        fig.savefig(
            "neutron_fit_debug.png",
            dpi=150,
            bbox_inches="tight"
        )


    return {
        "mu": mu,
        "sigma": sigma,
        "A": A,
        "m": m,
        "b": b,
        "chi2": chi2,
        "ndf": ndf,
        "chi2_ndf": chi2_ndf,
        "params": popt,
        "covariance": pcov,
        "yield": neutron_yield,
        "yield_err": neutron_yield_err
    }, fig


def compute_yield_uncertainty(popt, pcov, fit_min, fit_max, bin_width, fit_ok=True, eps=1e-6):

    """
    Propagate the Gaussian+polynomial fit's parameter covariance matrix
    to an uncertainty on the extracted (background-subtracted) Gaussian
    yield, using the delta method:

        Var(yield) = J^T . Cov(popt) . J

    J is evaluated numerically (central finite differences) since the
    yield integral has no convenient closed form once it is clipped to
    a finite window.

    Returns 0.0 if the fit failed or the covariance matrix is not
    finite/well-defined (e.g. curve_fit returned inf on a poorly
    constrained parameter), since in that case the parameter
    uncertainties themselves are not meaningful.
    """

    if not fit_ok:
        return 0.0

    pcov = np.asarray(pcov, dtype=float)

    if not np.all(np.isfinite(pcov)):
        print("Warning: non-finite covariance matrix, yield uncertainty set to 0")
        return 0.0

    def yield_func(params):

        A, mu, sigma, m, b = params

        gaussian = lambda x: A * np.exp(-0.5 * ((x - mu) / sigma)**2)

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


def plot_mx_histogram(df, output_name="Mx_debug.png"):
    
    # Handle pandas / awkward
    if isinstance(df, pd.DataFrame):
        mx = df["Mx_epiX"].dropna().to_numpy()
    else:
        mx = np.asarray(df["Mx_epiX"])
        mx = mx[~pd.isna(mx)]

    mx = mx[mx != -9999]

    print("Number of Mx entries:", len(mx))
    print("Mx min:", np.min(mx))
    print("Mx max:", np.max(mx))
    print("Mx mean:", np.mean(mx))

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(
        mx,
        bins=200,
        histtype="step"
    )

    ax.set_xlabel(r"$M_X(e\pi)$ [GeV]")
    ax.set_ylabel("Counts")
    ax.set_title("Missing Mass Distribution")

    plt.savefig(
        output_name,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)



def compute_epiN_misID(df, cutMask):

    """
    pi -> K mis-ID extraction

    denominator:
        EB PID pions + kaons + protons
        (all neutron peak events)

    numerator:
        EB PID kaons + BDT
        (pions reconstructed as kaons)

    Both yields come from independent Gaussian + polynomial fits.
    Uncertainties are propagated from each fit's own covariance-derived
    yield uncertainty (see compute_yield_uncertainty), not from a
    Poisson counting-statistics approximation on the yield.
    """

    # -------------------------
    # denominator fit
    # -------------------------

    denom_fit, denom_fig = gaussian_fitter(
        df,
        output_png=False,
        peak_width=peak_width
    )

    n_total = denom_fit["yield"]
    n_total_err = denom_fit["yield_err"]


    # -------------------------
    # numerator fit
    # -------------------------

    df_fake = df[cutMask]

    if len(df_fake) == 0:
        return 0,0


    fake_fit, fake_fig = gaussian_fitter(
        df_fake,
        output_png=False,
        peak_width=peak_width
    )

    n_fake = fake_fit["yield"]
    n_fake_err = fake_fit["yield_err"]


    # -------------------------
    # ratio
    # -------------------------

    if n_total <= 0:
        return 0,0


    misID = n_fake/n_total


    # Standard error propagation for a ratio R = n_fake / n_total,
    # using each fit's actual yield uncertainty rather than assuming
    # sqrt(1/n) Poisson statistics. This treats the two fits as
    # statistically independent (they are fit to disjoint data
    # samples: the BDT-selected kaons vs. the full pi+K+p sample);
    # if that independence assumption is not appropriate for your
    # analysis, a covariance term would need to be added here.
    if n_fake > 0 and n_total > 0:
        error = misID * np.sqrt(
            (n_fake_err / n_fake)**2 + (n_total_err / n_total)**2
        )
    else:
        error = 0


    return misID, error



def compute_pion_efficiency(df):

    """
    Pion efficiency:

        EB-PID pions neutron yield
        ----------------------------
        EB-PID pions + kaons + protons neutron yield

    Both yields are extracted from Gaussian + polynomial fits
    to the neutron peak, and uncertainties are propagated from
    each fit's covariance-derived yield uncertainty.
    """

    # --------------------------------
    # Denominator:
    # EB PID pion + kaon + proton
    # --------------------------------

    total_fit, _ = gaussian_fitter(
        df,
        peak_width=peak_width
    )

    n_total = total_fit["yield"]
    n_total_err = total_fit["yield_err"]


    # --------------------------------
    # Numerator:
    # EB PID pions only
    # --------------------------------

    df_pi = df[df["pid"] == 211]

    pion_fit, _ = gaussian_fitter(
        df_pi,
        peak_width=peak_width
    )

    n_pi = pion_fit["yield"]
    n_pi_err = pion_fit["yield_err"]


    if n_total <= 0:
        return 0,0


    efficiency = n_pi / n_total


    # standard ratio error propagation using the fits' own
    # covariance-derived yield uncertainties
    if n_pi > 0 and n_total > 0:
        error = efficiency * np.sqrt(
            (n_pi_err / n_pi)**2 + (n_total_err / n_total)**2
        )
    else:
        error = 0


    return efficiency, error



def compute_epiN_contamination(df_SIDIS, misID, misID_err, efficiency, efficiency_err, cutMask):

    """
    Propagates uncertainty through the full contamination chain:

        n_true_pi = n_reco_pi / efficiency
        n_fake    = misID * n_true_pi
        contam    = n_fake / n_k

    n_reco_pi and n_k are raw event counts (not fit yields), so Poisson
    counting statistics (sqrt(N)) is the appropriate uncertainty for
    them. misID and efficiency carry their own propagated uncertainties
    from the Gaussian-fit yields (misID_err, efficiency_err). All
    quantities are treated as statistically independent and combined
    in quadrature at each step.
    """

    df_pi = df_SIDIS[df_SIDIS["pid"] == 211]

    n_reco_pi = len(df_pi)
    n_reco_pi_err = np.sqrt(n_reco_pi) if n_reco_pi > 0 else 0

    if efficiency <= 0:
        return 0,0


    # efficiency correction
    n_true_pi = n_reco_pi / efficiency

    if n_reco_pi > 0 and efficiency > 0:
        rel_err_n_true_pi = np.sqrt(
            (n_reco_pi_err / n_reco_pi)**2 + (efficiency_err / efficiency)**2
        )
    else:
        rel_err_n_true_pi = 0

    n_true_pi_err = n_true_pi * rel_err_n_true_pi


    # expected pion leakage
    n_fake = misID*n_true_pi

    if n_fake > 0 and misID > 0:
        rel_err_n_fake = np.sqrt(
            (misID_err / misID)**2 + rel_err_n_true_pi**2
        )
    else:
        rel_err_n_fake = 0

    n_fake_err = n_fake * rel_err_n_fake


    # selected kaons
    n_k = len(df_SIDIS[cutMask])
    n_k_err = np.sqrt(n_k) if n_k > 0 else 0


    if n_k <= 0:
        return 0,0


    contamination = n_fake/n_k


    if n_fake > 0 and n_k > 0:
        err = contamination * np.sqrt(
            rel_err_n_fake**2 + (n_k_err / n_k)**2
        )
    else:
        err = 0


    return contamination, err

def contamination_pipeline(
    df_SIDIS,
    df_epiN,
    cutMask_SIDIS,
    cutMask,
    title
):

    # denominator:
    # EB PID pion + kaon + proton
    denom_result, denom_fig = gaussian_fitter(
        df_epiN,
        output_png=True,
        peak_width=peak_width,
        title=title + " : EB PID all"
    )


    # numerator:
    # EB PID kaons + BDT
    fake_result, fake_fig = gaussian_fitter(
        df_epiN[cutMask],
        output_png=True,
        peak_width=peak_width,
        title=title + " : EB K + BDT",
        Rlim=1.05
    )


    # numerator:
    # EB PID pions
    pion_result, pion_fig = gaussian_fitter(
        df_epiN[df_epiN["pid"] == 211],
        output_png=True,
        peak_width=peak_width,
        title=title + " : EB PID pions"
    )


    misID, misID_err = compute_epiN_misID(
        df_epiN,
        cutMask
    )


    efficiency, efficiency_err = compute_pion_efficiency(
        df_epiN
    )


    contamination, contamination_err = compute_epiN_contamination(
        df_SIDIS,
        misID,
        misID_err,
        efficiency,
        efficiency_err,
        cutMask_SIDIS
    )


    return (
        [denom_fig, fake_fig, pion_fig],
        misID,
        misID_err,
        efficiency,
        efficiency_err,
        contamination,
        contamination_err
    )
    

####################################################################
print("opening root file")



# ----------------------------
# User setting
# ----------------------------

N_FILES = 50  # <-- change this


# ----------------------------
# Columns
# ----------------------------

cols = [
    "pid",
    "p",
    "theta",
    "beta",
    "chi2pid",
    "rich_RQ",
    "vz",
    "bdt_pass",
    "rich_best_PID",
    "rich_best_ntot",
    "bdt_score"
]

kinematics = [
    "Mx_eKX",
    "Mx_epiX",
    "Mx_epX",
    "Q2",
    "W",
    "y"
]

cols.extend(kinematics)


# ----------------------------
# Find ROOT files
# ----------------------------

data_dir = "/work/clas12/CooperBe/MLStuff/scored_data_v02/"

files = sorted(
    glob.glob(data_dir + "*.root")
)


if len(files) == 0:
    raise FileNotFoundError(
        "No ROOT files found"
    )


# take requested number of files
files_to_use = files[:N_FILES]


print(f"Found {len(files)} ROOT files")
print(f"Using {len(files_to_use)} ROOT files")


# ----------------------------
# Load files
# ----------------------------

dfs = []

for f in files_to_use:

    print("Opening:", f)

    df_temp = uproot.open(
        f + ":PhysicsEvents"
    ).arrays(
        cols
    )

    # ----------------------------
    # Convert to pandas if needed
    # ----------------------------

    if isinstance(df_temp, ak.Array):

        df_temp = ak.to_dataframe(
            df_temp
        )

        # uproot/awkward can create a multi-index
        # for nested structures, remove it
        df_temp = df_temp.reset_index(drop=True)


    elif not isinstance(df_temp, pd.DataFrame):

        raise TypeError(
            f"Unknown data type: {type(df_temp)}"
        )


    dfs.append(df_temp)


# combine all files
df = pd.concat(
    dfs,
    ignore_index=True
)


print("Loaded events:", len(df))
print("Columns:", df.columns.tolist())
#################################################################################

mx_plots=[]
metric_plots=[]


outDir="../../figures/Data_Application/epiN/thinner/"
plot_mx_histogram(df)


pEdges = au.makeBinEdges(2.75,5,10)
tEdges = au.makeBinEdges(5,20,1)

tbins = au.makeBins(df, "theta", binEdges=tEdges)
n_theta = len(tbins)
n_p = len(pEdges) - 1

print("starting binning loop")

mds=[]
mds_er=[]
ef=[]
ef_er=[]
con=[]
con_er=[]


for i, tbin in enumerate(tbins):

    # create storage for this theta bin
    mds.append([])
    mds_er.append([])
    ef.append([])
    ef_er.append([])
    con.append([])
    con_er.append([])


    pbins = au.makeBins(
        tbin,
        "p",
        binEdges=pEdges
    )


    for j, pbin in enumerate(pbins):

        title = (
            f"$M_X(e\\pi)$ : "
            f"{pEdges[j]:.2f} < p < {pEdges[j+1]:.2f} GeV/c, "
            f"{tEdges[i]:.1f} < $\\theta$ < {tEdges[i+1]:.1f}$^\\circ$"
        )


        # Apply SIDIS cuts for contamination calculation
        pbin_SIDIS = au.apply_Sidis_Cuts(pbin)


        # ----------------------------
        # epiN BDT kaon selection
        # ----------------------------

        passes = (
            (pbin["pid"] == 321) &
            (pbin["bdt_pass"] == True)
        )


        # ----------------------------
        # SIDIS BDT kaon selection
        # ----------------------------

        passes_SIDIS = (
            (pbin_SIDIS["pid"] == 321) &
            (pbin_SIDIS["bdt_pass"] == True)
        )


        # ----------------------------
        # Run extraction pipeline
        # ----------------------------

        (
            outFigs,
            misId,
            misId_err,
            efficiency,
            efficiency_err,
            contamination,
            contamination_err
        ) = contamination_pipeline(
            pbin_SIDIS,
            pbin,
            passes_SIDIS,
            passes,
            title
        )


        # Store all neutron fit plots
        # (denominator, fake K, pion)
        mx_plots.append({
            "theta_bin": i,
            "p_bin": j,
            "figures": outFigs
        })


        # Store metrics theta -> p
        mds[i].append(misId)
        mds_er[i].append(misId_err)

        ef[i].append(efficiency)
        ef_er[i].append(efficiency_err)

        con[i].append(contamination)
        con_er[i].append(contamination_err)


print("pipeline fully evaluated, making mx pdf")


print("pipeline fully evaluated, making mx pdf")

##################################################################################################



with PdfPages(outDir + "mx_perbin.pdf") as pdf:

    for entry in mx_plots:

        theta_bin = entry["theta_bin"]
        p_bin = entry["p_bin"]

        figs = entry["figures"]


        # one page for this (theta,p) bin
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(8,12),
            constrained_layout=True
        )


        labels = [
            "Denominator: EB PID π + K + p",
            "Mis-ID numerator: EB PID K + BDT",
            "Efficiency numerator: EB PID π"
        ]


        for ax, old_fig, label in zip(
            axes,
            figs,
            labels
        ):

            old_ax = old_fig.axes[0]


            # ----------------------------
            # Identify Line2D objects that belong to errorbar containers
            # (data line + cap lines) so we don't double-plot them below
            # ----------------------------

            errorbar_lines = set()

            for c in old_ax.containers:
                if isinstance(c, matplotlib.container.ErrorbarContainer):
                    data_line, caplines, barlinecols = c.lines
                    errorbar_lines.add(data_line)
                    for cap in caplines:
                        errorbar_lines.add(cap)


            # ----------------------------
            # Copy fit curves
            # ----------------------------

            for line in old_ax.lines:

                if line in errorbar_lines:
                    continue  # skip — drawn once via containers loop below

                line_label = line.get_label()

                ax.plot(
                    line.get_xdata(),
                    line.get_ydata(),
                    linestyle=line.get_linestyle(),
                    marker=line.get_marker(),
                    markersize=line.get_markersize(),
                    label=line_label
                )


            # ----------------------------
            # Copy errorbar data points
            # ----------------------------

            for container in old_ax.containers:

                if isinstance(
                    container,
                    matplotlib.container.ErrorbarContainer
                ):

                    data_line = container.lines[0]

                    # Extract y-errors
                    yerr = None

                    if len(container.lines) > 2:

                        barlinecols = container.lines[2]

                        if len(barlinecols) > 0:

                            segments = barlinecols[0].get_segments()

                            if len(segments) > 0:

                                half_heights = [
                                    abs(seg[0][1] - seg[1][1]) / 2
                                    for seg in segments
                                ]

                                yerr = np.array([
                                    half_heights,
                                    half_heights
                                ])


                    ax.errorbar(
                        data_line.get_xdata(),
                        data_line.get_ydata(),
                        yerr=yerr,
                        fmt=data_line.get_marker(),
                        markersize=data_line.get_markersize(),
                        linestyle="none",
                        capsize=2,
                        label=container.get_label()
                    )


            # ----------------------------
            # Copy fit information text
            # ----------------------------

            for text in old_ax.texts:

                ax.text(
                    text.get_position()[0],
                    text.get_position()[1],
                    text.get_text(),
                    transform=ax.transAxes,
                    verticalalignment=text.get_verticalalignment(),
                    horizontalalignment=text.get_horizontalalignment()
                )


            # ----------------------------
            # Formatting
            # ----------------------------

            ax.set_xlim(
                old_ax.get_xlim()
            )

            ax.set_ylim(
                old_ax.get_ylim()
            )

            ax.set_xlabel(
                r"$M_X(e\pi)$ [GeV]"
            )

            ax.set_ylabel(
                "Counts"
            )

            ax.set_title(label)

            ax.legend(
                fontsize=8
            )


        fig.suptitle(
            f"p = {pEdges[p_bin]:.2f} - {pEdges[p_bin+1]:.2f} GeV/c, "
            f"θ = {tEdges[theta_bin]:.1f} - {tEdges[theta_bin+1]:.1f}°",
            fontsize=14
        )


        pdf.savefig(fig)

        plt.close(fig)


print("pdf saved as "+outDir+"mx_perbin.pdf")


###############################################################################################################################
print("making mis_id plot:")

p_centers = 0.5*(pEdges[:-1] + pEdges[1:])
n_theta = len(tbins)

metric_plots = []


# ----------------------------
# Mis-ID plot
# ----------------------------

fig, ax = plt.subplots(figsize=(8,6))

for i in range(n_theta):

    ax.errorbar(
        p_centers,
        mds[i],
        yerr=mds_er[i],
        marker="o",
        linestyle="none",
        capsize=3,
        label=(
            f"{tEdges[i]:.1f} < θ < "
            f"{tEdges[i+1]:.1f}°"
        )
    )

ax.set_xlabel("p [GeV/c]")
ax.set_ylabel("π → K mis-ID")
ax.set_title("π → K Mis-ID vs Momentum")
ax.legend()

metric_plots.append(fig)

fig.savefig(
    outDir+"misID_vs_p.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


print("mis_id_plot done. making efficiency plot")


# ----------------------------
# Efficiency plot
# ----------------------------

fig, ax = plt.subplots(figsize=(8,6))

for i in range(n_theta):

    ax.errorbar(
        p_centers,
        ef[i],
        yerr=ef_er[i],
        marker="o",
        linestyle="none",
        capsize=3,
        label=(
            f"{tEdges[i]:.1f} < θ < "
            f"{tEdges[i+1]:.1f}°"
        )
    )

ax.set_xlabel("p [GeV/c]")
ax.set_ylabel("Pion efficiency")
ax.set_title("π Efficiency vs Momentum")
ax.legend()

metric_plots.append(fig)

fig.savefig(
    outDir+"pion_efficiency_vs_p.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


print("efficiency plot done. making contamination plot")


# ----------------------------
# Contamination plot
# ----------------------------

fig, ax = plt.subplots(figsize=(8,6))

for i in range(n_theta):

    ax.errorbar(
        p_centers,
        con[i],
        yerr=con_er[i],
        marker="o",
        linestyle="none",
        capsize=3,
        label=(
            f"{tEdges[i]:.1f} < θ < "
            f"{tEdges[i+1]:.1f}°"
        )
    )

ax.set_xlabel("p [GeV/c]")
ax.set_ylabel("Kaon contamination")
ax.set_title("SIDIS Kaon Contamination vs Momentum")
ax.legend()

metric_plots.append(fig)

fig.savefig(
    outDir+"kaon_contamination_vs_p.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)


print("All plots complete! creating pdf")


# ----------------------------
# Save metric PDF
# ----------------------------

with PdfPages(outDir + "metrics.pdf") as pdf:

    failed = 0

    for plot in metric_plots:
        if plot is not None:
            pdf.savefig(plot)
            plt.close(plot)
        else:
            failed += 1

print("Metric plots missing:", failed)
print("pdf saved as "+outDir+" metrics.pdf")
#########################################################################################################################


import math
import awkward as ak

def compute_contamination_ak(arr, pid=None):
    """
    Computes contamination among reconstructed K+ candidates.

    Parameters
    ----------
    arr : awkward.Array
        Input array with fields "pid" and "mc_matching_pid".
    pid : int or None, optional
        If None, computes total contamination
        (all non-321 truth particles reconstructed as K+).
        Otherwise computes the contamination contribution
        from the specified MC PID.

    Returns
    -------
    r : float
        Contamination fraction.
    rErr : float
        Statistical uncertainty.
    """
    temp = arr[arr["pid"] == 321]

    if pid is None:
        # Total contamination
        a = ak.sum(temp["mc_matching_pid"] != 321)
    else:
        # Contribution from one particle species
        a = ak.sum(temp["mc_matching_pid"] == pid)

    b = ak.num(temp, axis=0) if temp.ndim == 1 else len(temp)

    r = 0.0
    rErr = 99.0
    if b != 0:
        r = a / b
    if a != 0:
        rErr = r * math.sqrt((1 / a) + (1 / b))
    return r, rErr




print("performing MC corss-check")
###############################################################################################################################
# MC comparison
###############################################################################################################################
cols.append("mc_matching_pid")
df_mc = uproot.open("~/ML_Files/MC_scored/pid_training_v2.root:PhysicsEvents").arrays(cols, library="pd")

df_data = uproot.open("~/ML_Files/MC_scored/pid_training_v2.root:PhysicsEvents").arrays(cols, library="pd")
df_pass_mc=df_mc[df_mc["bdt_pass"]==True]


# ----------------------------
# Bin the MC data the same way as the data
# (same theta/p edges as above)
# ----------------------------

print("binning MC data for contamination comparison")

mc_con = []
mc_con_er = []

tbins_mc = au.makeBins(df_mc, "theta", binEdges=tEdges)
n_theta_mc = len(tbins_mc)

for i, tbin_mc in enumerate(tbins_mc):

    mc_con.append([])
    mc_con_er.append([])

    pbins_mc = au.makeBins(
        tbin_mc,
        "p",
        binEdges=pEdges
    )

    for j, pbin_mc in enumerate(pbins_mc):

        value, error = compute_contamination_ak(pbin_mc)

        mc_con[i].append(value)
        mc_con_er[i].append(error)

print("MC contamination binning complete")


##################################################################################################
# Plot 1: MC contamination only
##################################################################################################

fig, ax = plt.subplots(figsize=(8,6))

for i in range(n_theta_mc):

    ax.errorbar(
        p_centers,
        mc_con[i],
        yerr=mc_con_er[i],
        marker="o",
        linestyle="none",
        capsize=3,
        label=(
            f"{tEdges[i]:.1f} < θ < "
            f"{tEdges[i+1]:.1f}°"
        )
    )

ax.set_xlabel("p [GeV/c]")
ax.set_ylabel("Kaon contamination")
ax.set_title("MC Kaon Contamination vs Momentum")
ax.legend()

fig.savefig(
    outDir+"mc_kaon_contamination_vs_p.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

print("MC-only contamination plot saved")


##################################################################################################
# Plot 2: MC overlaid with the epiN-truth contamination
##################################################################################################

fig, ax = plt.subplots(figsize=(8,6))

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

n_overlay = min(n_theta, n_theta_mc)

for i in range(n_overlay):

    color = colors[i % len(colors)]

    ax.errorbar(
        p_centers,
        con[i],
        yerr=con_er[i],
        marker="o",
        linestyle="none",
        capsize=3,
        color=color,
        label=(
            f"epiN truth: {tEdges[i]:.1f} < θ < "
            f"{tEdges[i+1]:.1f}°"
        )
    )

    ax.errorbar(
        p_centers,
        mc_con[i],
        yerr=mc_con_er[i],
        marker="s",
        linestyle="none",
        capsize=3,
        color=color,
        markerfacecolor="none",
        label=(
            f"MC: {tEdges[i]:.1f} < θ < "
            f"{tEdges[i+1]:.1f}°"
        )
    )

ax.set_xlabel("p [GeV/c]")
ax.set_ylabel("Kaon contamination")
ax.set_title("SIDIS Kaon Contamination: epiN Truth vs MC")
ax.legend(fontsize=7)

fig.savefig(
    outDir+"contamination_mc_vs_epiN_truth_vs_p.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

print("MC vs epiN-truth overlay plot saved")

# ------------------------------------------------------------
# Save contamination results as CSV
# ------------------------------------------------------------

contamination_rows = []

for i in range(n_theta):

    for j in range(n_p):

        contamination_rows.append({
            "p_lo": pEdges[j],
            "p_hi": pEdges[j+1],
            "contamination_initial": con[i][j],
            "contamination_initial_err": con_er[i][j],
            "theta_lo": tEdges[i],
            "theta_hi": tEdges[i+1],
        })


contamination_df = pd.DataFrame(contamination_rows)

contamination_csv = "../uncertainty/" + "epiN_contamination_binned_PLOTTING.csv"

contamination_df.to_csv(
    contamination_csv,
    index=False
)

print(f"Saved contamination CSV: {contamination_csv}")
print(contamination_df)


