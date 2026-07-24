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



peak_width=0.15

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


def gaussian_fitter(df, output_png=False, peak_width=peak_width, title=None):

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

    p0 = [
        max(fit_counts.max(), 1),
        neutron_mass,
        0.02,
        0.0,
        np.median(fit_counts)
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
        f"Yield = {neutron_yield:.0f}\n"
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
        "yield": neutron_yield
    }, fig


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


    # -------------------------
    # ratio
    # -------------------------

    if n_total <= 0:
        return 0,0


    misID = n_fake/n_total


    # binomial approximation
    if n_fake > 0:
        error = misID*np.sqrt(
            (1/n_fake)+(1/n_total)
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
    to the neutron peak.
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


    if n_total <= 0:
        return 0,0


    efficiency = n_pi / n_total


    # statistical uncertainty
    if n_pi > 0:
        error = efficiency*np.sqrt(
            (1/n_pi) + (1/n_total)
        )
    else:
        error = 0


    return efficiency, error



def compute_epiN_contamination(df_SIDIS, misID, efficiency, cutMask):

    df_pi = df_SIDIS[df_SIDIS["pid"] == 211]

    n_reco_pi = len(df_pi)

    if efficiency <= 0:
        return 0,0


    # efficiency correction
    n_true_pi = n_reco_pi / efficiency


    # expected pion leakage
    n_fake = misID*n_true_pi


    # selected kaons
    n_k = len(df_SIDIS[cutMask])


    if n_k <= 0:
        return 0,0


    contamination = n_fake/n_k


    if n_fake > 0:
        err = contamination*np.sqrt(
            (1/n_fake)+(1/n_k)
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
        title=title + " : EB K + BDT"
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
        efficiency,
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

N_FILES = 16   # <-- change this


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

data_dir = "/work/clas12/CooperBe/MLStuff/scored_data_v01/"

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


outDir="../../figures/Data_Application/epiN/"
plot_mx_histogram(df)


pEdges = au.makeBinEdges(2.5,5,10)
tEdges = au.makeBinEdges(10,20,1)

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
            # Copy fit curves
            # ----------------------------

            for line in old_ax.lines:

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

                                yerr = np.array([
                                    [
                                        abs(seg[0][1] - seg[1][1])
                                        for seg in segments
                                    ],
                                    [
                                        abs(seg[0][1] - seg[1][1])
                                        for seg in segments
                                    ]
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
