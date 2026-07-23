from matplotlib.backends.backend_pdf import PdfPages

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import uproot
import sys
import os

from pathlib import Path

sys.path.append("../../scripts/")
import common_functions as au


############################################################
# Functions
############################################################

def compute_misID(df, cutMask):

    num = len(df[cutMask])
    den = len(df)

    if num == 0 or den == 0:
        return 0, 0

    misid = num / den

    err = misid * math.sqrt(
        (1/num) + (1/den)
    )

    return misid, err



def compute_pion_eff(df):

    num = len(df[df["pid"] == 211])
    den = len(df)

    if num == 0 or den == 0:
        return 0, 0

    eff = num / den

    err = eff * math.sqrt(
        (1/num) + (1/den)
    )

    return eff, err



def compute_contamination(df, eff, misId):

    k_mask = (
        (df["bdt_pass"] == True) &
        (df["pid"] == 321)
    )

    n_k = len(df[k_mask])

    n_pi = len(
        df[df["pid"] == 211]
    )


    # Correct back to true pion population
    if eff > 0:
        n_true_pi = n_pi / eff
    else:
        n_true_pi = 0


    # Fake kaons produced by pion leakage
    n_fake_k = misId * n_true_pi


    if n_k > 0:
        contam = n_fake_k / n_k
    else:
        contam = 0


    if n_fake_k > 0 and n_k > 0:
        err = contam * math.sqrt(
            (1/n_fake_k) +
            (1/n_k)
        )
    else:
        err = 0


    return contam, err



############################################################
# Load data
############################################################

outDir = "../../figures/Data_Application/epiN_misId/attempt2/"

Path(outDir).mkdir(
    parents=True,
    exist_ok=True
)


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
    "Mx_eKX",
    "Mx_epiX",
    "Mx_epX",
    "Q2",
    "W",
    "y",
    "rich_best_ntot",
    "bdt_score"
]


data_epiN = uproot.open(
    "~/ML_Files/data_epiN_v02/scored/epiN_dataset.root:PhysicsEvents"
).arrays(
    cols,
    library="pd"
)


data_SIDIS = uproot.open(
    "~/ML_Files/scored_data_v01/nSidis_005046.root:PhysicsEvents"
).arrays(
    cols,
    library="pd"
)


data_SIDIS = au.apply_Sidis_Cuts(data_SIDIS)



############################################################
# Binning
############################################################

pEdge = au.makeBinEdges(2.5,5,10)

tEdge = au.makeBinEdges(
    10,
    20,
    1
)


pCenters = (
    pEdge[:-1] + pEdge[1:]
)/2



theta_bins_epi = au.makeBins(
    data_epiN,
    "theta",
    binEdges=tEdge
)


theta_bins_sidis = au.makeBins(
    data_SIDIS,
    "theta",
    binEdges=tEdge
)



############################################################
# Compute all quantities once
############################################################

misid_results = []
eff_results = []
contam_results = []


for i in range(len(theta_bins_epi)):

    mvals = []
    merrs = []

    evals = []
    eerrs = []

    cvals = []
    cerrs = []


    epi_p_bins = au.makeBins(
        theta_bins_epi[i],
        "p",
        binEdges=pEdge
    )


    sidis_p_bins = au.makeBins(
        theta_bins_sidis[i],
        "p",
        binEdges=pEdge
    )


    for j in range(len(epi_p_bins)):

        epi_bin = epi_p_bins[j]
        sidis_bin = sidis_p_bins[j]


        # BDT kaon selection on true pion sample
        kcut = (
            (epi_bin["bdt_pass"] == True) &
            (epi_bin["pid"] == 321)
        )


        misid, misid_err = compute_misID(
            epi_bin,
            kcut
        )


        eff, eff_err = compute_pion_eff(
            epi_bin
        )


        contam, contam_err = compute_contamination(
            sidis_bin,
            eff,
            misid
        )


        mvals.append(misid)
        merrs.append(misid_err)

        evals.append(eff)
        eerrs.append(eff_err)

        cvals.append(contam)
        cerrs.append(contam_err)


    misid_results.append(
        (mvals,merrs)
    )

    eff_results.append(
        (evals,eerrs)
    )

    contam_results.append(
        (cvals,cerrs)
    )



############################################################
# Plot helper
############################################################

def make_overlay_plot(results, ylabel, title, filename):

    plt.figure(figsize=(8,6))


    for i,(vals,errs) in enumerate(results):

        plt.errorbar(
            pCenters,
            vals,
            yerr=errs,
            fmt="o",
            markersize=4,
            capsize=3,
            label=
            fr"${tEdge[i]:.1f}^\circ<\theta<{tEdge[i+1]:.1f}^\circ$"
        )


    plt.xlabel("Momentum (GeV)")
    plt.ylabel(ylabel)

    plt.title(title)

    plt.xticks(pEdge)

    plt.xlim(
        pEdge[0],
        pEdge[-1]
    )

    plt.legend(
        fontsize=8
    )

    plt.grid(False)

    plt.tight_layout()


    plt.savefig(
        outDir + filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()



############################################################
# Produce plots
############################################################


make_overlay_plot(
    misid_results,
    r"$\pi\rightarrow K$ mis-ID",
    "BDT pion mis-ID",
    "misID_vs_p_theta_overlay.png"
)


make_overlay_plot(
    eff_results,
    "EB pion efficiency",
    "Pion efficiency",
    "efficiency_vs_p_theta_overlay.png"
)


make_overlay_plot(
    contam_results,
    "K contamination",
    "SIDIS K contamination estimate",
    "contamination_vs_p_theta_overlay.png"
)

########################################################################################################################################


############################################################
# Event count plots (no error bars)
############################################################


def compute_misID_counts(df):

    numerator = len(
        df[
            (df["bdt_pass"] == True) &
            (df["pid"] == 321)
        ]
    )

    denominator = len(df)

    return numerator, denominator



def compute_eff_counts(df):

    numerator = len(
        df[df["pid"] == 211]
    )

    denominator = len(df)

    return numerator, denominator



def compute_contamination_counts(df, eff, misId):

    selected_k = len(
        df[
            (df["bdt_pass"] == True) &
            (df["pid"] == 321)
        ]
    )


    n_reco_pi = len(
        df[df["pid"] == 211]
    )


    # Same propagation used in contamination calculation
    if eff > 0:
        n_true_pi = n_reco_pi / eff
    else:
        n_true_pi = 0


    fake_k = misId * n_true_pi


    return fake_k, selected_k



def make_count_overlay_plot(results, ylabel, title, filename):

    plt.figure(figsize=(8,6))


    for i,(nums,dens) in enumerate(results):

        plt.plot(
            pCenters,
            nums,
            marker="o",
            linestyle="None",
            label=
            fr"Numerator ${tEdge[i]:.1f}^\circ<\theta<{tEdge[i+1]:.1f}^\circ$"
        )


        plt.plot(
            pCenters,
            dens,
            marker="o",
            linestyle="None",
            label=
            fr"Denominator ${tEdge[i]:.1f}^\circ<\theta<{tEdge[i+1]:.1f}^\circ$"
        )


    plt.xlabel("Momentum (GeV)")
    plt.ylabel(ylabel)

    plt.title(title)

    plt.xticks(pEdge)

    plt.xlim(
        pEdge[0],
        pEdge[-1]
    )

    plt.legend(
        fontsize=7,
        ncol=2
    )

    plt.grid(False)

    plt.tight_layout()


    plt.savefig(
        outDir + filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()



############################################################
# Calculate count arrays
############################################################


misid_counts = []
eff_counts = []
contam_counts = []


for i in range(len(theta_bins_epi)):


    mis_num = []
    mis_den = []

    eff_num = []
    eff_den = []

    cont_num = []
    cont_den = []


    epi_p_bins = au.makeBins(
        theta_bins_epi[i],
        "p",
        binEdges=pEdge
    )


    sidis_p_bins = au.makeBins(
        theta_bins_sidis[i],
        "p",
        binEdges=pEdge
    )


    for j in range(len(epi_p_bins)):


        epi_bin = epi_p_bins[j]
        sidis_bin = sidis_p_bins[j]


        ################################################
        # Mis-ID counts
        ################################################

        n,d = compute_misID_counts(
            epi_bin
        )

        mis_num.append(n)
        mis_den.append(d)



        ################################################
        # Efficiency counts
        ################################################

        n,d = compute_eff_counts(
            epi_bin
        )

        eff_num.append(n)
        eff_den.append(d)



        ################################################
        # Contamination counts
        ################################################

        misid,_ = compute_misID(
            epi_bin,
            (
                (epi_bin["bdt_pass"] == True) &
                (epi_bin["pid"] == 321)
            )
        )


        eff,_ = compute_pion_eff(
            epi_bin
        )


        n,d = compute_contamination_counts(
            sidis_bin,
            eff,
            misid
        )

        cont_num.append(n)
        cont_den.append(d)



    misid_counts.append(
        (mis_num,mis_den)
    )

    eff_counts.append(
        (eff_num,eff_den)
    )

    contam_counts.append(
        (cont_num,cont_den)
    )



############################################################
# Produce count plots
############################################################


make_count_overlay_plot(
    misid_counts,
    "Events",
    "BDT pion mis-ID numerator and denominator",
    "misID_event_counts.png"
)


make_count_overlay_plot(
    eff_counts,
    "Events",
    "Pion efficiency numerator and denominator",
    "efficiency_event_counts.png"
)


make_count_overlay_plot(
    contam_counts,
    "Events",
    "K contamination numerator and denominator",
    "contamination_event_counts.png"
)


############################################################
# 2D pion efficiency heatmap
############################################################

p_edges = np.linspace(0.5, 5.0, 11)      # 10 p bins
theta_edges = np.linspace(0.0, 35.0, 11) # 10 theta bins

eff_map = np.zeros((len(theta_edges)-1, len(p_edges)-1))

for i in range(len(theta_edges)-1):

    theta_low = theta_edges[i]
    theta_high = theta_edges[i+1]

    theta_bin = data_epiN[
        (data_epiN["theta"] >= theta_low) &
        (data_epiN["theta"] < theta_high)
    ]

    for j in range(len(p_edges)-1):

        p_low = p_edges[j]
        p_high = p_edges[j+1]

        p_bin = theta_bin[
            (theta_bin["p"] >= p_low) &
            (theta_bin["p"] < p_high)
        ]

        eff, _ = compute_pion_eff(p_bin)

        eff_map[i, j] = eff


plt.figure(figsize=(8,6))

mesh = plt.pcolormesh(
    p_edges,
    theta_edges,
    eff_map,
    cmap="coolwarm",
    shading="auto",
    vmin=0,
    vmax=1
)

plt.colorbar(mesh, label="EB Pion Efficiency")

plt.xlabel("Momentum (GeV)")
plt.ylabel(r"$\theta$ (deg)")
plt.title("Pion Efficiency with epi(N) event Truth")

plt.xlim(0.5,5)
plt.ylim(0,35)

plt.tight_layout()

plt.savefig(
    outDir + "pion_efficiency_heatmap.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

plt.figure(figsize=(8,6))
data_epiN=data_epiN[data_epiN["p"]<2.2]
plt.hist(
    data_epiN["Mx_epiX"],
    bins=100,
    range=(0.85, 1.05),
    histtype="step"
)

plt.xlabel(r"$M_X(e\pi)$ (GeV)")
plt.ylabel("Events")
plt.title(r"Data $ep \rightarrow e\pi^+(n) p<2.2 $")

plt.tight_layout()

plt.savefig(
    outDir + "epiN_Mx_epiX_histogram.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Count plots finished.")
print("Done.")