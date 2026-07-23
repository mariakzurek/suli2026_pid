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
# User config: single theta range
############################################################

THETA_LOW  = 10.0
THETA_HIGH = 20.0


outDir = "../../figures/Data_Application/epiN_misId/comparisons/"

Path(outDir).mkdir(
    parents=True,
    exist_ok=True
)


############################################################
# Data-driven (EB PID / SIDIS) functions
############################################################

def compute_misID_data(df, cutMask):

    num = len(df[cutMask])
    den = len(df)

    if num == 0 or den == 0:
        return 0, 0

    misid = num / den

    err = misid * math.sqrt(
        (1/num) + (1/den)
    )

    return misid, err



def compute_pion_eff_data(df):

    num = len(df[df["pid"] == 211])
    den = len(df)

    if num == 0 or den == 0:
        return 0, 0

    eff = num / den

    err = eff * math.sqrt(
        (1/num) + (1/den)
    )

    return eff, err



def compute_contamination_data(df, eff, misId):

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
# MC-truth functions
############################################################

def compute_misID_MC(df):

    true_pi = df[df["mc_matching_pid"] == 211]

    if len(true_pi) == 0:
        return 0, 0

    fake_k = true_pi[
        (true_pi["pid"] == 321) &
        (true_pi["bdt_pass"] == True)
    ]

    num = len(fake_k)
    den = len(true_pi)

    if num == 0:
        return 0, 0

    value = num / den

    err = value * math.sqrt(
        (1/num) + (1/den)
    )

    return value, err



def compute_pion_eff_MC(df):

    true_pi = df[
        df["mc_matching_pid"] == 211
    ]

    if len(true_pi) == 0:
        return 0, 0

    reco_pi = true_pi[
        true_pi["pid"] == 211
    ]

    num = len(reco_pi)
    den = len(true_pi)

    if num == 0:
        return 0, 0

    value = num / den

    err = value * math.sqrt(
        (1/num) + (1/den)
    )

    return value, err



def compute_contamination_MC(df):

    fake_k = df[
        (df["mc_matching_pid"] == 211) &
        (df["pid"] == 321) &
        (df["bdt_pass"] == True)
    ]

    selected_k = df[
        (df["pid"] == 321) &
        (df["bdt_pass"] == True)
    ]

    n_fake = len(fake_k)
    n_k = len(selected_k)

    if n_k == 0:
        return 0, 0

    value = n_fake / n_k

    if n_fake > 0:
        err = value * math.sqrt(
            (1/n_fake) + (1/n_k)
        )
    else:
        err = 0

    return value, err



############################################################
# Load data (data-driven method)
############################################################

cols_data = [
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
    cols_data,
    library="pd"
)


data_SIDIS = uproot.open(
    "~/ML_Files/scored_data_v01/nSidis_005036.root:PhysicsEvents"
).arrays(
    cols_data,
    library="pd"
)


data_SIDIS = au.apply_Sidis_Cuts(data_SIDIS)



############################################################
# Load MC (MC-truth method)
############################################################

cols_mc = cols_data + ["mc_matching_pid"]


df_mc = uproot.open(
    "~/ML_Files/MC_scored/pid_training_v2.root:PhysicsEvents"
).arrays(
    cols_mc,
    library="pd"
)
df_mc = au.apply_Sidis_Cuts(df_mc)


############################################################
# Binning: single theta range, binned in momentum
############################################################

pEdge = au.makeBinEdges(2.5, 5, 10)

pCenters = (
    pEdge[:-1] + pEdge[1:]
) / 2


# Restrict every sample to the single user-specified theta range
epi_theta = data_epiN[
    (data_epiN["theta"] >= THETA_LOW) &
    (data_epiN["theta"] < THETA_HIGH)
]

sidis_theta = data_SIDIS[
    (data_SIDIS["theta"] >= THETA_LOW) &
    (data_SIDIS["theta"] < THETA_HIGH)
]

mc_theta = df_mc[
    (df_mc["theta"] >= THETA_LOW) &
    (df_mc["theta"] < THETA_HIGH)
]


epi_p_bins = au.makeBins(
    epi_theta,
    "p",
    binEdges=pEdge
)

sidis_p_bins = au.makeBins(
    sidis_theta,
    "p",
    binEdges=pEdge
)

mc_p_bins = au.makeBins(
    mc_theta,
    "p",
    binEdges=pEdge
)


############################################################
# Compute data-driven quantities vs p
############################################################

misid_data_vals, misid_data_errs = [], []
eff_data_vals, eff_data_errs = [], []
contam_data_vals, contam_data_errs = [], []


for j in range(len(epi_p_bins)):

    epi_bin = epi_p_bins[j]
    sidis_bin = sidis_p_bins[j]

    kcut = (
        (epi_bin["bdt_pass"] == True) &
        (epi_bin["pid"] == 321)
    )

    misid, misid_err = compute_misID_data(epi_bin, kcut)
    eff, eff_err = compute_pion_eff_data(epi_bin)
    contam, contam_err = compute_contamination_data(sidis_bin, eff, misid)

    misid_data_vals.append(misid)
    misid_data_errs.append(misid_err)

    eff_data_vals.append(eff)
    eff_data_errs.append(eff_err)

    contam_data_vals.append(contam)
    contam_data_errs.append(contam_err)


############################################################
# Compute MC-truth quantities vs p
############################################################

misid_mc_vals, misid_mc_errs = [], []
eff_mc_vals, eff_mc_errs = [], []
contam_mc_vals, contam_mc_errs = [], []


for pbin in mc_p_bins:

    m, e = compute_misID_MC(pbin)
    eff, ee = compute_pion_eff_MC(pbin)
    contam, ce = compute_contamination_MC(pbin)

    misid_mc_vals.append(m)
    misid_mc_errs.append(e)

    eff_mc_vals.append(eff)
    eff_mc_errs.append(ee)

    contam_mc_vals.append(contam)
    contam_mc_errs.append(ce)


############################################################
# Plot helper: overlay data-driven vs MC-truth
############################################################

def make_comparison_plot(data_vals, data_errs, mc_vals, mc_errs, ylabel, title, filename):

    plt.figure(figsize=(8, 6))

    plt.errorbar(
        pCenters,
        data_vals,
        yerr=data_errs,
        fmt="o",
        markersize=4,
        capsize=3,
        label="Data-driven (EB PID)"
    )

    plt.errorbar(
        pCenters,
        mc_vals,
        yerr=mc_errs,
        fmt="s",
        markersize=4,
        capsize=3,
        label="MC truth"
    )

    plt.xlabel("Momentum (GeV)")
    plt.ylabel(ylabel)

    plt.title(
        title + fr" (${THETA_LOW:.1f}^\circ<\theta<{THETA_HIGH:.1f}^\circ$)"
    )

    plt.xticks(pEdge)
    plt.xlim(pEdge[0], pEdge[-1])

    plt.legend(fontsize=9)

    plt.grid(False)

    plt.tight_layout()

    plt.savefig(
        outDir + filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()



############################################################
# Produce comparison plots
############################################################

make_comparison_plot(
    misid_data_vals, misid_data_errs,
    misid_mc_vals, misid_mc_errs,
    r"$\pi\rightarrow K$ mis-ID",
    "BDT pion mis-ID: data-driven vs MC truth",
    "misID_comparison.png"
)

make_comparison_plot(
    eff_data_vals, eff_data_errs,
    eff_mc_vals, eff_mc_errs,
    "Pion efficiency",
    "Pion efficiency: data-driven vs MC truth",
    "efficiency_comparison.png"
)

make_comparison_plot(
    contam_data_vals, contam_data_errs,
    contam_mc_vals, contam_mc_errs,
    "K contamination",
    "K contamination: data-driven vs MC truth",
    "contamination_comparison.png"
)


from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

############################################################
# Output
############################################################

Path(outDir).mkdir(parents=True, exist_ok=True)

pdf_name = outDir + "Mx_epiX_overlay_by_p.pdf"

############################################################
# Missing-mass cut
############################################################

df_SIDIS = uproot.open(
    "~/ML_Files/scored_data_v01/nSidis_005046.root:PhysicsEvents"
).arrays(
    cols_data,
    library="pd"
)

data_cut = data_epiN[
    (data_epiN["Mx_epiX"] > 0.85) &
    (data_epiN["Mx_epiX"] < 1.05)
]

sidis_cut = df_SIDIS[
    (df_SIDIS["Mx_epiX"] > 0.85) &
    (df_SIDIS["Mx_epiX"] < 1.05)
]

############################################################
# Momentum bins
############################################################

p_edges = np.linspace(0.5, 5.0, 11)

############################################################
# PDF
############################################################

with PdfPages(pdf_name) as pdf:

    for i in range(len(p_edges)-1):

        pmin = p_edges[i]
        pmax = p_edges[i+1]

        data_bin = data_cut[
            (data_cut["p"] >= pmin) &
            (data_cut["p"] < pmax)
        ]

        sidis_bin = sidis_cut[
            (sidis_cut["p"] >= pmin) &
            (sidis_cut["p"] < pmax)
        ]

        plt.figure(figsize=(8,6))

        plt.hist(
            data_bin["Mx_epiX"],
            bins=60,
            range=(0.85,1.05),
            density=True,
            histtype="step",
            linewidth=2,
            label=f"Data ({len(data_bin)})"
        )

        plt.hist(
            sidis_bin["Mx_epiX"],
            bins=60,
            range=(0.85,1.05),
            density=True,
            histtype="step",
            linewidth=2,
            label=f"SIDIS ({len(sidis_bin)})"
        )

        plt.xlabel(r"$M_X(e\pi)$ (GeV)")
        plt.ylabel("Normalized Counts")
        plt.title(
            fr"${pmin:.2f} < p < {pmax:.2f}$ GeV"
        )

        plt.legend()
        plt.tight_layout()

        pdf.savefig()
        plt.close()

print("Saved:", pdf_name)

############################################################
# Broad comparison (p < 2.2 GeV)
############################################################

data_low = data_cut[
    data_cut["p"] < 2.2
]

sidis_low = sidis_cut[
    sidis_cut["p"] < 2.2
]

plt.figure(figsize=(8,6))

plt.hist(
    data_low["Mx_epiX"],
    bins=60,
    range=(0.85,1.05),
    density=True,
    histtype="step",
    linewidth=2,
    label=f"Data ({len(data_low)})"
)

plt.hist(
    sidis_low["Mx_epiX"],
    bins=60,
    range=(0.85,1.05),
    density=True,
    histtype="step",
    linewidth=2,
    label=f"SIDIS ({len(sidis_low)})"
)

plt.xlabel(r"$M_X(e\pi)$ (GeV)")
plt.ylabel("Normalized Counts")
plt.title(r"$p<2.2$ GeV")

plt.legend()

plt.tight_layout()

plt.savefig(
    outDir + "Mx_epiX_overlay_p_lt_2p2.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()
print("Done.")