from matplotlib.backends.backend_pdf import PdfPages
import argparse
import pathlib
import importlib

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import json
import math
import uproot
import sys
import awkward as ak
sys.path.append("../../scripts/")

from pathlib import Path
import common_functions as au
from baseline_chi2pid import passes_kplus_chi2pid_cut


def compute_misId(df, cutMask):
    num_df = df[cutMask]
    num = len(num_df)
    den = len(df)

    if num == 0 or den == 0:
        return 0, 0

    misid = num / den
    rErr = misid * math.sqrt((1/num) + (1/den))

    return misid, rErr

def compute_eff(df, cutMask):
    num_df = df[cutMask]
    num = len(num_df)
    den = len(df)

    if num == 0 or den == 0:
        return 0, 0

    eff = num / den
    rErr = eff * math.sqrt((1/num) + (1/den))

    return eff, rErr

def compute_contam(df, rate_file):
    
    # Convert awkward array to pandas DataFrame if needed
    if not isinstance(df, pd.DataFrame):
        df = ak.to_dataframe(df)

    # Load rate table if a filename is passed
    if isinstance(rate_file, str):
        rate_df = pd.read_csv(rate_file)
    else:
        rate_df = rate_file


    # Identify this bin
    theta = df["theta"].mean()
    p = df["p"].mean()

    rate_bin = rate_df[
        (rate_df["theta_low"] <= theta) &
        (theta < rate_df["theta_high"]) &
        (rate_df["p_low"] <= p) &
        (p < rate_df["p_high"])
    ]

    if len(rate_bin) != 1:
        return 0, 0


    mis_id = rate_bin["mis_id"].iloc[0]
    eff = rate_bin["eff"].iloc[0]


    # Number of pion candidates in this bin
    N_pi = len(
        df[df["pid"] == 211]
    )

    # Number of BDT kaon candidates in this bin
    N_k_bdt = len(
        df[(df["bdt_pass"] == True)&(df["pid"] == 321)]
    )


    if eff == 0 or N_pi == 0 or N_k_bdt == 0:
        return 0, 0


    # Estimated fake kaons
    fake_k = mis_id * (N_pi / eff)

    # Contamination fraction
    contam = fake_k / N_k_bdt


    # Statistical uncertainty from event counts
    rel_err = np.sqrt(
        (1 / N_pi) +
        (1 / N_k_bdt)
    )

    contam_err = contam * rel_err


    return contam, contam_err
    

grid = pd.DataFrame(columns=["theta_low","theta_high", "p_low","p_high", "mis_id", "eff"])

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


# ==================================================
# LOAD DATA PION SAMPLE
# Already contains:
#   - EB selection
#   - neutron missing mass selection
# ==================================================

outDir="../../figures/Data_Application/epiN_misId/"

data_epiN = uproot.open(
    "~/ML_Files/data_epiN_v02/scored/epiN_dataset.root:PhysicsEvents"
).arrays(cols, library="pd")

k_n=len(data_epiN[data_epiN["pid"]==321])
pi_n=len(data_epiN[data_epiN["pid"]==211])
p_n=len(data_epiN[data_epiN["pid"]==2212])
print(f"K+ num {k_n}")
print(f"Pi+ num {pi_n}")
print(f"P num {p_n}")

pEdge = au.makeBinEdges(1, 5, 10)
tEdge = au.makeBinEdges(15, 20, 3)
#tEdge = au.makeBinEdges(5, 35, 1)


pCenters = (
    pEdge[:-1] + pEdge[1:]
) / 2



# ==================================================
# Split DATA into theta bins
# ==================================================

theta_bins = au.makeBins(
    data_epiN,
    "theta",
    binEdges=tEdge
)



# ==================================================
# Plot all theta bins
# ==================================================

plt.figure(figsize=(8,6))



for i, theta_bin in enumerate(theta_bins):

    vals = []
    errs = []

    p_bins = au.makeBins(
        theta_bin,
        "p",
        binEdges=pEdge
    )

    for j, pbin in enumerate(p_bins):

        passes = (
            (pbin["bdt_pass"] == True)&
            (pbin["pid"] == 321)
        )

        val, er = compute_misId(pbin, passes)
        print(val)
        vals.append(val)
        errs.append(er)

        grid.loc[len(grid)] = [
            tEdge[i],
            tEdge[i + 1],
            pEdge[j],
            pEdge[j + 1],
            val,
            np.nan
        ]



    plt.errorbar(
        pCenters,
        vals,
        yerr=errs,
        fmt='o',
        capsize=3,
        markersize=4,
        label=fr"${tEdge[i]:.1f}^\circ < \theta < {tEdge[i+1]:.1f}^\circ$"
    )



plt.xlabel("Momentum (GeV)")
plt.ylabel(r"$\pi \rightarrow K$ mis-ID")

plt.title(
    r"BDT $\pi\rightarrow K$ mis-ID from $ep\rightarrow e\pi^+(n)$"
)


plt.xticks(pEdge)
plt.xlim(
    pEdge[0],
    pEdge[-1]
)

plt.grid(False)

plt.legend(
    fontsize=8
)

plt.tight_layout()


plt.savefig(
    outDir + "epiN_pion_misId_vs_p_theta_bins.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


plt.figure(figsize=(8,6))



for i, theta_bin in enumerate(theta_bins):

    vals = []
    errs = []

    p_bins = au.makeBins(
        theta_bin,
        "p",
        binEdges=pEdge
    )

    for j, pbin in enumerate(p_bins):

        passes = (
            pbin["pid"] == 211    # or whatever your efficiency selection should be
        )

        val, er = compute_eff(pbin, passes)

        vals.append(val)
        errs.append(er)

        # Update the existing row in the dataframe
        mask = (
            (grid["theta_low"] == tEdge[i]) &
            (grid["theta_high"] == tEdge[i + 1]) &
            (grid["p_low"] == pEdge[j]) &
            (grid["p_high"] == pEdge[j + 1])
        )

        grid.loc[mask, "eff"] = val

    theta_low = tEdge[i]
    theta_high = tEdge[i + 1]

    plt.errorbar(
        pCenters,
        vals,
        yerr=errs,
        fmt='o',
        capsize=3,
        markersize=4,
        label=fr"${theta_low:.1f}^\circ < \theta < {theta_high:.1f}^\circ$"
    )

plt.xlabel("Momentum (GeV)")
plt.ylabel("Efficiency")

plt.title("epi(N) EB pion efficiency")


plt.xticks(pEdge)
plt.xlim(
    pEdge[0],
    pEdge[-1]
)

plt.grid(False)

plt.legend(
    fontsize=8
)

plt.tight_layout()


plt.savefig(
    outDir + "epiN_pion_eff_vs_p_theta_bins.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

grid.to_csv(outDir + "eff_misId_grid.csv", index=False)



#######################################################

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




data_SIDIS= uproot.open(
    "~/ML_Files/scored_data_v01/nSidis_005046.root:PhysicsEvents"
).arrays(cols, library="pd")

data_SIDIS=au.apply_Sidis_Cuts(data_SIDIS)


theta_bins = au.makeBins(
    data_SIDIS,
    "theta",
    binEdges=tEdge
)




for i, theta_bin in enumerate(theta_bins):

    vals = []
    errs = []

    p_bins = au.makeBins(
        theta_bin,
        "p",
        binEdges=pEdge
    )

    for j, pbin in enumerate(p_bins):

        passes = (
            pbin["pid"] == 211    # or whatever your efficiency selection should be
        )

        val, er = compute_contam(pbin,(outDir+"eff_misId_grid.csv"))
        
        vals.append(val)
        errs.append(er)



    theta_low = tEdge[i]
    theta_high = tEdge[i + 1]

    plt.errorbar(
        pCenters,
        vals,
        yerr=errs,
        fmt='o',
        capsize=3,
        markersize=4,
        label=fr"${theta_low:.1f}^\circ < \theta < {theta_high:.1f}^\circ$"
    )

plt.xlabel("Momentum (GeV)")
plt.ylabel("contamination")

plt.title("K+ epi(N) validated BDT contamination")


plt.xticks(pEdge)
plt.xlim(
    pEdge[0],
    pEdge[-1]
)

plt.grid(False)

plt.legend(
    fontsize=8
)

plt.tight_layout()


plt.savefig(
    outDir + "epiN_pion_contam_vs_p_theta_bins.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


