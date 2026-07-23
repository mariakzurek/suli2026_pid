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


def compute_misID_MC(df):

    true_pi = df[df["mc_matching_pid"] == 211]

    if len(true_pi) == 0:
        return 0,0


    fake_k = true_pi[
        (true_pi["pid"] == 321) &
        (true_pi["bdt_pass"] == True)
    ]


    num = len(fake_k)
    den = len(true_pi)


    if num == 0:
        return 0,0


    value = num/den

    err = value * math.sqrt(
        (1/num)+(1/den)
    )

    return value,err



def compute_pion_eff_MC(df):

    true_pi = df[
        df["mc_matching_pid"] == 211
    ]

    if len(true_pi)==0:
        return 0,0


    reco_pi = true_pi[
        true_pi["pid"] == 211
    ]


    num=len(reco_pi)
    den=len(true_pi)


    if num==0:
        return 0,0


    value=num/den


    err=value*math.sqrt(
        (1/num)+(1/den)
    )


    return value,err




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


    n_fake=len(fake_k)
    n_k=len(selected_k)


    if n_k==0:
        return 0,0


    value=n_fake/n_k


    if n_fake>0:

        err=value*math.sqrt(
            (1/n_fake)+(1/n_k)
        )

    else:
        err=0


    return value,err



############################################################
# Load MC
############################################################


outDir="../../figures/MC_Application/epiN_misId/"

Path(outDir).mkdir(
    parents=True,
    exist_ok=True
)



cols=[
    "pid",
    "mc_matching_pid",
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



df_mc = uproot.open(
    "~/ML_Files/MC_scored/pid_training_v2.root:PhysicsEvents"
).arrays(
    cols,
    library="pd"
)



############################################################
# Binning
############################################################


pEdge=au.makeBinEdges(
    2.5,
    5,
    10
)


tEdge=au.makeBinEdges(
    10,
    20,
    3
)


pCenters=(
    pEdge[:-1]+pEdge[1:]
)/2



theta_bins=au.makeBins(
    df_mc,
    "theta",
    binEdges=tEdge
)



############################################################
# Calculate arrays
############################################################


misid_results=[]
eff_results=[]
contam_results=[]



for theta_bin in theta_bins:


    p_bins=au.makeBins(
        theta_bin,
        "p",
        binEdges=pEdge
    )


    mvals=[]
    merrs=[]

    evals=[]
    eerrs=[]

    cvals=[]
    cerrs=[]


    for pbin in p_bins:


        m,e=compute_misID_MC(
            pbin
        )

        eff,ee=compute_pion_eff_MC(
            pbin
        )

        contam,ce=compute_contamination_MC(
            pbin
        )


        mvals.append(m)
        merrs.append(e)

        evals.append(eff)
        eerrs.append(ee)

        cvals.append(contam)
        cerrs.append(ce)



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
# Plotting
############################################################


def make_overlay_plot(results,ylabel,title,filename):


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


    plt.xlabel(
        "Momentum (GeV)"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.xticks(
        pEdge
    )

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
        outDir+filename,
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()



make_overlay_plot(
    misid_results,
    r"$\pi\rightarrow K$ mis-ID",
    "MC Truth BDT pion mis-ID",
    "MC_misID_vs_p_theta.png"
)



make_overlay_plot(
    eff_results,
    "Pion efficiency",
    "MC Truth pion efficiency",
    "MC_efficiency_vs_p_theta.png"
)



make_overlay_plot(
    contam_results,
    "K contamination",
    "MC Truth K contamination",
    "MC_contamination_vs_p_theta.png"
)

############################################################
# 2D MC pion efficiency heatmap
############################################################

p_edges = np.linspace(0.5, 5.0, 11)
theta_edges = np.linspace(0.0, 35.0, 11)

eff_map = np.zeros((len(theta_edges)-1, len(p_edges)-1))

for i in range(len(theta_edges)-1):

    theta_low = theta_edges[i]
    theta_high = theta_edges[i+1]

    theta_bin = df_mc[
        (df_mc["theta"] >= theta_low) &
        (df_mc["theta"] < theta_high)
    ]

    for j in range(len(p_edges)-1):

        p_low = p_edges[j]
        p_high = p_edges[j+1]

        p_bin = theta_bin[
            (theta_bin["p"] >= p_low) &
            (theta_bin["p"] < p_high)
        ]

        true_pi = p_bin[
            p_bin["mc_matching_pid"] == 211
        ]

        reco_pi = true_pi[
            true_pi["pid"] == 211
        ]

        if len(true_pi) > 0:
            eff_map[i, j] = len(reco_pi) / len(true_pi)
        else:
            eff_map[i, j] = 0


plt.figure(figsize=(8,6))

mesh = plt.pcolormesh(
    p_edges,
    theta_edges,
    eff_map,
    shading="auto",
    cmap="coolwarm",
    vmin=0,
    vmax=1
)

plt.colorbar(mesh, label="MC Pion Efficiency")

plt.xlabel("Momentum (GeV)")
plt.ylabel(r"$\theta$ (deg)")
plt.title("Pion Efficiency with MC Truth")

plt.xlim(0.5,5)
plt.ylim(0,35)

plt.tight_layout()

plt.savefig(
    outDir + "mc_pion_efficiency_heatmap.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

############################################################
# MC Missing Mass Histogram (after neutron cut)
############################################################

mc_cut = df_mc[
    (df_mc["Mx_epiX"] > 0.85) &
    (df_mc["Mx_epiX"] < 1.05)
]


mc_cut=mc_cut[mc_cut["p"]<2.2]

plt.figure(figsize=(8,6))

plt.hist(
    mc_cut["Mx_epiX"],
    bins=100,
    range=(0.85, 1.05),
    histtype="step",
    linewidth=2
)

plt.xlabel(r"$M_X(e\pi)$ (GeV)")
plt.ylabel("Events")
plt.title("MC $ep\\rightarrow e\\pi(n)$ Missing Mass p<2.2")

plt.tight_layout()

plt.savefig(
    outDir + "MC_Mx_epiX_hist.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Done.")