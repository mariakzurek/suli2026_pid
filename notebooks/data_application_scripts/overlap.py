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
import uproot
import sys
sys.path.append("../../scripts/")

from pathlib import Path
import common_functions as au
from baseline_chi2pid import passes_kplus_chi2pid_cut

cols = ["pid", "p", "theta", "beta", "chi2pid", "rich_RQ", "vz", "bdt_pass", "rich_best_PID", "rich_RQ", "rich_best_ntot", "bdt_score"]
kinematics =["Mx_eKX","Mx_epiX","Mx_epX", "Q2", "W", "y"]

width=0.15
mass=0.93

for kin in kinematics:
    cols.append(kin)
    
df = uproot.open("~/ML_Files/scored_data_v01/nSidis_005046.root:PhysicsEvents").arrays(cols, library="pd")

outDir="../../figures/Data_Application/epiN/"

df1=df[(df["Mx_epiX"]>(mass-width))&(df["Mx_epiX"]<(mass+width))]

tEdge = au.makeBinEdges(0, 36, 10)
pEdge = au.makeBinEdges(0.5, 5, 10)

tBins = au.makeBins(df1, "theta", binEdges=tEdge)

vals = np.zeros((len(tBins), len(pEdge)-1), dtype=int)

for i, tbin in enumerate(tBins):
    pBins = au.makeBins(tbin, "p", binEdges=pEdge)

    for j, pbin in enumerate(pBins):
        vals[i, j] = len(pbin)


plt.figure(figsize=(8, 10))

im = plt.imshow(
    vals,
    origin="lower",
    aspect="auto",
    cmap="coolwarm",
    interpolation="nearest"
)

# Colorbar
cbar = plt.colorbar(im)
cbar.set_label("Number of Events")

# Tick labels at bin centers
plt.xticks(
    np.arange(len(pEdge) - 1),
    [f"{0.5*(pEdge[i]+pEdge[i+1]):.2f}" for i in range(len(pEdge)-1)],
    rotation=45
)

plt.yticks(
    np.arange(len(tEdge) - 1),
    [f"{0.5*(tEdge[i]+tEdge[i+1]):.0f}" for i in range(len(tEdge)-1)]
)

plt.xlabel("Momentum p (GeV)")
plt.ylabel(r"$\theta$ (deg)")
plt.title("Event Counts")

plt.tight_layout()
plt.savefig(outDir + "event_heatmap.png", dpi=150)
plt.close()




df2=au.apply_Sidis_Cuts(df)

tBins = au.makeBins(df2, "theta", binEdges=tEdge)


vals2 = np.zeros((len(tBins), len(pEdge)-1), dtype=int)
for i, tbin in enumerate(tBins):
    pBins = au.makeBins(tbin, "p", binEdges=pEdge)

    for j, pbin in enumerate(pBins):
        vals2[i, j] = len(pbin)


plt.figure(figsize=(8, 10))

im = plt.imshow(
    vals2,
    origin="lower",
    aspect="auto",
    cmap="coolwarm",
    interpolation="nearest"
)

# Colorbar
cbar = plt.colorbar(im)
cbar.set_label("Number of Events")

# Tick labels at bin centers
plt.xticks(
    np.arange(len(pEdge) - 1),
    [f"{0.5*(pEdge[i]+pEdge[i+1]):.2f}" for i in range(len(pEdge)-1)],
    rotation=45
)

plt.yticks(
    np.arange(len(tEdge) - 1),
    [f"{0.5*(tEdge[i]+tEdge[i+1]):.0f}" for i in range(len(tEdge)-1)]
)

plt.xlabel("Momentum p (GeV)")
plt.ylabel(r"$\theta$ (deg)")
plt.title("SIDIS Event Counts")

plt.tight_layout()
plt.savefig(outDir + "SIDIS_event_heatmap.png", dpi=150)
plt.close()