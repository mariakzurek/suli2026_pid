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

for kin in kinematics:
    cols.append(kin)
    
df = uproot.open("~/ML_Files/scored_data_v01/nSidis_005046.root:PhysicsEvents").arrays(cols, library="pd")
#df=df[df["rich_best_ntot"]>3]
#df=df[df["rich_RQ"]>0.2]
#df=df[df["rich_best_PID"]!=-9999]
df=au.apply_Sidis_Cuts(df)

outDir="../../figures/Data_Application/contams/"

# PDF that will hold each MC/DATA pair side-by-side, one pair per page
pdf = PdfPages(outDir+"contams_comparison.pdf")


# In[3]:


import math
import numpy as np
import matplotlib.pyplot as plt

# Define momentum bins
pEdges = au.makeBinEdges(2.5, 5, 10)

# Select RICH angular range
df_rich = df[df["theta"] < 11.25]
# = df_rich[
 #   (df_rich["rich_best_PID"] == 321) |
  #  (df_rich["rich_best_PID"] == 221) |
   # (df_rich["rich_best_PID"] == 2212)
#]
df_rich = df_rich[
    (df_rich["rich_best_PID"]!=-9999)
]


###############################################################################
df_val = pd.read_parquet("/work/clas12/CooperBe/MLStuff/dataset_v03/val.parquet")

#df_val=au.apply_Sidis_Cuts(df_val)
mod, mod_df = au.load_model_and_data("/work/clas12/CooperBe/MLStuff/tier2All/model_v02/model.joblib", df_val)


# In[3]:


tStart = 5
tEnd = 11.25
tBinNum =5

pStart = 2.5
pEnd = 5
pBinNum = 10

pBinEdges = np.linspace(pStart, pEnd, pBinNum + 1)
tBinEdges = np.linspace(tStart, tEnd, tBinNum + 1)


# In[4]:


csvFile = "optimized_thresholds_MC.csv"

if Path(csvFile).exists():
    print(f"Loading optimized thresholds from {csvFile}")
    results_df = pd.read_csv(csvFile)
else:
    print("Optimized threshold file not found. Running optimization...")
    results_df = au.optimizeFOM(
        mod_df,
        tBinEdges,
        pBinEdges,
        outputCSV=csvFile,
        deviation=0.03
    )

cols = ["pid", "mc_matching_pid", "p", "theta", "beta", "chi2pid", "rich_RQ", "rich_best_PID", "rich_best_ntot", "vz","ftof_energy_1B", "ftof_time_1B", "ftof_path_1B", "ecin_path", "ecin_energy", "ecin_time"]
kinematics =["Mx_eKX","Mx_epiX","Mx_epX", "Q2", "W", "y"]

for kin in kinematics:
    cols.append(kin)

df_test = uproot.open("/volatile/clas12/cooperb/SULI/pid_training_v2.root:PhysicsEvents").arrays(cols, library="pd")

#df_test=pd.read_parquet("/work/clas12/CooperBe/MLStuff/dataset_v03/test.parquet")
df_test=df_test[df_test["mc_matching_pid"]!=-9999]
feature_names=au.get_feature_names("/work/clas12/CooperBe/MLStuff/tier2All/model_v02/model.joblib")
print(feature_names)
df_test = au.apply_model_to_df(mod, df_test, feature_names)
df_test=au.apply_Sidis_Cuts(df_test)


outDir="../../figures/Data_Application/contams/"

###############################################################################################################################

import numpy as np
import matplotlib.pyplot as plt

# Controls whether acceptance values are written in each square
SHOW_VALUES = False

# Define bins
p_edges = np.linspace(2.5, 5.0, 100)       # 10 p bins
theta_edges = np.linspace(0, 11.25, 100)      # 10 theta bins

# Store acceptance values
acceptance = np.zeros((len(theta_edges)-1, len(p_edges)-1))

# Loop through theta bins
theta_bins = au.makeBins(df_test, "theta", binEdges=theta_edges)

for i, theta_bin in enumerate(theta_bins):

    # Now split this theta bin into momentum bins
    p_bins = au.makeBins(theta_bin, "p", binEdges=p_edges)

    for j, p_bin in enumerate(p_bins):

        # Define RICH cut
        cut_RICH = (
            #(p_bin["rich_best_PID"] != -9999) &
            #(p_bin["rich_best_ntot"] > 3) &
            (p_bin["rich_RQ"] > 0.2)
        )

        before = len(p_bin)
        after = len(p_bin[cut_RICH])

        acceptance[i, j] = after


# -----------------------------
# Plot heatmap
# -----------------------------

plt.figure(figsize=(9,7))

plt.imshow(
    acceptance,
    origin="lower",
    aspect="auto",
    extent=[
        p_edges[0],
        p_edges[-1],
        theta_edges[0],
        theta_edges[-1]
    ],
    cmap='coolwarm'
)

plt.colorbar(label="Events with RQ>2")

plt.xlabel(r"$p$ (GeV/c)")
plt.ylabel(r"$\theta$ (deg)")
plt.title("RICH RQ events >0.2")


# Add values to squares
if SHOW_VALUES:
    for i in range(len(theta_edges)-1):
        for j in range(len(p_edges)-1):

            p_center = (p_edges[j] + p_edges[j+1]) / 2
            theta_center = (theta_edges[i] + theta_edges[i+1]) / 2

            plt.text(
                p_center,
                theta_center,
                f"{acceptance[i,j]:.3f}",
                ha="center",
                va="center",
                fontsize=7
            )


plt.savefig("MC_RQ.png", dpi=150, bbox_inches="tight")
plt.close()

# keep MC RQ acceptance array so it can be paired with DATA RQ below
acceptance_MC_RQ = acceptance.copy()
p_edges_MC_RQ = p_edges.copy()
theta_edges_MC_RQ = theta_edges.copy()

######################################################################################

# Controls whether acceptance values are written in each square
SHOW_VALUES = False

# Define bins
p_edges = np.linspace(2.5, 5.0, 100)       # 10 p bins
theta_edges = np.linspace(0, 11.25, 100)      # 10 theta bins

# Store acceptance values
acceptance = np.zeros((len(theta_edges)-1, len(p_edges)-1))

# Loop through theta bins
theta_bins = au.makeBins(df_rich, "theta", binEdges=theta_edges)

for i, theta_bin in enumerate(theta_bins):

    # Now split this theta bin into momentum bins
    p_bins = au.makeBins(theta_bin, "p", binEdges=p_edges)

    for j, p_bin in enumerate(p_bins):

        # Define RICH cut
        cut_RICH = (
            #(p_bin["rich_best_PID"] != -9999) &
            #(p_bin["rich_best_ntot"] > 3) &
            (p_bin["rich_RQ"] > 0.2)
        )

        before = len(p_bin)
        after = len(p_bin[cut_RICH])

        acceptance[i, j] = after


# -----------------------------
# Plot heatmap
# -----------------------------

plt.figure(figsize=(9,7))

plt.imshow(
    acceptance,
    origin="lower",
    aspect="auto",
    extent=[
        p_edges[0],
        p_edges[-1],
        theta_edges[0],
        theta_edges[-1]
    ],
    cmap='coolwarm'
)

plt.colorbar(label="Events with RQ>2")

plt.xlabel(r"$p$ (GeV/c)")
plt.ylabel(r"$\theta$ (deg)")
plt.title("RICH RQ events >0.2")


# Add values to squares
if SHOW_VALUES:
    for i in range(len(theta_edges)-1):
        for j in range(len(p_edges)-1):

            p_center = (p_edges[j] + p_edges[j+1]) / 2
            theta_center = (theta_edges[i] + theta_edges[i+1]) / 2

            plt.text(
                p_center,
                theta_center,
                f"{acceptance[i,j]:.3f}",
                ha="center",
                va="center",
                fontsize=7
            )


plt.savefig(outDir+"DATA_RQ.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Combine MC_RQ and DATA_RQ side by side into one PDF page ---
fig, (axMC, axDATA) = plt.subplots(1, 2, figsize=(16, 7))

imMC = axMC.imshow(
    acceptance_MC_RQ,
    origin="lower",
    aspect="auto",
    extent=[p_edges_MC_RQ[0], p_edges_MC_RQ[-1], theta_edges_MC_RQ[0], theta_edges_MC_RQ[-1]],
    cmap='coolwarm'
)
fig.colorbar(imMC, ax=axMC, label="Events with RQ>2")
axMC.set_xlabel(r"$p$ (GeV/c)")
axMC.set_ylabel(r"$\theta$ (deg)")
axMC.set_title("MC: RICH RQ events >0.2")

imDATA = axDATA.imshow(
    acceptance,
    origin="lower",
    aspect="auto",
    extent=[p_edges[0], p_edges[-1], theta_edges[0], theta_edges[-1]],
    cmap='coolwarm'
)
fig.colorbar(imDATA, ax=axDATA, label="Events with RQ>2")
axDATA.set_xlabel(r"$p$ (GeV/c)")
axDATA.set_ylabel(r"$\theta$ (deg)")
axDATA.set_title("DATA: RICH RQ events >0.2")

pdf.savefig(fig)
plt.close(fig)

############################################################################################

import numpy as np
import matplotlib.pyplot as plt

# Controls whether acceptance values are written in each square
SHOW_VALUES = False

# Define bins
p_edges = np.linspace(0, 5.0, 100)       # 10 p bins
theta_edges = np.linspace(0, 11.25, 100)      # 10 theta bins

# Store acceptance values
acceptance = np.zeros((len(theta_edges)-1, len(p_edges)-1))

# Loop through theta bins
theta_bins = au.makeBins(df_test, "theta", binEdges=theta_edges)

for i, theta_bin in enumerate(theta_bins):

    # Now split this theta bin into momentum bins
    p_bins = au.makeBins(theta_bin, "p", binEdges=p_edges)

    for j, p_bin in enumerate(p_bins):

        # Define RICH cut
        cut_RICH = (
            #(p_bin["rich_best_PID"] != -9999) &
            (p_bin["rich_best_ntot"] > 3)
            #(p_bin["rich_RQ"] > 0.2)
        )

        before = len(p_bin)
        after = len(p_bin[cut_RICH])

        acceptance[i, j] = after


# -----------------------------
# Plot heatmap
# -----------------------------

plt.figure(figsize=(9,7))

plt.imshow(
    acceptance,
    origin="lower",
    aspect="auto",
    extent=[
        p_edges[0],
        p_edges[-1],
        theta_edges[0],
        theta_edges[-1]
    ],
    cmap='coolwarm'
)

plt.colorbar(label="Events with npho>3")

plt.xlabel(r"$p$ (GeV/c)")
plt.ylabel(r"$\theta$ (deg)")
plt.title("RICH npho events >3")


# Add values to squares
if SHOW_VALUES:
    for i in range(len(theta_edges)-1):
        for j in range(len(p_edges)-1):

            p_center = (p_edges[j] + p_edges[j+1]) / 2
            theta_center = (theta_edges[i] + theta_edges[i+1]) / 2

            plt.text(
                p_center,
                theta_center,
                f"{acceptance[i,j]:.3f}",
                ha="center",
                va="center",
                fontsize=7
            )


plt.savefig(outDir+"MC_npho.png", dpi=150, bbox_inches="tight")
plt.close()

# keep MC npho acceptance array so it can be paired with DATA npho below
acceptance_MC_npho = acceptance.copy()
p_edges_MC_npho = p_edges.copy()
theta_edges_MC_npho = theta_edges.copy()

######################################################################################

# Controls whether acceptance values are written in each square
SHOW_VALUES = False

# Define bins
p_edges = np.linspace(2.5, 5.0, 100)       # 10 p bins
theta_edges = np.linspace(0, 11.25, 100)      # 10 theta bins

# Store acceptance values
acceptance = np.zeros((len(theta_edges)-1, len(p_edges)-1))

# Loop through theta bins
theta_bins = au.makeBins(df_rich, "theta", binEdges=theta_edges)

for i, theta_bin in enumerate(theta_bins):

    # Now split this theta bin into momentum bins
    p_bins = au.makeBins(theta_bin, "p", binEdges=p_edges)

    for j, p_bin in enumerate(p_bins):

        # Define RICH cut
        cut_RICH = (
            #(p_bin["rich_best_PID"] != -9999) &
            (p_bin["rich_best_ntot"] > 3)
            #(p_bin["rich_RQ"] > 0.2)
        )

        before = len(p_bin)
        after = len(p_bin[cut_RICH])

        acceptance[i, j] = after


# -----------------------------
# Plot heatmap
# -----------------------------

plt.figure(figsize=(9,7))

plt.imshow(
    acceptance,
    origin="lower",
    aspect="auto",
    extent=[
        p_edges[0],
        p_edges[-1],
        theta_edges[0],
        theta_edges[-1]
    ],
    cmap='coolwarm'
)

plt.colorbar(label="Events with npho>3")

plt.xlabel(r"$p$ (GeV/c)")
plt.ylabel(r"$\theta$ (deg)")
plt.title("RICH npho events >3")


# Add values to squares
if SHOW_VALUES:
    for i in range(len(theta_edges)-1):
        for j in range(len(p_edges)-1):

            p_center = (p_edges[j] + p_edges[j+1]) / 2
            theta_center = (theta_edges[i] + theta_edges[i+1]) / 2

            plt.text(
                p_center,
                theta_center,
                f"{acceptance[i,j]:.3f}",
                ha="center",
                va="center",
                fontsize=7
            )


plt.savefig(outDir+"DATA_npho.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Combine MC_npho and DATA_npho side by side into one PDF page ---
fig, (axMC, axDATA) = plt.subplots(1, 2, figsize=(16, 7))

imMC = axMC.imshow(
    acceptance_MC_npho,
    origin="lower",
    aspect="auto",
    extent=[p_edges_MC_npho[0], p_edges_MC_npho[-1], theta_edges_MC_npho[0], theta_edges_MC_npho[-1]],
    cmap='coolwarm'
)
fig.colorbar(imMC, ax=axMC, label="Events with npho>3")
axMC.set_xlabel(r"$p$ (GeV/c)")
axMC.set_ylabel(r"$\theta$ (deg)")
axMC.set_title("MC: RICH npho events >3")

imDATA = axDATA.imshow(
    acceptance,
    origin="lower",
    aspect="auto",
    extent=[p_edges[0], p_edges[-1], theta_edges[0], theta_edges[-1]],
    cmap='coolwarm'
)
fig.colorbar(imDATA, ax=axDATA, label="Events with npho>3")
axDATA.set_xlabel(r"$p$ (GeV/c)")
axDATA.set_ylabel(r"$\theta$ (deg)")
axDATA.set_title("DATA: RICH npho events >3")

pdf.savefig(fig)
plt.close(fig)

############################################################################################

import numpy as np
import matplotlib.pyplot as plt

# Controls whether acceptance values are written in each square
SHOW_VALUES = False

# Define bins
p_edges = np.linspace(2.5, 5.0, 100)       # 10 p bins
theta_edges = np.linspace(0, 11.25, 100)      # 10 theta bins

# Store acceptance values
acceptance = np.zeros((len(theta_edges)-1, len(p_edges)-1))

# Loop through theta bins
theta_bins = au.makeBins(df_test, "theta", binEdges=theta_edges)

for i, theta_bin in enumerate(theta_bins):

    # Now split this theta bin into momentum bins
    p_bins = au.makeBins(theta_bin, "p", binEdges=p_edges)

    for j, p_bin in enumerate(p_bins):

        # Define RICH cut
        cut_RICH = (
            (p_bin["rich_best_PID"] != -9999)
            #(p_bin["rich_best_ntot"] > 3) &
            #(p_bin["rich_RQ"] > 0.2)
        )

        before = len(p_bin)
        after = len(p_bin[cut_RICH])

        acceptance[i, j] = after


# -----------------------------
# Plot heatmap
# -----------------------------

plt.figure(figsize=(9,7))

plt.imshow(
    acceptance,
    origin="lower",
    aspect="auto",
    extent=[
        p_edges[0],
        p_edges[-1],
        theta_edges[0],
        theta_edges[-1]
    ],
    cmap='coolwarm'
)

plt.colorbar(label="Events with rich pid")

plt.xlabel(r"$p$ (GeV/c)")
plt.ylabel(r"$\theta$ (deg)")
plt.title("RICH events with pid")



# Add values to squares
if SHOW_VALUES:
    for i in range(len(theta_edges)-1):
        for j in range(len(p_edges)-1):

            p_center = (p_edges[j] + p_edges[j+1]) / 2
            theta_center = (theta_edges[i] + theta_edges[i+1]) / 2

            plt.text(
                p_center,
                theta_center,
                f"{acceptance[i,j]:.3f}",
                ha="center",
                va="center",
                fontsize=7
            )


plt.savefig(outDir+"MC_pid.png", dpi=150, bbox_inches="tight")
plt.close()

# keep MC pid acceptance array so it can be paired with DATA pid below
acceptance_MC_pid = acceptance.copy()
p_edges_MC_pid = p_edges.copy()
theta_edges_MC_pid = theta_edges.copy()

######################################################################################

# Controls whether acceptance values are written in each square
SHOW_VALUES = False

# Define bins
p_edges = np.linspace(2.5, 5.0, 100)       # 10 p bins
theta_edges = np.linspace(0, 11.25, 100)      # 10 theta bins

# Store acceptance values
acceptance = np.zeros((len(theta_edges)-1, len(p_edges)-1))

# Loop through theta bins
theta_bins = au.makeBins(df_rich, "theta", binEdges=theta_edges)

for i, theta_bin in enumerate(theta_bins):

    # Now split this theta bin into momentum bins
    p_bins = au.makeBins(theta_bin, "p", binEdges=p_edges)

    for j, p_bin in enumerate(p_bins):

        # Define RICH cut
        cut_RICH = (
            (p_bin["rich_best_PID"] != -9999)
            #(p_bin["rich_best_ntot"] > 3) &
            #(p_bin["rich_RQ"] > 0.2)
        )

        before = len(p_bin)
        after = len(p_bin[cut_RICH])

        acceptance[i, j] = after


# -----------------------------
# Plot heatmap
# -----------------------------

plt.figure(figsize=(9,7))

plt.imshow(
    acceptance,
    origin="lower",
    aspect="auto",
    extent=[
        p_edges[0],
        p_edges[-1],
        theta_edges[0],
        theta_edges[-1]
    ],
    cmap='coolwarm'
)

plt.colorbar(label="Events with rich pid")

plt.xlabel(r"$p$ (GeV/c)")
plt.ylabel(r"$\theta$ (deg)")
plt.title("RICH events with pid")


# Add values to squares
if SHOW_VALUES:
    for i in range(len(theta_edges)-1):
        for j in range(len(p_edges)-1):

            p_center = (p_edges[j] + p_edges[j+1]) / 2
            theta_center = (theta_edges[i] + theta_edges[i+1]) / 2

            plt.text(
                p_center,
                theta_center,
                f"{acceptance[i,j]:.3f}",
                ha="center",
                va="center",
                fontsize=7
            )


plt.savefig(outDir+"DATA_pid.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Combine MC_pid and DATA_pid side by side into one PDF page ---
fig, (axMC, axDATA) = plt.subplots(1, 2, figsize=(16, 7))

imMC = axMC.imshow(
    acceptance_MC_pid,
    origin="lower",
    aspect="auto",
    extent=[p_edges_MC_pid[0], p_edges_MC_pid[-1], theta_edges_MC_pid[0], theta_edges_MC_pid[-1]],
    cmap='coolwarm'
)
fig.colorbar(imMC, ax=axMC, label="Events with rich pid")
axMC.set_xlabel(r"$p$ (GeV/c)")
axMC.set_ylabel(r"$\theta$ (deg)")
axMC.set_title("MC: RICH events with pid")

imDATA = axDATA.imshow(
    acceptance,
    origin="lower",
    aspect="auto",
    extent=[p_edges[0], p_edges[-1], theta_edges[0], theta_edges[-1]],
    cmap='coolwarm'
)
fig.colorbar(imDATA, ax=axDATA, label="Events with rich pid")
axDATA.set_xlabel(r"$p$ (GeV/c)")
axDATA.set_ylabel(r"$\theta$ (deg)")
axDATA.set_title("DATA: RICH events with pid")

pdf.savefig(fig)
plt.close(fig)

# Close out the combined PDF (writes it to disk)
pdf.close()


################################################################################################### MC EPI(N) tests

import os
import numpy as np
import matplotlib.pyplot as plt

p_edges = np.linspace(0.5, 5.0, 25)
theta_edges = np.linspace(5, 35, 15)

outDir = "../../figures/Data_Application/"
os.makedirs(outDir, exist_ok=True)


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

data_epiN = uproot.open(
    "~/ML_Files/data_epiN_v02/scored/epiN_dataset.root:PhysicsEvents"
).arrays(cols, library="pd")


# ==================================================
# LOAD MC TRAINING SAMPLE
# Select truth pions
# ==================================================

mc_training = pd.read_parquet(
    "/work/clas12/CooperBe/MLStuff/dataset_v03/train.parquet"
)


#mc_true_pions = mc_training[
 #   mc_training["mc_matching_pid"] == 211
#]
mc_true_pions = mc_training[
    mc_training["mc_matching_pid"] == 211
]


# Optional if you want exclusive neutron peak MC:
#
# mc_true_pions = mc_true_pions[
#     (mc_true_pions["Mx_epiX"] > 0.85) &
#     (mc_true_pions["Mx_epiX"] < 1.05)
# ]


# ==================================================
# Define common binning
# ==================================================




# ==================================================
# Create count arrays
# ==================================================

data_counts = np.zeros(
    (len(p_edges)-1, len(theta_edges)-1)
)

mc_counts = np.zeros_like(data_counts)



# ==================================================
# Fill DATA p-theta counts
# ==================================================

data_p_bins = au.makeBins(
    data_epiN,
    "p",
    binEdges=p_edges
)


for ip, data_p_slice in enumerate(data_p_bins):

    data_theta_bins = au.makeBins(
        data_p_slice,
        "theta",
        binEdges=theta_edges
    )

    for it, data_pt_slice in enumerate(data_theta_bins):
        data_counts[ip, it] = len(data_pt_slice)



# ==================================================
# Fill MC p-theta counts
# ==================================================

mc_p_bins = au.makeBins(
    mc_true_pions,
    "p",
    binEdges=p_edges
)


for ip, mc_p_slice in enumerate(mc_p_bins):

    mc_theta_bins = au.makeBins(
        mc_p_slice,
        "theta",
        binEdges=theta_edges
    )

    for it, mc_pt_slice in enumerate(mc_theta_bins):
        mc_counts[ip, it] = len(mc_pt_slice)



# ==================================================
# Print sanity checks
# ==================================================

print("DATA events:", data_counts.sum())
print("MC true pion events:", mc_counts.sum())

print("DATA max bin:", data_counts.max())
print("MC max bin:", mc_counts.max())



# ==================================================
# DATA heatmap
# ==================================================

plt.figure(figsize=(8,6))

plt.pcolormesh(
    p_edges,
    theta_edges,
    data_counts.T,
    cmap="coolwarm",
    vmin=0,
    vmax=data_counts.max()
)

plt.colorbar(label="Number of events")

plt.xlabel("Momentum p (GeV)")
plt.ylabel(r"$\theta$ (deg)")
plt.title("DATA: EB pion sample from neutron peak")

plt.tight_layout()

plt.savefig(
    outDir + "epiN_data_p_theta_coverage.png",
    dpi=150
)

plt.close()



# ==================================================
# MC heatmap
# ==================================================

plt.figure(figsize=(8,6))

plt.pcolormesh(
    p_edges,
    theta_edges,
    mc_counts.T,
    cmap="coolwarm",
    vmin=0,
    vmax=mc_counts.max()
)

plt.colorbar(label="Number of events")

plt.xlabel("Momentum p (GeV)")
plt.ylabel(r"$\theta$ (deg)")
plt.title("MC: True pion from BDT training sample")

plt.tight_layout()

plt.savefig(
    outDir + "epiN_mc_p_theta_coverage.png",
    dpi=150
)

plt.close()



# ==================================================
# Coverage comparison
# ==================================================

outside_data = (
    (data_counts > 0) &
    (mc_counts == 0)
)

print(
    "DATA bins outside MC coverage:",
    outside_data.sum()
)


if outside_data.sum() > 0:
    print("WARNING: Some DATA pion bins are outside MC pion coverage.")
else:
    print("MC covers all DATA pion bins.")





