#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Maria's Script that has the chi2pid cut functions
import numpy as np

# ---------------------------------------------------------------------------
# K+ pass-2 chi2pid parameters (pid_cuts.java lines 438–439)
# ---------------------------------------------------------------------------
MU_KPLUS = 0.082
SIGMA_KPLUS = 0.985

# Lower-bound exponential parameters for p > 2 GeV/c (pid_cuts.java line 443)
LOWER_C = 1.2        # plateau value the exponential decays away from
LOWER_P0 = 2.0       # momentum at which the exponential kicks in
LOWER_TAU = 0.6      # decay constant in GeV/c

# Upper-bound exponential parameters for p > 2.5 GeV/c (pid_cuts.java line 445)
UPPER_C = 2.6        # plateau value the exponential decays away from
UPPER_P0 = 2.5       # momentum at which the upper bound starts contracting
UPPER_TAU = 0.3      # decay constant in GeV/c

# ---------------------------------------------------------------------------
# π+ pass-2 chi2pid parameters (pid_cuts.java lines 428–429)
# ---------------------------------------------------------------------------
MU_PIPLUS = -0.067
SIGMA_PIPLUS = 0.956

# Upper-bound exponential parameters for π+ at p > 3.5 GeV (pid_cuts.java line 433)
# Lower bound stays flat at μ−3σ for all p (no lower-bound exponential for π+)
PIPLUS_UPPER_C = -0.55
PIPLUS_UPPER_P0 = 3.5
PIPLUS_UPPER_TAU = 0.55


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _const_plus_exponential(c: float, a: float, tau: float,
                             p0: float, p: np.ndarray) -> np.ndarray:
    """
    Evaluate the helper function used by pid_cuts.java (line 263–264).

        f(p) = c + a * exp( -(p - p0) / tau )

    Parameters
    ----------
    c   : asymptotic plateau (value as p → ∞)
    a   : amplitude of the exponential term at p = p0
    tau : decay constant (GeV/c); controls how quickly the exponential falls
    p0  : reference momentum (GeV/c) at which the exponential term equals *a*
    p   : track momentum array (GeV/c)

    Returns
    -------
    np.ndarray, same shape as *p*
    """
    return c + a * np.exp(-(p - p0) / tau)


# ---------------------------------------------------------------------------
# K+ cut
# ---------------------------------------------------------------------------

def passes_kplus_chi2pid_cut(chi2pid: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Pass-2 K+ chi2pid cut, momentum-dependent in 3 regimes.

    Direct Python translation of the K+ block in
    ``charged_hadron_pass2_chi2pid_cut`` (pid_cuts.java lines 437–447).

    Parameters
    ----------
    chi2pid : array-like
        REC::Particle.chi2pid for each track (pull of beta against K+
        hypothesis).  Can be a numpy array or any array-like (pandas Series,
        list).
    p : array-like
        Track momentum magnitude in GeV/c.

    Returns
    -------
    np.ndarray of bool, same shape as inputs. True = passes cut.

    Examples
    --------
    Apply to a pandas DataFrame::

        df["baseline_pass"] = passes_kplus_chi2pid_cut(
            df["chi2pid"].to_numpy(), df["p"].to_numpy()
        )
    """
    chi2pid = np.asarray(chi2pid, dtype=float)
    p = np.asarray(p, dtype=float)

    lo_flat = MU_KPLUS - 3.0 * SIGMA_KPLUS   # -2.873
    hi_flat = MU_KPLUS + 3.0 * SIGMA_KPLUS   # +3.037

    lo_exp = _const_plus_exponential(LOWER_C, lo_flat - LOWER_C, LOWER_TAU, LOWER_P0, p)
    hi_exp = _const_plus_exponential(UPPER_C, hi_flat - UPPER_C, UPPER_TAU, UPPER_P0, p)

    # Regime 1: p < 2  → symmetric flat window
    # Regime 2: 2 < p < 2.5 → lower bound contracts, upper stays flat
    # Regime 3: p >= 2.5 → both bounds contract
    # Note: tracks exactly at p == 2.0 are not covered by the Java branches
    # (Java uses p<2 and p>2), so they fall to the else (regime 3).
    # We replicate that behaviour with the np.where nesting below.

    lower = np.where(p < 2.0, lo_flat, lo_exp)
    upper = np.where(p < 2.5, hi_flat, hi_exp)

    return (chi2pid > lower) & (chi2pid < upper)


# ---------------------------------------------------------------------------
# π+ cut
# ---------------------------------------------------------------------------

def passes_piplus_chi2pid_cut(chi2pid: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Pass-2 π+ chi2pid cut, momentum-dependent in 2 regimes.

    Direct Python translation of the π+ block (pid == 211) in
    ``charged_hadron_pass2_chi2pid_cut`` (pid_cuts.java lines 427–435).

    Regimes
    -------
    p < 3.5 GeV/c  : symmetric window  mu ± 3*sigma
                     lower = μ − 3σ = −2.935
                     upper = μ + 3σ = +2.801
    p ≥ 3.5 GeV/c  : lower bound stays flat at μ − 3σ = −2.935;
                     upper bound contracts exponentially, asymptoting to
                     PIPLUS_UPPER_C = −0.55 as p → ∞.
                     upper = const_plus_exponential(−0.55,
                                                    μ+3σ+0.55,
                                                    0.55, 3.5, p)

    Parameters
    ----------
    chi2pid : array-like
        REC::Particle.chi2pid for each track (pull of beta against π+
        hypothesis).  Can be a numpy array or any array-like (pandas Series,
        list).
    p : array-like
        Track momentum magnitude in GeV/c.

    Returns
    -------
    np.ndarray of bool, same shape as inputs. True = passes cut.

    Notes
    -----
    The lower bound is flat (μ − 3σ) for all momenta — the π+ cut has no
    lower-bound exponential regime, unlike K+.  The upper bound asymptotes
    to −0.55 at high momentum, reflecting the narrowing of the pion/kaon
    separation in chi2pid space.

    Examples
    --------
    Apply to a pandas DataFrame::

        df["baseline_piplus"] = passes_piplus_chi2pid_cut(
            df["chi2pid"].to_numpy(), df["p"].to_numpy()
        )
    """
    chi2pid = np.asarray(chi2pid, dtype=float)
    p = np.asarray(p, dtype=float)

    lo_flat = MU_PIPLUS - 3.0 * SIGMA_PIPLUS   # -2.935
    hi_flat = MU_PIPLUS + 3.0 * SIGMA_PIPLUS   # +2.801

    # Upper-bound exponential: asymptotes to PIPLUS_UPPER_C = -0.55
    hi_exp = _const_plus_exponential(
        PIPLUS_UPPER_C,
        hi_flat - PIPLUS_UPPER_C,       # amplitude = hi_flat + 0.55 = 3.351
        PIPLUS_UPPER_TAU,
        PIPLUS_UPPER_P0,
        p,
    )

    # Regime 1: p < 3.5 → symmetric flat window
    # Regime 2: p ≥ 3.5 → lower stays flat, upper contracts exponentially
    lower = lo_flat  # constant for all p
    upper = np.where(p < 3.5, hi_flat, hi_exp)

    return (chi2pid > lower) & (chi2pid < upper)


# ---------------------------------------------------------------------------
# Loose flat-cut reference (older, momentum-independent, species-agnostic)
# ---------------------------------------------------------------------------

def passes_loose_chi2pid_cut(chi2pid: np.ndarray, p: np.ndarray,
                              threshold: float = 3.0) -> np.ndarray:
    """
    Loose chi2pid cut: |chi2pid| < threshold (default 3).

    This is the older, momentum-independent cut.  The numeric form is
    identical for K+ and π+ (and any other species), so the function is
    provided once under a species-agnostic name.  Use it for comparison and
    orientation only — the production baseline is the species-specific
    pass-2, momentum-dependent cut (``passes_kplus_chi2pid_cut`` or
    ``passes_piplus_chi2pid_cut``).

    Parameters
    ----------
    chi2pid   : array-like, REC::Particle.chi2pid
    p         : array-like, track momentum (GeV/c) — accepted but not used
                (kept so call sites have a uniform signature)
    threshold : float, half-width of the symmetric window (default 3)

    Returns
    -------
    np.ndarray of bool
    """
    chi2pid = np.asarray(chi2pid, dtype=float)
    return np.abs(chi2pid) < threshold


# Backward-compatible alias so any existing code importing the old name works.
passes_kplus_loose_chi2pid_cut = passes_loose_chi2pid_cut


# ---------------------------------------------------------------------------
# __main__: sanity-check assertions + optional visualisation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    # -----------------------------------------------------------------------
    # K+ spot-check assertions
    # -----------------------------------------------------------------------
    # p = 1.0, chi2pid = 0.0  →  well inside flat window → True
    assert passes_kplus_chi2pid_cut(np.array([0.0]), np.array([1.0]))[0], \
        "FAIL: p=1.0, chi2pid=0.0 should pass"

    # p = 1.0, chi2pid = -3.0  →  lower bound = -2.873; -3.0 < -2.873 → False
    assert not passes_kplus_chi2pid_cut(np.array([-3.0]), np.array([1.0]))[0], \
        "FAIL: p=1.0, chi2pid=-3.0 should NOT pass (below lower bound -2.873)"

    # p = 3.0, chi2pid = -2.0
    #   lower = 1.2 + (-4.073) * exp(-1/0.6) ≈ 0.432
    #   -2.0 < 0.432 → False
    assert not passes_kplus_chi2pid_cut(np.array([-2.0]), np.array([3.0]))[0], \
        "FAIL: p=3.0, chi2pid=-2.0 should NOT pass (below lower bound ≈ 0.432)"

    # p = 3.0, chi2pid = 1.0
    #   lower ≈ 0.432, upper = 2.6 + 0.437*exp(-0.5/0.3) ≈ 2.682
    #   0.432 < 1.0 < 2.682 → True
    assert passes_kplus_chi2pid_cut(np.array([1.0]), np.array([3.0]))[0], \
        "FAIL: p=3.0, chi2pid=1.0 should pass (between ≈0.432 and ≈2.682)"

    print("All K+ spot-check assertions passed.")

    # -----------------------------------------------------------------------
    # π+ spot-check assertions
    # -----------------------------------------------------------------------
    # Precompute flat bounds for reference in comments:
    #   lo_flat = -0.067 - 3*0.956 = -2.935
    #   hi_flat = -0.067 + 3*0.956 = +2.801

    # p = 1.0, chi2pid = 0.0  →  flat window: -2.935 < 0 < 2.801 → True
    assert passes_piplus_chi2pid_cut(np.array([0.0]), np.array([1.0]))[0], \
        "FAIL π+: p=1.0, chi2pid=0.0 should pass (well inside flat window)"

    # p = 1.0, chi2pid = 3.0  →  3.0 > hi_flat=2.801 → False
    assert not passes_piplus_chi2pid_cut(np.array([3.0]), np.array([1.0]))[0], \
        "FAIL π+: p=1.0, chi2pid=3.0 should NOT pass (above upper bound 2.801)"

    # p = 4.0, chi2pid = 0.0
    #   lower = -2.935 (flat)
    #   upper = -0.55 + 3.351 * exp(-(4.0-3.5)/0.55)
    #         = -0.55 + 3.351 * exp(-0.9091)
    #         = -0.55 + 3.351 * 0.4027 ≈ -0.55 + 1.350 ≈ 0.800
    #   -2.935 < 0.0 < 0.800 → True
    assert passes_piplus_chi2pid_cut(np.array([0.0]), np.array([4.0]))[0], \
        "FAIL π+: p=4.0, chi2pid=0.0 should pass (between -2.935 and ≈0.800)"

    # p = 5.0, chi2pid = 1.0
    #   upper = -0.55 + 3.351 * exp(-(5.0-3.5)/0.55)
    #         = -0.55 + 3.351 * exp(-2.7273) ≈ -0.55 + 0.219 ≈ -0.331
    #   chi2pid=1.0 > upper=-0.331 → False (fails upper bound)
    assert not passes_piplus_chi2pid_cut(np.array([1.0]), np.array([5.0]))[0], \
        "FAIL π+: p=5.0, chi2pid=1.0 should NOT pass (above upper bound ≈-0.331)"

    # p = 5.0, chi2pid = -1.0
    #   lower=-2.935, upper≈-0.331
    #   -2.935 < -1.0 < -0.331 → True
    assert passes_piplus_chi2pid_cut(np.array([-1.0]), np.array([5.0]))[0], \
        "FAIL π+: p=5.0, chi2pid=-1.0 should pass (between -2.935 and ≈-0.331)"

    print("All π+ spot-check assertions passed.")

    # -----------------------------------------------------------------------
    # Print cut-window tables
    # -----------------------------------------------------------------------
    p_check = np.array([1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])

    # K+ window
    lo_flat_k = MU_KPLUS - 3.0 * SIGMA_KPLUS
    hi_flat_k = MU_KPLUS + 3.0 * SIGMA_KPLUS
    lo_exp_k = _const_plus_exponential(LOWER_C, lo_flat_k - LOWER_C, LOWER_TAU, LOWER_P0, p_check)
    hi_exp_k = _const_plus_exponential(UPPER_C, hi_flat_k - UPPER_C, UPPER_TAU, UPPER_P0, p_check)
    lower_k = np.where(p_check < 2.0, lo_flat_k, lo_exp_k)
    upper_k = np.where(p_check < 2.5, hi_flat_k, hi_exp_k)

    # π+ window
    lo_flat_pi = MU_PIPLUS - 3.0 * SIGMA_PIPLUS
    hi_flat_pi = MU_PIPLUS + 3.0 * SIGMA_PIPLUS
    hi_exp_pi = _const_plus_exponential(
        PIPLUS_UPPER_C, hi_flat_pi - PIPLUS_UPPER_C,
        PIPLUS_UPPER_TAU, PIPLUS_UPPER_P0, p_check)
    lower_pi = np.full_like(p_check, lo_flat_pi)
    upper_pi = np.where(p_check < 3.5, hi_flat_pi, hi_exp_pi)

    print(f"\nK+ chi2pid cut window:")
    print(f"{'p (GeV/c)':>10}  {'lower bound':>12}  {'upper bound':>12}")
    print("-" * 38)
    for p_val, lo, hi in zip(p_check, lower_k, upper_k):
        print(f"{p_val:>10.2f}  {lo:>12.4f}  {hi:>12.4f}")

    print(f"\nπ+ chi2pid cut window:")
    print(f"{'p (GeV/c)':>10}  {'lower bound':>12}  {'upper bound':>12}")
    print("-" * 38)
    for p_val, lo, hi in zip(p_check, lower_pi, upper_pi):
        print(f"{p_val:>10.2f}  {lo:>12.4f}  {hi:>12.4f}") 


# In[2]:


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
sys.path.append("../scripts/")

from pathlib import Path
import common_functions as au

#importlib.reload(au)


# In[3]:


df_val = pd.read_parquet("/work/clas12/CooperBe/MLStuff/dataset_v03/val.parquet")

#df_val=au.apply_Sidis_Cuts(df_val)
mod, mod_df = au.load_model_and_data("/work/clas12/CooperBe/MLStuff/tier2All/model_v02/model.joblib", df_val)
allPlots=[]


# In[4]:


tStart = 5
tEnd = 35
tBinNum = 5

pStart = 0.5
pEnd = 5
pBinNum = 10

tBinEdges = np.linspace(tStart, tEnd, tBinNum + 1)
pBinEdges = np.linspace(pStart, pEnd, pBinNum + 1)


# In[5]:


thresholds = np.linspace(0.0, 0.95, 100)
results = []

thetaBins = au.makeBins(
    df=mod_df,
    variable="theta",
    binEdges=tBinEdges
)

for i in range(len(thetaBins)):

    pBins = au.makeBins(
        df=thetaBins[i],
        variable="p",
        binEdges=pBinEdges
    )

    for j in range(len(pBins)):

        df_bin = pBins[j]

        if len(df_bin) == 0:
            continue

        mc = df_bin["mc_matching_pid"].to_numpy()
        scores = df_bin["score"].to_numpy()
        is_K = (mc == 321)
        is_pi = (mc == 211)

        if np.sum(is_K) == 0:
            continue

        best_t = np.nan
        best_fom = -np.inf

        for t in thresholds:
            accepted = scores > t
            N_K = np.sum(accepted & is_K)
            N_pi = np.sum(accepted & is_pi)
            denom = np.sqrt(N_K + N_pi)
            if denom == 0:
                continue
            fom = N_K / denom
            if fom > best_fom:
                best_fom = fom
                best_t = t

        results.append({
        "theta_low": tBinEdges[i],
        "theta_high": tBinEdges[i + 1],
        "p_low": pBinEdges[j],
        "p_high": pBinEdges[j + 1],
        "best_threshold": best_t,
        "best_fom": best_fom
        })


# In[6]:


results_df = pd.DataFrame(results)
results_df.to_csv("../figures/full_range/FOM/optimized_thresholdsV5.csv", index=False)


# In[7]:


def apply_bdt_cut(df, threshold_df):

    p = df["p"].to_numpy()
    theta = df["theta"].to_numpy()
    score = df["score"].to_numpy()

    pass_bdt = np.zeros(len(df), dtype=bool)

    for _, row in threshold_df.iterrows():

        if np.isnan(row["best_threshold"]):
            continue

        bin_mask = (
            (p >= row["p_low"]) & (p < row["p_high"]) &
            (theta >= row["theta_low"]) & (theta < row["theta_high"])
        )

        pass_bdt |= bin_mask & (score > row["best_threshold"])

    return pass_bdt


# In[ ]:





# In[ ]:


df_test=pd.read_parquet("/work/clas12/CooperBe/MLStuff/dataset_v03/test.parquet")
df_test=df_test[df_test["mc_matching_pid"]!=-9999]
feature_names=au.get_feature_names("/work/clas12/CooperBe/MLStuff/tier2All/model_v02/model.joblib")
print(feature_names)
df_test = au.apply_model_to_df(mod, df_test, feature_names)


# In[ ]:


mask= apply_bdt_cut(df_test,pd.DataFrame(results))
print("created mask")



# In[ ]:


import os
results_df = pd.DataFrame(results)

# Momentum bin centers
results_df["pCenter"] = (results_df["p_low"] + results_df["p_high"]) / 2

outdir = "../figures/optimized/"
os.makedirs(outdir, exist_ok=True)

for (thetaLow, thetaHigh), group in results_df.groupby(["theta_low", "theta_high"]):

    group = group.sort_values("pCenter")

    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(
        group["pCenter"],
        group["best_fom"],
        marker="o",
        linewidth=0
    )

    # Put ticks at the actual p-bin edges
    ax.set_xticks(pBinEdges)

    ax.set_xlim(pBinEdges[0], pBinEdges[-1])

    ax.set_xlabel("Momentum (GeV/c)")
    ax.set_ylabel(r"FOM")
    ax.set_title(
        f"Optimal FOM\n"
        f"{thetaLow:.1f}$^\\circ$ ≤ θ < {thetaHigh:.1f}$^\\circ$"
    )

    ax.grid(False)

    filename = (
        f"FOM_theta_{thetaLow:.0f}_{thetaHigh:.0f}.png"
    )

    fig.savefig(os.path.join(outdir, filename), dpi=150)
    allPlots.append(fig)
    plt.close(fig)


# In[ ]:


chiComp=[]

results_df = pd.DataFrame(results)

# Momentum bin centers
results_df["pCenter"] = (results_df["p_low"] + results_df["p_high"]) / 2

outdir = "../figures/optimized/"
os.makedirs(outdir, exist_ok=True)

allPlots = []

for (thetaLow, thetaHigh), group in results_df.groupby(["theta_low", "theta_high"]):

    group = group.sort_values("pCenter")

    # -------------------------------------------------
    # BUILD CHI2PID-BASED FOM PER BIN
    # -------------------------------------------------
    chi2_fom_list = []

    for j in range(len(group)):

        p_low = group.iloc[j]["p_low"]
        p_high = group.iloc[j]["p_high"]

        # you must reconstruct bin selection from original df
        df_bin = mod_df[
            (mod_df["theta"] >= thetaLow) &
            (mod_df["theta"] < thetaHigh) &
            (mod_df["p"] >= p_low) &
            (mod_df["p"] < p_high)
        ]

        if len(df_bin) == 0:
            chi2_fom_list.append(np.nan)
            continue

        mc = df_bin["mc_matching_pid"].to_numpy()
        is_K = (mc == 321)
        is_pi = (mc == 211)

        if np.sum(is_K) == 0:
            chi2_fom_list.append(np.nan)
            continue

        # ---------------------------------------------
        # APPLY YOUR CHI2PID CUT FUNCTION
        # ---------------------------------------------
        accepted = passes_kplus_chi2pid_cut(
            df_bin["chi2pid"].to_numpy(),
            df_bin["p"].to_numpy()
        )

        N_K = np.sum(accepted & is_K)
        N_pi = np.sum(accepted & is_pi)

        denom = np.sqrt(N_K + N_pi)

        chi2_fom = (N_K / denom) if denom != 0 else np.nan

        chi2_fom_list.append(chi2_fom)

    # -------------------------------------------------
    # PLOT
    # -------------------------------------------------
    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(
        group["pCenter"],
        group["best_fom"],
        marker="o",
        linestyle="",
        label="MLP optimized"
    )

    ax.plot(
        group["pCenter"],
        chi2_fom_list,
        marker="s",
        linestyle="",
        label="chi2pid cut"
    )

    ax.set_xticks(pBinEdges)
    ax.set_xlim(pBinEdges[0], pBinEdges[-1])

    ax.set_xlabel("Momentum (GeV/c)")
    ax.set_ylabel("FOM")
    ax.set_title(
        f"Optimal FOM Comparison\n"
        f"{thetaLow:.1f} < theta < {thetaHigh:.1f}"
    )

    ax.legend()
    ax.grid(False)

    filename = f"COMPARE_FOM_theta_{thetaLow:.0f}_{thetaHigh:.0f}.png"

    fig.savefig(os.path.join(outdir, filename), dpi=150)
    chiComp.append(fig)
    plt.close(fig)
with PdfPages("/work/clas12/CooperBe/Argonne2026/suli2026_pid/figures/full_range/FOM/" + "BDT_Chi2_FOM_comparison.pdf") as pdf:
    for plot in chiComp:
        pdf.savefig(plot)


# In[ ]:


fomBDTScans=[]
import os
import numpy as np
import matplotlib.pyplot as plt

thresholds = np.linspace(0.0, 0.95, 100)

outdir = "../figures/optimized/FOM_scans_BDT/"
os.makedirs(outdir, exist_ok=True)

for i in range(len(thetaBins)):

    theta_low = tBinEdges[i]
    theta_high = tBinEdges[i + 1]

    pBins = au.makeBins(
        df=thetaBins[i],
        variable="p",
        binEdges=pBinEdges
    )

    for j in range(len(pBins)):

        df_bin = pBins[j]

        if len(df_bin) == 0:
            continue

        mc = df_bin["mc_matching_pid"].to_numpy()
        scores = df_bin["score"].to_numpy()

        is_K = (mc == 321)
        is_pi = (mc == 211)

        if np.sum(is_K) == 0:
            continue

        fom_values = []

        for t in thresholds:
            accepted = scores > t

            N_K = np.sum(accepted & is_K)
            N_pi = np.sum(accepted & is_pi)

            denom = np.sqrt(N_K + N_pi)

            if denom == 0:
                fom = 0.0
            else:
                fom = N_K / denom

            fom_values.append(fom)

        # -------------------------------------------------
        # PLOT PER BIN
        # -------------------------------------------------
        fig, ax = plt.subplots(figsize=(7,5))

        ax.plot(thresholds, fom_values, linewidth=1)

        ax.set_xlabel("MLP Threshold")
        ax.set_ylabel(r"FOM")

        ax.set_title(
            f"FOM vs Threshold\n"
            f"{theta_low:.1f}< theta <{theta_high:.1f}\n"
            f"{pBinEdges[j]:.2f}< p <{pBinEdges[j+1]:.2f}"
        )

        ax.grid(True)

        filename = (
            f"FOMscan_theta{i}_p{j}.png"
        )

        fig.savefig(os.path.join(outdir, filename), dpi=150)
        fomBDTScans.append(fig)
        plt.close(fig)
with PdfPages("/work/clas12/CooperBe/Argonne2026/suli2026_pid/figures/full_range/FOM/" + "FOM_BDT_PerBin.pdf") as pdf:
    for plot in fomBDTScans:
        pdf.savefig(plot)


# In[ ]:


from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np
import matplotlib.pyplot as plt

fomBDTScans2 = []

thresholds = np.linspace(0.0, 0.95, 100)

outdir = "../figures/full_range_FOM/FOM_scans_BDT2/"
os.makedirs(outdir, exist_ok=True)

for i in range(len(thetaBins)):

    theta_low = tBinEdges[i]
    theta_high = tBinEdges[i + 1]

    pBins = au.makeBins(
        df=thetaBins[i],
        variable="p",
        binEdges=pBinEdges
    )

    for j in range(len(pBins)):

        df_bin = pBins[j]

        if len(df_bin) == 0:
            continue

        mc = df_bin["mc_matching_pid"].to_numpy()
        scores = df_bin["score"].to_numpy()

        is_K = (mc == 321)
        is_pi = (mc == 211)

        if np.sum(is_K) == 0:
            continue

        fom_values = []
        numerator_values = []
        denominator_values = []

        for t in thresholds:

            accepted = scores > t

            N_K = np.sum(accepted & is_K)
            N_pi = np.sum(accepted & is_pi)

            denom = np.sqrt(N_K + N_pi)

            if denom == 0:
                fom = 0.0
            else:
                fom = N_K / denom

            fom_values.append(fom)
            numerator_values.append(N_K)
            denominator_values.append(denom)

        # -------------------------------------------------
        # PLOT PER BIN
        # -------------------------------------------------
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(7, 10),
            sharex=True
        )

        # FOM
        axes[0].plot(thresholds, fom_values, linewidth=1)
        axes[0].set_ylabel(r"FOM")
        axes[0].set_title(
            f"FOM vs Threshold\n"
            f"{theta_low:.1f}< theta <{theta_high:.1f}\n"
            f"{pBinEdges[j]:.2f}< p <{pBinEdges[j+1]:.2f}"
        )
        axes[0].grid(True)

        # Numerator
        axes[1].plot(thresholds, numerator_values, linewidth=1)
        axes[1].set_ylabel(r"$N_K$")
        axes[1].grid(True)

        # Denominator
        axes[2].plot(thresholds, denominator_values, linewidth=1)
        axes[2].set_xlabel("MLP Threshold")
        axes[2].set_ylabel(r"$\sqrt{N_K+N_\pi}$")
        axes[2].grid(True)

        plt.tight_layout()

        fomBDTScans2.append(fig)


# -------------------------------------------------
# SAVE MULTI-PAGE PDF
# -------------------------------------------------
pdf_path = (
    "/work/clas12/CooperBe/Argonne2026/suli2026_pid/figures/full_range/FOM/num_denom_FOM_BDT_PerBin.pdf"
)

with PdfPages(pdf_path) as pdf:
    for plot in fomBDTScans2:
        pdf.savefig(plot)
        plt.close(plot)


# In[ ]:


fomChi2Scans = []

thresholds_chi = np.linspace(0, 3.0, 100)

outdir = "../figures/optimized/FOM_scans_chi2/"
os.makedirs(outdir, exist_ok=True)

allPlots = []

for i in range(len(thetaBins)):

    theta_low = tBinEdges[i]
    theta_high = tBinEdges[i + 1]

    pBins = au.makeBins(
        df=thetaBins[i],
        variable="p",
        binEdges=pBinEdges
    )

    for j in range(len(pBins)):

        df_bin = pBins[j]

        if len(df_bin) == 0:
            continue

        mc = df_bin["mc_matching_pid"].to_numpy()
        chi2 = df_bin["chi2pid"].to_numpy()

        is_K = (mc == 321)
        is_pi = (mc == 211)

        if np.sum(is_K) == 0:
            continue

        fom_values = []

        for t in thresholds_chi:

            accepted = np.abs(chi2) < t

            N_K = np.sum(accepted & is_K)
            N_pi = np.sum(accepted & is_pi)

            denom = np.sqrt(N_K + N_pi)

            if denom == 0:
                fom = 0.0
            else:
                fom = N_K / denom

            fom_values.append(fom)

        # -------------------------------------------------
        # PLOT PER BIN
        # -------------------------------------------------
        fig, ax = plt.subplots(figsize=(7,5))

        ax.plot(thresholds_chi, fom_values, linewidth=1)

        ax.set_xlabel(r"$|chi2pid|$ cut")
        ax.set_ylabel(r"FOM")

        ax.set_title(
            f"FOM vs |chi2pid| cut\n"
            f"{theta_low:.1f}< theta <{theta_high:.1f}\n"
            f"{pBinEdges[j]:.2f}< p <{pBinEdges[j+1]:.2f}"
        )

        ax.grid(True)

        fig.savefig(os.path.join(outdir, f"FOMscan_chi2_theta{i}_p{j}.png"), dpi=150)

        allPlots.append(fig)

        plt.close(fig)

# -------------------------------------------------
# PDF OUTPUT
# -------------------------------------------------
with PdfPages(
    "/work/clas12/CooperBe/Argonne2026/suli2026_pid/figures/full_range/FOM/FOM_chi2_PerBin.pdf"
) as pdf:
    for plot in allPlots:
        pdf.savefig(plot)


# In[ ]:


vals = []
errs = []

for i in range(tBinNum):

    theta_low = tBinEdges[i]
    theta_high = tBinEdges[i + 1]

    theta_mask = (
        (df_test["theta"].to_numpy() >= theta_low) &
        (df_test["theta"].to_numpy() < theta_high)
    )

    vals_theta = []
    errs_theta = []

    for j in range(pBinNum):

        p_low = pBinEdges[j]
        p_high = pBinEdges[j + 1]

        p_mask = (
            (df_test["p"].to_numpy() >= p_low) &
            (df_test["p"].to_numpy() < p_high)
        )

        # -------------------------------------------------
        # 1. SUBSET FIRST (IMPORTANT CHANGE)
        # -------------------------------------------------
        bin_mask = theta_mask & p_mask
        df_bin = df_test[bin_mask].copy()

        if len(df_bin) == 0:
            vals_theta.append(0.0)
            errs_theta.append(0.0)
            continue

        # -------------------------------------------------
        # 2. APPLY BDT ONLY ON SUBSET
        # -------------------------------------------------
        bdt_mask_bin = apply_bdt_cut(df_bin, results_df)

        # -------------------------------------------------
        # 3. EFFICIENCY ON SAME SUBSET
        # -------------------------------------------------
        v, e = au.compute_efficiency(df_bin, bdt_mask_bin)

        vals_theta.append(v)
        errs_theta.append(e)

    vals.append(vals_theta)
    errs.append(errs_theta)

vals = np.array(vals)
errs = np.array(errs)


# In[ ]:


import matplotlib.pyplot as plt
import os
import numpy as np

outdir = "../figures"
os.makedirs(outdir, exist_ok=True)

fig, ax = plt.subplots(figsize=(8,6))

p_centers = (pBinEdges[:-1] + pBinEdges[1:]) / 2

for i in range(tBinNum):

    y = np.array(vals[i])
    e = np.array(errs[i])

    mask = np.isfinite(y) & (y != 0)

    ax.errorbar(
        p_centers[mask],
        y[mask],
        yerr=e[mask],
        fmt='o',
        capsize=3,
        label=fr"${tBinEdges[i]:.1f}< theta <{tBinEdges[i+1]:.1f}$"
    )

ax.set_ylim(0, 1.1)
ax.set_xlim(pBinEdges[0], pBinEdges[-1])

edges = np.linspace(pBinEdges[0], pBinEdges[-1], len(pBinEdges))
ax.set_xticks(pBinEdges)

ax.set_xlabel("Momentum (GeV/c)")
ax.set_ylabel("Efficiency")
ax.set_title("Efficiency  (Using Optimized MLP)")

ax.grid(False)
ax.legend()


outpath = "../figures/full_range/FOM/efficiency_vs_p_THRESHOLD.png"
fig.savefig(outpath, dpi=150)
allPlots.append(fig)
plt.show()


# In[ ]:


vals = []
errs = []

for i in range(tBinNum):

    theta_low = tBinEdges[i]
    theta_high = tBinEdges[i + 1]

    theta_mask = (
        (df_test["theta"].to_numpy() >= theta_low) &
        (df_test["theta"].to_numpy() < theta_high)
    )

    vals_theta = []
    errs_theta = []

    for j in range(pBinNum):

        p_low = pBinEdges[j]
        p_high = pBinEdges[j + 1]

        p_mask = (
            (df_test["p"].to_numpy() >= p_low) &
            (df_test["p"].to_numpy() < p_high)
        )

        # -------------------------------------------------
        # 1. SUBSET FIRST (IMPORTANT CHANGE)
        # -------------------------------------------------
        bin_mask = theta_mask & p_mask
        df_bin = df_test[bin_mask].copy()

        if len(df_bin) == 0:
            vals_theta.append(0.0)
            errs_theta.append(0.0)
            continue

        # -------------------------------------------------
        # 2. APPLY BDT ONLY ON SUBSET
        # -------------------------------------------------
        bdt_mask_bin = apply_bdt_cut(df_bin, results_df)
        df_bin=df_bin[bdt_mask_bin]
        # -------------------------------------------------
        # 3. EFFICIENCY ON SAME SUBSET
        # -------------------------------------------------
        v, e = au.compute_contamination(df_bin)

        vals_theta.append(v)
        errs_theta.append(e)

    vals.append(vals_theta)
    errs.append(errs_theta)

vals = np.array(vals)
errs = np.array(errs)


# In[ ]:


import matplotlib.pyplot as plt
import os
import numpy as np

outdir = "../figures"
os.makedirs(outdir, exist_ok=True)

fig, ax = plt.subplots(figsize=(8,6))

p_centers = (pBinEdges[:-1] + pBinEdges[1:]) / 2

for i in range(tBinNum):

    y = np.array(vals[i])
    e = np.array(errs[i])

    mask = np.isfinite(y) & (y != 0)

    ax.errorbar(
        p_centers[mask],
        y[mask],
        yerr=e[mask],
        fmt='o',
        capsize=3,
        label=fr"${tBinEdges[i]:.1f}< theta <{tBinEdges[i+1]:.1f}$"
    )

ax.set_ylim(0, 1.1)
ax.set_xlim(pBinEdges[0], pBinEdges[-1])

edges = np.linspace(pBinEdges[0], pBinEdges[-1], len(pBinEdges))
ax.set_xticks(pBinEdges)

ax.set_xlabel("Momentum (GeV/c)")
ax.set_ylabel("Contamination")
ax.set_title("Contamination (Using Optimized MLP)")
ax.set_ylim(0,0.5)

ax.grid(False)
ax.legend()


outpath = "../figures/full_range/FOM/contamination_vs_p_THRESHOLD.png"
fig.savefig(outpath, dpi=150)
allPlots.append(fig)
plt.show()


# In[ ]:


with PdfPages("/work/clas12/CooperBe/Argonne2026/suli2026_pid/figures/full_range/FOM/" + "FOM_Maximized.pdf") as pdf:
    for plot in allPlots:
        pdf.savefig(plot)


# In[ ]:




