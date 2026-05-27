"""
baseline_chi2pid.py
-------------------
Python reference implementations of the pass-2 K+ and π+ chi2pid cuts used in
CLAS12 RGA analysis, plus a species-agnostic loose cut.

Cuts implemented
~~~~~~~~~~~~~~~~
* **K+** branch of ``charged_hadron_pass2_chi2pid_cut``
  (pid_cuts.java lines 437–447, three-regime, momentum-dependent)
* **π+** branch of ``charged_hadron_pass2_chi2pid_cut``
  (pid_cuts.java lines 427–435, two-regime, momentum-dependent)
* **Loose flat cut** |chi2pid| < 3, species-agnostic (same numeric form for
  K+ and π+); provided for comparison against the species-specific pass-2 cuts

Source file for all Java references::

    processing_classes/src/extended_kinematic_fitters/pid_cuts.java
    lines 263–265  (helper ``const_plus_exponential``)

Note: proton and other species cuts are not implemented here because they are
outside the scope of the SULI 2026 ML-PID project.

Typical usage on a pandas DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    import pandas as pd
    import numpy as np
    from scripts.baseline_chi2pid import passes_kplus_chi2pid_cut
    from scripts.baseline_chi2pid import passes_piplus_chi2pid_cut

    df = pd.read_parquet("training_ntuple.parquet")
    df["baseline_kplus"] = passes_kplus_chi2pid_cut(
        df["chi2pid"].to_numpy(), df["p"].to_numpy()
    )
    df["baseline_piplus"] = passes_piplus_chi2pid_cut(
        df["chi2pid"].to_numpy(), df["p"].to_numpy()
    )
"""

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

    # -----------------------------------------------------------------------
    # Optional visualisation
    # -----------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import os

        script_dir = os.path.dirname(os.path.abspath(__file__))
        fig_dir = os.path.join(script_dir, "..", "figures")
        os.makedirs(fig_dir, exist_ok=True)

        p_grid = np.arange(0.5, 5.05, 0.05)

        # ---- K+ boundary curves ----
        lo_exp_curve_k = _const_plus_exponential(
            LOWER_C, lo_flat_k - LOWER_C, LOWER_TAU, LOWER_P0, p_grid)
        hi_exp_curve_k = _const_plus_exponential(
            UPPER_C, hi_flat_k - UPPER_C, UPPER_TAU, UPPER_P0, p_grid)
        lower_curve_k = np.where(p_grid < 2.0, lo_flat_k, lo_exp_curve_k)
        upper_curve_k = np.where(p_grid < 2.5, hi_flat_k, hi_exp_curve_k)

        # ---- π+ boundary curves ----
        hi_exp_curve_pi = _const_plus_exponential(
            PIPLUS_UPPER_C, hi_flat_pi - PIPLUS_UPPER_C,
            PIPLUS_UPPER_TAU, PIPLUS_UPPER_P0, p_grid)
        lower_curve_pi = np.full_like(p_grid, lo_flat_pi)
        upper_curve_pi = np.where(p_grid < 3.5, hi_flat_pi, hi_exp_curve_pi)

        # ---- Plot 1: K+ window ----
        chi2_grid = np.arange(-5.0, 5.05, 0.1)
        P, C = np.meshgrid(p_grid, chi2_grid)
        mask_k = passes_kplus_chi2pid_cut(C.ravel(), P.ravel()).reshape(P.shape)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pcolormesh(P, C, mask_k.astype(float), cmap="Blues", alpha=0.35,
                      shading="auto")
        ax.plot(p_grid, lower_curve_k, "b-", lw=2, label="lower bound")
        ax.plot(p_grid, upper_curve_k, "r-", lw=2, label="upper bound")
        ax.axhline(lo_flat_k, color="b", ls="--", lw=1,
                   label=f"flat lower = {lo_flat_k:.3f}")
        ax.axhline(hi_flat_k, color="r", ls="--", lw=1,
                   label=f"flat upper = {hi_flat_k:.3f}")
        ax.axvline(2.0, color="gray", ls=":", lw=1)
        ax.axvline(2.5, color="gray", ls=":", lw=1)
        ax.set_xlabel("p (GeV/c)")
        ax.set_ylabel("chi2pid")
        ax.set_title("Pass-2 K+ chi2pid cut window\n"
                     "(blue shaded = passes; dashed = flat reference)")
        ax.legend(fontsize=8)
        ax.set_xlim(0.5, 5.0)
        ax.set_ylim(-5, 5)
        out_path_k = os.path.join(fig_dir, "baseline_chi2pid_window.png")
        fig.savefig(out_path_k, dpi=150, bbox_inches="tight")
        print(f"\nK+ plot saved to {out_path_k}")
        plt.close(fig)

        # ---- Plot 2: π+ window ----
        mask_pi = passes_piplus_chi2pid_cut(C.ravel(), P.ravel()).reshape(P.shape)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pcolormesh(P, C, mask_pi.astype(float), cmap="Oranges", alpha=0.35,
                      shading="auto")
        ax.plot(p_grid, lower_curve_pi, "darkorange", lw=2, ls="-",
                label="lower bound (flat)")
        ax.plot(p_grid, upper_curve_pi, "red", lw=2, ls="-",
                label="upper bound")
        ax.axhline(lo_flat_pi, color="darkorange", ls="--", lw=1,
                   label=f"flat lower = {lo_flat_pi:.3f}")
        ax.axhline(hi_flat_pi, color="red", ls="--", lw=1,
                   label=f"flat upper = {hi_flat_pi:.3f}")
        ax.axhline(PIPLUS_UPPER_C, color="purple", ls=":", lw=1,
                   label=f"upper asymptote = {PIPLUS_UPPER_C:.2f}")
        ax.axvline(3.5, color="gray", ls=":", lw=1, label="p = 3.5 GeV/c")
        ax.set_xlabel("p (GeV/c)")
        ax.set_ylabel("chi2pid")
        ax.set_title("Pass-2 π+ chi2pid cut window\n"
                     "(orange shaded = passes; dashed = flat reference)")
        ax.legend(fontsize=8)
        ax.set_xlim(0.5, 5.0)
        ax.set_ylim(-5, 5)
        out_path_pi = os.path.join(fig_dir, "baseline_chi2pid_window_piplus.png")
        fig.savefig(out_path_pi, dpi=150, bbox_inches="tight")
        print(f"π+ plot saved to {out_path_pi}")
        plt.close(fig)

        # ---- Plot 3: Combined K+ and π+ windows ----
        fig, ax = plt.subplots(figsize=(9, 5))
        # K+ shading and bounds
        ax.fill_between(p_grid, lower_curve_k, upper_curve_k,
                        alpha=0.18, color="steelblue", label="_nolegend_")
        ax.plot(p_grid, lower_curve_k, color="steelblue", lw=2,
                label="K+ lower bound")
        ax.plot(p_grid, upper_curve_k, color="steelblue", lw=2, ls="--",
                label="K+ upper bound")
        # π+ shading and bounds
        ax.fill_between(p_grid, lower_curve_pi, upper_curve_pi,
                        alpha=0.18, color="darkorange", label="_nolegend_")
        ax.plot(p_grid, lower_curve_pi, color="darkorange", lw=2,
                label="π+ lower bound")
        ax.plot(p_grid, upper_curve_pi, color="darkorange", lw=2, ls="--",
                label="π+ upper bound")
        ax.axhline(PIPLUS_UPPER_C, color="purple", ls=":", lw=1,
                   label=f"π+ upper asymptote = {PIPLUS_UPPER_C:.2f}")
        ax.set_xlabel("p (GeV/c)")
        ax.set_ylabel("chi2pid")
        ax.set_title("Pass-2 chi2pid cut windows: K+ (blue) vs π+ (orange)\n"
                     "(solid = lower bound, dashed = upper bound)")
        ax.legend(fontsize=8)
        ax.set_xlim(0.5, 5.0)
        ax.set_ylim(-5, 5)
        out_path_both = os.path.join(fig_dir, "baseline_chi2pid_window_both.png")
        fig.savefig(out_path_both, dpi=150, bbox_inches="tight")
        print(f"Combined plot saved to {out_path_both}")
        plt.close(fig)

    except ImportError:
        print("\nmatplotlib not available — skipping plots.")
