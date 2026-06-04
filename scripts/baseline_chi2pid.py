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
* **Per-run-period flat μ ± Nσ cut** ``passes_per_run_chi2pid_cut``
  (pid_cuts.java lines 362–458, ``charged_hadron_chi2pid_cut``):
  flat-in-momentum windows with run-period-specific μ and σ for all five
  charged-hadron species (π±, K±, p) in both FD and CD, covering RGA Sp19
  (runnum 6616–6783) and MC (runnum 11); falls back to |chi2pid| < 4 for
  other runs.  A helper ``classify_runperiod`` is also provided.

Source file for all Java references::

    processing_classes/src/extended_kinematic_fitters/pid_cuts.java
    lines 263–265  (helper ``const_plus_exponential``)
    lines 362–458  (``charged_hadron_chi2pid_cut``)

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

# ──────────────────────────────────────────────────────────────────────────
# charged_hadron_chi2pid_cut — per-run-period μ ± Nσ windows
# ──────────────────────────────────────────────────────────────────────────
# Source: pid_cuts.charged_hadron_chi2pid_cut, lines 362–458
# Window: μ - N·σ < chi2pid < μ + N·σ
# N = 3.0 for pions/kaons, 3.5 for protons
# Additionally for FD: p ≥ 5.0 GeV → False (hard reject)
# For CD: p < 0.25 GeV → False, plus protons p ≥ 1.5 GeV → False
# Fall-through for runs outside Sp19/MC: |chi2pid| < 4.0

# (run_period, detector) → {pid: (mu, sigma, N)}
PER_RUN_CHI2PID = {
    # RGA Sp19 inbending (runnum 6616–6783)
    ("sp19", "FD"): {
         211:  (-0.05, 1.08, 3.0),
        -211:  (-0.02, 1.08, 3.0),
         321:  ( 0.03, 1.08, 3.0),
        -321:  (-0.14, 1.30, 3.0),
         2212: ( 0.36, 1.36, 3.5),
    },
    ("sp19", "CD"): {
         211:  (-0.18, 1.76, 3.0),
        -211:  (-0.21, 1.68, 3.0),
         321:  ( 0.59, 2.06, 3.0),
        -321:  (-0.20, 1.72, 3.0),
         2212: ( 0.75, 2.15, 3.5),
    },
    # MC (runnum == 11)
    ("mc", "FD"): {
         211:  ( 0.06, 1.23, 3.0),
        -211:  (-0.03, 1.26, 3.0),
         321:  (-0.02, 1.31, 3.0),
        -321:  (-0.04, 1.04, 3.0),
         2212: ( 0.07, 1.35, 3.5),
    },
    ("mc", "CD"): {
         211:  ( 0.09, 1.57, 3.0),
        -211:  (-0.02, 1.52, 3.0),
         321:  (-0.14, 1.75, 3.0),
        -321:  (-0.37, 1.44, 3.0),
         2212: ( 0.45, 1.91, 3.5),
    },
}

# Fallback half-width used when run period is neither sp19 nor mc
CHI2PID_FALLBACK_ABS = 4.0

# ──────────────────────────────────────────────────────────────────────────
# pass_1_pion_generic_chi2pid_cut — Stefan Diehl pass-1 pion-only cut
# ──────────────────────────────────────────────────────────────────────────
# Source: pid_cuts.pass_1_pion_generic_chi2pid_cut, lines 334-360
# Pion-only. No kaon or proton version exists.
# Formula:
#   C = 0.88 (π+) or 0.93 (π−)
#   p < 2.44 GeV/c: |chi2pid| < 3*C  (symmetric, flat)
#   p ≥ 2.44 GeV/c:
#     lower: chi2pid > -3*C
#     upper: chi2pid < C * (a0 + a1*exp(-p/τ1) + a2*exp(-p/τ2))
#       where a0=0.00869, a1=14.98587, τ1=1.18236, a2=1.81751, τ2=4.86394

PASS1_PION_C_PLUS  = 0.88   # π+
PASS1_PION_C_MINUS = 0.93   # π−
PASS1_PION_P_TRANSITION = 2.44   # GeV/c

# Upper-bound sum-of-exponentials coefficients
PASS1_PION_A0   = 0.00869
PASS1_PION_A1   = 14.98587
PASS1_PION_TAU1 = 1.18236
PASS1_PION_A2   = 1.81751
PASS1_PION_TAU2 = 4.86394


def classify_runperiod(runnum):
    """Return 'sp19' for RGA Sp19 (runnum 6616–6783), 'mc' for runnum == 11,
    or None for all other run numbers.

    Parameters
    ----------
    runnum : int or array-like of int

    Returns
    -------
    str or None   (scalar input)
    np.ndarray of object dtype containing 'sp19', 'mc', or None  (array input)
    """
    scalar = np.ndim(runnum) == 0
    runnum = np.atleast_1d(np.asarray(runnum, dtype=int))
    result = np.where(
        (runnum >= 6616) & (runnum <= 6783),
        "sp19",
        np.where(runnum == 11, "mc", None),
    )
    return result[0] if scalar else result


def passes_per_run_chi2pid_cut(
    chi2pid,
    p,
    pid,
    runnum,
    detector,
):
    """
    Python translation of ``pid_cuts.charged_hadron_chi2pid_cut``
    (pid_cuts.java lines 362–458).

    Applies flat μ ± N·σ windows whose constants depend on run period,
    particle species, and detector region.  Falls back to |chi2pid| < 4.0
    for runs not covered by an explicit period.

    Parameters
    ----------
    chi2pid : array-like
        REC::Particle.chi2pid for each track.
    p : array-like
        Track momentum magnitude in GeV/c.
    pid : array-like of int
        PDG particle ID (211, -211, 321, -321, 2212, …).
    runnum : array-like of int
        Run number (e.g. 6616 for RGA Sp19, 11 for MC).
    detector : array-like of str
        Detector region for each track: ``'FD'`` (Forward Detector) or
        ``'CD'`` (Central Detector).

    Returns
    -------
    np.ndarray of bool, same length as inputs.  True = passes cut.

    Notes
    -----
    Hard rejections mirroring the Java source:

    * **FD**: p ≥ 5.0 GeV → False regardless of chi2pid or run period.
    * **CD**: p < 0.25 GeV → False; proton (pid == 2212) with p ≥ 1.5 GeV → False.
    * Any PID not in {211, -211, 321, -321, 2212} always falls through to the
      fallback |chi2pid| < 4.0 window (matching the Java ``return Math.abs(chi2pid) < 4.0``).

    Examples
    --------
    Apply to a pandas DataFrame that contains 'detector' ('FD'/'CD') and
    'runnum' columns::

        df["per_run_pass"] = passes_per_run_chi2pid_cut(
            df["chi2pid"].to_numpy(),
            df["p"].to_numpy(),
            df["pid"].to_numpy(),
            df["runnum"].to_numpy(),
            df["detector"].to_numpy(),
        )
    """
    chi2pid = np.asarray(chi2pid, dtype=float)
    p       = np.asarray(p,       dtype=float)
    pid     = np.asarray(pid,     dtype=int)
    runnum  = np.asarray(runnum,  dtype=int)
    detector = np.asarray(detector, dtype=object)

    n = chi2pid.shape[0]
    result = np.zeros(n, dtype=bool)

    # Classify run periods once
    is_sp19 = (runnum >= 6616) & (runnum <= 6783)
    is_mc   = (runnum == 11)
    is_fd   = (detector == "FD")
    is_cd   = (detector == "CD")

    for i in range(n):
        chi = chi2pid[i]
        pi  = p[i]
        sp  = pid[i]

        if is_fd[i]:
            # Hard momentum rejection
            if pi >= 5.0:
                result[i] = False
                continue

            period_key = None
            if is_sp19[i]:
                period_key = "sp19"
            elif is_mc[i]:
                period_key = "mc"

            if period_key is not None:
                params = PER_RUN_CHI2PID.get((period_key, "FD"), {}).get(sp)
                if params is not None:
                    mu, sigma, N = params
                    result[i] = (mu - N * sigma) < chi < (mu + N * sigma)
                    continue
            # Fall through to fallback
            result[i] = abs(chi) < CHI2PID_FALLBACK_ABS

        elif is_cd[i]:
            # Hard momentum rejections
            if sp == 2212 and pi >= 1.5:
                result[i] = False
                continue
            if pi < 0.25:
                result[i] = False
                continue

            period_key = None
            if is_sp19[i]:
                period_key = "sp19"
            elif is_mc[i]:
                period_key = "mc"

            if period_key is not None:
                params = PER_RUN_CHI2PID.get((period_key, "CD"), {}).get(sp)
                if params is not None:
                    mu, sigma, N = params
                    result[i] = (mu - N * sigma) < chi < (mu + N * sigma)
                    continue
            # Fall through to fallback
            result[i] = abs(chi) < CHI2PID_FALLBACK_ABS

        else:
            # Track is neither FD nor CD — apply fallback
            result[i] = abs(chi) < CHI2PID_FALLBACK_ABS

    return result


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
# Pass-1 pion cut (Stefan Diehl)
# ---------------------------------------------------------------------------

def passes_pass1_pion_chi2pid_cut(chi2pid, p, pid):
    """
    Stefan Diehl's pass-1 chi2pid cut for pions (π+ and π−).

    Parameters
    ----------
    chi2pid : array-like
        REC::Particle.chi2pid values.
    p : array-like
        Track momentum in GeV/c.
    pid : int or array-like
        PDG code.  Only 211 and -211 are supported; other PIDs return False.

    Returns
    -------
    np.ndarray of bool: True = track passes the cut.

    Notes
    -----
    - Symmetric ±3C window below p=2.44 GeV/c
    - Asymmetric above: lower bound stays at -3C, upper bound is a
      sum-of-exponentials in p that decays toward C * 0.00869 as p→∞
    - π+ uses C=0.88, π− uses C=0.93 (slightly looser)
    """
    chi2pid = np.asarray(chi2pid, dtype=float)
    p       = np.asarray(p,       dtype=float)

    # Broadcast pid to array if scalar
    pid_arr = np.asarray(pid)
    if pid_arr.ndim == 0:
        pid_arr = np.full(chi2pid.shape, int(pid_arr), dtype=int)
    else:
        pid_arr = pid_arr.astype(int)

    # Per-track C factor; 0.0 for unsupported PIDs (will always fail cuts)
    C = np.where(pid_arr == 211, PASS1_PION_C_PLUS,
        np.where(pid_arr == -211, PASS1_PION_C_MINUS, 0.0))

    lower = -3.0 * C

    # Upper bound
    #   p <  2.44: +3*C  (flat)
    #   p >= 2.44: C * (a0 + a1*exp(-p/τ1) + a2*exp(-p/τ2))
    upper_high = C * (
        PASS1_PION_A0
        + PASS1_PION_A1 * np.exp(-p / PASS1_PION_TAU1)
        + PASS1_PION_A2 * np.exp(-p / PASS1_PION_TAU2)
    )
    upper = np.where(p < PASS1_PION_P_TRANSITION, 3.0 * C, upper_high)

    # Unsupported PID: C==0 makes both bounds 0, forcing failure; mask
    # explicitly so the semantics are unambiguous.
    valid_pid = (pid_arr == 211) | (pid_arr == -211)

    return valid_pid & (chi2pid > lower) & (chi2pid < upper)


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
    # classify_runperiod spot-checks
    # -----------------------------------------------------------------------
    assert classify_runperiod(6616) == "sp19", "FAIL: 6616 should be sp19"
    assert classify_runperiod(6783) == "sp19", "FAIL: 6783 should be sp19"
    assert classify_runperiod(11)   == "mc",   "FAIL: 11 should be mc"
    assert classify_runperiod(6000) is None,   "FAIL: 6000 should be None"
    assert classify_runperiod(7000) is None,   "FAIL: 7000 should be None"
    # Array form
    periods = classify_runperiod(np.array([6616, 11, 6000]))
    assert list(periods) == ["sp19", "mc", None], "FAIL: array classify_runperiod"
    print("All classify_runperiod assertions passed.")

    # -----------------------------------------------------------------------
    # passes_per_run_chi2pid_cut spot-checks
    # -----------------------------------------------------------------------
    # Helper: single-track wrapper
    def _prcut(chi, momentum, species, run, det):
        return passes_per_run_chi2pid_cut(
            np.array([chi], dtype=float),
            np.array([momentum], dtype=float),
            np.array([species], dtype=int),
            np.array([run], dtype=int),
            np.array([det], dtype=object),
        )[0]

    # ---- FD hard rejection: p >= 5.0 ----
    assert not _prcut(0.0, 5.0, 211, 6700, "FD"), \
        "FAIL: FD p=5.0 should hard-reject regardless of chi2pid"
    assert not _prcut(0.0, 6.0, 2212, 6700, "FD"), \
        "FAIL: FD p=6.0 should hard-reject"

    # ---- FD Sp19 pi+ (pid=211): mu=-0.05, sigma=1.08, N=3.0 ----
    # window: (-0.05 - 3.24, -0.05 + 3.24) = (-3.29, 3.19)
    assert     _prcut( 0.0,  1.0,  211, 6700, "FD"), "FAIL: FD Sp19 pi+, chi2=0 should pass"
    assert     _prcut( 3.0,  1.0,  211, 6700, "FD"), "FAIL: FD Sp19 pi+, chi2=3.0 should pass"
    assert not _prcut( 3.5,  1.0,  211, 6700, "FD"), "FAIL: FD Sp19 pi+, chi2=3.5 should NOT pass"
    assert not _prcut(-3.5,  1.0,  211, 6700, "FD"), "FAIL: FD Sp19 pi+, chi2=-3.5 should NOT pass"

    # ---- FD Sp19 proton (pid=2212): mu=0.36, sigma=1.36, N=3.5 ----
    # window: (0.36 - 4.76, 0.36 + 4.76) = (-4.40, 5.12)
    assert     _prcut( 0.0, 1.0, 2212, 6700, "FD"), "FAIL: FD Sp19 proton, chi2=0 should pass"
    assert not _prcut(-5.0, 1.0, 2212, 6700, "FD"), "FAIL: FD Sp19 proton, chi2=-5 should NOT pass"

    # ---- FD MC pi- (pid=-211): mu=-0.03, sigma=1.26, N=3.0 ----
    # window: (-0.03 - 3.78, -0.03 + 3.78) = (-3.81, 3.75)
    assert     _prcut( 0.0, 1.0, -211, 11, "FD"), "FAIL: FD MC pi-, chi2=0 should pass"
    assert not _prcut( 4.0, 1.0, -211, 11, "FD"), "FAIL: FD MC pi-, chi2=4 should NOT pass"

    # ---- CD hard rejections ----
    # p < 0.25
    assert not _prcut(0.0, 0.20, 211, 6700, "CD"), \
        "FAIL: CD p=0.20 should hard-reject"
    # proton p >= 1.5
    assert not _prcut(0.0, 1.5, 2212, 6700, "CD"), \
        "FAIL: CD proton p=1.5 should hard-reject"
    assert not _prcut(0.0, 2.0, 2212, 6700, "CD"), \
        "FAIL: CD proton p=2.0 should hard-reject"

    # ---- CD Sp19 K- (pid=-321): mu=-0.20, sigma=1.72, N=3.0 ----
    # window: (-0.20 - 5.16, -0.20 + 5.16) = (-5.36, 4.96)
    assert     _prcut( 0.0, 0.5, -321, 6700, "CD"), "FAIL: CD Sp19 K-, chi2=0 should pass"
    assert not _prcut( 5.5, 0.5, -321, 6700, "CD"), "FAIL: CD Sp19 K-, chi2=5.5 should NOT pass"

    # ---- CD MC K+ (pid=321): mu=-0.14, sigma=1.75, N=3.0 ----
    # window: (-0.14 - 5.25, -0.14 + 5.25) = (-5.39, 5.11)
    assert     _prcut( 0.0, 0.5,  321, 11, "CD"), "FAIL: CD MC K+, chi2=0 should pass"
    assert not _prcut( 6.0, 0.5,  321, 11, "CD"), "FAIL: CD MC K+, chi2=6 should NOT pass"

    # ---- Fallback for unknown run period ----
    # runnum=7000 is neither sp19 nor mc → |chi2pid| < 4.0
    assert     _prcut( 3.9, 1.0, 211, 7000, "FD"), "FAIL: fallback chi2=3.9 should pass"
    assert not _prcut( 4.0, 1.0, 211, 7000, "FD"), "FAIL: fallback chi2=4.0 should NOT pass"
    assert not _prcut(-4.0, 1.0, 211, 7000, "FD"), "FAIL: fallback chi2=-4.0 should NOT pass"

    # ---- Neither FD nor CD → fallback ----
    assert     _prcut( 0.0, 1.0, 211, 6700, "OTHER"), "FAIL: non-FD/CD fallback should pass chi2=0"
    assert not _prcut( 4.5, 1.0, 211, 6700, "OTHER"), "FAIL: non-FD/CD fallback chi2=4.5 should NOT pass"

    print("All passes_per_run_chi2pid_cut assertions passed.")

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
    # passes_pass1_pion_chi2pid_cut spot-checks
    # -----------------------------------------------------------------------
    # Helper: single-track wrapper
    def _p1cut(chi, momentum, species):
        return passes_pass1_pion_chi2pid_cut(
            np.array([chi],      dtype=float),
            np.array([momentum], dtype=float),
            np.array([species],  dtype=int),
        )[0]

    # ---- π+ (C=0.88), p < 2.44: flat ±3*0.88 = ±2.64 window ----
    assert     _p1cut( 0.0,  1.0,  211), "FAIL p1 π+: chi2=0 should pass"
    assert     _p1cut( 2.5,  1.0,  211), "FAIL p1 π+: chi2=2.5 should pass (< 2.64)"
    assert not _p1cut( 2.7,  1.0,  211), "FAIL p1 π+: chi2=2.7 should NOT pass (> 2.64)"
    assert     _p1cut(-2.5,  1.0,  211), "FAIL p1 π+: chi2=-2.5 should pass (> -2.64)"
    assert not _p1cut(-2.7,  1.0,  211), "FAIL p1 π+: chi2=-2.7 should NOT pass (< -2.64)"

    # ---- π− (C=0.93), p < 2.44: flat ±3*0.93 = ±2.79 window ----
    assert     _p1cut( 0.0,  1.0, -211), "FAIL p1 π−: chi2=0 should pass"
    assert     _p1cut( 2.7,  1.0, -211), "FAIL p1 π−: chi2=2.7 should pass (< 2.79)"
    assert not _p1cut( 2.9,  1.0, -211), "FAIL p1 π−: chi2=2.9 should NOT pass (> 2.79)"

    # ---- π+ (C=0.88), p >= 2.44: lower = -2.64, upper = exponential ----
    # At p=3.0:
    #   upper = 0.88*(0.00869 + 14.98587*exp(-3/1.18236) + 1.81751*exp(-3/4.86394))
    #         = 0.88*(0.00869 + 14.98587*0.07904 + 1.81751*0.53914)
    #         ≈ 0.88*(0.00869 + 1.18439 + 0.97974)
    #         ≈ 0.88*2.17282 ≈ 1.912
    _upper_p1_p3 = 0.88 * (
        PASS1_PION_A0
        + PASS1_PION_A1 * math.exp(-3.0 / PASS1_PION_TAU1)
        + PASS1_PION_A2 * math.exp(-3.0 / PASS1_PION_TAU2)
    )
    assert     _p1cut(  0.0, 3.0, 211), \
        "FAIL p1 π+: p=3.0, chi2=0 should pass (between -2.64 and ≈1.912)"
    assert     _p1cut( _upper_p1_p3 - 0.01, 3.0, 211), \
        "FAIL p1 π+: p=3.0, just below upper bound should pass"
    assert not _p1cut( _upper_p1_p3 + 0.01, 3.0, 211), \
        "FAIL p1 π+: p=3.0, just above upper bound should NOT pass"
    assert not _p1cut(-2.7,  3.0, 211), \
        "FAIL p1 π+: p=3.0, chi2=-2.7 should NOT pass (< lower bound -2.64)"
    assert     _p1cut(-2.5,  3.0, 211), \
        "FAIL p1 π+: p=3.0, chi2=-2.5 should pass (> lower bound -2.64)"

    # ---- Unsupported PID always returns False ----
    assert not _p1cut(0.0, 1.0,  321), "FAIL p1: K+ pid should return False"
    assert not _p1cut(0.0, 1.0, 2212), "FAIL p1: proton pid should return False"

    # ---- Scalar pid broadcast ----
    result_scalar_pid = passes_pass1_pion_chi2pid_cut(
        np.array([0.0, 0.0]), np.array([1.0, 1.0]), 211
    )
    assert result_scalar_pid.dtype == bool, "FAIL p1: dtype should be bool"
    assert result_scalar_pid[0] and result_scalar_pid[1], \
        "FAIL p1: scalar pid broadcast should pass for chi2=0, p=1"

    print("All passes_pass1_pion_chi2pid_cut assertions passed.")

    # -----------------------------------------------------------------------
    # Print pass-1 pion cut-window table
    # -----------------------------------------------------------------------
    p_check_p1 = np.array([1.0, 2.0, 2.44, 3.0, 4.0, 5.0, 6.0])
    upper_high_p1 = (
        PASS1_PION_A0
        + PASS1_PION_A1 * np.exp(-p_check_p1 / PASS1_PION_TAU1)
        + PASS1_PION_A2 * np.exp(-p_check_p1 / PASS1_PION_TAU2)
    )
    for label, C_val in [("π+ (C=0.88)", PASS1_PION_C_PLUS),
                          ("π− (C=0.93)", PASS1_PION_C_MINUS)]:
        lower_p1 = -3.0 * C_val
        upper_flat_p1 = 3.0 * C_val
        upper_p1 = np.where(
            p_check_p1 < PASS1_PION_P_TRANSITION,
            upper_flat_p1,
            C_val * upper_high_p1,
        )
        print(f"\nPass-1 {label} chi2pid cut window:")
        print(f"{'p (GeV/c)':>10}  {'lower bound':>12}  {'upper bound':>12}")
        print("-" * 38)
        for pv, hi in zip(p_check_p1, upper_p1):
            print(f"{pv:>10.2f}  {lower_p1:>12.4f}  {hi:>12.4f}")

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

        # ---- Plot 4: Pass-1 pion window (π+ and π−) ----
        upper_high_grid = (
            PASS1_PION_A0
            + PASS1_PION_A1 * np.exp(-p_grid / PASS1_PION_TAU1)
            + PASS1_PION_A2 * np.exp(-p_grid / PASS1_PION_TAU2)
        )
        lower_curve_p1_plus  = np.full_like(p_grid, -3.0 * PASS1_PION_C_PLUS)
        lower_curve_p1_minus = np.full_like(p_grid, -3.0 * PASS1_PION_C_MINUS)
        upper_curve_p1_plus  = np.where(
            p_grid < PASS1_PION_P_TRANSITION,
            3.0 * PASS1_PION_C_PLUS,
            PASS1_PION_C_PLUS * upper_high_grid,
        )
        upper_curve_p1_minus = np.where(
            p_grid < PASS1_PION_P_TRANSITION,
            3.0 * PASS1_PION_C_MINUS,
            PASS1_PION_C_MINUS * upper_high_grid,
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.fill_between(p_grid, lower_curve_p1_plus, upper_curve_p1_plus,
                        alpha=0.18, color="steelblue", label="_nolegend_")
        ax.plot(p_grid, lower_curve_p1_plus, color="steelblue", lw=2,
                label="π+ lower (−3C)")
        ax.plot(p_grid, upper_curve_p1_plus, color="steelblue", lw=2, ls="--",
                label="π+ upper")
        ax.fill_between(p_grid, lower_curve_p1_minus, upper_curve_p1_minus,
                        alpha=0.18, color="darkorange", label="_nolegend_")
        ax.plot(p_grid, lower_curve_p1_minus, color="darkorange", lw=2,
                label="π− lower (−3C)")
        ax.plot(p_grid, upper_curve_p1_minus, color="darkorange", lw=2, ls="--",
                label="π− upper")
        ax.axvline(PASS1_PION_P_TRANSITION, color="gray", ls=":", lw=1,
                   label=f"p = {PASS1_PION_P_TRANSITION} GeV/c")
        ax.set_xlabel("p (GeV/c)")
        ax.set_ylabel("chi2pid")
        ax.set_title("Pass-1 pion chi2pid cut (Diehl): π+ (blue) vs π− (orange)\n"
                     "(solid = lower bound, dashed = upper bound)")
        ax.legend(fontsize=8)
        ax.set_xlim(0.5, 5.0)
        ax.set_ylim(-5, 5)
        out_path_p1 = os.path.join(fig_dir, "baseline_chi2pid_window_pass1_pion.png")
        fig.savefig(out_path_p1, dpi=150, bbox_inches="tight")
        print(f"Pass-1 pion plot saved to {out_path_p1}")
        plt.close(fig)

    except ImportError:
        print("\nmatplotlib not available — skipping plots.")
