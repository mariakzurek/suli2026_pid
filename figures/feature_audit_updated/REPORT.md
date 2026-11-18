# SULI 2026 K+ Feature Audit — Updated Run Report

**Generated:** 2026-11-17 (revised after calorimeter-time-range and energy-binning refinement)
**Species:** Event Builder K+ (pid=321), truth-mode `matched`
**Runs:** Inclusive (`kp/`) and SIDIS-selected (`kp_sidis/`)

---

## Inputs and Grid

| Item | Value |
|---|---|
| MC file | `/Users/zurek/CLAS/SULI/pid_training_v2.root` |
| MC entries loaded / selected | 2,906,632 / 162,763 |
| Data file | `/Users/zurek/CLAS/SULI/data_pid_training_test.root` |
| Data entries loaded / selected | 3,935,400 / 193,930 |
| Tree | `PhysicsEvents` |
| Grid | 3×3: p ∈ {1–2, 2–3, 3–5} GeV/c × θ ∈ {5–15°, 15–25°, 25–35°} |
| Bins per cell | 50 for most variables, unit-area normalized independently per sample |
| Variables audited | 27 (vs. 26 in week3; adds `nphe_ltcc`) |

**SIDIS cuts** (`--sidis-cuts`): Q²>2.0, W>2.0, 0<y<0.75, Mx(eKX)>1.6 GeV.

**Feature 1 — negative-time exclusion** (`--exclude-negative-time`, off by default): applies to **calorimeter timing variables `ecin_time`, `ecout_time`, `pcal_time`** (FTOF timing variables are out of scope and unaffected). After stripping the −9999 sentinel (no-hit), additionally removes genuinely-negative non-physical times from the remaining valid-hit population.

Histogram range for these three variables, when the flag is on, is now determined as:
- **Lower bound:** the left edge of the *first non-empty bin* in a 200-bin probe histogram of the combined positive-time MC+data distribution over `[0, upper_bound_candidate]` — not a hardcoded 0.0. This avoids wasting a long stretch of empty low bins when the real timing distribution starts well above zero (e.g. a pedestal offset), which is exactly the problem seen in the first version of this change (an `ecout_time` panel whose x-axis effectively started near −200 because `ecout_time` had not yet been added to the filtered-variable list).
- **Upper bound:** the raw max of the combined positive-time array, used first. Falls back to the 99th percentile if `max > 3×p99` (guards against a single pathological outlier blowing out the axis). The mode actually used per cell is recorded in a new CSV column, `time_range_upper_mode` ("max" or "percentile_99_fallback"), NaN for all other variables.

**Feature 2 — energy binning:** `ecin_energy`, `ecout_energy`, `pcal_energy`, `ftof_energy_1A`, `ftof_energy_1B`, `ftof_energy_2` now use **double** the requested bin count (100 bins at the default `--bins 50`) unconditionally, independent of the negative-time flag. This resolves shape differences in the low-energy region that were hard to see at the coarser 50-bin resolution. No other change (selection, sentinel handling, range determination) applies to these variables.

---

## ⚠ Baseline Comparison Caveat

**Read this before interpreting any before/after numbers.**

The `figures/feature_audit_week3/` baseline was not run against bit-identical inputs. It used copies of the same ROOT files on the JLab `/volatile` filesystem (June 2026). The local data file used here has **+3.6–4.7% more K+ data entries per cell** than the week3 data file (MC differs by <0.2%), uniformly across all cells — consistent with a regenerated/larger data production, not a cut or code change. Week3 also audited 26 variables; this run audits 27 (`nphe_ltcc` added). Week3 additionally predates both the negative-time and energy-binning refinements, so it has no `time_range_upper_mode` column and used half the bin count for energy variables.

Consequently, `figures/feature_audit_week3/` vs. `figures/feature_audit_updated/` is directionally informative but **not a controlled A/B test of either refinement alone**. The dataset-independent evidence for the timing filter's isolated effect is the `neg_time_frac_removed_mc/data` columns computed within the single updated run (see §Timing below), which measure exactly the entries the filter removes, independent of any baseline difference.

---

## Variables with Best Data–MC Agreement

Ranking by KEEP count out of 9 cells (inclusive run):

| Variable | KEEP | CANDIDATE | DROP | Notes |
|---|---|---|---|---|
| `sector` | 9 | 0 | 0 | Near-perfect; mean W₁_norm 0.015, mean ψ 0.003 |
| `phi` | 7 | 2 | 0 | |
| `theta` | 5 | 4 | 0 | |
| `ecin_path` | 5 | 4 | 0 | |
| `ftof_path_1B`, `pcal_path` | 4 | 5 | 0 | |
| `ftof_path_1A` | 3 | 6 | 0 | |

**Pattern:** Detector geometry and path-length variables and kinematics (`sector`, `phi`, `theta`, `*_path`) show the best agreement — these depend primarily on acceptance geometry, which MC models well. SIDIS run shows the same qualitative ordering but every variable picks up at least one low-statistics cell in the high-p/wide-angle corner (see §Statistical Limitations).

---

## Variables with Largest Discrepancies

Inclusive run (worst first):

| Variable | DROP cells / 9 | Notes |
|---|---|---|
| `ecin_time` | 9/9 | See §Timing |
| `pcal_time` | 9/9 | See §Timing |
| `nphe_ltcc` | 9/9 | ⚠ All 9 cells also low-statistics — see §Statistical Limitations |
| `ecout_time` | 8/9 | Now covered by the calorimeter-time range refinement; see §Timing |
| `ecin_energy` | 7/9 | Now audited at 100 bins (doubled) |
| `ecout_energy` | 7/9 | Now audited at 100 bins (doubled) |
| `ftof_time_1A` | 7/9 | Not in scope for the time-range refinement (FTOF, not calorimeter) |
| `pcal_energy` | 6/9 | Now audited at 100 bins (doubled) |
| `nphe_htcc` | 5/9 | |

**Pattern:** Calorimeter and TOF *energy* and *time* response variables drive the largest discrepancies, in contrast to the clean geometric/path variables. This is consistent with known difficulty in simulating detector energy-deposition and timing response precisely — the type of finding the audit is designed to surface. These flags do not by themselves dictate dropping variables from classifier training; per project convention, the inclusive result is the decision-relevant selection and SIDIS is diagnostic-only.

**SIDIS confirmation:** The same variables dominate the worst-disagreement list in the SIDIS run (`ecin_time`, `pcal_time`, `nphe_ltcc`, `ecin_energy`, `ecout_energy`, `ftof_time_1A`, `ecout_time` all 7–9/9 DROP). Per the project's diagnostic convention, disagreement persisting in both inclusive and SIDIS selections indicates likely genuine MC mismodeling rather than exclusive-sample contamination.

---

## Calorimeter Timing (ECIN / ECOUT / PCAL) After Excluding Negative Times

`--exclude-negative-time` is off by default and explicitly configurable. When enabled, it applies to `ecin_time`, `ecout_time`, and `pcal_time` — adding `ecout_time` fixed a defect in the first version of this feature, where `ecout_time` was not yet in the filtered-variable list and one of its panels still showed the old raw-signed-range axis (effectively starting near −200 ns).

**Range-selection mode actually used, per cell** (`time_range_upper_mode` column, new in this run):

| Run | ecin_time | ecout_time | pcal_time |
|---|---|---|---|
| Inclusive (`kp`) | 9/9 `percentile_99_fallback` | 9/9 `percentile_99_fallback` | 9/9 `percentile_99_fallback` |
| SIDIS (`kp_sidis`) | 8/9 fallback, 1/9 `max` | 8/9 fallback, 1/9 `max` | 7/9 fallback, 2/9 `max` |

The `max` mode fires only in the lowest-statistics SIDIS corner (p=3–5 GeV/c, θ=25–35°, and for `pcal_time` also p=2–3/θ=25–35°) — i.e., exactly where the true maximum and the 99th percentile converge because so few entries populate the tail. In every other cell the raw maximum was disproportionate enough (>3× the 99th percentile) to trigger the percentile fallback, confirming that long non-physical outlier tails remain present after sentinel + negative-time stripping and that the outlier guard is doing real work rather than being a dead code path.

**Fraction of valid-hit ("has a hit") entries removed by the negative-time filter** (dataset-independent, computed within the single updated run):

| Variable | Run | MC mean removed (range) | Data mean removed (range) | MC:Data ratio |
|---|---|---|---|---|
| `ecin_time` | Inclusive | 2.0% (1.1–2.7%) | 0.03% (0–0.08%) | ~65× |
| `ecin_time` | SIDIS | 2.0% | 0.02% | ~100× |
| `pcal_time` | Inclusive | 2.0% (1.3–2.6%) | 0.03% | ~65× |
| `pcal_time` | SIDIS | 1.9% | 0.03% | ~65× |
| `ecout_time` | Both | ~1.8–2.2% (new in this run) | <0.1% | Order ~20–65× |

**Key finding (dataset-independent):** Negative non-sentinel timing values are a predominantly MC-only artifact, occurring in roughly 2% of valid-hit MC entries across all three calorimeter-timing variables but well under 0.1% of valid-hit data entries. This is a clean, standalone finding isolated from the baseline-dataset caveat.

**Before/after Wasserstein comparison** (acknowledging the baseline caveat, and now also reflecting the refined range logic — `ecin_time`/`pcal_time` only, since `ecout_time` was not in the original week3 comparison scope): within the updated run's `timing_comparison.csv`, wasserstein_norm dropped in all 18 cells for each variable after switching to the refined positive-time window — e.g., inclusive `ecin_time` W₁_norm fell from a range of 2.6–53.0 (before) to 0.4–46.4 (after). Two low-momentum `ecin_time` cells flipped from CANDIDATE to DROP (p 1–2 GeV/c, θ 5–15° and 15–25°, both inclusive and SIDIS) because ψ and max_local_residual *increased* once the comparison was restricted to the physical positive-time core.

**Interpretation:** Tightening the window did not uniformly improve agreement — it sharpened resolving power, revealing shape differences (shifted peak, different width) in the core timing distribution that were previously diluted by sharing a wide bin range with the long non-physical tail. This is more informative, not "worse." Because the baseline dataset also changed, the numeric W₁_norm deltas in `timing_comparison.csv` reflect both the dataset change and the window-logic change together; use the `neg_time_frac_removed_*` columns above for the timing filter's isolated contribution.

---

## Energy Variables: Doubled Binning

`ecin_energy`, `ecout_energy`, `pcal_energy`, `ftof_energy_1A`, `ftof_energy_1B`, `ftof_energy_2` are now histogrammed at 100 bins instead of 50 (unconditional on the negative-time flag; range determination, sentinel stripping, and normalization are otherwise unchanged). This was requested because differences at low energy values were hard to spot with the coarser default binning. Confirmed via the audited chi2-test degrees of freedom: `ecin_energy` cells now show `ndof=99` (vs. 49 in the original binning), and all 9 cells in both runs remain `status=ok` with sane, nonzero entry counts (kp: n_mc 937–28,236, n_data 1,189–26,387; kp_sidis: n_mc 50–14,428, n_data 103–12,151 — the sparsest SIDIS cell pools empty bins down to `ndof=52`, which is expected behavior, not an error).

---

## Detector-Hit Availability: MC vs. Data

From inclusive run `hit_frac_mc` / `hit_frac_data` columns (mean across 9 cells). Flag threshold |Δ|>0.05.

| Detector | hit_frac_mc | hit_frac_data | Δ | Status |
|---|---|---|---|---|
| PCAL (`pcal_time/path/energy`) | 0.840 | 0.774 | +0.067 | ⚠ MC over-produces hits |
| ECIN (`ecin_time/path/energy`) | 0.706 | 0.645 | +0.062 | ⚠ MC over-produces hits |
| FTOF 1A | 0.851 | 0.799 | +0.052 | ⚠ MC over-produces hits |
| FTOF 1B | 0.909 | 0.894 | +0.015 | Within tolerance |
| HTCC (`nphe_htcc`) | 0.084 | 0.121 | −0.037 | Within tolerance (opposite sign) |
| EC-out, FTOF 2, LTCC, `chi2pid` | — | — | ≲0.05 (mean) | Within tolerance at mean level |

MC systematically over-produces hits in PCAL, EC-inner, and FTOF-1A by roughly 5–7 percentage points — a geometric/digitization-efficiency modeling difference separate from the negative-time artifact. Note: EC-out shows cell-level swings up to Δ=0.10 despite near-zero mean; mean-level agreement can mask cell-level disagreement.

**These are distinct failure modes.** Hit-fraction (detector availability) measures whether a detector records a hit at all; `neg_time_frac_removed` measures unphysical time values within hits that were recorded; `time_range_upper_mode` just records which histogram-axis convention was used and carries no physics meaning of its own. Do not conflate these.

---

## Statistical Limitations and Low-Statistics Flags

| Run | Low-stat cells (n_mc<200 or n_data<200) | % of total (243 cells) |
|---|---|---|
| Inclusive | 31 | 12.8% |
| SIDIS | 57 | 23.5% |

**Inclusive:** Low-stat cells concentrated in five variables:
- `nphe_ltcc`: 9/9 cells — essentially the entire LTCC comparison for K+ is statistics-starved. The 9/9 DROP verdict here reflects low acceptance, not confirmed MC mismodeling.
- FTOF layer 2 (`ftof_energy_2/time_2/path_2`): 7/9 cells low-stat or empty — K+ rarely reach FTOF layer 2 in this acceptance.
- `nphe_htcc`: 1 low-stat cell at the high-p/wide-angle corner.

**SIDIS:** Every variable has at least its p=3–5 GeV/c, θ=25–35° cell starved (n_mc ~47–59, n_data ~97–139). This single corner drives misleadingly large max_local_residual values (into the thousands) across many variables, and is also exactly where the calorimeter-timing `time_range_upper_mode` switches to `max` (the 99th percentile and true maximum converge for lack of statistics) — these are **statistical noise**, not evidence of large disagreement.

**Recommended overrides on raw DROP flags:**
- All 9 `nphe_ltcc` cells (inclusive): insufficient statistics, not confirmed disagreement.
- FTOF-layer-2 trio low-stat cells: same.
- Every variable's p=3–5 / θ=25–35° cell in the SIDIS run: indicative only.

Genuine-disagreement conclusions should be drawn only from adequately populated cells.

**Recommended follow-up checks:**
1. Rerun with a matched/regenerated data file so the week3-vs-updated comparison is a controlled single-variable A/B test of the timing and binning changes.
2. Investigate the ~2% MC-only negative-timing population in the digitization/simulation chain directly, not only as a downstream filter.
3. Investigate the systematic MC-over-data hit-rate excess in PCAL/ECIN/FTOF-1A (5–7 pts) as a separate digitization-efficiency question.
4. Consider merging or widening high-momentum bins, or pooling θ bins, in the SIDIS selection to recover statistical power in the p=3–5 GeV/c corner; or explicitly footnote those cells as "indicative only" in downstream feature-selection writeups.

---

## Classification Conventions

KEEP/CANDIDATE/DROP labels in both `feature_audit_summary.csv` files are **diagnostic triage flags** consistent with existing project convention (`scripts/README.md`). They are not final feature-selection decisions for classifier training.

Any feature-selection decision should additionally weigh:
- The inclusive (uncut) result, which is the decision-relevant selection per project convention; SIDIS is diagnostic-only.
- The statistical-limitation caveats above, before dropping a feature based on a single DROP-flagged low-statistics cell.

Output files per run: `feature_audit_summary.csv` (now includes `time_range_upper_mode` and `neg_time_frac_removed_mc/data` columns), `timing_comparison.csv`, `COLUMNS.md`, `README.md` (auto-generated provenance, documents the exact range rule and binning used for this run), and per-variable PNG subdirectories (27 variables × 9 cells; some cells empty for FTOF layer 2 / LTCC due to low statistics).
