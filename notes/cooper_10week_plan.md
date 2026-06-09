# Cooper — 10-Week SULI Project Plan

**Channel:** `ep → e' p K+ X` SIDIS, RGA pass-2, FD-only.

---

## 1. Project statement

Cooper will build a machine-learning classifier that decides, for each forward-detector charged track that the CLAS12 Event Builder has labeled K+, whether the track really is a K+ or is a pion that the standard PID has misidentified. He will train the classifier on `clasdis` Monte Carlo, where the truth particle identity is known. The primary model is a gradient-boosted decision tree (BDT, LightGBM), trained first. A small MLP is trained as a second model family for comparison. He will compare both against the existing Event-Builder-plus-chi2pid cut. He will measure how much the ML classifier reduces pion contamination in the `ep → e' p K+ X` analysis sample, using one MC-truth benchmark and one data-driven benchmark (e.g., ep->epi+(n) missing mass), with RICH as a cross-check where it has coverage. As a secondary research question, Cooper will also measure the size of the K→π contamination — true kaons the Event Builder labeled as pions — in MC, to assess whether the K+ kinematic phase space can be extended by recovering these tracks. If the measured contamination is significant, the classifier scope will be extended to include kaon recovery from the EB-π+ sample. The final product is a calibrated, reproducible classifier plus a quantified statement of how much better than the baseline it is, in each (p, theta) bin, for the EB-K+ refinement task and (if pursued) for kaon recovery.

## 2. Project goals (measurable)

- **G1.** Produce a trained, calibrated probabilistic K+ PID classifier (BDT as primary; MLP as comparison; final choice by performance) on `clasdis` MC, with calibrated probability output (Platt and/or isotonic).
- **G2.** Benchmark MC-truth efficiency and pion-to-kaon contamination of the classifier in (p, theta) bins, against the standard EB + pass-2 momentum-dependent chi2pid cut as baseline (`scripts/baseline_chi2pid.py:passes_kplus_chi2pid_cut` in the analysis repo).
- **G3.** Measure pion-to-kaon mis-ID rate on RGA pass-2 data in (p, theta) bins using one data-driven method (`ep → e h+ (n)` missing mass, contamination measured per (p, theta) bin), with RICH cross-check where its acceptance covers the bin.
- **G4.** Measure the K→π contamination rate in MC — the fraction of true K+ tracks the Event Builder labels as π+ — as a function of (p, theta). Decide based on the measured rate whether to extend the classifier to recover kaons from the EB-π+ sample.
- **G5.** Quote a single headline number for the `ep → e' p K+ X` analysis: "at fixed kaon efficiency `eps`, ML reduces pion contamination from `C_baseline` to `C_ML` in bin X."
- **G6.** Deliver a working code repository (training pipeline, model file, evaluation scripts), a poster, and a written report.

## 3. Final deliverables

1. **Training MC ntuple pipeline** on `rich_studies`: `processing_mc_pid_training.groovy` plus a converter branch in `convert_txt_to_root.cpp`, plus an ifarm submission script. Reproducible from a clean checkout with a single command.
2. **Analysis-channel ntuple** for `ep → e' p K+ X` via `processing_mc_three_particles.groovy` called with `p1=2212, p2=321`. Extended to add ToF + calorimeter per-hadron features.
3. **Trained classifier:** Python scikit-learn `Pipeline` (preprocessor + estimator + calibrator), saved as `joblib`/`pickle`. Plus loader script and worked example.
4. **Per-(p, theta)-bin contamination numbers from two estimators:**
   - MC truth (clasdis validation set)
   - Missing mass method, with RICH cross-check where its acceptance covers the bin
5. **Headline improvement number** for `ep → e' p K+ X`: ML pion contamination at matched kaon efficiency vs EB+chi2pid baseline.
6. **Poster** (DOE SULI format) and **written report** (~5-10 pages, conference-note style).
7. **Code repo:** clean, with `README.md`, environment file (`environment.yml` or `requirements.txt`), one-command training script, evaluation notebook.

---

## 4. Week-by-week schedule

---

### Week 1 — Onboarding, environment, first MC ntuple

**Theme.** Get to the point where he can run the training-MC pipeline end-to-end on one HIPO file and look at the output in Python.

**Calendar reality.** Cooper arrives Tuesday. Tuesday and Wednesday are JLab lab
orientation (badging, IT provisioning, building access). Project work starts Thursday
at the earliest — and Thursday may be a half day if orientation overflows. Effective
project time in Week 1 is roughly 1.5 to 2 days. The four tasks below will likely
span Thursday through the following Monday or Tuesday (rolling into calendar Week 2).

**Cooper's tasks.**
- **Task 1 (first project day, ~half day).** Read this plan and the onboarding doc
  (`cooper_day1_and_week1.md`) front to back. Meet Maria. Confirm the ifarm path to
  the clasdis MC files (Decision Point D1). Verify ifarm SSH, `/work/` and `/volatile/`
  quotas, `module load clas12/pro`. Confirm git access to both repos.
- **Task 2 (~2 hours, after Task 1).** Run `hipo-utils -dump` on one `clasdis` file.
  **Confirm whether `RICH::Particle` bank is present and non-empty.** 
- **Task 3 (~half day, second project day).** Clone both repos. Run
  `processing_calibration.groovy` on ONE small `clasdis` file end-to-end (build-chain
  smoke test). Then run `processing_mc_pid_training.groovy` on the same file. Open the
  resulting ROOT TTree in Python with `uproot`; print `.shape`, `.head()`,
  `.describe()`, `.value_counts()` on `mc_matching_pid`.
- **Task 4 (~half day, third project day or early Week 2).** Make at minimum 2
  diagnostic plots: beta vs p colored by truth class, and chi2pid distribution per
  truth class. Save as PNGs, commit to the analysis repo. Send Maria a brief Slack
  message with the plots and any questions. This is the Week 1 summary — no
  separate written document required.

**Maria's tasks.**
- Kickoff meeting with Cooper on his first project day (Thursday or Friday). Walk
  through this plan, the channel definition, the deliverables, and the calendar of
  meetings.
- Confirm Cooper has the ifarm paths to (a) the clasdis MC files and (b) the RGA
  pass-2 data skim Cooper will eventually use.
- Review Cooper's plots and Slack/email summary. Send written feedback before the
  following week's first meeting.
- By end of Week 2: read Cooper's 1-page written summary (due end of Week 2, not
  Week 1).

**Done when.**
- One ROOT file from `processing_mc_pid_training.groovy` exists on disk with the full
  57-column feature set.
- Two plots (beta vs p, three truth classes; chi2pid per truth class) committed to
  the analysis repo.
- Brief Slack/email summary with plots sent to Maria.

**Risks / dependencies.**
- ifarm access not yet provisioned by arrival day. Severity H. Mitigation: Maria pushes
  JLab IT before Cooper arrives. If still delayed, Cooper does reading and Python setup
  locally in the interim; full pipeline catches up once access is granted.
- `RICH::Particle` empty on `clasdis`. Severity H. See Risk register.
- Cooper underwater on Python. Severity M. Mitigation: pair-code on Task 3/4 with a postdoc for the first uproot session.

**Fallback / scope-down.** If `processing_mc_pid_training.groovy` does not run cleanly
by the end of Task 3, fall back to using `processing_calibration.groovy` output (which
has most of the features needed) and defer the new Groovy script's debug to Week 2.
Tasks 3 and 4 slide into Week 2 without blocking the overall schedule.

---

### Week 2 — Baseline definition and MC-truth feature audit

**Theme.** Define the baseline (EB + pass-2 momentum-dependent chi2pid cut; see `scripts/baseline_chi2pid.py`) precisely, in code, and produce the first MC-truth contamination measurement against that baseline. Audit the feature distributions for MC-truth pi+ vs K+ vs p.

> **Status (end of Week 2).** Baseline contamination work is **complete**: β vs p plots, χ²pid distributions, contamination matrix, and the nine-panel 2D map are done. The data/MC variable agreement audit (Steps 2c–2d below) was not finished and is **carried over into Week 3** — see Week 3 for the continued task description.

**Cooper's tasks.**
- **End of week: write the 1-page written summary** (moved from Week 1). Write
  `~/CLAS/SULI/notes/week1_summary.md`. Four sections: (1) what is in the ntuple —
  row count, class balance, RICH bank populated or not; (2) what the truth classes look
  like in the plots; (3) anything broken or surprising; (4) at least 3 specific
  questions for Maria. Send to Maria by EOD Friday of Week 2.
- Scale up `processing_mc_pid_training.groovy` to a larger `clasdis` sample (10-20 files, or whatever runs in <2 hours on ifarm interactive). Stage output on `/volatile/`.

**Step 2a — Define PID metrics.**

We use the same efficiency / purity / contamination / mis-ID conventions as the
SIDIS group standard (see `notes/2026-04-21_SidisMeeting_PID-studies.extracted.md`
for definitions and notation). Cooper must apply these consistently in plots,
tables, the report, and the poster. For our K+ analysis:

- **Purity of K+ sample**: `P^K = N(K→K) / Σ_i N(i→K)`
  — fraction of EB-ID'd K+ that are truly K+
- **Contamination of K+ by species i**: `C^{i→K} = N(i→K) / Σ_k N(k→K)`
  — fraction of EB-ID'd K+ that are species i (π+, p, etc.)
  Closure: `P^K + Σ_i C^{i→K} = 1`
- **K+ efficiency**: `ε^K = N(K→K) / Σ_j N(K→j)`
  — fraction of true K+ correctly ID'd as K+
- **K+ mis-ID to species j**: `M^{K→j} = N(K→j) / Σ_k N(K→k)`
  — fraction of true K+ wrongly ID'd as species j (π+, p, etc.)
  Closure: `ε^K + Σ_j M^{K→j} = 1`

Key convention: purity/contamination normalize to DETECTED counts and depend on
the π:K:p production rate (use them when reporting on a sample composition).
Efficiency/mis-ID normalize to TRUE counts and depend only on the detector (use
them when characterizing detector or classifier performance).

The matrix `N(i→j)` is what `processing_mc_pid_training.groovy` produces.
Compute it in Python via `pd.crosstab(df["mc_matching_pid"], df["pid"])`, where
rows are true species and columns are EB-assigned species (`pid` is the EB-assigned PID, column 9 of the ntuple). All four metrics above
are derived from this one table. Keep the raw crosstab in every results CSV so
that the numbers are always reproducible.

**Step 2b — Baseline contamination measurement with formal metrics.**

Write a Python script `compute_baseline.py` that:
  - Reads the training ntuple.
  - Applies the baseline cut: EB pid == 321 AND `passes_kplus_chi2pid_cut(chi2pid, p)` AND `1.0 < p < 5.0` AND FD-only AND fiducial flags pass.
  - For each (p, theta) bin (start with 8 p-bins from 1.0 to 5.0 GeV, 2 theta sub-bins per p-bin — confirm grid with Maria), reports the full N(i→K) count matrix and the derived metrics:
    - `N(K→K)`, `N(π→K)`, `N(p→K)` — raw counts (numerators for purity and contamination)
    - `P^K` (purity), `C^{π→K}` (pion contamination), `C^{p→K}` (proton contamination) — ratios to detected K+ count
    - `ε^K` (kaon efficiency), `M^{K→π}` (kaon mis-ID to π+), `M^{K→p}` (kaon mis-ID to p) — ratios to true K+ count
- Produce the following diagnostic figures for K+:
  - **2D maps in (p, theta):** `N(π→K)`, `N(K→K)`, `N(p→K)` (raw count maps); then `C^{π→K}`, `P^K`, `C^{p→K}` (purity/contamination maps); then `ε^K`, `M^{K→π}`, `M^{K→p}` (efficiency/mis-ID maps). Nine panels in a 3×3 layout.
  - **1D plots vs p at fixed theta:** at two representative theta slices (e.g., ~9° and ~25°), plot `P^K`, `C^{π→K}`, `C^{p→K}` vs p in one panel row, and `ε^K`, `M^{K→π}`, `M^{K→p}` vs p in a second panel row. These are the key diagnostic plots for the report and poster.

**Step 2c — Data/MC variable-agreement audit.** *(Not completed in Week 2 — carried over into Week 3. See Week 3 Task 3a for the full task description and output requirements.)*

The PI's standing requirement: *"if MC and data variables disagree, we cannot use them for ML."* Before any ML training begins, audit every relevant variable in the training ntuple against the RGA pass-2 data distribution for the EB-K+ subsample. Do the comparison in coarse (p, θ) slices — use 9 cells: p in [1, 2], [2, 3], [3, 5] GeV crossed with θ in [5°, 15°], [15°, 25°], [25°, 35°]. In each cell, overlay MC (EB-K+ tracks) and data (EB-K+ tracks) 1D distributions. The per-cell decision is determined by the three-metric drift system (PSI, Wasserstein-norm, max local residual) described in Week 3 Task 3a — not by a single KS threshold.

The per-variable decision is: **KEEP** (use in training), **CANDIDATE** (evaluate further before including), or **DROP** (exclude from training until understood).

**Group 1 — Kinematics** (compare distributions even though these are not ML training features — required for the reweighting decision, Step 2d below):
  - `p` (col 10), `theta` (col 11), `phi` (col 12), `vz` (col 13), `sector` (col 14)

**Group 2 — ML training features** (these MUST agree well; if MC/data disagree, the feature is excluded from training):
  - `beta` (col 16), `chi2pid` (col 17)
  - `ftof_energy_1A` (col 18), `ftof_energy_1B` (col 19)
  - `ftof_time_1A` (col 20), `ftof_time_1B` (col 21)
  - `ftof_path_1A` (col 22), `ftof_path_1B` (col 23)
  - `ecin_energy` (col 24), `ecout_energy` (col 25)
  - `ecin_time` (col 26), `ecout_time` (col 27)
  - `ecin_path` (col 28), `ecout_path` (col 29)

**Group 3 — Candidate additional features** (Connor dropped these; Cooper evaluates whether they agree well enough to include):
  - `pcal_energy` (col 31), `pcal_time` (col 32), `pcal_path` (col 33)
  - `ftof_energy_2` (col 34), `ftof_time_2` (col 35), `ftof_path_2` (col 36)

For each variable, confirm the hit-fraction (fraction of tracks with a non-missing/-9999 value) in MC and in data. Save all overlaid distribution plots to `figures/feature_audit/`. The authoritative machine-readable record is `figures/feature_audit/feature_audit_summary.csv`; the audit narrative is written up in Section 5 of the Week 1-2 report (Task 3c).

**Step 2d — (p, θ) data/MC comparison for the reweighting decision.** *(Not completed in Week 2 — carried over into Week 3. See Week 3 Task 3a for the full task description and output requirements.)*

Even though `p` and `theta` are not ML training features, comparing their 2D distributions between MC and data is critical for Task 3 (Connor's reweighting). Connor's reweighting computes 2D data/MC ratios in (p, θ) bins and reweights MC events to match data; before deciding whether to use that approach, we need to know whether the distributions actually differ and by how much.

Cooper must:
  - Plot the 2D (p, θ) distribution for MC (EB-K+ tracks) and for data (EB-K+ tracks) side by side, using the same binning as the analysis grid.
  - Plot the ratio map data/MC in (p, θ).
  - Write a one-paragraph interpretation: if the ratio map is roughly flat (data/MC ≈ 1 everywhere), no reweighting is needed. If it varies significantly across (p, θ) space, reweighting is warranted and should be implemented in Week 4.
  - Save figures to `figures/feature_audit/ptheta_data_mc_ratio.png`. The one-paragraph interpretation of the ratio map goes into Section 5 of the Week 1-2 report (Task 3c).

- **Measure K→π mis-ID in MC:** for each (p, theta) bin, compute `M^{K→π}` = fraction of true K+ tracks the Event Builder labels as π+. This is the kaon-recovery diagnostic: a large `M^{K→π}` means the EB-π+ sample contains a recoverable population of true kaons. Plot as a 2D map in (p, theta). Record results in `notes/kpi_contamination.md` and flag to Maria by end of week.
- Read scikit-learn user guide §1.10 (decision trees) and §1.11 (ensemble methods). One page of notes per section in `/notes/sklearn_reading.md`.

**Maria's tasks.**
- Mid-week meeting (30 min) to review the baseline contamination numbers (`P^K`, `C^{π→K}`, `C^{p→K}`, `ε^K`, `M^{K→π}`). Sanity-check against prior work and physical intuition.
- Confirm bin edges in (p, theta) Cooper should use. Decide now if a non-default grid is wanted.
- Review the feature audit (Steps 2c and 2d). Decide which flagged features are excluded from ML training and whether the (p, θ) ratio map warrants reweighting in Week 4. Any feature where MC/data shapes disagree goes on the exclusion list immediately — do not defer this decision to Week 4.
- Read Cooper's 1-page written summary by EOD Friday; send written feedback before Week 3.

**Done when.**
- `baseline_contamination_table.csv` with one row per (p, theta) bin, columns {p_low, p_high, theta_low, theta_high, N_KtoK, N_pitoK, N_ptoK, P_K, C_pitoK, C_ptoK, eps_K, M_Ktopi, M_Ktop, and corresponding statistical uncertainties}.
- Nine-panel 2D map figure (N-counts, purity/contamination, efficiency/mis-ID) committed to `/figures/baseline_2d_maps.png`.
- Two-row 1D-vs-p figure at two theta slices committed to `/figures/baseline_1d_vs_p.png`.
- **Baseline plots + contamination matrix complete (done).** β vs p plots, χ²pid distributions, and contamination matrix produced and committed.
- Feature audit (Steps 2c–2d): MC/data distribution overlays, per-feature KEEP / CANDIDATE / DROP table, and 2D (p, θ) ratio map **carried into Week 3** (not a Week 2 completion criterion).
- `M^{K→π}` 2D map in `/figures/kpi_misid_map.png`; results documented in `notes/kpi_contamination.md`.
- `week1_summary.md` written and sent to Maria.

**Risks / dependencies.**
- ntuple statistics insufficient at high p. Severity M. Mitigation: scale up the file list early in the week; do not wait until Friday.
- Column names in the ntuple don't match variable names used in prior notes or literature. Severity L. Mitigation: write a `feature_map.py` that translates once for all downstream code.
- Data/MC feature disagreement affects many variables (Steps 2c/2d). Severity M. Mitigation: run the audit early in the week so flagged features are identified before any ML training begins. The combined three-metric drift decision (PSI, Wasserstein-norm, max local residual — detailed in Week 3 Task 3a) is robust to sample-size effects that make a KS-only threshold unreliable; a reduced feature set still yields a valid classifier.

**Fallback / scope-down.** If the feature audit reveals broken features (e.g. FTOF time always zero), table the broken features and proceed with whichever subset is correct. Do not block on fixing the groovy script in Week 2; come back to it in Week 3 if needed.

---

### Week 3 — Data/MC audit; production-scale ntuples; Week 1-2 report

**Theme.** Finish the data/MC variable agreement audit carried over from Week 2. Scale up ntuple production to the full MC sample and a meaningful data sample using slurm batch jobs. Complete and submit the Week 1-2 written report.

**Cooper's tasks.**

**Task 3a — Finish the data/MC variable-agreement audit (carried over from Week 2).**

This is the continuation of Steps 2c and 2d from Week 2. The goal is the same: establish, for every candidate ML training feature, whether MC and data distributions agree well enough to trust the feature in training, and produce KEEP / CANDIDATE / DROP decisions that gate which variables enter the ML pipeline.

*Procedure.* Run `scripts/compare_mc_data.py` over the variable groups defined in Week 2 (Group 1 kinematics, Group 2 ML features, Group 3 candidate features). The script compares MC (EB-K+ tracks) against data (EB-K+ tracks) independently in each of the 9 coarse (p, θ) cells (p in [1, 2], [2, 3], [3, 5] GeV × θ in [5°, 15°], [15°, 25°], [25°, 35°]). For each variable in each cell it computes three drift metrics, derives a per-cell `drift_decision`, generates the overlaid 1D distribution PNG, and writes all results to `figures/feature_audit/feature_audit_summary.csv`. Do not treat the script as a black box. Cooper must understand what each metric measures and be able to defend the decisions to Maria.

The audit is a batch operation, not interactive work. Run it from the project root using the species driver script:

```bash
python scripts/audit_species.py \
    --mc   /volatile/clas12/<username>/SULI/mc_pid_training_full.root \
    --data /volatile/clas12/<username>/SULI/data_pid_training.root \
    --species kp \
    --vars all_audit kinematics \
    --outdir figures/feature_audit/kp
```

`audit_species.py` is the canonical entry point for the SULI workflow; the underlying engine (`compare_mc_data.py`) is a generic library that `audit_species.py` wraps with the project-specific species cut and truth-match logic. By default (`--truth-mode matched`), the audit covers EB-labeled K⁺ tracks that the MC truth match accepted — this is the correct sample for ML feature drift because the classifier operates on the EB-labeled population including its mis-IDs; the alternative truth modes (`pure`, `off`) are documented in `audit_species.py`'s module docstring for the rare cases where they apply. Cooper auditing other species later (π⁺ for the Δφ classifier, p for the proton channel) reruns the same script with `--species pip` or `--species p` — no new tooling required.

This writes every per-cell PNG, populates the summary CSV, and prints the per-variable summary table to stdout. Iterating on a single variable while debugging is fast: replace `--vars all_audit kinematics` with `--vars beta` (or any single variable name) and re-run in seconds. Do **not** run the full audit inside a Jupyter notebook — the ~25-variable × 9-cell render budget plus the multi-million-row DataFrames pinned in the kernel make this fragile and slow. Use Jupyter only for *inspecting* outputs: load the CSV with pandas, look at flagged variables, open the relevant PNGs. For unattended runs on very large data samples, wrap the same CLI in a slurm submission (same pattern as Task 3b's MC processing).

*What the three metrics measure.* The `wasserstein_normalized` metric is the Wasserstein-1 (earth-mover's) distance between the MC and data distributions, divided by the IQR of the combined sample. It measures how much probability mass you would need to move, in units of the variable's typical spread, to turn MC into data. Dividing by the IQR makes it scale-free: the same threshold applies to sharply-peaked beta and broad theta without any per-variable tuning. A value `< 0.05` means the distributions agree well; `≥ 0.20` means the disagreement is obvious by eye. The `psi_score` metric is the Population Stability Index, a standard feature-drift diagnostic used in credit-risk modeling. It uses quantile binning that auto-adapts to the variable's shape, so the thresholds are not arbitrary — they are the same tiers the industry has validated across a wide range of distributions: `< 0.10` is stable, `0.10–0.25` requires attention, `≥ 0.25` is a major shift. The `max_local_residual` metric is the maximum absolute quantile residual across all quantile bins — it catches localized disagreement that the integrated metrics average away. This is the PID-relevant failure mode: MC and data can agree in the bulk but disagree in a tail that drives mis-identification. A value `< 0.30` means no bin disagrees by more than 30%; `≥ 0.80` means the worst bin is wildly off. Each metric catches a failure mode the others miss. Using all three prevents a single number from hiding a real disagreement, which was the central weakness of any single-statistic threshold rule.

For every variable that lands CANDIDATE or DROP on any metric, Cooper must open the per-cell PNGs and confirm the disagreement is real — not driven by a single empty bin, a sentinel value (-9999) he missed, or a binning artifact. The workflow is numbers-first-then-eyes, not numbers-only. Numerical metrics do not override visual inspection; they direct where to look.

*Per-cell decision.* The `drift_decision` column in `feature_audit_summary.csv` gives the authoritative call for each (variable, p-bin, θ-bin) cell. It is computed automatically by `classify_drift(...)` using the rule: **DROP** if at least two metrics flag DROP; **CANDIDATE** if any one metric flags CANDIDATE or DROP; **KEEP** if all metrics flag KEEP. NaN metrics (e.g., a constant variable in a low-statistics cell) are skipped; a cell where every non-NaN metric is KEEP resolves to KEEP.

*Per-variable aggregation.* After the per-cell decisions are in, aggregate to a single variable-level decision: the variable is **DROP** if any cell is DROP; **CANDIDATE** if no cell is DROP but at least one is CANDIDATE; **KEEP** if every cell is KEEP; **UNKNOWN** if every cell is UNKNOWN (too few entries to compute anything in any cell). The rationale is direct: an ML feature that fails in even one kinematic region leaks data/MC differences into classifier predictions in that region. The per-cell metric maxima that support the variable-level decision are recorded in `figures/feature_audit/feature_audit_summary.csv` (machine-readable) and written up in Section 5 of the Week 1-2 report (human-readable narrative).

*KS and χ² legacy columns.* `ks_distance`, `ks_pvalue`, `chi2`, `chi2_per_ndof`, `chi2_pvalue`, `mean_rel_diff`, `max_abs_rel_diff`, and `ks_flag` are still written to `feature_audit_summary.csv` as sanity checks. At Cooper's sample sizes, KS p-values are essentially meaningless — any real sample will reject the null — so they are not decision drivers. However: if `drift_decision = KEEP` but the KS distance is large, that is a sign the new metrics missed something (or the quantile binning for the local-residual panel was misconfigured) — investigate before trusting the decision. If `drift_decision = DROP` but KS looks fine, trust the drift decision; KS is blind to localized tail disagreements and shape changes that don't shift the bulk.

*Special handling for Group 1 kinematics.* `p`, `theta`, `phi`, `vz`, and `sector` are not ML training features, so their `drift_decision` does not gate inclusion. But the drift metrics on `p` and `theta` directly inform the Step 2d reweighting decision: if both are KEEP, no reweighting is likely needed; if either is CANDIDATE or DROP, the 2D (p, θ) ratio map will show structure and reweighting should be implemented in Week 4. Run the three-metric audit on Group 1 first, record the per-variable decisions, then proceed to the 2D ratio map (Step 2d, continuing below) with those decisions already in hand.

*Outputs:*
  - `figures/feature_audit/<var>/<var>_pXX-XX_thetaXX-XX.png` — per-cell overlays (auto-generated).
  - `figures/feature_audit/feature_audit_summary.csv` — per-cell metrics including `drift_decision` (auto-generated). Cooper appends a `decision_notes` column after running the audit: one free-text entry per row capturing any visual-cross-check observation that overrides the auto-decision or that Maria should see (e.g. "disagreement driven by single empty bin," "tail bump matches a known FTOF artifact," "hit-fraction collapse below p=2 GeV"). Empty string when the auto-decision stands without commentary.
  - `figures/feature_audit/README.md` — short operator note (≤ 20 lines): what's in the directory, what each CSV column means, pointer to the report section where the audit is written up. This is reference material for anyone walking up to the directory cold, not narrative.
  - `figures/feature_audit/ptheta_data_mc_ratio.png` — 2D ratio map (path unchanged).

**Task 3b — Production-scale ntuples.**

Cooper has so far run the processing scripts on one HIPO file each. For the variable audit to be statistically meaningful and for Week 4+ training to work, much larger samples are needed. These runs must go through slurm batch submission — **do not run interactively** at this scale.

*MC sample:* process the **full** RICH-on `clasdis` sample at `/work/clas12/zurek/SULI/clasdis_rich_on/`. Passing `n_files=0` to the script processes all available files in that directory — this is the same directory Cooper has been using for one-file tests. Submit as a slurm array job. Add the submission script to `suli2026_pid/slurm/` (create the directory if it does not exist yet). Document the script in the repo README.

*Data sample:* process at least 10–20 RGA Fa18 inbending pass-2 HIPO files. Input directory: `/cache/clas12/rg-a/production/recon/fall2018/torus-1/pass2/main/train/nSidis/`. The full sample is much larger, but 10–20 files are sufficient for the variable-audit statistics. Submit via slurm (same pattern as MC).

*Outputs:*
  - MC ntuple from full RICH-on `clasdis` sample written to `/volatile/clas12/<username>/SULI/`. Record total file count, event count, and output size.
  - Data ntuple from ≥ 10 Fa18 inbending files written to `/volatile/clas12/<username>/SULI/`. Record file count and event count.

**Task 3c — Finish the Week 1-2 report.**

The LaTeX template in `suli2026_pid/report/` still has placeholder text and TODO figure captions. Maria has been leaving comments on the Overleaf project. This task closes those out and gets the report to a clean draft.

Concretely:
  - **Review every Overleaf comment Maria has left and resolve each one.** Either address the question or concern in the text, or reply to the comment with reasoning if you disagree. No comment should be left unacknowledged.
  - **Read the report end to end** — not just edit sections in isolation. Make sure the narrative flows: introduction → method → results → discussion.
  - Fill in remaining placeholders: abstract, introduction prose, section 4 contamination discussion, summary section.
  - **Section 5 — audit findings.** Write up the data/MC audit results based on `figures/feature_audit/feature_audit_summary.csv` and the `decision_notes` column Cooper added. The section must contain: (1) a typeset table of the variable-level KEEP/CANDIDATE/DROP decisions with the worst per-cell PSI, W/IQR, and max local residual for each variable; (2) per-variable prose for every CANDIDATE and DROP variable explaining what the disagreement looks like and what was visually confirmed; (3) the one-paragraph reweighting recommendation based on the (p, θ) ratio map — if the ratio map is roughly flat, no reweighting is needed; if it varies significantly across (p, θ) space, reweighting is warranted and will be implemented in Week 4. Use the per-cell PNGs as supporting figures where they sharpen the narrative — do not include all of them.
  - Replace every TODO figure caption with a real caption describing what is actually shown (axes, units, color coding, key takeaway).
  - Copy produced PNGs into `report/figures/`.
  - Compile locally with `pdflatex` (or via Overleaf) and confirm the PDF builds without errors or unresolved references.

- Submit a slurm job array to process the **full** `clasdis` sample through `processing_mc_pid_training.groovy` (see Task 3b above). Run `processing_mc_three_particles.groovy` with `p1=2212, p2=321` on the same files to produce the analysis-channel ntuple for `ep → e' p K+ X`.
- Define the train / validation / test split. Split at the file level (not event level), randomly, with a fixed seed. Recommended ratio: ~70% train, ~15% validation, ~15% test. Record the file lists in the repo as `train_files.txt`, `val_files.txt`, `test_files.txt`.
- Compute baseline contamination on the **test set only** to get the final baseline numbers.
- Read scikit-learn user guide §3.1 (cross-validation), §3.3 (metrics), §1.16 (calibration). One page of notes per section.

**Maria's tasks.**
- Review the feature audit results: `figures/feature_audit/feature_audit_summary.csv` (with `decision_notes` filled in) and Section 5 of the Week 1-2 report. Confirm the KEEP / CANDIDATE / DROP decisions. Decide whether the (p, θ) ratio map warrants reweighting in Week 4.
- Approve the train/val/test split.
- 1-hour meeting: walk through the baseline contamination plot. Decide whether the (p, theta) binning needs revision before Week 4.
- Check in with Cooper on Python/ML progress. If reading is not getting done, adjust Week 4 expectations.
- **Make decision on hyperon-tagged kaon-truth channel** (see Decision Points). This decision blocks Week 8 planning.
- **Sign off on the Week 1-2 report draft** after all Overleaf comments are resolved and the PDF compiles clean.
- If Cooper has not used slurm arrays before, help set up the submission script at the start of the week — do not let this block ntuple production past Tuesday.

**Done when.**
- `figures/feature_audit/feature_audit_summary.csv` present with `drift_decision` column populated for all variables and cells, and Cooper's `decision_notes` column filled in (empty strings allowed where the auto-decision stands without commentary).
- `figures/feature_audit/` populated with per-variable per-cell overlay PNGs and `ptheta_data_mc_ratio.png`.
- `figures/feature_audit/README.md` written (≤ 20 lines: directory contents, CSV column meanings, pointer to Section 5 of the Week 1-2 report).
- Reweighting recommendation (one paragraph) written in Section 5 of the Week 1-2 report.
- MC ntuple from full RICH-on `clasdis` sample exists on `/volatile/clas12/<username>/SULI/`; event count and output size documented.
- Data ntuple from ≥ 10 Fa18 inbending HIPO files exists on `/volatile/clas12/<username>/SULI/`; file count and event count documented.
- Slurm submission script committed to `suli2026_pid/slurm/`.
- Analysis-channel ntuple for `ep → e' p K+ X` produced and validated (e, p, K+ kinematic distributions look sane).
- Train/val/test split locked. File lists in repo.
- Baseline contamination numbers on the test set documented in `notes/baseline_final.md`.
- All Overleaf comments resolved or replied to. Report PDF compiles without errors. PI signs off on the draft.

**Risks / dependencies.**
- **Full-sample MC production takes longer than expected.** The RICH-on `clasdis` sample is ~318 files at roughly 10 min each — approximately 50 CPU-hours total. This must be parallelized via a slurm array; a single interactive run is not feasible. Mitigation: submit the array early Monday. Maria helps set up the submission script if Cooper has not used slurm arrays before. If the queue is slow, run a 10-job test array first to validate the script, then submit the full array.
- **Data sample is not cached or access is slow.** Files in `/cache/` may need to be staged from tape. Severity M. Mitigation: check file availability with `ls -lh` on the target directory at the start of the week; if files need staging, submit the staging request immediately and work on the audit and report while waiting.
- slurm jobs fail or queue indefinitely. Severity M. Mitigation: run a small test array (5–10 jobs) before submitting the full batch. Check output logs before assuming all jobs completed.
- **Three drift metrics give conflicting signals on edge-case variables.** Severity L. This is expected on variables near the CANDIDATE/DROP boundary — one metric flags DROP while the two others say KEEP. The combined `drift_decision` rule handles the common cases, but edge cases require visual cross-check of the per-cell PNGs to determine whether the flagging metric is responding to a real shape difference or a low-statistics artifact. Visual inspection resolves the ambiguity; do not resolve it by ignoring the metric.
- Truth-matching efficiency on `clasdis` not what Connor reports (he used |Δφ| < 6°, |Δθ| < 2°). Severity L. Mitigation: log the match efficiency as a sanity number.
- Report polish takes longer than expected if many Overleaf comments require substantive revisions. Severity M. Mitigation: address comments in priority order — physics content first, prose last. A clean compile with all comments resolved is the minimum bar.

**Fallback / scope-down.** If full-sample MC processing does not complete by Friday, proceed to Week 4 with whatever is on disk and re-run the remainder in the background. Do not block Week 4 on ntuple completion. If the report draft is not signed off by Friday, carry the final PI sign-off into the first day of Week 4.

---

### Week 4 — First model: BDT (LightGBM); comparison protocol; probability calibration

**Theme.** Train a gradient-boosted decision tree on the detector features. Compare to baseline on the test set. Establish the comparison protocol. Calibrate probabilities.

**Cooper's tasks.**
- Install LightGBM (`conda install -c conda-forge lightgbm`). Train a first BDT: `lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)`. Fit on the train set. Predict probabilities on the test set. LightGBM handles missing values (-9999 or NaN) natively — no imputation needed for the first pass.
- Apply (p, theta) reweighting: 15x15 grid in (p, theta), reweight MC distribution to data EB-K+ distribution. Apply as `sample_weight` to the BDT fit. **Sanity check**: data here means the EB-K+ distribution in the actual RGA pass-2 data subset Cooper will eventually evaluate on. If Cooper has not yet pulled the data distribution, do an MC-only fit first and add the reweighting in Week 5 — flag the limitation.
- For each (p, theta) bin in the analysis grid, sweep the BDT score threshold from 0 to 1 and compute (eff, contam) for each threshold. Plot the (eff, contam) ROC-like curve in one example bin (low p, low theta) and one challenging bin (high p, mid theta). Overlay the baseline point (EB + pass-2 momentum-dependent chi2pid cut gives one (eff, contam) point per bin).
- Establish the comparison protocol for "ML vs baseline":
  - Option A: at matched efficiency (BDT threshold set to give same kaon efficiency as baseline), what is BDT contamination? Quote `(C_baseline - C_BDT) / C_baseline` per bin.
  - Option B: at matched contamination, what is BDT efficiency? Quote `(eff_BDT - eff_baseline) / eff_baseline` per bin.
  - **Use both.** Report both numbers per bin.
- **Probability calibration.** Apply `CalibratedClassifierCV` with `method='sigmoid'` (Platt) to the BDT, using a held-out calibration set (carved from train; do NOT use the test set). Produce a reliability diagram before and after calibration. LightGBM probabilities are often already well-calibrated; this is a quick check, not a week of work.
- Produce the canonical 1-page summary plot: contamination vs (p, theta) for {baseline, BDT at matched eff}.

**Maria's tasks.**
- Sit with Cooper at least once to walk through the LightGBM training code. This is the first time Cooper writes an ML pipeline; do not let mistakes calcify.
- Review the comparison protocol. Confirm Option A + Option B is the right framing.
- Decide: do we attempt per-(p, theta)-bin threshold optimization (FOM = `N_K / sqrt(N_K + N_pi)`) in Week 4, or defer to Week 5? Recommendation: defer. Use a single global BDT threshold for the Week 4 comparison; optimize per-bin in Week 5.

**Done when.**
- `bdt_v1.pkl` saved (BDT + calibrator).
- Test-set contamination plot for BDT vs baseline, both Option A and Option B numbers, in `/figures/bdt_vs_baseline.png` and a CSV table.
- Reliability diagram for BDT before and after calibration in `/figures/bdt_reliability.png`.
- 2-page written internal report: "what the BDT learned, where it helps, where it doesn't."

**Risks / dependencies.**
- BDT performance disappointing or anomalous (e.g. perfectly tracks chi2pid). Severity M. Mitigation: this is data and reality. If BDT only learns chi2pid, that is a real finding. Diagnose why; check feature importances immediately.
- Reweighting depends on RGA data sample Cooper does not yet have on hand. Severity M. Mitigation: do the unweighted fit first; add reweighting in Week 5.

**Fallback / scope-down.** If the BDT training hits a wall (e.g. memory at full sample size), train on a subset (5M tracks is plenty for a first pass) and document. Full-statistics training can be re-run in Week 6.

---

### Week 5 — Per-bin BDT threshold optimization; apply to SIDIS channel; first headline number

**Theme.** Complete per-bin threshold optimization. Apply the BDT to the analysis-channel ntuple. Quote the first headline improvement number.

**Cooper's tasks.**
- Implement per-(p, theta)-bin threshold optimization using FOM = `N_K / sqrt(N_K + N_pi)`. Sweep BDT score threshold 0 to 0.95 per bin; pick the maximum. Use the validation set for threshold selection — do NOT use the test set. Tabulate the thresholds for each bin.
- Apply the optimized BDT to the **analysis-channel ntuple** for `ep → e' p K+ X`. For each event, run the BDT on the K+ candidate, accept the event if BDT score > the bin-optimal threshold. Compare to the baseline-cut accepted event count and pion-truth contamination.
- Produce the first headline number: "for the `ep → e' p K+ X` MC sample, at fixed kaon efficiency `eps_0`, baseline gives contamination `C_baseline = X%`, BDT gives `C_BDT = Y%`, improvement = `(X-Y)/X = Z%`."
- Plot M_X(e' p K+) distributions for {all events, baseline-cut events, BDT-cut events}, color-coded by truth class. This shows visually how the cut cleans up the missing-mass spectrum.
- Write a 1-page section of the eventual report: "MC-truth comparison of BDT to EB+chi2pid baseline on the `ep → e' p K+ X` channel." This is a writeup deposit, not the final report.

**Maria's tasks.**
- Review the headline number. Sanity-check against intuition: does the improvement scale with momentum the way you'd expect?
- **Mid-project review with Cooper.** 1 hour. Are we on track? Is the scope right? Adjust Week 6-10 ambition.

**Done when.**
- Per-bin BDT thresholds in a CSV.
- Headline improvement number quoted, with statistical uncertainty.
- M_X(e' p K+) plot in `/figures/`.
- Mid-project writeup section in `/report/section_bdt_truth.md`.

**Risks / dependencies.**
- The improvement is small or negative. Severity M. Mitigation: if BDT doesn't beat baseline, that is information. The channel context (SIDIS, different kinematic distribution) means results may differ from prior work. Document honestly. Then diagnose: are the features informative? Is the reweighting correct?
- Per-bin optimization overfits because some bins have low MC statistics. Severity M. Mitigation: use the validation set for threshold selection, not the test set. Apply on the test set only once.

**Fallback / scope-down.** If per-bin optimization is taking more than 2 days, use a single global threshold from the validation set. The improvement number changes but the pipeline still works.

---

### Week 6 — MLP as second model family; BDT vs MLP comparison; final model decision

**Theme.** Train the MLP as a second model family. Compare BDT vs MLP head-to-head. Decide which is the final model.

**Cooper's tasks.**
- **MLP.** Use `sklearn.neural_network.MLPClassifier` or a small PyTorch net. Start small: 2 hidden layers of 64 units, ReLU, Adam, early stopping. Apply `StandardScaler` to features first (mandatory for MLP; not needed for BDT). Impute -9999 with per-feature median before scaling, and add binary missing-indicator features (`feature_X_present`) for PCAL and FTOF layer 2 — missingness is physically informative. Save as `mlp_v1.pkl`. Apply `CalibratedClassifierCV` (Platt) and produce a reliability diagram.
- **Comparison.** Apply the same per-bin threshold optimization to the MLP. Compare contamination vs (p, theta) for {baseline, BDT, MLP}. Compute Brier score, log-loss, and AUC for each on the test set. Tabulate.
- **Calibration check for BDT.** Also check isotonic calibration for the BDT (in addition to Platt from Week 4). Produce both reliability diagrams. The better one goes into `bdt_final.pkl`.
- **Model decision.** Pick whichever has the best contamination at fixed efficiency on the analysis-channel test set, with a calibration that passes the reliability-diagram eyeball test.

**Maria's tasks.**
- Review calibration plots. Decide whether isotonic or Platt is the production choice.
- Make the final model family decision (BDT vs MLP). This drives all downstream work.

**Done when.**
- MLP trained, calibrated, saved as `mlp_v1.pkl`.
- Comparison table: baseline | BDT | MLP, columns = {Brier, log-loss, AUC, contam at matched eff per bin}.
- Reliability diagrams for both models, calibrated and uncalibrated.
- Final model family decided and documented.

**Risks / dependencies.**
- MLP underperforms because of missing features (-9999 propagating through). Severity M. Mitigation: the missing-indicator approach described above handles this explicitly.
- MLP training is slow on ifarm CPU. Severity M. Mitigation: keep the architecture small (2 × 64). If still too slow, use a GPU node (`--partition=gpu`) or fall back to sklearn's `MLPClassifier` which is faster for smaller networks.

**Fallback / scope-down.** Drop MLP entirely. BDT alone is the final model. The story is still complete — one model family, well-tuned and calibrated, is better than two models done poorly.

---

### Week 7 — Model refinement; hyperparameter tuning; feature ablation

**Theme.** Take the chosen model (from Week 6) and squeeze the last real performance out of it. Understand which features are actually driving the gain.

**Cooper's tasks.**
- **Hyperparameter tuning.** Run a grid or random search on the chosen model's top 3–4 hyperparameters (for LightGBM: `n_estimators`, `learning_rate`, `max_depth`, `min_child_samples`; for MLP: learning rate, hidden layer sizes, dropout). Use the validation set for selection; apply the winner to the test set only once.
- **Feature ablation.** Drop one feature group at a time (e.g., remove all FTOF variables; remove ECAL; remove HTCC nphe; remove chi2pid). Retrain and compare AUC and per-bin contamination. Quantify: "removing X degrades contamination by Y% in the worst bin." This tells you which features are load-bearing.
- **Feature importance / SHAP.** For LightGBM, extract the built-in feature importance (gain). Optionally compute SHAP values for a random subset of test tracks. Plot the top-10 features by importance. This section should be presentable in the report and poster.
- Final model choice confirmed. Save as `model_final.pkl`. Refactor the training pipeline as a single command: `python train.py --config config.yaml --output model_final.pkl`. README explains how to reproduce from scratch.

**Maria's tasks.**
- Sign off on final model choice and hyperparameter settings.
- Review the feature ablation results. Flag any surprising finding (e.g., chi2pid alone explains 90% of the gain — or doesn't).

**Done when.**
- `model_final.pkl` committed. README updated.
- Training pipeline reproducible from one command.
- Feature importance plot in `/figures/feature_importance.png`.
- Feature ablation table: per-group AUC and contamination change, in `/notes/feature_ablation.md`.

**Risks / dependencies.**
- Hyperparameter tuning takes all week and yields negligible gain. Severity L. Mitigation: cap the tuning at 2 days. If the default hyperparameters from Week 4 are already near-optimal (common with LightGBM), document that and move on to ablation.
- Feature ablation reveals that chi2pid dominates and nothing else matters. Severity M (scientifically interesting, not a blocker). Mitigation: report honestly. The ML still provides a calibrated probability, which the hard cut does not.

**Fallback / scope-down.** If ablation takes longer than projected, reduce it to 3 feature groups (FTOF, ECAL, everything else). The key finding is whether chi2pid alone is sufficient — answer that question and stop if pressed for time.

---

### Week 8 — Exclusive missing-mass method: full implementation, MC validation, first contamination numbers

**Theme.** Implement the exclusive missing-mass data-driven method end-to-end. Validate it against MC truth. Produce the first data-side contamination table.

**Cooper's tasks.**
- Pull the RGA pass-2 data ntuples for the neutron-tagged sample (`ep → e h+ X`, neutron-tagged). Paths confirmed with Maria; if not yet on hand, use MC-only missing-mass method as described in the fallback below.
- **Full missing-mass method implementation.** Produce the M_X(e h+) spectrum for h+ = (EB-pi+, EB-K+, EB-p) separately, in all (p, theta) bins. Identify the neutron peak at M_X ≈ 0.94 GeV. Sideband-subtract (signal window: 0.85–1.05 GeV; sidebands: 0.55–0.75 and 1.15–1.35 GeV). Count events under the neutron peak with EB-K+ ID; divide by total under-peak count = pi-to-K mis-ID rate per bin. Tabulate `C_miss_mass(p, theta)` for all bins.
- **MC validation of the missing-mass method.** Apply the identical procedure to MC. Compare `C_miss_mass^MC(p, theta)` to `C_MC_truth(p, theta)`. Plot the ratio per bin. This is the method closure test: if the missing-mass method recovers MC truth on MC, it is trustworthy on data.
- **First contamination numbers.** Compare `C_miss_mass(p, theta)` (data) to `C_MC_truth(p, theta)` (MC). Plot side by side with error bars. If they agree within stated systematics, that's the primary result. If not, investigate and document honestly.
- Begin drafting the report section "Data-driven validation" — describe the exclusive missing-mass method, present the closure test, present the data results.

**Maria's tasks.**
- Review missing-mass method closure-test plot. Sanity check: neutron peak should be well-resolved; closure ratio should be ≈1 within errors.
- Decide whether to consult with the method's originators for sanity check on method details.
- Mid-week meeting on the first contamination numbers.

**Done when.**
- Missing-mass method contamination table complete for all (p, theta) bins, in CSV (`C_miss_mass_data.csv`).
- MC closure plot for the missing-mass method in `/figures/miss_mass_closure.png`.
- Missing-mass vs MC-truth comparison plot in `/figures/miss_mass_vs_mc_truth.png`.

**Risks / dependencies.**
- Data ntuples not yet produced or not on Cooper's path. Severity H if not resolved. Mitigation: use MC-only missing-mass method (apply the method to MC; check closure against MC truth). Validates the method even before data is ready.
- Missing-mass method and MC-truth disagree by >2σ in many bins. Severity M-H. Mitigation: this is real physics. Investigate: does the data have a contamination the MC doesn't capture? Is the neutron-peak sideband subtraction biased? Document the discrepancy honestly; do not paper over it.
- The pi-to-K mis-ID rate is dominated by combinatoric background, not real mis-ID. Severity M. Mitigation: the sideband subtraction handles this; verify the sideband regions are genuinely background-only by checking the M_X shape.

**Fallback / scope-down.** If data ntuples are unavailable, MC-only missing-mass method (closure test only) is the deliverable for the week. The data result moves to Week 9.

---

### Week 9 — ML applied to neutron-tagged sample on data; ML-vs-baseline contamination plot; report and poster drafts

**Theme.** Apply the final ML model to the data-side neutron-tagged sample. Produce the definitive ML-vs-baseline comparison. Draft the full report and poster.

**Cooper's tasks.**
- **Apply final ML classifier to the neutron-tagged sample on data.** For each event in the neutron-tagged sample, run `model_final` on the K+ candidate. Apply the per-(p, theta)-bin threshold. Compare `C_ML(p, theta)` to `C_baseline(p, theta)` and `C_miss_mass(p, theta)`. This is the primary result: ML reduces contamination in data, measured by an independent data-driven method.
- **ML-vs-baseline contamination plot.** Per (p, theta) bin: plot {baseline, ML, missing-mass method} contamination with error bars. This is the headline figure for the poster. Make it publication-quality.
- **RICH cross-check (where coverage exists).** In the RICH acceptance (`p > 1.75 GeV, theta < 20°`, sector 4): select `ep → e pi+ X` with the pi+ RICH-tagged (`best_PID == 211`, `RQ > 0.2`, `N_photons > 3`). The fraction with EB-K+ ID gives an independent contamination estimate. Overlay RICH cross-check points on the comparison plot for the bins where they exist. This is a cross-check, not a standalone method — a handful of bins is sufficient.
- **Report draft.** Full pass on the ~10-page written report. Sections: (1) introduction and channel; (2) baseline and ML pipeline; (3) MC-truth comparison; (4) data-driven validation (exclusive missing-mass method, RICH cross-check); (5) systematic uncertainties; (6) conclusion and headline number.
- **Poster draft.** Take the report's figures and tables, lay them out for the SULI poster format. First pass.

**Maria's tasks.**
- Review the ML-vs-baseline contamination plot. This is the primary deliverable — it needs to be right.
- Review report draft Sections 1-4 by mid-week. Return written comments.
- Review poster draft Friday. Return comments by Sunday.

**Done when.**
- ML-vs-baseline contamination plot (with missing-mass method and RICH cross-check) in `/figures/contam_ml_vs_baseline.png`.
- Report draft v1 in `/report/report_v1.md`.
- Poster draft v1.

**Risks / dependencies.**
- ML-vs-missing-mass method disagree in some bins. Severity M. Mitigation: report honestly. If the ML reduces contamination relative to baseline but the missing-mass method and MC-truth disagree on the absolute level, that is a systematic uncertainty, not a failure. Quote the spread.
- RICH coverage is narrow (sector 4, limited p-theta bins). Severity L (expected). Mitigation: present what's available. RICH is a cross-check, not the primary result.

**Fallback / scope-down.** If the RICH cross-check is incomplete by Wednesday, drop it. Missing-mass method + MC-truth + ML is sufficient for the report and poster.

---

### Week 10 — Report final; poster final; handoff

**Theme.** Polish. Submit.

**Cooper's tasks.**
- Report final pass. Address all of Maria's comments. Add missing figures, redo any plot that's substandard. Aim for the report to be self-contained — a stranger should be able to read it and understand what was done.
- Poster final pass. Print test (one full-size print on Monday so any layout issues surface early).
- Code repo cleanup. README rewrite. One-command training reproducibility test (delete everything, re-clone, re-run, confirm same model). Tag a release.
- Practice talk for SULI symposium. Maria + 1-2 lab members as audience.
- Handoff document: 2-page memo to whoever picks this up next (likely a future SULI student or graduate student). What works, what doesn't, what's the next obvious thing to try.

**Maria's tasks.**
- Final report review.
- Attend practice talk.
- Submit poster to SULI symposium per DOE deadline.

**Done when.**
- Report final, PDF, committed to repo as `report_final.pdf`.
- Poster final, PDF, submitted.
- Repo tagged `v1.0`.
- Handoff memo written.
- SULI symposium talk delivered.

**Risks / dependencies.**
- Cooper runs out of time on polish. Severity M. Mitigation: have a "minimum viable poster" version ready by Wednesday. Polish on top of a working version, not from scratch.

**Fallback / scope-down.** Cut the practice talk if time-critical. Cut the handoff memo length to 1 page.

---

## 5. Decision points table

| # | Decision | By when | Who decides | What depends on it |
|---|---|---|---|---|
| D1 | Confirm Cooper's ifarm paths to clasdis MC and RGA pass-2 data | Week 1 Day 1 | Maria + Cooper | All processing work, all weeks |
| D2 | RICH bank present and non-empty on clasdis MC | Week 1 Day 2 | Cooper (verification) | Whether RICH cross-checks are available at all |
| D3 | Hyperon-tagged kaon-truth channel (e' K+ Lambda, e' K+ Sigma0, both, or skip) | Week 3 | Maria | Week 8-9 cross-checks, scope of "independent kaon-truth tag" |
| D4 | Extend `processing_three_particles` with ToF/calo per-hadron features, or postpone | Week 3 | Maria | Strength of per-hadron PID features in analysis-channel ntuple; Week 5 evaluation |
| D5 | (p, theta) bin edges | Week 2 | Maria | All contamination tables and plots |
| D6 | Final model family (BDT / MLP) for the production result | Week 6 | Maria + Cooper | Pipeline freeze, all later evaluation |
| D7 | Calibration method (Platt, isotonic, or both) for the production result | Week 6-7 | Maria | Final model deliverable |

---

## 6. Risk register

| # | Risk | Sev | Mitigation | Fallback |
|---|---|---|---|---|
| R1 | `RICH::Particle` bank empty on clasdis MC | H | Cooper verifies Day 2 with `hipo-utils -dump`. If empty, RICH cross-check moves from MC to data-only; the 14 RICH variables are not in the MC training ntuple (no big loss, they're not in features anyway). | Drop RICH as a sanity comparison on MC. RICH still used on data as a cross-check in Week 9 where sector-4 coverage exists. |
| R2 | Cooper's ifarm access delayed | H | Maria pushes JLab IT pre-arrival. Backup laptop environment with local copy of one HIPO file for offline work. | Cooper does Python and reading work in Week 1, full pipeline catches up Week 2. |
| R3 | Training MC statistics insufficient at high p (>3 GeV) where contamination is worst | M-H | Use full `clasdis` inbending sample. Quote MC stat uncertainty per bin. | If high-p bins are still stat-limited, merge bins. Honest binning at the cost of resolution. |
| R4 | Missing-mass method and MC-truth disagree beyond systematics | M | Investigate honestly: does the data have a contamination the MC doesn't capture? Is the sideband subtraction biased? | The disagreement is a finding. Report it. Quote it as a systematic uncertainty rather than papering over it. |
| R5 | MLP underperforms or takes too long | M | Keep architecture small (2 × 64). If still too slow, use a GPU node or sklearn MLPClassifier. | Drop MLP. BDT alone is enough for the report. |
| R6 | `processing_three_particles` extension for per-hadron ToF/calo not done in time | M | Decide D4 in Week 3. If postponed, use existing 3-particle ntuple without ToF/calo for analysis-channel evaluation; apply ML using the training-MC features only (which are per-track, indexed by REC::Particle row). | Do the analysis using only the training-ntuple features; analysis-channel ntuple becomes an event-selection helper only, not a per-hadron PID input. |
| R7 | Cooper underwater on Python in first 2 weeks | M | Pair-code Day 4-5. Sklearn reading is non-negotiable Week 2-3. Maria checks in mid-week. | Reduce Week 4-5 model-training ambition; lean on Maria for sklearn pipeline construction in Week 4. |

---

## 7. Scope-down list

If behind schedule at Week 5, drop in this order:

1. **Drop MLP.** Keep BDT only. (Saves ~3 days in Week 6.)
2. **Drop isotonic calibration**, keep only Platt (sigmoid) calibration. (Saves ~1 day.)
3. **Drop per-(p, theta)-bin threshold optimization.** Use a single global threshold chosen on the validation set. (Saves ~2 days in Week 5.)
4. **Drop the analysis-channel ntuple extension for ToF/calo per-hadron variables** if not done by Week 6. Use existing `processing_mc_three_particles.groovy` output without ToF/calo per-hadron features. Apply the ML classifier to candidate K+ tracks using their kinematics + a join back to the training ntuple's per-track features.
5. **Drop the hyperon-tagged kaon-truth cross-channel** (whichever was picked in D3). Rely on MC truth + missing-mass method for the kaon-truth side.
6. **Drop feature ablation studies.** Keep hyperparameter tuning only. Feature importance plot alone is sufficient for the report.
7. **Drop the RICH cross-check in Week 9.** Missing-mass method + MC-truth + ML is the complete story.

Do not drop: the headline improvement number, the report, the poster. These are the contractual SULI deliverables.

---

## Appendix: notes on conventions and gotchas

- **FD-only.** All training and evaluation done with `status` in [2000, 4000). Verified by `generic_tests.forward_detector_cut`. CD tracks excluded from this entire project.
- **chi2pid as feature, not cut.** The training script does NOT cut on chi2pid. The baseline DOES apply the pass-2 momentum-dependent chi2pid cut (`passes_kplus_chi2pid_cut` in `scripts/baseline_chi2pid.py`); the older loose form `|chi2pid| < 3` is not the production baseline. The ML uses chi2pid as one input feature among many. This is the central methodological point: we are using ML to learn a better cut than the standard one, not to replace chi2pid.
- **MC truth matching.** Geometric: `|delta phi| < 6°, |delta theta| < 2°`. Same as Connor. Tracks with no MC match within this window are dropped from the training set (they cannot have a truth label).
- **Energy-loss and momentum corrections.** Applied at the groovy level via `analysis_fitter` infrastructure for electrons; not applied for K+ (Hayward's code has no kaon corrections; this is a known limitation common to cut-based and ML approaches equally).
- **Reproducibility.** All sklearn fits with `random_state=42`. All split file lists committed to repo. All slurm submission scripts committed. README must explain how to go from clean checkout to trained model in one command.
- **Honesty.** If a measurement disagrees with another, say so. The missing-mass method and MC-truth are independent by construction: if they disagree, that is part of the result, not a problem to paper over.
