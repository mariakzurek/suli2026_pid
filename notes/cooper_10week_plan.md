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

**Step 2c — Data/MC variable-agreement audit.** *(Not completed in Week 2 — carried over into Week 3. See Week 3 for the full task description and output requirements.)*

The PI's standing requirement: *"if MC and data variables disagree, we cannot use them for ML."* Before any ML training begins, audit every relevant variable in the training ntuple against the RGA pass-2 data distribution for the EB-K+ subsample. Do the comparison in coarse (p, θ) slices — use 9 cells: p in [1, 2], [2, 3], [3, 5] GeV crossed with θ in [5°, 15°], [15°, 25°], [25°, 35°]. In each cell, overlay MC (EB-K+ tracks) and data (EB-K+ tracks) 1D distributions. Flag any variable where KS distance > 0.05 or the shapes disagree visually.

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

For each variable, confirm the hit-fraction (fraction of tracks with a non-missing/-9999 value) in MC and in data. Save all overlaid distribution plots to `/figures/feature_audit/`. Summarize the per-feature KEEP / CANDIDATE / DROP decision table in `notes/feature_audit.md`.

**Step 2d — (p, θ) data/MC comparison for the reweighting decision.** *(Not completed in Week 2 — carried over into Week 3. See Week 3 for the full task description and output requirements.)*

Even though `p` and `theta` are not ML training features, comparing their 2D distributions between MC and data is critical for Task 3 (Connor's reweighting). Connor's reweighting computes 2D data/MC ratios in (p, θ) bins and reweights MC events to match data; before deciding whether to use that approach, we need to know whether the distributions actually differ and by how much.

Cooper must:
  - Plot the 2D (p, θ) distribution for MC (EB-K+ tracks) and for data (EB-K+ tracks) side by side, using the same binning as the analysis grid.
  - Plot the ratio map data/MC in (p, θ).
  - Write a one-paragraph interpretation: if the ratio map is roughly flat (data/MC ≈ 1 everywhere), no reweighting is needed. If it varies significantly across (p, θ) space, reweighting is warranted and should be implemented in Week 4.
  - Save figures to `/figures/feature_audit/ptheta_data_mc_ratio.png`. Document the interpretation in `notes/feature_audit.md`.

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
- Data/MC feature disagreement affects many variables (Steps 2c/2d). Severity M. Mitigation: run the audit early in the week so flagged features are identified before any ML training begins. A reduced feature set still yields a valid classifier.

**Fallback / scope-down.** If the feature audit reveals broken features (e.g. FTOF time always zero), table the broken features and proceed with whichever subset is correct. Do not block on fixing the groovy script in Week 2; come back to it in Week 3 if needed.

---

### Week 3 — Data/MC audit; production-scale ntuples; Week 1-2 report

**Theme.** Finish the data/MC variable agreement audit carried over from Week 2. Scale up ntuple production to the full MC sample and a meaningful data sample using slurm batch jobs. Complete and submit the Week 1-2 written report.

**Cooper's tasks.**

**Task 3a — Finish the data/MC variable-agreement audit (carried over from Week 2).**

This is the continuation of Steps 2c and 2d from Week 2. The goal is to establish, for every candidate ML training feature, whether MC and data distributions agree well enough to trust the feature in training.

*What the audit involves:* per-feature overlays of MC vs data 1D distributions in 9 coarse (p, θ) slices (p in [1, 2], [2, 3], [3, 5] GeV × θ in [5°, 15°], [15°, 25°], [25°, 35°]); KS-test flag per feature per slice (flag if KS distance > 0.05 or shapes disagree visually); KEEP / CANDIDATE / DROP decision per feature; and the 2D (p, θ) data/MC ratio map needed for the reweighting decision.

*Tooling:* the `compare_mc_data.py` script (being written separately) handles the per-feature comparison mechanics — generating the overlaid histograms and computing the KS statistic. Cooper should **use** this script AND understand what each test does: what the KS statistic measures, what the null hypothesis is, and why KS distance > 0.05 is the threshold. Do not treat the script as a black box.

*Outputs:*
  - `notes/feature_audit.md` — per-feature decision table (one row per variable, columns: feature name, hit-fraction MC, hit-fraction data, max KS distance across slices, decision, notes).
  - `figures/feature_audit/` — one overlay plot per feature per (p, θ) slice, plus the 2D (p, θ) data/MC ratio map at `figures/feature_audit/ptheta_data_mc_ratio.png`.
  - One-paragraph reweighting recommendation in `notes/feature_audit.md`: if the ratio map is roughly flat, no reweighting is needed; if it varies significantly, reweighting is warranted and will be implemented in Week 4.

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
  - Fill in remaining placeholders: abstract, introduction prose, section 4 contamination discussion, section 5 audit findings (once the audit from Task 3a is done), summary section.
  - Replace every TODO figure caption with a real caption describing what is actually shown (axes, units, color coding, key takeaway).
  - Copy produced PNGs into `report/figures/`.
  - Compile locally with `pdflatex` (or via Overleaf) and confirm the PDF builds without errors or unresolved references.

- Submit a slurm job array to process the **full** `clasdis` sample through `processing_mc_pid_training.groovy` (see Task 3b above). Run `processing_mc_three_particles.groovy` with `p1=2212, p2=321` on the same files to produce the analysis-channel ntuple for `ep → e' p K+ X`.
- Define the train / validation / test split. Split at the file level (not event level), randomly, with a fixed seed. Recommended ratio: ~70% train, ~15% validation, ~15% test. Record the file lists in the repo as `train_files.txt`, `val_files.txt`, `test_files.txt`.
- Compute baseline contamination on the **test set only** to get the final baseline numbers.
- Read scikit-learn user guide §3.1 (cross-validation), §3.3 (metrics), §1.16 (calibration). One page of notes per section.

**Maria's tasks.**
- Review the completed feature audit (`notes/feature_audit.md`). Confirm the KEEP / CANDIDATE / DROP decisions. Decide whether the (p, θ) ratio map warrants reweighting in Week 4.
- Approve the train/val/test split.
- 1-hour meeting: walk through the baseline contamination plot. Decide whether the (p, theta) binning needs revision before Week 4.
- Check in with Cooper on Python/ML progress. If reading is not getting done, adjust Week 4 expectations.
- **Make decision on hyperon-tagged kaon-truth channel** (see Decision Points). This decision is no longer on the critical path — the W8/W9 validation now runs on RICH-overlap (primary, on `ep → eKpX`) and exclusive-pion π-mis-ID (secondary, on the separate `ep → eπ⁺(n)` channel), not on a hyperon tag. D3 is a supplementary-sanity-check question at this point; skip is a defensible answer.
- **Sign off on the Week 1-2 report draft** after all Overleaf comments are resolved and the PDF compiles clean.
- If Cooper has not used slurm arrays before, help set up the submission script at the start of the week — do not let this block ntuple production past Tuesday.

**Done when.**
- `notes/feature_audit.md` complete with per-feature KEEP / CANDIDATE / DROP decision table.
- `figures/feature_audit/` populated with per-feature overlay plots and `ptheta_data_mc_ratio.png`.
- Reweighting recommendation (one paragraph) written in `notes/feature_audit.md`.
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

**Fallback / scope-down.** If the BDT training hits a wall (e.g. memory at full sample size), train on a subset (5M tracks is plenty for a first pass) and document. Full-statistics training can be re-run in Week 7 alongside the two new BDTs on the full momentum range.

---

### Week 5 — Diagnose Week-4 low-p underperformance, lock a feature tier, then per-bin optimize and apply to the analysis channel

**Theme.** Investigate first, optimize and apply second. Week 4 produced a working end-to-end BDT pipeline but on a minimal four-feature tier (`beta`, `ftof_energy_1B`, `ftof_time_1B`, `ftof_path_1B`; no `chi2pid`, no ECAL, no FTOF panel 1A), and the per-bin results show surprisingly poor performance at low momentum — exactly the regime where K/π separation should be easiest. Applying that model to the analysis channel before understanding the low-p behavior would propagate the problem into the headline number. Week 5 therefore splits cleanly in two: the first half tests whether the low-p underperformance is feature-set-fixable and commits to a tier; the second half runs per-bin threshold optimization and the first analysis-channel application on that locked tier.

The `week4-tier-flexible` branch decouples dataset schema from training features — the dataset is built once with the maximal feature set and the training selects a subset via `--features-file`. Three tier files are seeded as starting points: **tier 1** (the current four features), **tier 2** (tier 1 + `chi2pid` + FTOF 1A panel), and **tier 3** (tier 2 + ECAL `ecin_*` / `ecout_*` + `nphe_htcc`). These are not gospel; Maria and Cooper may revise tier composition as the comparison comes in.

**Cooper's tasks — first half (Tue–Wed): investigate and commit to a tier.**
- Run tier-2 and tier-3 trainings on the existing dataset (no rebuild needed; that is the whole point of the new branch). Hold all other hyperparameters fixed to the Week-4 configuration so the comparison is clean. Save model artifacts under `/work/clas12/$USER/SULI/` in a layout that makes "which tier produced this model" unambiguous.
- Build a side-by-side comparison: per-bin AUC, per-bin C^π→K, and the feature importance ranking for tier 1 vs tier 2 vs tier 3. The central question is whether adding `chi2pid` (tier 2) closes the low-p gap. If it does, that is the answer and the diagnosis is short. If it does not, the problem is structural and the second set of tasks below applies.
- If `chi2pid` does not fix low-p, diagnose per-bin: (a) plot per-bin event counts and class balance `n_K / n_π` — low-p EB-K+ samples are kaon-dominated and may be statistics-starved on the negative class; (b) plot per-bin score distributions colored by truth and judge whether the BDT separates the classes cleanly at low p or whether the score itself is confused; (c) re-examine `evaluate.py`'s `n<50` low-stat policy and bin-edge handling at low p, and confirm the training-vs-evaluation class-balance assumption (if they differ, the threshold sweep can produce misleading per-bin numbers without reweighting); (d) note that the Week-4 Platt calibration was a near no-op (pre/post AUC identical), so a residual kinematic bias in the raw scores would not have been absorbed.
- Commit to a tier for the rest of the project. Write the choice and its justification into `notes/cooper_week5_decisions.md` (one paragraph minimum), with the tier-comparison plots and the diagnosis as evidence. This is the deliverable Maria signs off on at mid-week before any per-bin optimization runs.

**Cooper's tasks — second half (Wed–Fri): per-bin optimization and analysis-channel application on the locked tier.**
- Per-(p, theta)-bin threshold optimization on the chosen tier using FOM = `N_K / sqrt(N_K + N_pi)`. Sweep BDT score threshold 0 to 0.95 per bin; pick the maximum. Use the validation set for threshold selection — never the test set. Tabulate the thresholds.
- Apply the optimized BDT to the analysis-channel ntuple (`ep → e' p K+ X`, from `processing_mc_three_particles.groovy`). For each event run the BDT on the K+ candidate and accept if the score exceeds the bin-optimal threshold. Compare accepted-event count and pion-truth contamination to the EB+chi2pid baseline. This stays on MC. Data application is **Week 8** (two SIDIS channels `eKX` and `eKpX` plus a separate exclusive-pion `eπ⁺(n)` channel; RICH-overlap on `eKpX` as primary cross-check and π-mis-ID on `eπ⁺(n)` as secondary); do not promote it here.
- Quote the first headline number on the chosen tier, not on tier 1: "for the `ep → e' p K+ X` MC sample, at fixed kaon efficiency `eps_0`, baseline gives contamination `C_baseline = X%`, BDT gives `C_BDT = Y%`, improvement = `(X-Y)/X = Z%`." If low-p remains problematic and the tier choice does not fix it, quote the number with an honest per-bin breakdown rather than burying the issue in a single average.
- Plot M_X(e' p K+) distributions for {all events, baseline-cut events, BDT-cut events}, color-coded by truth class.
- Write a 1-page section of the eventual report: "MC-truth comparison of BDT to EB+chi2pid baseline on the `ep → e' p K+ X` channel," covering both the tier choice and the headline number. Writeup deposit, not the final report.

**Maria's tasks.**
- Mid-week meeting now focuses on the tier comparison and the low-p diagnosis, not the headline number. Sign off on the tier choice first; the per-bin optimization run depends on it.
- Sit with Cooper through the tier comparison if needed — this is the first time the project has had to read a per-tier ablation and decide on a feature set under time pressure.
- Sign-off comes in two stages: (a) the tier choice, mid-week; (b) the headline number, end of week. Do not collapse them.
- **Mid-project review.** 1 hour. Are we on track? Is the scope right? Adjust Week 6–10 ambition. This conversation also absorbs whatever the low-p investigation concluded.

**Done when — first half.**
- Tier-2 and tier-3 trainings completed, model artifacts saved reproducibly on `/work/clas12/$USER/SULI/`.
- Side-by-side comparison table of tier 1 vs tier 2 vs tier 3 (per-bin AUC, per-bin C^π→K, feature importance).
- Written diagnosis (one paragraph minimum) of why low-p was poor in tier 1 and whether the higher tiers fix it.
- Locked tier choice for the rest of the project with brief rationale, in `notes/cooper_week5_decisions.md`.

**Done when — second half.**
- Per-bin BDT thresholds (on the locked tier) in a CSV.
- Headline improvement number quoted on the locked tier, with statistical uncertainty and a per-bin breakdown.
- M_X(e' p K+) plot in `/figures/`.
- Mid-project writeup section in `/report/section_bdt_truth.md`.

**Risks / dependencies.**
- Low-p underperformance is structural rather than feature-set-fixable (class imbalance, training-vs-evaluation class-balance mismatch, evaluation-code artifact at low stats, or a real kinematic bias the calibration cannot absorb). Severity M–H. Mitigation: if tier 3 does not close the gap either, Maria decides whether to scope the collaboration-meeting result down to mid/high-p only, with the low-p caveat documented honestly rather than hidden.
- Tier-comparison results are inconclusive — e.g. tier 2 partially improves low-p but with mixed signals on mid-p, or feature importance flips between tiers in ways that don't suggest a clean choice. Severity M. Mitigation: ship tier 2 (the obvious physics-motivated middle option) rather than spending the rest of the week chasing the cleanest possible ablation.
- Per-bin optimization overfits because some bins have low MC statistics. Severity M. Mitigation: use the validation set for threshold selection, not the test set. Apply on the test set only once.
- (p, theta) reweighting (originally Week 4) is now punted again — neither the tier comparison nor the headline number in Week 5 will be reweighted. Severity L–M. Mitigation: this is a tracked debt. Week 6 (collab meeting) has no analysis capacity for it; the natural pickup point is Week 7 when the two new BDTs are retrained on full-p — apply `sample_weight` reweighting at that retraining step or note the omission explicitly in the writeup so it doesn't get lost. If it slips past W7, it becomes a quoted systematic on the W9 RICH-vs-MC comparison rather than a re-training.

**Fallback / scope-down.** If the tier comparison does not resolve the low-p question by end-of-day Wednesday, ship the tier-2 model with honest per-bin numbers and a documented caveat about low-p, and run the second-half tasks on that. Do not spend Thursday and Friday chasing a structural issue at the expense of the analysis-channel application and the mid-project writeup; the chase, if needed, is picked up in Week 7 when the two new full-p BDTs get trained — that retraining is the natural moment to re-examine whether the low-p behavior is structural or was a current-p-range artifact.

---

### Week 6 — CLAS collaboration meeting

**Theme.** Cooper gives his collab-meeting talk on Tuesday and spends the rest of the week at the meeting as a participant. The talk itself carries the Weeks 4–5 material — tier comparison and locked tier choice from Week 5, per-bin FOM thresholds, headline contamination-vs-(p, θ) figure on the locked tier, and whatever first `apply_bdt` results were defensible by Monday. There are no plan-tracked deliverables from Wednesday onward; the week is deliberately absorbed by conference attendance and no new analysis threads open. Week 7 picks up threshold refinement and the two new full-p BDTs.

---

### Week 7 — Threshold tuning refinement; two new BDTs on the full momentum range

**Theme.** Two threads in parallel. First, extend the Week-5 per-bin FOM threshold work into a fuller sweep on the current BDT, now with the collab-meeting feedback incorporated — this is refinement of what was started, not a fresh formulation. Second, train two new BDTs on the **full momentum range** using the same tier features locked at the end of Week 5: a binary K⁺ vs non-K⁺ model (the current formulation, retrained without the `--p-max` cut) and a three-class K⁺ / π⁺ / p multiclass model. The tier question is closed for the remainder of the project; do not reopen it here. What is open is momentum coverage and target formulation.

**Prerequisite — parquet rebuild without the p-max cut.** The current parquet built by `scripts/training/build_dataset.py` applies `--p-max`, which caps the training momentum. Both new BDTs need the full momentum range, which means rebuilding the parquet (or building a `v02` parquet without the p-max cut) before either training can start. This is a Monday task, not something to discover on Wednesday. Bump the parquet suffix rather than overwriting `v01`, and record the manifest — the Week-5 threshold-refinement thread still runs against the current-p-range parquet, so both must coexist on disk.

**Cooper's tasks.**

*Thread 1 — threshold tuning on the existing BDT (current p-range).*
- Extend the Week-5 per-(p, θ)-bin FOM sweep. Widen the threshold grid where the Week-5 optima landed near a grid endpoint, add finer resolution near the optimum, and re-tabulate. Incorporate any per-bin methodology feedback Cooper carried back from the collab meeting — e.g., if the audience flagged low-stat bins or bin-edge behavior, address it here.
- Re-quote the headline contamination-vs-(p, θ) numbers on the current-p-range BDT with the refined thresholds. This is the reference number the two new full-range BDTs will be compared against in Week 8.
- Commit the refined threshold table and the updated headline plot. This work stays on the locked tier and the current parquet.

*Thread 2 — two new BDTs on the full momentum range.*
- Rebuild the parquet without the `--p-max` cut (see prerequisite above). Bump the dataset version — `datasets/v02/` or equivalent. Record the manifest.
- **BDT-1: binary K⁺ vs non-K⁺, full p-range.** Same tier features as the locked Week-5 choice. Same Week-4 fixed LightGBM hyperparameters (`n_estimators=200, learning_rate=0.05, max_depth=6, objective='binary', random_state=42`). Same Platt calibration on a held-out slice of train. Save under `/work/clas12/$USER/SULI/models/tier{N}_binary_fullp_vNN/`.
- **BDT-2: three-class K⁺ / π⁺ / p, full p-range.** Same tier features. LightGBM in multiclass mode (`objective='multiclass', num_class=3`). Class labels are truth species restricted to {K⁺, π⁺, p} — decide with Maria whether other truth species collapse into an "other" fourth class or are dropped from the training set; document the choice in the manifest. Save under `/work/clas12/$USER/SULI/models/tier{N}_3class_fullp_vNN/`.
- **First-pass evaluation of both new BDTs.** Per-(p, θ)-bin AUC (for binary) and per-class ROC / one-vs-rest AUC (for multiclass) on the validation set. Sanity-check that extending to full p does not degrade the mid-p region where the current model works. Do not run the full per-bin FOM optimization here — that lives in Week 8 alongside the model decision.

**Maria's tasks.**
- Approve the parquet rebuild plan and the version-bump convention Monday morning. This blocks Thread 2.
- Decide the multiclass class-set question (K⁺/π⁺/p only vs K⁺/π⁺/p/other) before Cooper trains BDT-2.
- Mid-week check-in on Thread-1 threshold refinement. Confirm the refined thresholds are ready to be the Week-8 comparison reference before Cooper hands them off.

**Done when.**
- Refined per-bin threshold table for the current-p-range BDT committed, with the updated headline plot.
- `v02` parquet (full p-range) built and manifest recorded.
- `bdt_binary_fullp` model artifact saved with wrapper dict and features list.
- `bdt_3class_fullp` model artifact saved with wrapper dict and features list.
- First-pass validation-set AUC (per bin, per class) for both new BDTs tabulated in `notes/cooper_week7_fullp_bdts.md`.

**Risks / dependencies.**
- Parquet rebuild takes longer than expected because the underlying MC ntuples do not have the full momentum coverage assumed. Severity M. Mitigation: check the p-distribution in the source ROOTs Monday before launching the rebuild; if the ntuples are themselves p-max-capped upstream, the "full p-range" is really "full available p-range" — document that and proceed.
- Multiclass BDT is worse than the binary K⁺ vs non-K⁺ in the K⁺ metrics that matter for this analysis. Severity L (this is the question Week 8 exists to answer). Mitigation: no mitigation needed; the answer is the point.
- Full-p training exposes low-stat regions at very high p that the current-p-range BDT never saw. Severity M. Mitigation: log per-bin train / val counts and flag any bin below a stat floor for exclusion from the Week-8 comparison. Do not silently include statistics-starved bins in the headline.

**Fallback / scope-down.** If Thread 2 slips, ship BDT-1 (binary full-p) only. BDT-2 (3-class) moves to early Week 8 as a parallel thread with the MLP. Do not drop the threshold-refinement Thread 1 — that is the deliverable the Week-6 collab talk asked for.

---

### Week 8 — MLP as second model family; final model decision; apply to data

**Theme.** Three parts, sequenced. First, train the MLP as a second model family on all three formulations that exist coming out of Week 7 (binary current-p-range, binary full-p-range, 3-class full-p-range) — this is what was originally Week 6, deferred to fit around the collab meeting. Second, make the final model decision: which family (BDT or MLP), which formulation (binary vs 3-class), which threshold strategy. This is the go/no-go for the analysis application. Third, apply the chosen model to data on two SIDIS channels (`eKX` primary, `eKpX` with proton tag for the RICH-overlap cross-check) plus a separate exclusive-pion channel (`eπ⁺(n)`) that provides an independent π-mis-ID cross-check.

**Prerequisite — two new data ntuples.** The SIDIS-inclusive data ntuple (`ep → eKX`, kaon-only in FD) already exists at `/volatile/clas12/zurek/SULI/data_v01/`. Part 3 needs two additional productions, neither of which exists yet, and both require Groovy work in `~/CLAS/SULI/clas12_analysis_software/processing_scripts/`:

1. **SIDIS-with-proton-tag ntuple (`ep → eKpX`), for the RICH-overlap primary cross-check.** Same SIDIS event topology as `eKX` but with a detected proton in FD alongside the kaon and electron. The proton tag reduces combinatorics in the RICH-overlap region and is where the primary cross-check runs. Extend `processing_data_pid_training.groovy` (or fork a sibling — decide with Maria) and run via the existing slurm wrapper. Output to `/volatile/clas12/$USER/SULI/data_eKpX_v01/`.
2. **Exclusive-pion ntuple (`ep → eπ⁺(n)`), for the π-mis-ID secondary cross-check.** A distinct exclusive channel: electron and π⁺ detected in FD with the recoil neutron reconstructed via missing mass and required to sit under the exclusive neutron peak. This gives a clean pion truth sample independent of the K-selection channel. The BDT is then run on this pion sample to measure how often true pions get scored as K — i.e., π-mis-ID-as-K. Write a new Groovy processing script (`processing_data_epiN.groovy` or equivalent) and run via slurm. Output to `/volatile/clas12/$USER/SULI/data_epiN_v01/`.

Both productions block their respective Part-3 cross-checks; start the Groovy work Monday.

**Cooper's tasks.**

*Part 1 — MLP as second model family.*
- Train MLPs matching all three BDT formulations from Weeks 5/7: binary current-p-range, binary full-p-range, 3-class full-p-range. `sklearn.neural_network.MLPClassifier` or a small PyTorch net, 2 hidden layers of 64 units, ReLU, Adam, early stopping. Apply `StandardScaler` first (mandatory for MLP; not needed for BDT), impute -9999 with per-feature median before scaling, add binary missing-indicator features for the physically-informative missing groups (PCAL, FTOF layer 2). Platt calibration and a reliability diagram per model.
- Head-to-head comparison: for each of the three formulations, tabulate {BDT, MLP} × {Brier, log-loss, AUC or one-vs-rest AUC, per-bin C^π→K at matched K⁺ efficiency}. Same test set discipline as before — touched exactly once for the reported numbers.

*Part 2 — final model decision.*
- Pick the family, the formulation, and the threshold strategy. Write the decision and its justification in `notes/cooper_week8_model_decision.md` with the head-to-head tables as evidence. Sign-off from Maria required before Part 3 begins; this is the pipeline-freeze moment for the analysis application. Save the chosen artifact as `model_final.joblib` under `/work/clas12/$USER/SULI/models/`.

*Part 3 — apply to data.* Two primary SIDIS channels carry the data-side result, plus a separate exclusive-pion channel that provides an independent π-mis-ID cross-check.
- **Primary SIDIS channel (`ep → eKX`).** Run `apply_bdt.py` (the Week-6-implementer PR) on `/volatile/clas12/zurek/SULI/data_v01/`. Apply the chosen per-bin threshold (or class-probability rule, for the 3-class case). This is the standard SIDIS channel — kaon detected in FD, no additional hadron requirement; produce ML-vs-baseline contamination-per-bin numbers as the primary data-side deliverable.
- **SIDIS-with-proton-tag channel (`ep → eKpX`).** Once the `data_eKpX_v01/` ntuple exists (see prerequisite), run `apply_bdt.py` on it. This is also a SIDIS channel — the detected proton is a tag that reduces combinatorics in the RICH-overlap region, not an exclusivity requirement. Do not call this the "exclusive" channel; the recoil system is still `X`. The point of this channel is to enable the RICH-overlap primary cross-check below on a cleaner sample.
- **Primary validation: RICH overlap in data (on `eKpX`).** In the (p, θ) region where the RICH is instrumented and providing PID (roughly `p > 1.75 GeV, θ < 20°`, sector 4 — reconfirm current RICH acceptance with Maria before drawing acceptance boundaries), extract the kaon-contamination number two ways on the same tracks: (a) from the MC-based ML pipeline the classifier provides, and (b) from RICH particle ID directly, using the RICH tags as near-truth-level PID. Compare bin by bin. This is the primary cross-check because RICH in its acceptance gives PID that is close to truth-level; disagreement between the ML estimate and the RICH estimate is a real result and drives Week 9's framing.
- **Secondary validation: π-mis-ID from a separate exclusive-pion channel (`ep → eπ⁺(n)`).** This cross-check does **not** run on the K-selection channel. On the `data_epiN_v01/` ntuple, select the exclusive-pion sample by requiring the missing mass `M_X(eπ⁺)` to sit inside the neutron peak — that mass cut gives a clean sample of true π⁺ tracks essentially free of K⁺ contamination, independent of any ML or chi2pid decision. Then run the trained BDT on those pion tracks and measure how often the classifier scores them as K — i.e., the π-mis-ID-as-K rate on real data. Compare bin by bin to the MC-predicted π-mis-ID rate on true pions from the training set. Note the asymmetry explicitly in the writeup: this channel constrains only **π-mis-ID-as-K**, not proton contamination and not K efficiency. It is a secondary cross-check to RICH, not a substitute, and it is measured on a physically separate channel rather than by reading the K-selection channel's own missing-mass distribution.

**Maria's tasks.**
- Confirm the Groovy strategy for both new ntuple productions (`eKpX` and `eπ⁺(n)`) Monday — fork vs extend for `eKpX`; new-script for the exclusive-pion channel. Cooper is blocked on Part 3 without this.
- Reconfirm current RICH acceptance boundaries and PID-tag conventions (`best_PID`, `RQ`, `N_photons` thresholds) before Cooper draws the RICH-overlap cross-check bins on the `eKpX` sample.
- Confirm the exclusive-neutron mass window for the `ep → eπ⁺(n)` selection (nominal ± width) before Cooper commits the pion-truth definition.
- Sign off on the Part-2 final model decision before Part 3 begins. This is the D6/D7 decision consolidation.
- Mid-week check-in on the RICH cross-check preliminary numbers and the π-mis-ID cross-check preliminary numbers.

**Done when.**
- MLPs trained for all three formulations; head-to-head BDT-vs-MLP comparison tables in `notes/cooper_week8_model_comparison.md`.
- Final model decision written and signed off; `model_final.joblib` committed under `/work/`.
- `data_eKpX_v01/` ntuple produced (SIDIS-with-proton-tag); event count and file count documented.
- `data_epiN_v01/` ntuple produced (exclusive-pion); event count and file count documented.
- `apply_bdt` run on the primary SIDIS channel (`eKX`); per-(p, θ)-bin ML-vs-baseline contamination table and plot committed.
- `apply_bdt` run on the SIDIS-with-proton-tag channel (`eKpX`); RICH-overlap cross-check plot (ML contamination estimate vs RICH-derived estimate, bin by bin in the RICH acceptance) committed as the week's primary validation figure.
- `apply_bdt` run on the exclusive-pion sample selected via the neutron-peak mass cut on `eπ⁺(n)`; π-mis-ID-as-K rate per (p, θ) bin, compared to the MC-predicted rate, committed as the secondary cross-check.

**Risks / dependencies.**
- Groovy work for either new ntuple (`eKpX` or `eπ⁺(n)`) slips past Monday. Severity M–H. Mitigation: start both Monday morning, not mid-week. For `eKpX`, if the extension is stuck by Tuesday, fall back to filtering the proton-tag topology out of the existing `data_v01/` ntuple in post-processing — worse cuts, larger intermediate files, but unblocks the RICH cross-check. For `eπ⁺(n)`, no post-hoc fallback is as clean: the exclusive-pion selection needs the electron-π⁺ topology that the K-selection ntuples do not carry. If that Groovy work slips, the π-mis-ID cross-check itself slips to Week 9.
- RICH acceptance in the current data is narrower than expected and the overlap sample is statistics-starved. Severity M. Mitigation: report the bins where the comparison is statistically meaningful; do not stretch the RICH cross-check into bins where the RICH sample is < ~100 tracks. If the RICH overlap is too thin to be a primary validation, the exclusive-pion π-mis-ID cross-check becomes the primary data-side validation and RICH degrades to sanity check — flag this to Maria immediately, do not paper over.
- Final model decision is genuinely close between BDT and MLP, or between binary and 3-class. Severity L. Mitigation: this is a real answer; pick the simpler formulation (binary BDT) unless one of the alternatives shows a decisive per-bin advantage. Document the closeness rather than manufacturing a false decisiveness.
- ML-vs-RICH comparison disagrees. Severity M–H. Mitigation: this is real physics — one of the two is missing something. Investigate honestly: is the MC training distribution wrong in the RICH region? Is the RICH tag selection contaminated by mis-tagged tracks? Document both possibilities and do not silently prefer one.
- Exclusive-pion sample is contaminated by non-neutron background under the mass peak. Severity M. Mitigation: the neutron-peak mass cut is not perfect; quote the residual non-pion fraction from MC and fold it into the π-mis-ID uncertainty band. If the residual is large enough to swamp the measurement, tighten the mass window and re-quote with reduced statistics.

**Fallback / scope-down.** If Part 3 slips, ship the primary SIDIS `apply_bdt` numbers on `eKX` and the RICH cross-check on `eKpX` (in whatever RICH-acceptance bins are populated); defer the exclusive-pion π-mis-ID cross-check to Week 9. If neither new ntuple is producible this week, both cross-checks slip to Week 9 and the SIDIS-inclusive `eKX` numbers become the sole data-side deliverable for Week 8.

---

### Week 9 — Deep-dive validation of the W8 data pass; poster-plot finalization

**Theme.** Two threads in parallel, both refinement rather than first-pass. The W8 apply-to-data run produced the first numbers on the primary SIDIS channels (`eKX` and `eKpX`) along with a first RICH-overlap cross-check on `eKpX` and a first π-mis-ID reading on the separate exclusive-pion channel (`eπ⁺(n)`). Week 9 sharpens those into defensible, publication-quality results: the RICH-vs-MC contamination comparison in the RICH-acceptance region gets its full systematic treatment as the primary validation, and the π-mis-ID cross-check on the exclusive-pion sample gets its full treatment as the secondary. In parallel, every plot destined for the CEU (or equivalent) poster is finalized this week — style, axes, labels, legends, colors, sizing. Treat plot finalization as its own workstream, not a Friday-afternoon cleanup pass on top of the analysis refinement. The two threads share Cooper's week but do not share deadlines: analysis refinement is signed off mid-week, plots are signed off by Friday.

No new model training this week. No new data production. The final model was locked in W8 Part 2; both new ntuples (`data_eKpX_v01/` and `data_epiN_v01/`) are locked coming into W9. If any of them show a defect discovered during refinement, document it and quote the effect on the number — do not retrain and do not reprocess.

**Cooper's tasks.**

*Thread 1 — refine the primary validation (RICH-vs-MC contamination in the RICH-acceptance region).*
- Take the W8 first-pass RICH-overlap plot as the starting point. Redraw the RICH-acceptance boundary explicitly: confirm sector 4, `p > 1.75 GeV, θ < 20°` (or whatever Maria reconfirmed at W8 start), and shade the acceptance region on the (p, θ) map so the reader sees exactly which bins the comparison covers and which are extrapolation.
- Build the coverage plot: per-(p, θ)-bin RICH-tagged track counts inside the acceptance, with the < ~100-track floor from W8 marked as an exclusion. Any bin below the floor drops out of the headline comparison and is called out separately as statistics-starved. Do not carry statistics-starved bins into the headline number.
- Full systematic uncertainty treatment on the ML contamination estimate in the RICH-acceptance region. Sources at minimum: (a) per-bin MC statistical uncertainty from the training/test split; (b) sensitivity to the per-bin threshold choice — vary the threshold by ± one grid step and re-quote; (c) sensitivity to the calibration — quote the number with and without Platt calibration; (d) if the (p, θ) reweighting debt from W5 is still un-resolved (see W7 risks), quote the sensitivity to including / excluding the reweighting. Systematic on the RICH-derived contamination side: (e) RICH tag purity — vary `RQ` and `N_photons` thresholds around Maria's nominal and quote the drift.
- Agreement / disagreement diagnosis, bin by bin. For every acceptance bin above the stats floor, compute `Δ = C_ML − C_RICH` and its combined uncertainty. Classify each bin as (i) agrees within uncertainty, (ii) disagrees but the disagreement points at a specific source (training MC mismodels the RICH region, RICH tag contaminated by mis-tagged tracks, per-bin threshold under- or over-tuned), or (iii) disagrees and the source is unclear. Write the diagnosis into `notes/cooper_week9_rich_diagnosis.md` bin by bin — no averaging until every disagreeing bin has been named.
- Produce the final headline RICH-vs-MC contamination plot: contamination-per-bin for {ML, RICH-derived} with combined error bars, acceptance boundary shaded, excluded bins greyed out. This is the primary validation figure for the report and the poster.

*Thread 2 — refine the secondary validation (π-mis-ID-as-K on the separate exclusive-pion channel `ep → eπ⁺(n)`).*
- Take the W8 first-pass π-mis-ID reading as the starting point. Refine the exclusive-neutron mass-window definition on `M_X(eπ⁺)`: quantify the residual non-pion background under the peak from MC (see W8 risk), and either tighten the window or apply a sideband subtraction to remove that background from the pion-truth sample. Then re-run the trained BDT on the cleaned pion sample and re-quote the per-(p, θ)-bin π-mis-ID-as-K rate.
- Full systematics on the π-mis-ID number: (a) vary the neutron mass-window boundaries and quote the drift; (b) if sideband subtraction is used, vary the sideband definition; (c) vary the per-bin BDT threshold by ± one grid step and re-quote (same convention as Thread 1); (d) MC statistical uncertainty on the reference π-mis-ID prediction; (e) sensitivity to Platt calibration (with vs without). Same discipline as Thread 1 — the number is only as defensible as its systematic band.
- Re-state the asymmetry that was flagged in W8 explicitly in the writeup: this channel constrains **π-mis-ID-as-K only**, not proton contamination, not K efficiency, and not the K→π direction. This asymmetry is what makes it a secondary rather than primary validation, and it is measured on a physically separate channel — the pion sample here has nothing to do with the K-selection channel's own missing-mass distribution. Do not let poster reviewers read the plot as a full validation of the ML result.
- Reconcile the π-mis-ID number from this channel with the RICH-derived kaon-contamination number from Thread 1 in the (p, θ) region where both are populated. The two are not directly the same quantity — the RICH number is `C^{π→K}` on the K-selection channel, and the exclusive-pion number is `M^{π→K}` on a pure-pion sample — but under the assumption that the classifier's π-mis-ID rate is a property of the classifier and not of the sample composition, the two constrain each other. If they are consistent under that assumption, that is the two-way cross-check the report leans on. If they disagree, that is a real result and joins the Thread-1 disagreement diagnosis rather than being buried.
- Produce the final π-mis-ID plot: per-(p, θ)-bin π-mis-ID-as-K rate on the exclusive-pion sample, with MC-predicted rate overlaid and combined error bars. Save as `/figures/pi_misid_epiN_final.png`.

*Thread 3 — finalize every poster plot.*
- Enumerate every plot destined for the CEU (or equivalent) poster in a checklist at the top of `notes/cooper_week9_poster_plots.md`. Candidates at minimum: the headline contamination-vs-(p, θ) figure (ML vs baseline on the locked model), the RICH-vs-MC comparison from Thread 1, the exclusive-pion π-mis-ID-as-K plot from Thread 2, the tier-comparison summary from W5, the per-bin FOM-threshold sweep example, and any feature-importance or reliability figure Maria wants on the poster.
- For each plot, run the finalization pass: consistent axis label conventions (units, symbols), consistent legend placement, consistent color mapping across figures (baseline is always one color, ML always another, RICH always a third — pick once, apply everywhere), font size legible at poster print scale (rule of thumb: axis labels ≥ 14 pt after scaling), and a size / aspect that fits the poster grid Maria specifies. Save the final versions under `/figures/poster/` with names that make the poster-slot assignment obvious.
- Print-test the poster-scale figures at least once mid-week on the largest display Cooper has access to. Screen-shrunk PNGs that read fine on a laptop routinely fall apart at poster scale — catch this Wednesday, not the night before submission.

**Maria's tasks.**
- Reconfirm the RICH-acceptance boundary at Monday's start, before Cooper draws Thread-1 systematic bands on a moving target.
- Sign off on the RICH-vs-MC disagreement diagnosis (Thread 1) mid-week before Cooper commits to the final systematic band. Bins classified as "disagrees, source unclear" need Maria's read before they are quoted in the report.
- Confirm the poster plot list (Thread 3) at the start of the week so Cooper is not finalizing plots that end up cut from the layout.
- Review the finalized poster plots by end-of-day Friday. Comments on style / labeling come Friday, not the following week — Week 10 is polish and submission, not another round of plot revision.

**Done when.**
- `notes/cooper_week9_rich_diagnosis.md` written, with per-bin agreement / disagreement classification and named sources for every disagreeing bin.
- Final RICH-vs-MC contamination plot in `/figures/rich_vs_mc_final.png` with acceptance shaded, excluded bins greyed, combined error bars.
- Final exclusive-pion π-mis-ID-as-K plot in `/figures/pi_misid_epiN_final.png` with MC-predicted rate overlaid and combined error bars.
- Reconciliation between the RICH-derived kaon-contamination number and the exclusive-pion π-mis-ID-as-K number documented in the diagnosis note, with the asymmetry-of-quantities caveat spelled out.
- `notes/cooper_week9_poster_plots.md` checklist complete; every listed plot has a finalized version under `/figures/poster/`.
- Mid-week print-test of poster-scale figures performed and any layout / legibility issues logged.

**Risks / dependencies.**
- RICH-vs-MC disagreement is large in more than a couple of bins and the sources are unclear. Severity M–H. Mitigation: this is the same honesty rule as W8 — the disagreement is a finding, not a failure. If diagnosis stalls, quote the disagreement as a systematic uncertainty on the ML result rather than manufacturing a false agreement. Maria decides mid-week whether unresolved disagreement gets its own section in the report.
- Systematic uncertainty treatment on the ML side balloons into a Week-long rabbit hole. Severity M. Mitigation: the five sources listed in Thread 1 are the scope; additional sources are noted as future work in the writeup rather than pursued this week. If (p, θ) reweighting was never resolved upstream, quote the sensitivity band and stop — do not turn W9 into a reweighting rescue.
- Poster plots proliferate beyond what fits the layout. Severity L. Mitigation: Maria's Monday confirmation of the plot list is the scope gate. New plots after Monday require an explicit swap-out, not an addition.
- The exclusive-pion π-mis-ID cross-check produces a number inconsistent with the RICH cross-check in the shared (p, θ) region under the "classifier π-mis-ID is a sample-independent property" assumption. Severity M. Mitigation: this joins the Thread-1 disagreement diagnosis and is documented, not resolved by picking one number over the other. If the disagreement is systematic, that assumption itself may be wrong — flag it explicitly rather than averaging past it.

**Fallback / scope-down.** If Thread 1 systematics stall, ship the RICH-vs-MC plot with the statistical-only error bars and a documented list of un-quantified systematics; the primary-validation claim degrades to "consistent within stats" rather than "consistent within full systematics." If Thread 3 slips, prioritize the headline contamination plot, the RICH-vs-MC plot, and the exclusive-pion π-mis-ID plot; everything else on the poster can be finalized early Week 10. Do not drop either the RICH or the exclusive-pion cross-check — they are the primary and secondary validations of the whole result, not scope-down candidates.

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
| D3 | Hyperon-tagged kaon-truth channel (e' K+ Lambda, e' K+ Sigma0, both, or skip) | Week 3 | Maria | Off critical path: W8/W9 validation is RICH-overlap on `eKpX` (primary) + exclusive-pion π-mis-ID on `eπ⁺(n)` (secondary). If used, hyperon-tag is a supplementary sanity check only; skip is defensible. |
| D4 | Extend `processing_three_particles` with ToF/calo per-hadron features, or postpone | Week 3 | Maria | Strength of per-hadron PID features in analysis-channel ntuple; Week 5 evaluation |
| D5 | (p, theta) bin edges | Week 2 | Maria | All contamination tables and plots |
| D6 | Final model family (BDT / MLP) and formulation (binary vs 3-class) for the production result | Week 8 | Maria + Cooper | Pipeline freeze, W8 Part 3 data application, W9 validation |
| D7 | Calibration method (Platt, isotonic, or both) for the production result | Week 8 | Maria | Final model deliverable; folded into the W8 Part 2 sign-off with D6 |

---

## 6. Risk register

| # | Risk | Sev | Mitigation | Fallback |
|---|---|---|---|---|
| R1 | `RICH::Particle` bank empty on clasdis MC | H | Cooper verifies Day 2 with `hipo-utils -dump`. If empty, RICH cross-check moves from MC to data-only; the 14 RICH variables are not in the MC training ntuple (no big loss, they're not in features anyway). | Drop RICH as a sanity comparison on MC. RICH remains the primary data-side validation in Week 8 (first pass in the RICH-acceptance region) and Week 9 (refined with full systematic band). |
| R2 | Cooper's ifarm access delayed | H | Maria pushes JLab IT pre-arrival. Backup laptop environment with local copy of one HIPO file for offline work. | Cooper does Python and reading work in Week 1, full pipeline catches up Week 2. |
| R3 | Training MC statistics insufficient at high p (>3 GeV) where contamination is worst | M-H | Use full `clasdis` inbending sample. Quote MC stat uncertainty per bin. | If high-p bins are still stat-limited, merge bins. Honest binning at the cost of resolution. |
| R4 | Missing-mass method and MC-truth disagree beyond systematics | M | Investigate honestly: does the data have a contamination the MC doesn't capture? Is the sideband subtraction biased? | The disagreement is a finding. Report it. Quote it as a systematic uncertainty rather than papering over it. |
| R5 | MLP underperforms or takes too long | M | Keep architecture small (2 × 64). If still too slow, use a GPU node or sklearn MLPClassifier. | Drop MLP. BDT alone is enough for the report. |
| R6 | `processing_three_particles` extension for per-hadron ToF/calo not done in time | M | Decide D4 in Week 3. If postponed, use existing 3-particle ntuple without ToF/calo for analysis-channel evaluation; apply ML using the training-MC features only (which are per-track, indexed by REC::Particle row). | Do the analysis using only the training-ntuple features; analysis-channel ntuple becomes an event-selection helper only, not a per-hadron PID input. |
| R7 | Cooper underwater on Python in first 2 weeks | M | Pair-code Day 4-5. Sklearn reading is non-negotiable Week 2-3. Maria checks in mid-week. | Reduce Week 4-5 model-training ambition; lean on Maria for sklearn pipeline construction in Week 4. |

---

## 7. Scope-down list

If behind schedule at Week 5, drop in this order:

1. **Drop MLP.** Keep BDT only. (Saves ~2–3 days in Week 8; the final model decision collapses to BDT-only across formulations.)
2. **Drop isotonic calibration**, keep only Platt (sigmoid) calibration. (Saves ~1 day.)
3. **Drop per-(p, theta)-bin threshold optimization.** Use a single global threshold chosen on the validation set. (Saves ~2 days in Week 5.)
4. **Drop the analysis-channel ntuple extension for ToF/calo per-hadron variables** if not done by Week 7. Use existing `processing_mc_three_particles.groovy` output without ToF/calo per-hadron features. Apply the ML classifier to candidate K+ tracks using their kinematics + a join back to the training ntuple's per-track features.
5. **Drop the hyperon-tagged kaon-truth cross-channel** (whichever was picked in D3). Rely on MC truth for the kaon-truth side and on the RICH-overlap cross-check on `eKpX` for the data-side kaon-contamination validation; the exclusive-pion `eπ⁺(n)` channel remains as the π-mis-ID secondary.
6. **Drop feature ablation studies.** Keep hyperparameter tuning only. Feature importance plot alone is sufficient for the report.
7. **Collapse the W7/W8 model matrix to binary K⁺ vs non-K⁺ only.** Drop the 3-class formulation (K⁺/π⁺/p) from both the Week-7 BDT training and the Week-8 MLP training. Keeps the binary current-p-range and binary full-p-range models; drops the two multiclass models. (Saves ~1–2 days in Week 7 and ~1 day in Week 8.) Do **not** drop the RICH cross-check — it is the primary data-side validation in W8/W9 and cannot be scoped down without gutting the result.

Do not drop: the headline improvement number, the report, the poster. These are the contractual SULI deliverables.

---

## Appendix: notes on conventions and gotchas

- **FD-only.** All training and evaluation done with `status` in [2000, 4000). Verified by `generic_tests.forward_detector_cut`. CD tracks excluded from this entire project.
- **chi2pid as feature, not cut.** The training script does NOT cut on chi2pid. The baseline DOES apply the pass-2 momentum-dependent chi2pid cut (`passes_kplus_chi2pid_cut` in `scripts/baseline_chi2pid.py`); the older loose form `|chi2pid| < 3` is not the production baseline. The ML uses chi2pid as one input feature among many. This is the central methodological point: we are using ML to learn a better cut than the standard one, not to replace chi2pid.
- **MC truth matching.** Geometric: `|delta phi| < 6°, |delta theta| < 2°`. Same as Connor. Tracks with no MC match within this window are dropped from the training set (they cannot have a truth label).
- **Energy-loss and momentum corrections.** Applied at the groovy level via `analysis_fitter` infrastructure for electrons; not applied for K+ (Hayward's code has no kaon corrections; this is a known limitation common to cut-based and ML approaches equally).
- **Reproducibility.** All sklearn fits with `random_state=42`. All split file lists committed to repo. All slurm submission scripts committed. README must explain how to go from clean checkout to trained model in one command.
- **Honesty.** If a measurement disagrees with another, say so. The missing-mass method and MC-truth are independent by construction: if they disagree, that is part of the result, not a problem to paper over.
