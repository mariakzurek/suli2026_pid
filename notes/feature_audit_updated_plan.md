# Updated Event-Builder K⁺ Feature Audit Plan

## Purpose and approach

Rerun the Event Builder K⁺ data/MC feature audit into a new, isolated
`figures/feature_audit_updated/` tree. The only analysis change is an opt-in
negative-time exclusion for `ecin_time` and `pcal_time`: after normal
sentinels are removed, remove genuinely negative timing values, use a
zero-anchored robust upper histogram bound, and recompute the existing shape
metrics on that filtered population. Existing detector-hit fractions remain a
separate sentinel/no-hit diagnostic and must not be reinterpreted as the
negative-time removal rate.

The generic engine receives only a reusable configurable mechanism; the SULI
CLI owns enabling it for these runs and recording the run provenance. This
preserves the no-flag behavior and all established drift thresholds.

## Requirements Summary

- Produce inclusive and SIDIS K⁺ audits with all default audit and kinematic
  variables, a 3×3 `(p, theta)` grid, 50 bins, matched truth mode, and the
  negative-time option enabled.
- Use the corrected ROOT inputs at `/Users/zurek/CLAS/SULI/pid_training_v2.root`
  and `/Users/zurek/CLAS/SULI/data_pid_training_test.root`; do not use the
  empty `files/` directory.
- Add per-cell MC/data fractions of post-sentinel timing entries excluded for
  negative time, separate from existing global and per-cell hit fractions.
- Produce new CSV/PNG/README/COLUMNS outputs, timing before/after comparison
  CSVs, and one evidence-based Markdown report.
- Leave frozen and unrelated audit output trees unchanged; do not alter the
  KEEP/CANDIDATE/DROP logic or treatment of variables other than the two
  specified timing variables.

## Verified implementation touchpoints

The following locations were verified against the current working copy:

- `scripts/compare_mc_data.py`: constants at lines 78–133;
  `classify_drift` at line 789; `_strip_sentinels` at 890;
  `compare_distribution` at 919 (selection through range selection at
  1069–1149, histogram normalization at 1151–1185, statistics at 1187–1240,
  and annotation at 1276–1314); `_empty_stats` at 1352; and
  `run_feature_audit` at 1395–1522.
- `scripts/audit_species.py`: `_write_audit_readme` at 319–354; parser at
  380–462; selected-data loading/cuts at 598–713; engine invocation at
  718–726; CSV/README/glossary handling at 729–751.
- `scripts/README.md`: CSV output glossary at 95–110 and unchanged drift
  decision thresholds at 44–59 and 130–136.

The current glossary-copy source path,
`figures/feature_audit/COLUMNS.md`, does not exist in this checkout. Existing
copies reside in frozen output trees such as
`figures/feature_audit_week3/COLUMNS.md`, which must not become an editable
source or be modified. Establish a maintained canonical glossary under
`scripts/` and copy from there instead.

## Assumptions and invariants

- Both ROOT files use tree `PhysicsEvents`. Their duplicate TTree cycles are
  intentional; uproot's bare `PhysicsEvents` selection resolves to the latest
  cycle, as the current loader already does. No cycle-selection change is
  needed.
- Full production input entry counts are 2,906,632 MC and 3,935,400 data;
  record these with the paths in each production README. Also record the
  post-selection counts actually used in the audit.
- `-9999` is the timing no-hit sentinel. Existing `hit_frac_*` fields remain
  based on this sentinel definition (and the existing `chi2pid` exception),
  before negative-time filtering.
- The default audit currently uses 50 bins and independent unit-integral
  normalization (`count / (in-range sample total × bin width)`) for MC and
  data. Filtering before histogramming will preserve this convention without
  modifying the normalization algorithm.
- The production commands run from the repository root. They intentionally
  pass `--outdir`, so automatic SIDIS output suffixing is irrelevant.

## Files to modify or create

| Path | Change |
|---|---|
| `scripts/compare_mc_data.py` | Add the generic opt-in timing filter, fraction statistics, documentation, and parameter plumbing. |
| `scripts/audit_species.py` | Add the SULI CLI option, forward it, record complete run provenance, and copy the canonical glossary. |
| `scripts/README.md` | Document the option and the added CSV fields without changing threshold guidance. |
| `scripts/COLUMNS.md` **(new)** | Canonical generated-output glossary, including the new timing-filter fields and their distinction from hit fractions. |
| `scripts/compare_timing_before_after.py` **(new)** | Reusable, documented CSV joiner for week-3 versus updated timing results. |
| `figures/feature_audit_updated/kp/…` **(generated)** | Inclusive audit outputs and `timing_comparison.csv`. |
| `figures/feature_audit_updated/kp_sidis/…` **(generated)** | SIDIS audit outputs and `timing_comparison.csv`. |
| `figures/feature_audit_updated/REPORT.md` **(generated after review)** | CSV-grounded summary of both production runs. |

## Open Questions

1. **Timing upper percentile — decision required before implementation.** Use
   the 99th percentile of the combined, post-sentinel, nonnegative MC+data
   timing values as the upper bound (recommended). It matches the engine's
   established robust range convention and prevents the known ECIN/PCAL raw
   outlier tails from determining the plot scale. The editor must name the
   chosen percentile in a code comment, output README, and this plan's
   completion evidence; do not silently replace it with a raw maximum.
2. **Non-applicable fraction convention — resolved recommendation.** Store
   `NaN` in `neg_time_frac_removed_mc` and `_data` when the flag is off or the
   variable is not in `TIME_VARIABLES`, rather than `0.0`. `NaN` truthfully
   means “filter was not applicable/not enabled,” while a real zero means the
   enabled filter inspected valid timing entries and found none negative.
   Apply the same `NaN` default in `_empty_stats`.
3. **Histogram-bound auditability — implementation decision.** The minimum
   required schema additions are the two removed-fraction columns. To make
   validation of the exact zero lower edge machine-checkable, the editor may
   additionally emit `hist_range_lo` and `hist_range_hi` for every successful
   row and document them in both glossaries. If avoiding those two broader
   schema fields, validation must instead explicitly inspect the histogram
   edges during the test run and preserve that evidence in the execution log.

## Todo

1. [ ] **Approve the two decisions above, then assign implementation to `editor`.**
   Record 99th percentile as the selected timing upper-bound rule unless the
   user directs otherwise. Keep the scope to this rerun; do not refactor the
   generic engine beyond the parameterized mechanism below.

2. [ ] **Add the generic timing-filter configuration in `scripts/compare_mc_data.py`.**
   Define module-level `TIME_VARIABLES = ["ecin_time", "pcal_time"]` adjacent
   to the feature/constants block, with a comment that it is only consulted
   when the caller explicitly enables exclusion. Add
   `exclude_negative_time: bool = False` to `compare_distribution` and
   `run_feature_audit`, and pass it through the latter’s per-cell call.
   Default `False` is mandatory for backward compatibility.

3. [ ] **Implement filtering and accounting at the correct point.** In
   `compare_distribution`, retain all current sentinel and `chi2pid`
   hit-fraction calculations unchanged. Immediately after `_strip_sentinels`,
   when the new flag is true and `variable in TIME_VARIABLES`, calculate for
   each sample:
   `count(arr < 0) / len(arr)` using the post-sentinel array, with a defined
   empty-array result of `NaN`; then replace the array by `arr[arr >= 0]`.
   Do not apply this path to any other variable. Add the two values to the
   successful stats dict and matching `NaN` values to `_empty_stats` so every
   CSV row has a stable schema.

4. [ ] **Apply the timing-specific range rule without changing other ranges.**
   For enabled timing filtering only, set the histogram lower bound to exactly
   `0.0` and set its upper bound to the approved combined nonnegative-value
   percentile. Include a teaching-style code comment explaining that raw
   extrema are unusable because of nonphysical timing outliers. Preserve the
   present 1st/99th-percentile range logic for every other case, including a
   timing variable when the flag is absent. Handle an empty filtered combined
   array and equal/degenerate percentile safely via the existing empty/range
   guards. Keep 50 bins from the caller and let existing per-sample
   normalization, KS/chi2/Wasserstein/PSI/max-local-residual, and
   `classify_drift` operate on the filtered arrays unchanged.

5. [ ] **Make plot and API documentation internally consistent.** Extend the
   `compare_distribution` docstring to define the flag, fractions, and the
   fact that `n_mc`/`n_data` for enabled timing variables are post-negative
   filter histogram populations while `n_hit_*` remain sentinel-only counts.
   Add a compact annotation line for filtered timing plots showing the MC/data
   removed fractions (or another clear equivalent), so visual plots agree
   with their CSV rows. Do not add an annotation or alter a plot for
   non-timing variables or disabled runs.

6. [ ] **Add and forward the CLI switch in `scripts/audit_species.py`.** Add
   `--exclude-negative-time` as `store_true`, defaulting to false, with help
   text limited to ECIN/PCAL timing and explaining that it removes
   non-sentinel values below zero. Print the resolved setting in the run
   preamble and forward it to `run_feature_audit`. Do not change the defaults
   for truth mode, bin count, normalization, VZ, or SIDIS cuts.

7. [ ] **Extend per-output provenance README generation.** Pass the resolved
   option plus production metadata into `_write_audit_readme`. Each README
   must state: absolute input paths; full input entry counts; loaded and final
   selected MC/data counts; species and MC/data selection expressions; truth
   mode; all applied cuts (including VZ and exact SIDIS values
   `Q2 > 2.0 GeV²`, `W > 2.0 GeV`, `0.0 < y < 0.75`, and
   `Mx_eKX > 1.6 GeV` for K⁺); all-default variable scope; the unchanged
   three-by-three p/theta edges; 50 bins; independent unit-area normalization;
   the enabled negative-time rule, fraction definition, and percentile
   rationale. State no SIDIS event-level cuts for the inclusive run. Capture
   counts from the actual loaded/selected DataFrames (and clearly label a
   limited `--max-rows` validation run if one is documented).

8. [ ] **Repair the glossary source without modifying frozen outputs.** Create
   `scripts/COLUMNS.md` as the maintained canonical glossary, based on the
   useful existing structure but updated for the new columns. Change the
   driver’s copy source to `scripts/COLUMNS.md`, so every new output directory
   receives a matching `COLUMNS.md`. Document
   `neg_time_frac_removed_mc` and `_data` as per-cell fractions among
   post-sentinel valid timing entries, populated only with the enabled timing
   filter and otherwise `NaN`; explicitly distinguish them from global and
   cell `hit_frac_*` sentinel/no-hit metrics. If histogram-bound fields are
   adopted, document them too. Update `scripts/README.md`’s CSV table and CLI
   documentation with the same definitions. Do not edit any
   `figures/feature_audit_week3/**/COLUMNS.md` file.

9. [ ] **Create `scripts/compare_timing_before_after.py` as reusable
   post-processing.** Give it the project teaching-document docstring
   (`WHAT IT DOES`, `WHEN TO USE IT`, `PITFALLS`) and argparse inputs for an
   original summary CSV, updated summary CSV, and output CSV. Restrict and
   validate to `ecin_time`/`pcal_time`; inner-join exactly by
   `variable,p_lo,p_hi,theta_lo,theta_hi`; require nine cells per variable in
   each input and fail clearly on duplicates/missing cells. Write one row per
   variable/cell with prefixed before/after key metrics
   (`ks_distance`, `chi2_per_ndof`, `wasserstein`, `wasserstein_norm`, `psi`,
   `max_local_residual`, `drift_decision`), counts, hit-fraction fields, and
   the updated negative-time fractions. Preserve original hit fractions beside
   updated values to demonstrate that hit availability is independent of the
   new filter. Do not fabricate before values for columns absent from week-3;
   leave such fields absent/`NaN` and document why.

10. [ ] **Run the fast pre-production validation twice in fresh temporary
    directories under `figures/feature_audit_updated/validation_*` (or an
    approved disposable path).** Use the real corrected absolute input paths,
    `--species kp --truth-mode matched --bins 50`, and a bounded
    `--max-rows` sample with `--vars ecin_time pcal_time beta`. First omit the
    new switch; then repeat the identical invocation with
    `--exclude-negative-time`. Example baseline command:

    ```bash
    python scripts/audit_species.py \
      --mc /Users/zurek/CLAS/SULI/pid_training_v2.root \
      --data /Users/zurek/CLAS/SULI/data_pid_training_test.root \
      --species kp --truth-mode matched --bins 50 --max-rows 100000 \
      --vars ecin_time pcal_time beta \
      --outdir figures/feature_audit_updated/validation_baseline
    ```

    Repeat with the same arguments, output directory
    `figures/feature_audit_updated/validation_negative_time`, and append
    `--exclude-negative-time`. The validation directory is disposable and is
    not a production deliverable.

11. [ ] **Evaluate validation pass/fail criteria before production.** Pass only
    if: (a) baseline output has `NaN` negative-time fractions and its existing
    numeric columns reproduce a separately repeated no-flag baseline to CSV
    formatting/floating tolerance; (b) every `beta` result is identical
    between baseline and flagged runs (apart from explicitly new, nonapplicable
    fields); (c) filtered ECIN/PCAL range begins at exactly 0.0 and uses the
    approved robust upper percentile; (d) their removal fractions are small,
    nonzero where populated and clearly unlike the roughly 45–55% sentinel
    no-hit fractions; (e) filtered timing `n_mc`/`n_data` equal post-sentinel
    counts minus removed negative entries; (f) each normalized plotted
    histogram integrates to approximately one via
    `sum(density * bin_width)` for MC and data; and (g) plot annotations show
    counts/fractions consistent with the CSV. Investigate and fix any failure
    before the full runs; do not paper over it by changing drift thresholds.

12. [ ] **Run the complete inclusive production audit after validation passes.**
    This command deliberately uses all default variables (`--vars` omitted),
    the default 3×3 grid, 50 bins, and explicit matched truth mode:

    ```bash
    python scripts/audit_species.py \
      --mc /Users/zurek/CLAS/SULI/pid_training_v2.root \
      --data /Users/zurek/CLAS/SULI/data_pid_training_test.root \
      --species kp --truth-mode matched --bins 50 \
      --exclude-negative-time \
      --outdir figures/feature_audit_updated/kp
    ```

    Verify the output has nine PNGs for each resolved default variable,
    `feature_audit_summary.csv`, `README.md`, and `COLUMNS.md`.

13. [ ] **Run the complete SIDIS diagnostic production audit after the inclusive
    run.**

    ```bash
    python scripts/audit_species.py \
      --mc /Users/zurek/CLAS/SULI/pid_training_v2.root \
      --data /Users/zurek/CLAS/SULI/data_pid_training_test.root \
      --species kp --truth-mode matched --bins 50 --sidis-cuts \
      --exclude-negative-time \
      --outdir figures/feature_audit_updated/kp_sidis
    ```

    Verify the README records the exact resolved SIDIS cuts and that this run
    has the same complete output structure. Do not limit variables, alter the
    grid, or reuse/overwrite a prior output directory.

14. [ ] **Generate and validate timing before/after artifacts.** Run the new
    utility separately for inclusive and SIDIS:

    ```bash
    python scripts/compare_timing_before_after.py \
      --before figures/feature_audit_week3/kp/feature_audit_summary.csv \
      --after figures/feature_audit_updated/kp/feature_audit_summary.csv \
      --output figures/feature_audit_updated/kp/timing_comparison.csv

    python scripts/compare_timing_before_after.py \
      --before figures/feature_audit_week3/kp_sidis/feature_audit_summary.csv \
      --after figures/feature_audit_updated/kp_sidis/feature_audit_summary.csv \
      --output figures/feature_audit_updated/kp_sidis/timing_comparison.csv
    ```

    Confirm each contains 18 keyed rows (two variables × nine cells), no
    duplicate keys, all required before/after metrics, and only updated-run
    removal fractions.

15. [ ] **Write `figures/feature_audit_updated/REPORT.md` only after both
    production summaries and timing-comparison CSVs exist.** Ground every
    conclusion in those files, citing cells/metrics rather than speculative
    prose. Cover both inclusive and SIDIS results: (1) best agreement
    variables, including all-cell KEEP/lowest-drift evidence; (2) largest
    discrepancies, DROP or persistent CANDIDATE cells; (3) ECIN/PCAL metric
    movement before versus after filtering and the removed fractions; (4)
    MC/data detector-hit availability differences from sentinel-based
    `hit_frac_*`, explicitly independent of the timing filter; (5) low-count
    cells based on `n_total_*_cell` and `n_hit_*_cell`, labelled as
    insufficient-statistics rather than physical disagreement with recommended
    follow-ups; and (6) the caveat that KEEP/CANDIDATE/DROP are diagnostic
    flags, not final feature-selection decisions.

16. [ ] **Perform final scope and preservation checks.** Confirm no HTML or
    browser artifacts were made; no file under `figures/feature_audit_week3/`,
    `figures/feature_audit/`, or `figures/feature_audit_NOCUT/` changed; no
    threshold/classification constants changed; and non-timing variables have
    no behavioral changes when the new option is omitted. Record the exact
    commands, validation outcome, output file counts, and any deviations in
    the execution handoff.

## Risks and mitigations

- **Outlier-dominated timing plots:** use the explicit combined positive-value
  percentile, never the raw maximum; validate the fixed zero lower edge.
- **Conflating absence of a hit with a negative measured time:** retain the
  existing sentinel-only hit fractions and add separately named removal
  fractions with unambiguous documentation.
- **Accidentally changing legacy analyses:** default the option off and compare
  no-flag validation outputs; write only to `feature_audit_updated` for this
  work.
- **Incomplete provenance:** pass actual raw/selected counts and resolved cuts
  to README generation rather than relying on generic prose.
- **Unreliable conclusions in sparse cells:** make low-statistics flags and
  follow-up recommendations a required report section rather than treating a
  drift flag as sufficient evidence.

## Acceptance criteria

The work is ready for review when the two approved production directories each
contain complete default-variable 3×3 PNG/CSV/README/glossary outputs; the
new fraction columns and documentation are consistent; both 18-row timing
comparison CSVs pass key validation; the final report is quantitatively based
on generated CSVs; validation passes all stated checks; and frozen/unrelated
outputs plus drift thresholds are unchanged.

## Non-goals

- No HTML, browser-rendered, notebook, or interactive artifacts.
- No edits to `figures/feature_audit_week3/`, `figures/feature_audit/`, or
  `figures/feature_audit_NOCUT/`.
- No change to KEEP/CANDIDATE/DROP thresholds, aggregation, or any metric
  implementation.
- No negative-time filtering or range-policy change for any variable other
  than `ecin_time` and `pcal_time` when the new flag is enabled.
- No unrelated audit-engine, ntuple-loader, selection, or training-pipeline
  refactor.
