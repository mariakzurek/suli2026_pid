# Feature audit — species K+ (pid=321)

Generated: 2026-11-18 00:52:32

## Input files
  MC   : /Users/zurek/CLAS/SULI/pid_training_v2.root
  Data : /Users/zurek/CLAS/SULI/data_pid_training_test.root

## Entry counts
  MC   loaded   : 2,906,632
  Data loaded   : 3,935,400
  MC   selected : 162,763  (after all cuts)
  Data selected : 193,930  (after all cuts)

## Selections applied
  MC   : (pid == 321) & (mc_matching_pid != -9999)
  Data : pid == 321
  Truth mode: matched
  Vertex-z cut: -8.0 < vz < 2.0 cm
  Q2 cut: not applied
  W cut: not applied
  y cut: not applied
  Mx cut: not applied

## Audit configuration
  Variables audited : beta, chi2pid, ftof_energy_1A, ftof_energy_1B, ftof_time_1A, ftof_time_1B, ftof_path_1A, ftof_path_1B, ecin_energy, ecout_energy, ecin_time, ecout_time, ecin_path, ecout_path, nphe_htcc, nphe_ltcc, pcal_energy, pcal_time, pcal_path, ftof_energy_2, ftof_time_2, ftof_path_2, p, theta, phi, vz, sector
  p bins (GeV/c)    : [1.0,2.0], [2.0,3.0], [3.0,5.0]
  theta bins (deg)  : [5.0,15.0], [15.0,25.0], [25.0,35.0]
  Histogram bins    : 50
  Normalization     : unit-area (count / (total × bin_width))

## Negative-time exclusion (ecin_time, ecout_time, pcal_time)
  --exclude-negative-time: ENABLED
  After sentinel removal (-9999), entries below 0 are removed for
  ecin_time, ecout_time, and pcal_time only.  These are non-physical
  timing values distinct from the -9999 no-hit sentinel.
  Histogram lower bound: 1st percentile (np.percentile(combined, 1.0))
  of the combined positive-time MC+data distribution, computed after
  sentinel and negative-value removal (never includes non-positive values).
  Histogram upper bound: raw max of combined positive-time array;
  fallback to 99th percentile if max > 3×p99 (outlier guard).
  time_range_upper_mode in CSV: 'max' or 'percentile_99_fallback'.
  neg_time_frac_removed_mc/data in CSV: fraction of post-sentinel
  valid-hit entries removed by this step (distinct from hit_frac_*).
  NaN for all non-timing variables.

## Column meanings
  See COLUMNS.md in this directory for the per-column glossary.
  See scripts/README.md (compare_mc_data.py Output section) for full metric definitions.
  See notes/cooper_10week_plan.md Task 3a for the full audit workflow.
