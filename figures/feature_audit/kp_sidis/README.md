# Feature audit — species K+ (pid=321)

Generated: 2026-06-15 11:48:58

## Selections applied
  MC   : (pid == 321) & (mc_matching_pid != -9999)
  Data : pid == 321
  Truth mode: matched
  Vertex-z cut: -8.0 < vz < 2.0 cm
  SIDIS cuts: enabled (--sidis-cuts)
  Q2 cut: 2.0 < Q2 < inf GeV²
  W cut: 2.0 < W < inf GeV
  y cut: 0.0 < y < 0.75
  Mx cut: 1.6 < Mx_eKX < inf GeV

## Input files
  MC   : /volatile/clas12/zurek/SULI/mc_v01/nb-clasdis-Q2_1.5-10712_8.root
  Data : /volatile/clas12/cooperb/SULI/data_v01/nSidis_005036.root

## Variables audited
  beta, chi2pid, ftof_energy_1A, ftof_energy_1B

## Column meanings
  See figures/feature_audit/COLUMNS.md for the per-column glossary.
  See scripts/README.md (compare_mc_data.py Output section) for full metric definitions.
  See notes/cooper_10week_plan.md Task 3a for the full audit workflow.
