# Feature audit — species K+ (pid=321)

Generated: 2026-06-09 14:02:47

## Selections applied
  MC   : (pid == 321) & (mc_matching_pid != -9999)
  Data : pid == 321
  Truth mode: matched
  Vertex-z cut: -8.0 < vz < 2.0 cm

## Input files
  MC   : /volatile/clas12/cooperb/SULI/pid_training_v2.root
  Data : /volatile/clas12/cooperb/SULI/data_pid_training_test.root

## Variables audited
  beta, chi2pid, ftof_energy_1A, ftof_energy_1B, ftof_time_1A, ftof_time_1B, ftof_path_1A, ftof_path_1B, ecin_energy, ecout_energy, ecin_time, ecout_time, ecin_path, ecout_path, nphe_htcc, pcal_energy, pcal_time, pcal_path, ftof_energy_2, ftof_time_2, ftof_path_2, p, theta, phi, vz, sector

## Column meanings
  See figures/feature_audit/COLUMNS.md for the per-column glossary.
  See scripts/README.md (compare_mc_data.py Output section) for full metric definitions.
  See notes/cooper_10week_plan.md Task 3a for the full audit workflow.
