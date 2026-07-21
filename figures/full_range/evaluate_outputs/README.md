# evaluate.py run provenance

Model: `/work/clas12/CooperBe/MLStuff/tier2All/model_v02/model.joblib`
Test set: `/work/clas12/CooperBe/MLStuff/dataset_v03/test.parquet`
p edges: [0.5, 1.4, 2.3, 3.2]
theta edges: [5.0, 15.0, 25.0, 35.0] deg
Threshold grid: 99 points [0.010, 0.990]
Low-stat threshold: n_label < 50

## Outputs
- `per_bin_sweep.csv` — eff_K, C_pi, C_p at each threshold for each bin
- `comparison_summary.csv` — matched-eff and matched-contam comparison
- `contam_vs_ptheta_baseline_vs_bdt.png` — headline heatmap (shared scale)
- `cp_to_K_map.png` — C^{p→K} at matched eff_K (Phase-4 input)

## Metric definitions
eff_K  = N(score>t & label==1) / N(label==1)
C_pi   = N(score>t & label==0) / N(score>t & label.notna())
C_p    = N(score>t & mc_matching_pid==2212) / N(score>t)

Baseline: passes_kplus_chi2pid_cut (scripts/baseline_chi2pid.py).
Matched-eff: BDT threshold where eff_K equals baseline eff_K; report BDT C_pi.
Matched-contam: BDT threshold where C_pi equals baseline C_pi; report BDT eff_K.
