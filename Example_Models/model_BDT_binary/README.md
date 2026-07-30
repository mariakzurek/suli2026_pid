# BDT training run provenance
Generated: 2026-07-13T20:34:37.359603Z

## Dataset
- Source: `/work/clas12/CooperBe/MLStuff/dataset_v03`
- Features (8): beta, ftof_energy_1B, ftof_time_1B, ftof_path_1B, chi2pid, ecin_path, ecin_energy, ecin_time
- Reweight map: None (unweighted)

## Hyperparameters
- `n_estimators`: 200
- `learning_rate`: 0.05
- `max_depth`: 6
- `objective`: binary
- `random_state`: 42
- `n_jobs`: -1
- `verbose`: 1
- calibration_frac: 0.2

## Metrics (validation set)
- auc_train_uncal: 0.90279
- auc_val_uncal: 0.90274
- auc_train_cal: 0.90279
- auc_val_cal: 0.90274
- brier_val_uncal: 0.12118
- brier_val_cal: 0.12288
- logloss_val_uncal: 0.38013
- logloss_val_cal: 0.38989
- n_train: 21302845
- n_cal: 4260569
- n_fit: 17042276
- n_val: 4544388
- k_frac_train: 0.62283
- k_frac_val: 0.62280

## Outputs
- `model.joblib` — wrapper dict {"model": calibrated LightGBM + Platt calibrator, "features": list of training feature names}
- `training_summary.csv` — AUC, Brier, log-loss for train/val pre/post cal
- `reliability_diagram.png` — calibration quality (on val set)
- `roc_val.png` — ROC curve on val set
- `feature_importance.png` / `.csv` — top-15 features by gain
