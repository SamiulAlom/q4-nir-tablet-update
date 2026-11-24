| file | notebook | cell | description |
|---|---|---|---|
| p4_splits.json | 01_audit_splits_CPU | splits | Frozen splits: Scenario I (200 repetitions, cross-validation folds), Scenario II (243 repetitions, group folds), pseudo-transfer splits and folds inside the source condition; holds the digest of the arrays it indexes |
| p4_shootout_parts.csv | 01_audit_splits_CPU | shootout | Shootout: tablets and assay summary in the organisers' three parts |
| p4_shootout_alignment.csv | 01_audit_splits_CPU | alignment | Shootout: spectra-only check of the row pairing of the two instruments per part, against two wrong pairings, and the pre-set rule's result |
| p4_shootout_alignment_suspects.csv | 01_audit_splits_CPU | alignment | Shootout: tablets whose partner spectrum is outside the 5 most similar rows in both directions (listed, not removed) |
| p4_copenhagen_groups.csv | 01_audit_splits_CPU | copenhagen | Copenhagen set: the 31 tablet groups with type, scale code and content range |
| p4_copenhagen_scales.csv | 01_audit_splits_CPU | copenhagen | Copenhagen set: groups, tablets and content range per scale code; tablets outside the source-condition range (scale codes 0 and 1) |
| p4_splits_summary.csv | 01_audit_splits_CPU | splits | Counts of the frozen splits, the pairing rule's result, and the hashes that bind splits and arrays |
| p4_input_hashes.csv | 01_audit_splits_CPU | arrays | SHA-256 of the two archives, of the data files read from them, and the digest of the arrays passed to the later notebooks |
| p4_versions_01.csv | 01_audit_splits_CPU | versions | Package versions of the notebook 01 run |
| p4_tuning_cv.csv | 02_tune_source_GPU | tuning | Source-only cross-validation: mean RMSE of every candidate (preprocessing and setting) per source set and family, full precision, with the chosen row marked |
| p4_tuning_folds.csv | 02_tune_source_GPU | tuning | Source-only cross-validation, every fold: RMSE per candidate, repeat, fold and seed, with the trees or epochs kept by early stopping; all PLS component counts |
| p4_cnn_checkpoints.csv | 02_tune_source_GPU | source CNNs | Source CNN checkpoints (three seeds per source set): file, hash, epochs, RMSE on their own training tablets |
| p4_tuning_seconds.csv | 02_tune_source_GPU | tuning | Wall-clock seconds of the tuning per source set and family on this PC |
| p4_versions_02.csv | 02_tune_source_GPU | versions | Package versions, GPU, driver, CUDA and cuDNN settings of the notebook 02 run |
| p4_tuning_chosen.json | 02_tune_source_GPU | tuning | Chosen preprocessing and settings per source set and family, the number of candidates searched, the pinned foundation-model weights, and hashes of code, splits, arrays and CNN checkpoints |
| p4_complete_02.json | 02_tune_source_GPU | outputs | Completion marker of notebook 02: run fingerprint and the hash of each of its output files |
| p4_curves_summary.csv | 03_update_curves_GPU | scores | RMSE and bias per part of the design, repetition, family, CNN seed, update, setting and update size n, computed from the saved 32-bit predictions |
| p4_pseudo_transfer.csv | 03_update_curves_GPU | pseudo-transfer | Pseudo-transfer inside the source scales: mean RMSE at n = 8 per direction, update method and setting (the table behind p4_update_settings.json) |
| p4_bench_s1.csv | 03_update_curves_GPU | Scenario I | Scenario I benchmark of deviation 1: each source model's prediction for all 460 test-part tablets measured on the source instrument |
| p4_curves_seconds.csv | 03_update_curves_GPU | timing | Wall-clock seconds per part file (repetition or choice of pool groups) and model family |
| p4_curves_timing.json | 03_update_curves_GPU | timing | Estimated hours of the full design from the mean seconds per part (input of the cost rule) |
| p4_versions_03.csv | 03_update_curves_GPU | versions | Package versions, GPU, design size and run fingerprint of the notebook 03 run |
| p4_pred_pseudo_01.npz | 03_update_curves_GPU | predictions | Per-tablet predictions of part pseudo_01: pred[repetition, method label, evaluation tablet] (32-bit), labels made of family, update, setting and n, eval_rows (row numbers in the arrays), y_true, rep |
| p4_pred_pseudo_10.npz | 03_update_curves_GPU | predictions | Per-tablet predictions of part pseudo_10: pred[repetition, method label, evaluation tablet] (32-bit), labels made of family, update, setting and n, eval_rows (row numbers in the arrays), y_true, rep |
| p4_pred_s2.npz | 03_update_curves_GPU | predictions | Per-tablet predictions of part s2: pred[repetition, method label, evaluation tablet] (32-bit), labels made of family, update, setting and n, eval_rows (row numbers in the arrays), y_true, rep |
| p4_pred_s1_12.npz | 03_update_curves_GPU | predictions | Per-tablet predictions of part s1_12: pred[repetition, method label, evaluation tablet] (32-bit), labels made of family, update, setting and n, eval_rows (row numbers in the arrays), y_true, rep |
| p4_pred_s1_21.npz | 03_update_curves_GPU | predictions | Per-tablet predictions of part s1_21: pred[repetition, method label, evaluation tablet] (32-bit), labels made of family, update, setting and n, eval_rows (row numbers in the arrays), y_true, rep |
| p4_update_settings.json | 03_update_curves_GPU | pseudo-transfer | Update settings: the pseudo-transfer choice for Scenario II with its table and provenance, and the main (middle) settings of Scenario I |
| p4_complete_03.json | 03_update_curves_GPU | outputs | Completion marker of notebook 03: run fingerprint, design size and the hash of each of its output files |
| p4_tab_curves.csv | 04_statistics_figures_CPU | tables | RMSE per method, setting and n over repetitions: median, 2.5th and 97.5th percentiles, median bias; CNN seeds averaged per repetition; main_setting marks the settings shown in the paper |
| p4_tab_opening.csv | 04_statistics_figures_CPU | tables | Opening table: source cross-validation RMSE (optimistic; Scenario II group-held-out), benchmark (Scenario I: source model on the 460 test-part tablets on the source instrument), level (b), and the no-update RMSE over repetitions |
| p4_tab_compare.csv | 04_statistics_figures_CPU | tables | Paired RMSE ratios per repetition: each method against PLS with slope and bias at the same n, and each family's own update against its slope and bias; median, percentiles, share of repetitions lower |
| p4_tab_primary.csv | 04_statistics_figures_CPU | tables | Pre-specified primary comparison: instrument 1 to 2, n = 10, each family against PLS (all with slope and bias), with the other methods shown beside it |
| p4_tab_budget.csv | 04_statistics_figures_CPU | tables | Recovery budget: median ratio of RMSE to the family's benchmark at every n, the first n at most 1.10 (or not reached), and the reversal flag; every setting |
| p4_tab_s2_range.csv | 04_statistics_figures_CPU | tables | Scenario II: RMSE over repetitions for evaluation tablets inside the source content range and above its largest value, with tablet counts |
| p4_tab_s2_groups.csv | 04_statistics_figures_CPU | tables | Scenario II: RMSE per evaluation group over the repetitions in which the group is evaluated |
| p4_tab_s2_logo.csv | 04_statistics_figures_CPU | tables | Scenario II: median paired ratio against PLS with slope and bias, and its range when one evaluation group at a time is left out |
| p4_tab_diff_regions.csv | 04_statistics_figures_CPU | tables | Difference spectra (differences of means over the same rows, no tablet pairing) summarised in five equal-width regions |
| p4_fig_curves.pdf | 04_statistics_figures_CPU | figures | Figure: RMSE against n, instrument 1 to 2 and scale codes 0+1 to 2; medians with 2.5th-97.5th percentile bands over repetitions |
| p4_fig_curves.png | 04_statistics_figures_CPU | figures | Same figure, 600 dpi |
| p4_fig_curves_s1_21.pdf | 04_statistics_figures_CPU | figures | Figure: RMSE against n, instrument 2 to 1 (secondary direction) |
| p4_fig_curves_s1_21.png | 04_statistics_figures_CPU | figures | Same figure, 600 dpi |
| p4_fig_diff_spectra.pdf | 04_statistics_figures_CPU | figures | Figure: mean and first-derivative difference spectra between instruments and between scale codes |
| p4_fig_diff_spectra.png | 04_statistics_figures_CPU | figures | Same figure, 600 dpi |
| p4_versions_04.csv | 04_statistics_figures_CPU | versions | Package versions of the notebook 04 run |
| p4_complete_04.json | 04_statistics_figures_CPU | outputs | Completion marker of notebook 04: input marker hashes and the hash of each output file |
| p4_sens_pairs_ratios.csv | 05_sensitivity_pairs_CPU | sensitivity | Sensitivity (plan section 8, first item; run after the results, deviation 4): paired RMSE ratios at n = 10 (U1) and n = 0 (U0), instrument 1 to 2, PLS and boosted trees on the 325 pair-averaged features against TabPFN and the 650-feature models |
| p4_sens_pairs_rmse.csv | 05_sensitivity_pairs_CPU | sensitivity | Sensitivity: RMSE over the 100 repetitions per method and n, PLS and boosted trees on 325 pair-averaged features (U0, U1), with the notebook 03 methods compared |
| p4_sens_pairs_tuning.csv | 05_sensitivity_pairs_CPU | sensitivity | Sensitivity: source-only tuning on 325 pair-averaged instrument-1 features (notebook 02's folds, grids and rule); CV RMSE is optimistic |
| p4_complete_05.json | 05_sensitivity_pairs_CPU | outputs | Completion marker of notebook 05: input hashes and the hash of each output file |
