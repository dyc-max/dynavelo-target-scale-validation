# Scientific stage and artifact map

| Stage | Archived source entry | Frozen data / metadata |
|---|---|---|
| Training seed 0 | repo/colab/train_hspc_colab.py; colab_training.py | 01_model_weights, original config/history |
| Training seeds 1/2 | repo/pipeline/train_hspc_local_resume.py; train_hspc_local_seed2_recovery.py | 01_model_weights, per-seed checkpoint hashes |
| Input assembly | repo/pipeline/assemble_model_inputs.py and adapters | 05_discovery_inputs |
| Sampling and sensitivity | repo/colab/screen_plan.py; screen_scope_plan_runtime_v2.py; dynavelo_streaming_core.py | 02_primary_sensitivity; 03_full_population_readouts |
| Numerical readout | repo/a100_production_20261114/cpu_fixed28_exact13_group_readout_v1.py; cpu_all28_axis_descriptive_readout_v1.py | 03_full_population_readouts; 08_provenance |
| GRNBoost2 | repo/a100_production_20261114/orthogonal_grnboost2_pilot_v1.py; orthogonal_grnboost2_full_v1.py | 04_grnboost2_full; 05_discovery_inputs |
| Column normalization and rank concordance | repo/local_reports/orthogonal_score_diagnostics_v1.py | 02_primary_sensitivity; manuscript Source_Data S1–S2 |
| Three-seed sign audit | archive/read_only_sign_audit.py | 02_primary_sensitivity and manuscript S6 |
| CEBPA external contrasts | work/CEBPA_expression_expansion_20261116_v1/analyse.R | 06_external_validation |
| Tumor DC | work/GSE195439_design_20261116_v1/analyse.R | 06_external_validation |
| Splenic DC | results/reports/xbp1_GSE52857_external_validation_20261116_v1/analyse_limma.R | 06_external_validation |
| Figures | tools/render_frozen_figures.py | manuscript/Source_Data; manuscript/Figures |

Scientific source paths above are beneath `archive/project/`, except the two
portable/archive entry paths explicitly indicated. The import closure includes
available local Python modules; dynamic imports and machine-specific execution
contracts still require the original runtime context. See REPRODUCIBILITY.md.
