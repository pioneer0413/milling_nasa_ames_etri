# H4 Model x Input Ratio Best RMSE Heatmap

각 셀은 해당 모델과 입력 신호 비중에서 가능한 H4 설정 중 가장 낮은 aggregate mean RMSE를 사용합니다. 낮은 RMSE일수록 밝은 색으로 표시했습니다.

- Figure PNG: `H4_model_input_ratio_best_rmse_heatmap.png`
- Figure SVG: `H4_model_input_ratio_best_rmse_heatmap.svg`
- Matrix CSV: `H4_model_input_ratio_best_rmse_matrix.csv`
- Cell source CSV: `H4_model_input_ratio_best_rmse_sources.csv`

## Used Latest Runs
- H4_S1_T1: 2026-06-08_161302_H4_S1_T1_input_signal_length_effect
- H4_S1_T2: 2026-06-08_191851_H4_S1_T2_raw_delta_meta_all_models
- H4_S1_T3: 2026-06-08_165151_H4_S1_T3_locv_target_restricted
- H4_S1_T4: 2026-06-08_193658_H4_S1_T4_delta_meta_only
- H4_S1_T5: 2026-06-08_171530_H4_S1_T5_with_case6
- H4_S1_T6: 2026-06-08_173840_H4_S1_T6_raw_only
- H4_S1_T7: 2026-06-08_174208_H4_S1_T7_delta_only
- H4_S1_T8: 2026-06-08_174655_H4_S1_T8_raw_delta_no_meta
- H4_S3_T1: 2026-06-08_201520_H4_S3_T1_entry_steady_exit
- H4_S3_T2: 2026-06-08_201620_H4_S3_T2_entry_steady
- H4_S3_T3: 2026-06-08_201721_H4_S3_T3_steady
- H4_S3_T4: 2026-06-08_201902_H4_S3_T4_noload_excluded_prefix

## Matrix

| model        |       10 |       20 |       30 |       40 |       50 |       60 |       70 |       80 |       90 |      100 |
|:-------------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| feature_gru  | 0.138947 | 0.112173 | 0.126979 | 0.113833 | 0.107003 | 0.104668 | 0.100585 | 0.094549 | 0.113379 | 0.111976 |
| XGBoost      | 0.139821 | 0.130414 | 0.122002 | 0.125303 | 0.118568 | 0.119528 | 0.118322 | 0.119141 | 0.114317 | 0.111153 |
| SVR          | 0.140969 | 0.143418 | 0.127707 | 0.121553 | 0.119679 | 0.123219 | 0.11728  | 0.113851 | 0.112223 | 0.109773 |
| RandomForest | 0.136565 | 0.12885  | 0.122805 | 0.128063 | 0.120612 | 0.119769 | 0.119451 | 0.120327 | 0.118594 | 0.116681 |
| Ridge        | 0.152817 | 0.15416  | 0.137217 | 0.132618 | 0.131149 | 0.124846 | 0.123208 | 0.125293 | 0.130954 | 0.126168 |
| MLP          | 0.227536 | 0.208855 | 0.185419 | 0.1777   | 0.167381 | 0.170238 | 0.165626 | 0.189812 | 0.177259 | 0.182926 |
