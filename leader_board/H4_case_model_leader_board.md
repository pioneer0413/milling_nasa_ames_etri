# H4 Case-wise Model Leader Board

각 H4 task의 최신 실행 결과에서 case별 RMSE 최저 조합을 정리했습니다. T0/S2 예비·시각화 결과는 제외하고 S1 T1-T8, S3 T1-T4를 포함했습니다.

## Files
- `H4_case_model_leader_board.csv`: case별 1위 조합
- `H4_case_model_top5.csv`: case별 상위 5개 후보
- `H4_case_model_winner_summary.csv`: model/task 조합별 승리 case 요약
- `H4_case_model_all_results.csv`: 정규화된 전체 후보 row

## Related Heatmap
- `H4_model_input_ratio_best_rmse_heatmap.png`: model x input signal ratio best RMSE heatmap
- `H4_model_input_ratio_best_rmse_heatmap.svg`: vector version of the heatmap
- `H4_model_input_ratio_best_rmse_matrix.csv`: heatmap cell values
- `H4_model_input_ratio_best_rmse_sources.csv`: source task/config selected for each heatmap cell

## Used Latest Runs
- H4_S1_T1: `2026-06-08_161302_H4_S1_T1_input_signal_length_effect`
- H4_S1_T2: `2026-06-08_191851_H4_S1_T2_raw_delta_meta_all_models`
- H4_S1_T3: `2026-06-08_165151_H4_S1_T3_locv_target_restricted`
- H4_S1_T4: `2026-06-08_193658_H4_S1_T4_delta_meta_only`
- H4_S1_T5: `2026-06-08_171530_H4_S1_T5_with_case6`
- H4_S1_T6: `2026-06-08_173840_H4_S1_T6_raw_only`
- H4_S1_T7: `2026-06-08_174208_H4_S1_T7_delta_only`
- H4_S1_T8: `2026-06-08_174655_H4_S1_T8_raw_delta_no_meta`
- H4_S3_T1: `2026-06-08_201520_H4_S3_T1_entry_steady_exit`
- H4_S3_T2: `2026-06-08_201620_H4_S3_T2_entry_steady`
- H4_S3_T3: `2026-06-08_201721_H4_S3_T3_steady`
- H4_S3_T4: `2026-06-08_201902_H4_S3_T4_noload_excluded_prefix`

## Case Winners

|   target_case | model        | task_id   |   input_length_pct | features       | segment           |       rmse |         r2 |        mae | second_best_model   |   margin_to_second_rmse |
|--------------:|:-------------|:----------|-------------------:|:---------------|:------------------|-----------:|-----------:|-----------:|:--------------------|------------------------:|
|             1 | XGBoost      | S3_T4     |                 80 | Raw+Delta+Meta | Entry+Steady+Exit | 0.044626   |   0.919187 | 0.0391441  | XGBoost             |             0.00401453  |
|             2 | XGBoost      | S1_T6     |                 90 | Raw            | -                 | 0.032673   |   0.94416  | 0.0286795  | XGBoost             |             0.00153847  |
|             3 | feature_gru  | S1_T4     |                 50 | Delta+Meta     | -                 | 0.0398018  |   0.911357 | 0.0330968  | feature_gru         |             3.65821e-05 |
|             4 | XGBoost      | S1_T1     |                 90 | Raw+Meta       | -                 | 0.0402358  |   0.914922 | 0.0299416  | XGBoost             |             0.000792625 |
|             5 | feature_gru  | S1_T4     |                 30 | Delta+Meta     | -                 | 0.0256616  |   0.988845 | 0.0218521  | feature_gru         |             0.00254428  |
|             6 | XGBoost      | S1_T5     |                100 | Raw+Delta+Meta | -                 | 0.00615645 | nan        | 0.00615645 | Ridge               |             0.000118531 |
|             7 | feature_gru  | S1_T4     |                 20 | Delta+Meta     | -                 | 0.0278175  |   0.962717 | 0.0234152  | XGBoost             |             0.00897467  |
|             8 | RandomForest | S1_T4     |                 10 | Delta+Meta     | -                 | 0.0519621  |   0.929343 | 0.0471507  | RandomForest        |             0.00794409  |
|             9 | MLP          | S1_T4     |                100 | Delta+Meta     | -                 | 0.0625625  |   0.938156 | 0.0538757  | feature_gru         |             0.00433818  |
|            10 | feature_gru  | S1_T4     |                 80 | Delta+Meta     | -                 | 0.0471517  |   0.95466  | 0.0417804  | feature_gru         |             0.00283576  |
|            11 | feature_gru  | S1_T4     |                 60 | Delta+Meta     | -                 | 0.0310685  |   0.979431 | 0.024672   | feature_gru         |             0.000211175 |
|            12 | SVR          | S1_T1     |                 70 | Raw+Meta       | -                 | 0.0468528  |   0.936932 | 0.0399313  | SVR                 |             0           |
|            13 | feature_gru  | S1_T4     |                 70 | Delta+Meta     | -                 | 0.0969146  |   0.955718 | 0.0628441  | feature_gru         |             0.0220986   |
|            14 | feature_gru  | S1_T4     |                 80 | Delta+Meta     | -                 | 0.0931556  |   0.92918  | 0.0716908  | feature_gru         |             0.00659726  |
|            15 | feature_gru  | S1_T4     |                 60 | Delta+Meta     | -                 | 0.0179905  |   0.993529 | 0.0142074  | feature_gru         |             0.00261281  |
|            16 | Ridge        | S1_T7     |                 80 | Delta          | -                 | 0.032064   |   0.973987 | 0.0257359  | feature_gru         |             0.0107854   |

## Winner Summary

|   rank | model        | task_id   |   input_length_pct | winning_cases   |   n_winning_cases |   mean_winner_rmse |
|-------:|:-------------|:----------|-------------------:|:----------------|------------------:|-------------------:|
|      1 | feature_gru  | S1_T4     |                 60 | 11,15           |                 2 |         0.0245295  |
|      2 | feature_gru  | S1_T4     |                 80 | 10,14           |                 2 |         0.0701537  |
|      3 | XGBoost      | S1_T5     |                100 | 6               |                 1 |         0.00615645 |
|      4 | feature_gru  | S1_T4     |                 30 | 5               |                 1 |         0.0256616  |
|      5 | feature_gru  | S1_T4     |                 20 | 7               |                 1 |         0.0278175  |
|      6 | Ridge        | S1_T7     |                 80 | 16              |                 1 |         0.032064   |
|      7 | XGBoost      | S1_T6     |                 90 | 2               |                 1 |         0.032673   |
|      8 | feature_gru  | S1_T4     |                 50 | 3               |                 1 |         0.0398018  |
|      9 | XGBoost      | S1_T1     |                 90 | 4               |                 1 |         0.0402358  |
|     10 | XGBoost      | S3_T4     |                 80 | 1               |                 1 |         0.044626   |
|     11 | SVR          | S1_T1     |                 70 | 12              |                 1 |         0.0468528  |
|     12 | RandomForest | S1_T4     |                 10 | 8               |                 1 |         0.0519621  |
|     13 | MLP          | S1_T4     |                100 | 9               |                 1 |         0.0625625  |
|     14 | feature_gru  | S1_T4     |                 70 | 13              |                 1 |         0.0969146  |
