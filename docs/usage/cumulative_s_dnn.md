Full initial grid:
```
python scripts/run_h3_s1_cumulative_h2_s2_revisit.py \
  --sensor-combinations current,vibration,acoustic,current_vibration,current_acoustic,vibration_acoustic,current_vibration_acoustic \
  --segments full_length,steady,entry,exit,entry_steady,entry_exit,steady_exit \
  --shifts A_to_B,A_to_C,B_to_A,B_to_C,C_to_A,C_to_B \
  --seeds 0,1,2 \
  --sequence-length 128 --max-epochs 100 --batch-size 4 --cv-folds 5 --quick-hidden-size 8
```

Extra seeds:
```
python scripts/run_h3_s1_cumulative_h2_s2_revisit.py \
  --seeds 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
```