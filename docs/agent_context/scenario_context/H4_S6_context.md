# Experiment Scenario: History-Aware Input-Ratio Effect on Feature-GRU VB Prediction

## 1. Scenario Setup

### 1.1 Purpose

H4_S1/H4_S5의 input-ratio 실험은 한 case의 **모든 run**을 동일한 ratio로 잘라 feature를 만들었다. 그러나 실제 마모 예측 상황에서는 **이미 완료된 이전 run들은 full-length 신호를 보유**하고 있고, **현재 깎고 있는 run만 부분 신호(r%)**만 관측된다.

본 시나리오의 목적은 이 현실적 가정을 반영하여, **이전 run은 full-length(100%), 예측 대상 run만 r%** 로 입력했을 때 현재 run의 VB를 얼마나 조기에 예측할 수 있는지 확인하는 것이다. run 간 시간 의존성을 GRU가 활용하도록 한다.

---

## 2. Research Questions

1. 이력 run을 full-length로 제공하고 현재 run만 부분 관측할 때, 입력 ratio(현재 run)를 늘리면 VB 예측 성능은 어떻게 변하는가?
2. 어느 정도의 현재 run 관측만으로 충분한 예측이 가능한가 (조기 예측 가능성)?

---

## 3. Dataset

* NASA Ames Milling Dataset

---

## 4. Key Difference vs H4_S1 / H4_S5

| 항목 | H4_S1/S5 (기존) | H4_S6 (본 시나리오) |
|------|-----------------|---------------------|
| 이력 run 신호 | r%로 truncate | **100% full-length** |
| 대상 run 신호 | r% | r% |
| 시퀀스 | case당 1개, timestep마다 예측 | **(case, 대상 run)마다 1개** |
| GRU head | 모든 timestep | **마지막 timestep (seq2one)** |
| ratio 적용 | 모든 run 동일 | **대상 run에만** |

---

## 5. Sample Construction

ratio r, 대상 run t (case 내 chronological 위치)마다 하나의 샘플:

```text
sequence = [ run_1@100%, run_2@100%, ..., run_{t-1}@100%, run_t@r% ]   (length t)
target   = VB[run_t]
```

* 이력 run(1..t-1): full-length feature
* 대상 run(t, 시퀀스 마지막): r% feature
* Delta 기준: 항상 `run_1@100%` 대비 (run_1은 항상 이력 → full)
* Meta(`DOC`, `feed`, `material`): timestep마다 append
* run_1(t=1, 이력 없음): 단일 step 시퀀스 `[run_1@r%]` (delta = feat@r% − feat@100%)

---

## 6. Sensor / Feature

* 모든 sensor 사용 (smcAC, smcDC, vib_table, vib_spindle, AE_table, AE_spindle)
* Statistics: Mean, RMS, Std, Peak → sensor당 4개 × 6 = 24 delta + meta 3 = **27 feature**

---

## 7. Model Setting

* **Sequence-to-one GRU** (마지막 hidden state로 스칼라 예측)
* Best H4_S5 config (`head32`): hidden=256, layers=3, dropout=0.1, head=32, lr=1e-3, weight_decay=1e-4, epochs=200, CosineAnnealingLR, grad_clip=1.0
* 가변 길이 시퀀스는 `pack_padded_sequence`로 처리, target VB는 train 통계로 표준화

---

## 8. Input Ratio Setting

* **per-ratio 모델**: r ∈ {10, 20, 30, 40, 50, 60, 70, 80, 90} 각각에 대해 독립적으로 학습/평가
* 100%는 제외 (대상 run도 full이면 truncation 없는 경우라 본 실험 관심 밖)

---

## 9. Preprocessing

* H4_S1과 동일: Case2-Run1·Case12-Run1 제외, Run1 VB NaN→0.0, 이후 선형보간(`method="index"`, `limit_area="inside"`)

---

## 10. Seed / Validation

* LOCV (15 cases)
* Random seed 3회 (0, 1, 2)

---

## 11. Evaluation / Aggregation

* RMSE, R2, MAE
* 각 seed에서 LOCV mean → 3-seed 평균을 ratio별 최종 성능으로 간주
* **본 시나리오는 task 정의가 달라 leader board와 직접 비교하지 않는다** (별도 트랙)

---

## 12. Expected Output

* ratio별 3-seed LOCV 지표 (`analysis/H4_S6_ratio_metrics.csv`)
* ratio vs RMSE/R2 곡선 (`figures/H4_S6_ratio_rmse_curve.png`)
* ratio×seed×case 상세 및 예측값 (`metrics/`, `predictions/`)
* 리포트 (`analysis/H4_S6_report.md`)

---
