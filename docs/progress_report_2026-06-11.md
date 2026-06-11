# 목표 달성 현황 보고서 (2026-06-11)

## 목표 요약

| 목표 | 조건 | 기준 | 현재 최고 성능 | 달성률 |
|------|------|------|---------------|--------|
| Goal 1 | 3-seed LOCV | RMSE ≤ 0.080 | 0.0908 (GRU @80%) | 88.1% |
| Goal 2 | ≤80% Input signal, 3-seed LOCV | RMSE ≤ 0.090 | 0.0908 (GRU @80%) | 99.1% |

---

## 현재 최고 성능 (H4 S8 기준, 3-seed LOCV)

**실험**: H4 S8 — XGBoost + Feature-GRU 앙상블, prefix 10-100% 스윕

| 모델 | Best prefix | RMSE | R² |
|------|-------------|------|----|
| Feature-GRU | 80% | **0.0908** | 0.6286 |
| Ensemble (GRU+XGB 0.5:0.5) | 80% | 0.0985 | 0.7011 |
| XGBoost | 100% | 0.1136 | 0.7050 |

- **Goal 1 갭**: 0.0908 − 0.080 = **+0.0108** (11.9% 초과)
- **Goal 2 갭**: 0.0908 − 0.090 = **+0.0008** (0.9% 초과, 사실상 경계)

---

## 병목 분석: 케이스별 RMSE (Feature-GRU @80%, 3-seed 평균)

| 케이스 | RMSE | 비고 |
|--------|------|------|
| Case 1 | **0.2427** | 단일 최대 이상치 — 전체 평균의 핵심 드라이버 |
| Case 12 | 0.1688 | |
| Case 8 | 0.1280 | |
| Case 13 | 0.1109 | |
| Case 4 | 0.1099 | |
| Case 2 | ~0.08 | |
| Cases 3,5,10,11,15,16 | < 0.06 | 우수 성능 그룹 |

**핵심 관찰**:
- 15개 케이스 중 8개가 RMSE > 0.08 (53%)
- **Case 1 단독 제거 시 평균 ≈ 0.080** — Goal 1 달성 임계점
- Case 1은 (1, 9) 페어로, 높은 DOC(1.5)+높은 feed(0.5)+iron 조건

---

## H6 실험 인사이트

### H6 S3: Exit 구간 유무에 따른 분리 분석

| 그룹 | 케이스 수 | RMSE |
|------|-----------|------|
| Non-exit cases | 7/15 | **0.0713** |
| Exit cases | 8/15 | **0.1494** |

→ Exit 구간 처리가 성능의 주요 분기점. Exit 케이스들이 평균을 0.0713에서 0.0908로 끌어올림.

### H6 S4: 조합 탐색 실험 (진행 중)
- Feature Set × Sensor Set × Input ratio × Feature type × Model의 체계적 조합 탐색
- 6가지 feature type: `raw`, `delta`, `raw_meta`, `delta_meta`, `raw_delta`, `raw_delta_meta`
- Resume + 예상 시간 추정 기능 포함

---

## 목표 달성 전략

### Goal 2 (RMSE ≤ 0.090 @ ≤80%) — 거의 달성

현재 0.0908 (0.9% 초과). 다음 접근으로 돌파 가능성 높음:

1. **H6 S4 조합 탐색 결과 활용**: 최적 Feature/Sensor 부분집합 발굴
2. **Fine-grained prefix**: 76-84% 구간 세밀 탐색 (H4 S5 방식, 2% step)
3. **Exit-aware masking**: H6 S3 결과 기반 exit 구간 처리 개선

### Goal 1 (RMSE ≤ 0.080) — 전략적 접근 필요

핵심 병목 = Case 1 (RMSE=0.243). 두 방향:

#### 방향 A: Case 1 자체 성능 개선
- Case 1은 (1,9) 페어 — 동일 조건 반복임에도 GRU가 크게 실패
- **원인 가설**: iron+high-DOC+high-feed 조건에서 run-to-run VB 점프 패턴이 비전형적
- **시도할 것**: case 1에 대한 전용 EDA (VB 단조 증가 확인, 센서 특이성)
- Delta feature가 case 1의 이상 run(run 1 기준 delta가 왜곡될 수 있음) 대응 여부 검토

#### 방향 B: 강건한 앙상블 / 전략적 케이스 가중치
- 현재 앙상블(0.5 GRU + 0.5 XGB)은 오히려 GRU(0.0908)보다 나쁨(0.0985)
- **이유**: XGB가 Case 1에서 더 나쁘거나, 단순 평균이 최적이 아님
- Case별 최적 모델 선택 (adaptive ensemble), 또는 uncertainty-weighted 앙상블

#### 방향 C: 구조적 모델 개선
- Exit 구간 제외 또는 별도 처리 (H6 S3: non-exit RMSE=0.0713 이미 달성)
- Meta-learning (H-MADG 계열): case 1 같은 hard case를 few-shot으로 적응
- Run-to-run monotonic prior 강화 (VB 단조 증가 제약)

---

## 권장 우선순위 작업

1. **H6 S4 실행** (현재 구현 완료): Feature/Sensor 조합 최적점 탐색
2. **Case 1 전용 EDA**: 왜 예측이 어려운지 원인 규명
3. **Exit-aware 전처리 재설계**: H6 S3 인사이트를 Feature-GRU에 적용
4. **Adaptive ensemble**: case별 모델 선택 또는 가중치 학습

---

## 실험 이력 요약

| 가설 | 내용 | Best RMSE | 비고 |
|------|------|-----------|------|
| H1 | Feature EDA / suitability | — | 분석 전용 |
| H2 | 기본 ML 벤치마크 | ~0.12 | |
| H3 | Steady cut 길이/위치 분석 | — | 분석 전용 |
| H4 S1-S4 | Feature-GRU 기본 실험 | 0.094 | |
| H4 S5 | HPO + Delta+Meta | 0.0908 | 3-seed LOCV |
| H4 S8 | XGB+GRU Ensemble | 0.0908 | GRU 단독과 동일 |
| H5 | Cases 13/14 난이도 분석 | — | |
| H6 S1-S3 | Exit-aware / VB 정규화 | 0.0713* | *non-exit only |
| H6 S4 | 조합 탐색 (진행 중) | TBD | |
