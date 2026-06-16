# Milling Tool-Wear Prediction (NASA Ames + PHM2010)

공구 마모량(VB, flank wear) 예측 연구 코드베이스. 핵심 가설은
**run 간 순차 의존성을 모델링하는 run-sequence 모델(RNN/LSTM/GRU)** 이
**run을 독립적으로 처리하는 run-independent 모델(RF/Ridge/SVR/XGBoost)** 대비
낮은 LOCV(Leave-One-Case-Out) RMSE를 달성하는지를 두 데이터셋에서 검증하는 것이다.

- **NASA Ames Milling**: VB \[mm\], 15개 case(case 6 제외), Delta+Meta feature
- **PHM2010**: VB_max = max(VB_flute_1, VB_flute_2, VB_flute_3) \[µm\], case 1/4/6, raw 통계 feature

---

## 1. 요구 사항

- **Python 3.12** (개발·검증 환경: 3.12 / Linux·WSL2)
- GPU는 선택 사항 — torch가 CUDA 가용 시 자동 사용, 없으면 CPU로 동작

## 2. 설치

```bash
git clone <repo-url>
cd 02_milling_v2

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

> CUDA 빌드의 torch가 필요하면 https://pytorch.org/get-started/locally/ 의
> 플랫폼별 명령으로 torch를 따로 설치한 뒤 나머지를 `pip install -r requirements.txt` 한다.

설치 검증:

```bash
python3 -c "import numpy, pandas, sklearn, torch, xgboost; print('ok', torch.cuda.is_available())"
```

## 3. 데이터

| 파일 | 저장소 포함 | 비고 |
|---|---|---|
| `datasets/nasa/process_info.csv` | ✅ | case/run/VB/공정 메타 |
| `datasets/phm2010/process_info.csv` | ✅ | case/run/VB_flute_1·2·3 |
| `datasets/phm2010/feature_vector.csv` | ✅ | 70 통계 feature (7 sensor × 10 stat) |
| `datasets/nasa/cutting_segment*/` | ✅ | 절삭 구간 분할 CSV |
| `datasets/nasa/raw_signal.csv` | ❌ (gitignored, ~119 MB) | 원시 센서 신호 |
| `datasets/phm2010/raw_signal.csv` | ❌ (gitignored, ~11 GB) | 원시 센서 신호 |

**중요**: PHM2010 feature 기반 실험(B1, E5 등)은 저장소에 포함된 CSV만으로
`raw_signal.csv` 없이 그대로 실행된다. 원시 신호는 NASA H-시리즈 및 signal 모델
(`build_feat_cache`, SignalCNN/GRU 등)에서만 필요하며, 별도로 확보해
위 경로에 두어야 한다.

## 4. 실행 예시

모든 스크립트는 **저장소 루트에서** 실행한다 (스크립트가 루트를 `sys.path`에 추가).
결과는 `experiments/executions/<EXP>/<SCN>/<timestamp>_*/` 에 기록된다 (gitignored).

```bash
# PHM2010 feature baseline (9 모델, LOCV-3) — raw_signal 불필요
python3 scripts/run_B1_S1_phm2010_feature_baseline.py

# PHM2010 Delta-feature 실험 (raw / delta / raw+delta)
python3 scripts/run_E5_S1_phm2010_delta_feature.py

# PHM2010 flute-label 증강 실험 (Arm A / Arm B)
python3 scripts/run_B1_S2_phm2010_flute_augment.py
```

테스트:

```bash
python3 -m pytest tests/ -q
```

## 5. 저장소 구조

```
scripts/                     실험 진입점 (run_*.py / analyze_*.py / plot_*.py)
milling_experiment_framework/  재사용 프레임워크 패키지 (별도 설치 불필요)
datasets/                    nasa/ , phm2010/  (대용량 raw_signal은 gitignored)
experiments/                 실행 산출물 (executions/는 gitignored)
configs/                     실험 설정
leaderboards/                결과 요약 / README
tests/                       pytest
docs/                        문서
```

## 6. 주의 사항 (심볼릭 링크)

이 저장소에는 개발 환경에 종속된 심볼릭 링크가 있다. 독립 clone에서는 깨질 수 있다.

- **`eda` → `../02_milling/eda/`** : 형제 디렉터리를 가리키는 추적된 링크.
  단독 clone에서는 깨지며, EDA·플롯 스크립트 4종
  (`plot_segment_sensor_overlays*.py`, `run_H1_S8_*`, `run_H7_S1_*`)만 이를 참조한다.
  **핵심 실험 러너는 영향 없음.** 필요 시 해당 경로의 EDA 자산을 별도로 마련한다.
- **`obsidian` → 개인 Obsidian vault (절대경로)** : gitignored 이므로 clone에
  포함되지 않는다. 노트 동기화용일 뿐 코드 실행과 무관하다.

GPU가 없으면 sequence 모델은 자동으로 CPU에서 동작한다 (학습 시간만 증가).
