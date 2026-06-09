# Scenario Setup

## Purpose

- Case 13,14가 다른 Case와 달리 예측에 어려움을 만드는 이유가 신호 패턴 때문일 것임을 검증

## Research Question

1. Case 13, 14의 신호 패턴이 만드는 특징은 VB와 관련성이 낮은가?
2. Case 13, 14의 신호 패턴이 만드는 특징은 다른 특징과 구분이 어려운가?

## Dataset and Case Setting

1. Dataset
    1. NASA Ames Milling Dataset

## Sensor Setting

- All Sensors
    - Current: [`smcAC`, `smcDC`]
    - Acoustic Emission: [`AE_spindle`, `AE_table`]
    - Vibration: [`vib_spindle`, `vib_table`]

## Feature Construction

- Statistics
    - Mean: `s.mean()`
    - Root Mean Square: `np.sqrt(s**2).mean())`
    - Standard Deviation: `s.std()`
    - Peak: `np.abs(s).max()`
- Delta Feature: Run1과 Run#의 Statistics Feature의 차
- Meta Feature: `DOC`, `Feed`, `Material`

## Expected Output

- CSV
- PNG