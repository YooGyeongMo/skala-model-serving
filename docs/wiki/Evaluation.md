# Evaluation

[Home](Home.md) | [Architecture](Architecture.md) | [API](API.md) | [Tradeoffs](Tradeoffs.md) | [Troubleshooting](Troubleshooting.md) | [Schedule](Schedule.md)

모든 값은 1회 측정입니다. 출처를 각 표 아래에 적었습니다.

## 1. 모델 성능 (검증 RMSE, 명)

| 조건 | RMSE | 비고 |
|---|---|---|
| 전일값 그대로 (보간본) | 2,523 | 기준선 |
| 7일 이동평균 (보간본) | 2,969 | 기준선 |
| Day2 MLflow 학습, 전체 1,035일 (보간 전) | 2,267 | seed 42, PC 가 달라도 동일 |
| Day2 MLflow 학습, 전체 1,035일 (보간 후) | **2,363** | 컨테이너 빌드 시 v1, 2026-11-05 재빌드에서도 `[GATE PASSED] rmse=2363` |
| ≤ 2025-08-31 학습 (974행) | 2,094 | 시연 1단계 |
| ≤ 2024-06-30 학습 (547행) | 1,961 | 드리프트 시연 1단계 |
| 배포 게이트 | ≤ 2,700 | |

- 전체 데이터 게이트 값(2,267 → 2,363) 은 보간 데이터(PR #7) 반영 전후의 차이입니다. 결항일 보간의 효과는 같은 학습 컷에서 평상시 구간만 잰 값(2,415 → 1,807) 으로 봅니다 ([Troubleshooting](Troubleshooting.md) E 항목).
- Day1 baseline (seed 미고정) 은 PC 와 실행마다 2,191 ~ 2,945 로 흔들렸습니다. 그래서 수치 비교는 seed 42 인 Day2 값으로 합니다.

출처: [`data/README.md`](../../data/README.md) 기초 통계, [`docs/IMPLEMENTATION_REPORT.md`](../IMPLEMENTATION_REPORT.md) 3장, 7장.

## 2. 서빙 지연

| 조합 | 기동 | 첫 `/predict` | 두 번째 |
|---|---|---|---|
| local, lazy | 0.22초 | 2.04초 | 0.013초 |
| mlflow, lazy | 0.20초 | 3.51초 | 0.016초 |
| mlflow, eager | 2.15초 | 0.096초 | 0.014초 |

- 대시보드 예측 카드 왕복: v1 44 ms → v3 187 ms (새 모델 lazy 로드 포함, D1 측정). 2026-11-05 재실행 컨테이너(eager) 에서 163 ms.
- 잘못된 입력 4종(19행, 21행, 음수, departures 누락) × 4모드 모두 422.

출처: IMPLEMENTATION_REPORT §A, §D.

## 3. 드리프트 시연 (수준 이동 +15%, 합성)

정상 배치와 드리프트 배치는 같은 41일(2025-05-21 ~ 06-30) 이고 뒤 21일만 ×1.15 입니다.

| 단계 | 기록 (출발 여객 37,000 고정) | 재실행 2026-11-05 (CSV 배치, 출발 여객 실제값) |
|---|---|---|
| v1 게이트 | 2,363 | 2,363 |
| 정상 배치 | RMSE 1,410 / bias +76 → ok | RMSE 1,191 / bias +86 → ok |
| 드리프트 1회차 | bias +1,661 → 게이트 1,893 → v2 | bias +1,958 → 게이트 1,893 → v2 (감지 → 승격 4.1초) |
| 드리프트 2회차 | bias +951 → 게이트 1,423 → v3 | bias +1,220 → 게이트 1,423 → v3 (3.4초) |
| 드리프트 3회차 | RMSE 1,619 / bias +239 → ok | RMSE 1,564 / bias +475 → ok |

- 판정 값(bias) 은 출발 여객 조건에 따라 달라지지만, 재학습 데이터는 같은 업로드 파일의 마지막 41행이라 게이트 RMSE(1,893, 1,423) 는 두 조건에서 같습니다.
- 재실행 근거: 컨테이너 `logs/aiops.log`

```
09:32:58,156 [WARN] drift detected - rmse=2450 bias=+1958 (excluding 0 anomaly day(s)) - triggering retrain
09:32:58,159 [INFO] retrain triggered (window=last_21_days)
09:33:02,240 [OK] new_rmse=1893 - production promoted: Airport_Arrivals_Predictor v2
09:33:10,932 [WARN] drift detected - rmse=1919 bias=+1220 (excluding 0 anomaly day(s)) - triggering retrain
09:33:10,933 [INFO] retrain triggered (window=last_21_days)
09:33:14,299 [OK] new_rmse=1423 - production promoted: Airport_Arrivals_Predictor v3
```

## 4. 드리프트 시연 (2025년 1분기 실데이터, 롤백 포함)

| 단계 | 결과 |
|---|---|
| ≤ 2024-06-30 학습 v1 | 게이트 1,961 |
| 정상 (2024-10-01 ~ 11-10) | RMSE 899 / bias −241 → ok |
| 하락 (~ 2025-01-21) | 이상치 1일 제외 bias −1,041 → 게이트 1,338 → v2 (16.7초) |
| 확정 (~ 02-11) | RMSE 3,169 / bias +199 → structure_drift, v2 확정 |
| 반등 (~ 02-25) | bias +686 → 게이트 902 → v3 |
| 재하락 (~ 03-18) | bias −1,008 → 게이트 **실패 2,742** → v3 미확정이라 **롤백 → v2** |
| 폭설 배치 | 이상치 1일(02-07, 6,088명) 제외 bias −103 → structure_drift 알림만 |

출처: `data/README.md` "드리프트 시연 순서". 롤백의 2,742 vs 2,700 은 간발이라 PC 에 따라 v4 승격으로 바뀔 수 있습니다.

## 5. 판정 정책 비교 (2025-01-01 ~ 10-31 하루 전진 시뮬레이션)

| 정책 | 재학습 | 결과 |
|---|---|---|
| 재학습 없음 | 0 | RMSE 3,038 |
| RMSE > 2,700 이면 재학습 | 약 3배 | RMSE 3,039 (개선 없음) |
| \|bias\| > 500 | 10회 (승격 9, 롤백 1) | RMSE 3,017 |
| \|bias\| > 1,500 | 1회 (3월) | |
| \|bias\| ≥ 2,000 | 0회 | |

출처: IMPLEMENTATION_REPORT §B, `docs/code_current.md` 3번 실험 6.

## 6. 컨테이너

| 항목 | 값 |
|---|---|
| 빌드 (`--no-cache`) | 8단계, pip 35.5초, 학습과 등록 32.6초, export 38.2초 |
| 이미지 | 3.26GB |
| `stale` 3단계 재현 | false → true → false (예측값은 세 단계 모두 39,980.96) |
| 버전 조회 무응답 | 848.03초 → 5.70초 |
