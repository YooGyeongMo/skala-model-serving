# Troubleshooting

[Home](Home.md) | [Architecture](Architecture.md) | [API](API.md) | [Tradeoffs](Tradeoffs.md) | [Evaluation](Evaluation.md) | [Schedule](Schedule.md)

형식: 증상 → 원인 → 해결 → 전후. 담당 표기는 작업자 코드(A 서빙, B 모니터링, C 학습과 배포, D 대시보드, E 데이터).

## 학습, 배포, 버전 조회 (C)

### C1. 승격 로그는 v2 인데 서버는 v1 으로 응답 [#6](https://github.com/wonn2k/skala-model-serving/pull/6) [#10](https://github.com/wonn2k/skala-model-serving/pull/10) [#12](https://github.com/wonn2k/skala-model-serving/pull/12)

- 증상: 컨테이너 안에서 `train_and_register.py` 를 따로 돌려 v2 승격 (`[GATE PASSED] rmse=2363 -> v2 promoted`). `/predict` 는 계속 `model_registry_version: "1"`.
- 원인: 서빙 모델이 `_model_cache` 에 남아 있다. v1, v2 가 seed 42 + 같은 데이터라 예측값까지 39,980.96 으로 같아 응답만으론 안 보인다.
- 해결: `GET /monitor/versions` 가 레지스트리 Production 과 서빙 모델을 나란히 반환, A 에게 `LoadedModel` 에 `run_id` 보관을 요청해([#9](https://github.com/wonn2k/skala-model-serving/pull/9)) run_id 기준으로 `stale` 판정.
- 전후: 판정 불가(항상 null) → 3단계 재현 v1/v1 false → v2/v1 **true** → 재기동 v2/v2 false. run_id 기준도 같은 3단계 확인 (`d97b47e5` → `f2b9ea8b`).

### C2. 트래킹 서버 무응답 시 조회가 14분 [#10](https://github.com/wonn2k/skala-model-serving/pull/10)

- 증상: `MLFLOW_TRACKING_URI=http://10.255.255.1:9999` 에서 `/monitor/versions` 848.03초.
- 원인: MLflow 기본 요청 타임아웃 120초 + 재시도 7회. sync 핸들러가 스레드풀 슬롯을 점유.
- 해결: `MLFLOW_HTTP_REQUEST_TIMEOUT=5`, `MLFLOW_HTTP_REQUEST_MAX_RETRIES=0` 을 `setdefault` 로.
- 전후: 848.03초 → 재시도 1회 10.68초 → 재시도 없음 5.70초. 연결 거부는 원래 0.54초.

### C3. sqlite URI 오타가 성공으로 보이고, 그 DB 가 커밋됐다 [#10](https://github.com/wonn2k/skala-model-serving/pull/10)

- 증상: `MLFLOW_TRACKING_URI=sqlite:////tmp/oops_typo_check.db` 에서 `production: null`, `registry_error: null`. 별도로 876KB `does_not_exist_typo.db` 가 main 에 들어감.
- 원인: MLflow 가 없는 sqlite 경로를 만나면 빈 DB 를 새로 만들고 조회는 0건으로 "성공". `git add -A` 가 그 파일을 쓸어 담았다.
- 해결: `tracking_uri` 를 응답에 노출해 어느 저장소를 봤는지 보이게 하고, 바이너리 제거, E 에게 `.gitignore` 에 `*.db`, `*.sqlite` 추가 요청.

### C4. MLflow 로더 구현 전 컨테이너 기동 실패 [#10](https://github.com/wonn2k/skala-model-serving/pull/10)

- 증상: `docker compose up` → `Application startup failed. Exiting.`
- 원인: Dockerfile 기본값이 `MODEL_SOURCE=mlflow` + eager 인데 `_load_from_mlflow()` 가 아직 `NotImplementedError` (A 담당 TODO).
- 해결: compose 에 `MODEL_SOURCE: ${MODEL_SOURCE:-mlflow}` 통로를 열어 `MODEL_SOURCE=local` 로 기동.
- 전후: 기동 실패 → `/health`, `/monitor/versions`, `/data/status`, `/predict` 모두 200 (예측 42,096.13).

### C5. 연속 배치를 보내면 판정이 뒤집힌다 [#18](https://github.com/wonn2k/skala-model-serving/pull/18)

- 증상: normal, falsealarm, drift 를 연속으로 보내면 normal 이 `retrain_triggered`, drift 가 `ok`.
- 원인: 앞 배치의 예측이 `recent_predictions` (최근 21건) 에 남아 다음 배치 판정에 섞인다.
- 해결: 배치마다 서버 재기동으로 창을 비우면 의도대로 갈림. 이후 B 가 승격 시 `recent_predictions.clear()` 를 넣었고, 41행 배치는 예측 21개로 창 전체를 교체한다는 점을 A 가 테스트로 고정.

### C6. 대시보드 "드리프트 배치 전송" 이 드리프트를 못 만든다 [#18](https://github.com/wonn2k/skala-model-serving/pull/18)

- 증상: 랜덤워크 드리프트 배치가 계속 `ok`. σ 를 3배로 올려도 같음.
- 원인: 랜덤워크가 37,000 중심 좌우 대칭이라 평균 치우침(bias) 이 생기지 않는다. 실제 일별 변동성 7.12% 대비 대시보드 σ 1.2% / 3.6%. 실제 수요는 평균회귀(자기상관 +0.663) 인데 랜덤워크는 누적 표류.
- 해결안 (D2 에 이관): 계단식 수준 이동 생성기 (앞 20일 37,000, 뒤 21일 27,000, 하루 변동 5%).
- 전후: 10 seed 실측 랜덤워크 0/10 → 계단식 **10/10** (bias 평균 −1,167, 최약 seed −740). 추세형(하루 −0.3%) 은 LSTM 이 따라가 bias −89 에 그침.

### C7. 난수 "정상 배치" 가 Production 을 교체 [#28](https://github.com/wonn2k/skala-model-serving/pull/28) (열림)

- 증상: 같은 CSV 를 보냈는데 값이 시간이 지나며 달라짐.
- 원인: 랜덤 정상 배치는 시드 없는 `Math.random()`. 우연히 bias 500 을 넘으면 실제 재학습과 승격이 일어나 모델이 바뀐다. 깨끗한 컨테이너에서 정상 버튼 4번 중 3번째에 승격, `production.version` 1 → 2.
- 확인: 승격 전 CSV rmse 1819.1711 / bias +259.04, 승격 후 1749.7946 / −130.23. 승격 전후 각각은 반복해도 소수점까지 동일 → 원인은 모델 교체 하나.
- 대응: 시연은 고정 CSV 만 쓰도록 기록.

## 서빙 API (A)

- `/monitor/versions` 가 항상 `stale=null` → 모델 객체에 `run_id` 가 없었음 → 선택한 버전 메타데이터에서 함께 보관 ([#9](https://github.com/wonn2k/skala-model-serving/pull/9)).
- MLflow 로드 `NotImplementedError` → Production 조회, 정확한 버전 고정 로드, 고정 scaler 결합 ([#5](https://github.com/wonn2k/skala-model-serving/pull/5)) → `/predict` 200, `model_registry_version=1`.
- `model_version` 이 항상 `"production"` 이라 버전 전환이 안 보임 → 기존 필드 유지 + `model_registry_version` 추가.
- 승격 순간 `reset_cache()` 부재로 `AttributeError` 500 → A 가 캐시 잠금과 함께 `reset_cache()` 추가, B 가 승격과 롤백 성공 뒤 호출.

## 모니터링 (B)

- 정상 배치(2024-05-21 ~ 06-30) 가 RMSE 2,820 으로 임계 초과 → `train_test_split` 이 마지막 20% 를 검증으로 떼어 이 구간이 학습에 안 들어감 + 06-29 하루 −7,598 → 학습 구간 안쪽(2024-04-15 ~ 05-25) 으로 교체, 2,374.
- 재학습 v2 가 다음 윈도우에서 v1 보다 나쁨 (3,918 vs 3,126) → 41행 fine-tuning 이 1분기 저점에 과적응 → 미확정 승격 롤백 도입 ([Tradeoffs T6](Tradeoffs.md)).

## 대시보드 (D1, D2)

- 배치 2건이 겹쳐 v6, v7 이 1.4초 간격으로 같은 `new_rmse=2233` 으로 승격 → 전송 중 버튼 비활성화와 재진입 방지 ([#14](https://github.com/wonn2k/skala-model-serving/pull/14)) → 이후 3건 순서 전송 시 승격 3회 겹침 없음. 서버 쪽 잠금은 남은 과제.
- `anomaly`, `structure_drift`, `rolled_back` 가 기본 "정상" 문구로 표시 → 상태 5종 분기 추가 ([#21](https://github.com/wonn2k/skala-model-serving/pull/21)).
- 생성 시각이 설계 문서는 문자열, 실제 응답은 Unix 밀리초 → 두 형식 모두 한국 시간 변환.

## 데이터, 통합 (E)

- 결항일 포함 학습으로 모델이 20일 이동평균이 됨 → 18일 보간 ([#7](https://github.com/wonn2k/skala-model-serving/pull/7)) → 평상시 RMSE 2,415 → 1,807.
- 학습 서브프로세스가 0초 만에 종료 → `subprocess.run(["python", ...])` 이 venv 가 아닌 시스템 Python 을 잡음 → `sys.executable`.
- `mlruns/` 를 지워도 레지스트리 번호가 v4 부터 이어짐 → 레지스트리는 `mlflow.db` 에 있음 → 시연 전 `mlflow.db`, `mlruns/`, `data/uploads/*.csv`, 모델 파일 모두 비움.
