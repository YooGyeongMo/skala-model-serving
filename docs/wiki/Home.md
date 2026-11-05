# 예측 모델 서비스 및 운영 관리 Wiki

제주공항 다음날 도착 여객 수를 LSTM 으로 예측해 FastAPI 로 서빙하고, MLflow 게이트, 드리프트 감지, 자동 fine-tuning, 재배포, 롤백까지 컨테이너 한 대 안에서 돌리는 서비스입니다.

## 한눈에

| 항목 | 값 |
|---|---|
| 예측 대상 | 다음날 도착 여객 수 (명) |
| 입력 | 최근 20일 (도착, 출발) |
| 데이터 | 한국공항공사 제주공항 일별 여객 1,035일 (2023-01-01 ~ 2025-10-31) |
| 배포 게이트 | 검증 RMSE ≤ 2,700명 |
| 드리프트 기준 | 최근 21건 \|bias\| > 500명 (하루 오차 10,000 초과는 이상치로 제외) |
| 재학습 | Production 가중치에서 최근 41행으로 3 epoch fine-tuning |
| 배포 | Docker 단일 컨테이너, MLflow sqlite |
| 팀 | 5명, 2일, PR 25개 |

## 페이지

| 페이지 | 내용 |
|---|---|
| [Architecture](Architecture.md) | 구성 요소, 데이터 흐름, 배포 구조, 상태가 어디에 남는지 |
| [API](API.md) | 엔드포인트 8개, 요청과 응답, 오류 |
| [Tradeoffs](Tradeoffs.md) | 결정 로그 15건 (상황, 선택지, 결정, 비용, 다시 볼 조건) |
| [Troubleshooting](Troubleshooting.md) | 증상, 원인, 해결, 전후 수치 |
| [Evaluation](Evaluation.md) | 모델 성능, 서빙 지연, 드리프트 시연 실측값 |
| [Schedule](Schedule.md) | 실제 타임라인, 일 쪼개기, 브랜치 전략 |

## 원자료

이 Wiki 의 수치는 모두 저장소 안 문서나 직접 실행에서 왔습니다.

- [`docs/code_current.md`](../code_current.md): 담당자별 실행 기록 (원자료)
- [`docs/IMPLEMENTATION_REPORT.md`](../IMPLEMENTATION_REPORT.md): 원자료를 정리한 구현 보고서
- [`data/README.md`](../../data/README.md): 데이터 출처, 가공 규칙, 시연 순서
- 2026-11-05 `main`(`c1f65aa`) 컨테이너 재실행 (README 캡처와 Evaluation 의 "재실행" 표)

[README 로 돌아가기](../../README.md)
