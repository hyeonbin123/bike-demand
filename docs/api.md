# API 계약

FastAPI 서비스(`src/bike_demand/api/`)가 지키는 요청·응답 형식. 데이터는 서비스용 PostgreSQL(`docs/serving-schema.md`)에서만 읽는다. 이 문서를 바꾸면 구현과 테스트도 함께 바꾼다.

## 공통
- 시각은 모두 KST ISO 8601 문자열(`2026-09-17T08:00:00+09:00`)
- 오류 응답 형식은 FastAPI 기본 `{"detail": ...}`
  - 404: 없는 대여소
  - 422: 요청 값 형식·범위 오류
  - 503: 판단에 필요한 최신 데이터가 없음(`detail`에 무엇이 없는지)
  - DB에 접속하지 못하면 모든 요청이 503 `{"detail": "database unavailable"}`
- 인증 없음(읽기 전용 공개 데이터)

### 어느 예측을 쓰나 (모든 응답 공통)
- `predictions`에는 예측 작업이 돌 때마다(단기예보 발표마다) 새 `model_version` 묶음이 쌓이고, 같은 시간을 여러 버전이 예측한다
- **쓸 버전 = `created_at`이 가장 늦은 행의 `model_version`** (가장 최근에 만든 묶음). 같으면 `model_version` 문자열이 큰 것
  - `/predictions/{station_id}`만 예외: **요청한 날짜(KST 0~23시)에 어느 대여소든 행이 있는 버전** 중에서 위 규칙으로 고른다(대여소별로 고르지 않음)
- 한 응답 안에서 **버전을 섞지 않는다**. 고른 버전에 필요한 시간이 없으면 그 시간은 없는 것으로 처리한다(아래 각 절의 규칙)
- 응답의 `model_version`은 고른 버전이다. 고를 버전이 없을 때만 `null`: 보통은 `predictions` 테이블이 비었을 때, `/predictions/{station_id}`는 요청한 날짜에 어느 대여소의 행도 없을 때
- 버전은 **전체 DB 기준으로 하나** 고른다(대여소·자치구·시간 구간마다 따로 고르지 않음). 그 버전에 어떤 대여소·시간이 없으면 이전 버전으로 채우지 않는다

### 숫자 반올림
- `expected_rentals`, `shortfall`: 소수 첫째 자리
- `predicted_rentals`: 소수 둘째 자리
- 좌표(`lat`, `lon`), 대수(`bike_count`, `rack_count`, `docks`)는 반올림하지 않고 저장된 값 그대로
- 계산(합계, 비교, 정렬)은 반올림 전 값으로 한다

## `GET /health`
```json
{"status": "ok", "database": "ok", "latest_snapshot_at": "2026-09-17T08:10:03+09:00", "latest_prediction_hour": "2026-09-17T23:00:00+09:00"}
```
- DB에 접속하지 못하면 503 `{"detail": "database unavailable"}`
- 스냅샷·예측이 하나도 없으면 해당 값은 `null`(상태는 `ok`)

## `GET /stations`
쿼리: `district`(선택, 자치구 이름 정확히 일치)

```json
[
  {"station_id": "ST-1121", "station_no": 1653, "station_name": "노원역1번출구", "district": "노원구",
   "lat": 37.655, "lon": 127.061, "docks": 15}
]
```
- `station_id` 순서. 좌표를 모르는 대여소는 `lat`·`lon`이 `null`

## `GET /stations/{station_id}`
대여소 정보 + 가장 최근 스냅샷 + 지금 시각부터 앞으로 6시간의 시간별 예측.

```json
{
  "station": {"station_id": "ST-1121", "station_no": 1653, "station_name": "노원역1번출구", "district": "노원구",
              "lat": 37.655, "lon": 127.061, "docks": 15},
  "snapshot": {"fetched_at": "2026-09-17T08:10:03+09:00", "bike_count": 4, "rack_count": 15},
  "predictions": [
    {"hour_start": "2026-09-17T08:00:00+09:00", "predicted_rentals": 3.2},
    {"hour_start": "2026-09-17T09:00:00+09:00", "predicted_rentals": 1.9}
  ],
  "model_version": "v1-..."
}
```
- `snapshot`: 없으면 `null`
- `predictions`: 현재 시각이 든 시간부터 6시간 중 고른 버전에 이 대여소의 값이 있는 시간만(순서대로). 없으면 빈 목록(이때도 `model_version`은 고른 버전, `predictions` 테이블이 비었을 때만 `null`)

## `GET /shortage-risk`
곧 자전거가 부족해질 대여소 목록. 서비스의 핵심 응답.

쿼리
| 이름 | 기본 | 범위 | 뜻 |
|---|---|---|---|
| `hours` | 3 | 1~6 | 앞으로 몇 시간을 볼지 |
| `limit` | 50 | 1~500 | 최대 몇 곳 |
| `district` | 없음 | | 자치구 필터 |
| `max_snapshot_age_minutes` | 30 | 1~180 | 이보다 오래된 스냅샷을 가진 대여소는 뺌 |

계산 (대여소마다)
- `as_of` = 그 대여소의 가장 최근 스냅샷 시각 (`max_snapshot_age_minutes` 안쪽인 것만)
- 보는 구간 = [`as_of`, `as_of` + `hours`시간)
- `expected_rentals` = 구간과 겹치는 시간별 예측의 합. 구간에 일부만 걸친 시간은 **겹친 분의 비율만큼** 곱한다 (예: `as_of` 08:20, `hours`=3 → 08시 예측 × 40/60 + 09시 + 10시 + 11시 × 20/60)
- `bike_count` = 그 스냅샷의 자전거 수
- `shortfall` = `expected_rentals` − `bike_count`
- 반납은 예측하지 않으므로 이 값은 "반납이 없을 때 모자랄 수 있는 대수"다. 응답과 화면 설명에 이 한계를 적는다
- 필요한 시간의 예측이 하나라도 없는 대여소는 뺀다

정렬·필터: `shortfall > 0`인 대여소만, `shortfall` 큰 순, 같으면 `station_id` 순.

```json
{
  "hours": 3,
  "model_version": "v1-...",
  "generated_at": "2026-09-17T08:12:00+09:00",
  "note": "반납은 반영하지 않은 값",
  "stations": [
    {"station_id": "ST-1121", "station_name": "노원역1번출구", "district": "노원구", "lat": 37.655, "lon": 127.061,
     "as_of": "2026-09-17T08:10:03+09:00", "bike_count": 1, "expected_rentals": 6.4, "shortfall": 5.4}
  ]
}
```
- 503은 **필터 전 전체 데이터** 기준으로만 낸다
  - `predictions` 테이블이 비었으면 503 `{"detail": "no predictions"}` (먼저 확인)
  - 모든 대여소를 통틀어 `max_snapshot_age_minutes` 안의 스냅샷이 하나도 없으면 503 `{"detail": "no recent snapshot"}`
- 그 밖에 자치구 필터, 필요한 시간의 예측이 빠진 대여소 제외, `shortfall > 0` 조건 때문에 남는 대여소가 없으면 **200과 빈 `stations` 목록**
- 반올림은 위 "숫자 반올림" 규칙

## `GET /predictions/{station_id}`
쿼리: `date`(필수, `YYYY-MM-DD`, KST 날짜)

```json
{"station_id": "ST-1121", "date": "2026-09-17", "model_version": "v1-...",
 "hours": [{"hour_start": "2026-09-17T00:00:00+09:00", "predicted_rentals": 0.1}]}
```
- 그날(KST 0~23시) 어느 대여소든 행이 있는 버전 중 `created_at`이 가장 늦은 버전에서, 이 대여소의 시간들(없는 시간은 목록에서 빠짐)
- 예: 같은 날 A 대여소는 옛 버전에만, B는 새 버전에만 있으면 A의 응답은 새 버전 + 빈 `hours`(옛 버전으로 채우지 않음)
- 없는 대여소 404. 그날 행이 전혀 없으면 `model_version`은 `null`, `hours`는 빈 목록
