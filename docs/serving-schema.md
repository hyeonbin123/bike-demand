# 서비스 DB 스키마

서비스용 PostgreSQL 16(`docker compose up -d db`, 호스트 포트 55452). 원본과 정제·집계 데이터는 Parquet와 DuckDB(`data/`)에 있고, 여기에는 API가 바로 읽을 것만 둔다. 테이블 정의는 `src/bike_demand/serving/models.py`, 마이그레이션은 `alembic/versions/`.

```bash
docker compose up -d db
uv run alembic upgrade head
uv run python -m bike_demand.serving.load stations
uv run python -m bike_demand.serving.load realtime --day 2026-09-16
uv run python -m bike_demand.serving.load forecasts
```

접속 문자열은 환경 변수 `DATABASE_URL`, 없으면 `postgresql+psycopg://bike:bike@127.0.0.1:55452/bike_demand`.

## 시간대
모든 시각은 `timestamptz`. 애플리케이션 연결은 세션 시간대를 `Asia/Seoul`로 열어 +09:00으로 읽는다. 적재할 때 들어오는 문자열은 모두 오프셋이 붙은 ISO 8601이다.

## 테이블

### `stations`
| 컬럼 | 형식 | 설명 |
|---|---|---|
| `station_id` | varchar(20) PK | 대여소ID (`ST-xxx`) |
| `station_no` | int | 대여소번호. 실시간에만 있는 곳은 이름 앞 번호("102. …")에서 읽음 |
| `station_name`, `district` | varchar | 이름, 자치구(실시간에만 있는 곳은 자치구 없음) |
| `lat`, `lon` | double | 좌표. 대여소 정보에 없는 곳은 빈 값 |
| `docks` | int | 거치대 수 |
| `source` | varchar(20) | `history`(과거 이력의 `dim_stations`) 또는 `realtime`(실시간 스냅샷에만 있음) |
| `updated_at` | timestamptz | 마지막으로 넣거나 고친 시각 |

- `load stations`: `dim_stations` 전체를 upsert. 같은 대여소가 `realtime`으로 먼저 들어와 있었다면 `history` 값으로 바뀐다
- `load realtime`: 스냅샷에 처음 보는 대여소만 `realtime`으로 추가하고, 있는 대여소는 고치지 않는다

### `realtime_snapshots`
| 컬럼 | 형식 | 설명 |
|---|---|---|
| `station_id` | varchar(20) PK | |
| `fetched_at` | timestamptz PK | 수집을 시작한 시각(한 스냅샷의 모든 대여소가 같은 값) |
| `bike_count` | int | 세워진 자전거 수. 거치대 수보다 많을 수 있음(거치대 밖 주차) |
| `rack_count` | int | 거치대 수 |

인덱스 `ix_realtime_snapshots_fetched_at`. 같은 (대여소, 시각)은 건너뛴다.

### `weather_forecasts`
기상청 단기예보를 긴 형식 그대로. PK (`base_datetime`, `fcst_datetime`, `category`, `nx`, `ny`), `value`는 문자열(`강수없음` 같은 값이 있어서). 같은 키는 건너뛴다(한 발표의 값은 바뀌지 않음).

### `predictions`
| 컬럼 | 형식 | 설명 |
|---|---|---|
| `station_id` | varchar(20) PK | |
| `hour_start` | timestamptz PK | 예측하는 1시간의 시작 |
| `model_version` | varchar(100) PK | 모델과 입력(예보 발표 시각)을 구분하는 이름 |
| `predicted_rentals` | double | 그 시간의 예상 대여 수 |
| `created_at` | timestamptz | |

인덱스 `ix_predictions_hour_start`. 예측 작업(T10)이 채운다. API는 가장 최근 `model_version`을 쓴다(`docs/api.md`).

## 적재 규칙
- 모든 적재는 멱등이다. 같은 입력을 다시 넣어도 행이 늘지 않는다(스냅샷·예보는 `ON CONFLICT DO NOTHING`, 대여소는 upsert)
- 새로 넣은 행 수는 적재 전후 행 수의 차이로 센다. 같은 테이블에 적재 작업을 동시에 돌리지 않는다는 전제다(Airflow에서 작업별 동시 실행 1)
- 테스트(`tests/test_serving_load.py`)는 같은 서버에 임시 DB를 만들어 마이그레이션을 적용하고 끝나면 지운다
