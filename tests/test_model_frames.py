# [무엇을 검증하나] 측정용 행·특징을 만드는 model/frames.py와 지표 계산 model/metrics.py의 기본
# 규칙을 확인한다: 추세 기간 계산, 대여소 패턴을 학습 기간(history)으로만 계산하기(평가 기간 값 누수
# 방지), 학습에 없던 새 대여소의 처리, 평가 기간에 대여가 없는 대여소 제외, 3시간 합 MAE 등 지표,
# 그리고 load_frame이 돌려주는 행 순서가 (station_code, day_index, hour_of_day) 오름차순으로
# 고정되는지(docs/experiments.md '학습 행 순서 고정' 절).
# [왜 중요한가] 이 프로젝트의 핵심 원칙이 '예측할 때 알 수 있는 것만 특징으로'다(README 설계 판단).
# 평가 기간의 대여가 대여소 패턴에 섞이면 validation 점수가 실제보다 좋게 나온다. 행 순서는
# LightGBM의 구간 표본과 bagging이 행 번호로 고르기 때문에 같은 자료·seed로 다시 학습해도 다른
# 모델이 나오는 원인이었다(SQL에 ORDER BY가 없어 DuckDB가 실행마다 다른 순서를 돌려줌).

import numpy as np
import pytest

# small_warehouse fixture(conftest)는 학습 기간 ST-1의 평일 8시 2대, 평가 기간 100대로 일부러 크게
# 다르게 만든 warehouse다.
from bike_demand.model import frames
from bike_demand.model.metrics import evaluate


# 학습 기간이 2025-01-01에 끝나면 '마지막 6개월'은 2024-07~12, '그 1년 전 같은 6개월'은
# 2023-07~12다(docs/experiments.md v1의 대여소 추세 정의).
def test_trend_windows():
    last, prior = frames.trend_windows(("2023-01-01", "2025-01-01"))
    assert last == ("2024-07-01", "2025-01-01")
    assert prior == ("2023-07-01", "2024-01-01")


# 평가 행(1/16~)의 profile_mean과 기준선 B0이 학습 기간 평균 2대인지 본다(평가 기간 100대가 섞이면
# 값이 커진다). 평가 행의 실제 대여는 100대 그대로다.
# 학습 기간에 없던 ST-2: 패턴·추세 특징은 결측(LightGBM이 결측으로 처리), 기준선 B0은 전체 대여소의
# 같은 요일구분×시간 평균으로 채운다(측정 전 보충 규칙).
def test_profile_uses_history_only_and_baselines_fall_back(small_warehouse):
    window = frames.Window(rows=("2024-01-16", "2024-02-01"), history=("2024-01-01", "2024-01-16"))
    frame = frames.load_frame(small_warehouse, window, only_active_stations=True)
    st1 = frame["station_code"] == 0
    weekday_8 = (frame["hour_of_day"] == 8) & (frame["is_offday"] == 0)

    # 평가 기간의 100대가 섞이지 않고 학습 기간 평균(2대)
    assert np.allclose(frame["profile_mean"][st1 & weekday_8], 2.0)
    assert np.allclose(frame["b0"][st1 & weekday_8], 2.0)
    # 학습 기간에 없던 ST-2: 패턴 특징은 결측, 기준선은 전체 대여소 평균(여기서는 ST-1뿐이라 2.0)
    st2 = frame["station_code"] == 1
    assert np.isnan(frame["profile_mean"][st2]).all()
    assert np.allclose(frame["b0"][st2 & weekday_8], 2.0)
    assert np.isnan(frame["station_trend"][st2]).all()
    assert frame["rentals"][st1 & weekday_8].min() == 100


# only_active_stations=True면 그 기간에 대여가 있는 대여소만 평가 행에 넣는다(계획의 '평가 대상
# 행'). fixture의 ST-2는 1/16 이후에만 격자 행이 있어 1/1~1/15에는 거르지 않아도 나오지 않는다.
# 그래서 이 기간에 대여 0건인 ST-2 격자 행을 더해, 거르는 조건이 있어야만 ST-1만 남게 만든다.
def test_inactive_stations_are_dropped_from_evaluation(small_warehouse):
    small_warehouse.execute("""
        insert into int_station_hour_grid
        select 'ST-2', hour_start, 0 from dim_hours where hour_start < '2024-01-16'
    """)
    window = frames.Window(rows=("2024-01-01", "2024-01-16"), history=("2024-01-01", "2024-01-16"))
    frame = frames.load_frame(small_warehouse, window, only_active_stations=True)
    assert set(np.unique(frame["station_code"])) == {0.0}
    # 거르지 않으면 대여 0건인 ST-2 행도 들어온다. 위 단언이 거르는 조건을 실제로 확인한다는 뜻이다.
    unfiltered = frames.load_frame(small_warehouse, window, only_active_stations=False)
    assert set(np.unique(unfiltered["station_code"])) == {0.0, 1.0}


# 격자 테이블의 물리 순서를 (대여소 내림차순, 시각 내림차순)으로 뒤집은 warehouse를 만든다. 작은
# 자료는 DuckDB가 넣은 순서대로 돌려줘 우연히 정렬돼 보일 수 있으므로, 정렬 검사가 실제로 뜻이
# 있도록 일부러 뒤집는다(뒤집지 않은 fixture는 정렬 없이도 사전순으로 나온다).
def reverse_grid(db):
    db.execute(
        "create table reversed_grid as select * from int_station_hour_grid "
        "order by station_id desc, hour_start desc"
    )
    db.execute("drop table int_station_hour_grid")
    db.execute("alter table reversed_grid rename to int_station_hour_grid")
    return db


# (station_code, day_index, hour_of_day) 오름차순이면 lexsort가 항등 순열을 돌려준다.
def is_sorted(frame) -> bool:
    order = np.lexsort((frame["hour_of_day"], frame["day_index"], frame["station_code"]))
    return bool(np.array_equal(order, np.arange(len(order))))


# 격자 행을 거꾸로 넣어도 load_frame 결과는 (station_code, day_index, hour_of_day) 사전순이어야
# 한다. 수준 특징(with_levels)을 붙이는 조인이 더해져도 같다. 정렬이 없으면 DuckDB가 넣은
# 순서(거꾸로) 그대로 돌려줘 실패한다(고치기 전 확인: 같은 자리에 같은 행이 오는 수 0).
@pytest.mark.parametrize("kwargs", [{}, {"with_levels": True}])
def test_load_frame_rows_are_sorted_even_when_the_grid_is_reversed(small_warehouse, kwargs):
    reverse_grid(small_warehouse)
    window = frames.Window(rows=("2024-01-01", "2024-02-01"), history=("2024-01-01", "2024-01-16"))
    frame = frames.load_frame(small_warehouse, window, only_active_stations=False, **kwargs)
    assert len(frame["rentals"]) == 31 * 24 + 16 * 24  # ST-1 한 달 + ST-2 1/16~
    assert is_sorted(frame)
    # 첫 행은 ST-1(번호 0)의 1월 1일 0시, 마지막 행은 ST-2(번호 1)의 1월 31일 23시
    assert (frame["station_code"][0], frame["hour_of_day"][0]) == (0.0, 0.0)
    assert (frame["station_code"][-1], frame["hour_of_day"][-1]) == (1.0, 23.0)
    assert frame["day_index"][0] == np.datetime64("2024-01-01", "D").astype(np.int64)


# 같은 자료를 다른 물리 순서로 넣은 두 warehouse에서 load_frame 결과가 열마다 값까지 같아야
# 한다(결측 위치 포함). 행 순서가 고정돼야 같은 입력에서 같은 LightGBM 모델이 나온다.
def test_load_frame_is_the_same_for_two_insertion_orders(small_warehouse):
    window = frames.Window(rows=("2024-01-01", "2024-02-01"), history=("2024-01-01", "2024-01-16"))
    forward = frames.load_frame(small_warehouse, window, False, with_levels=True)
    # 같은 테이블들을 붙인 메모리 DB(other)에 복사한 뒤 격자만 거꾸로 놓고, USE로 기본 카탈로그를
    # 바꿔 load_frame의 SQL(테이블 이름에 카탈로그가 없음)이 그쪽을 읽게 한다
    small_warehouse.execute("attach ':memory:' as other")
    for table in ("dim_stations", "dim_hours", "int_station_hour_grid"):
        small_warehouse.execute(f"create table other.{table} as select * from {table}")
    small_warehouse.execute("use other")
    reverse_grid(small_warehouse)
    assert small_warehouse.execute(
        "select station_id from int_station_hour_grid limit 1"
    ).fetchone() == ("ST-2",)  # 뒤집힌 격자를 읽고 있는지
    backward = frames.load_frame(small_warehouse, window, False, with_levels=True)
    assert list(forward) == list(backward)
    for name in forward:
        np.testing.assert_array_equal(forward[name], backward[name], err_msg=name)


# sort_frame은 모든 열을 같은 순열로 다시 놓아 행의 짝(특징·목표)이 깨지지 않아야 한다. 손으로 셀 수
# 있는 4행: 정렬 키 (대여소, 날짜, 시각)가 (1,5,0), (0,5,3), (0,5,1), (1,4,23)이면 (0,5,1) →
# (0,5,3) → (1,4,23) → (1,5,0) 순이고 rentals도 그 순서(30, 20, 40, 10)로 따라간다.
def test_sort_frame_reorders_every_column_together():
    frame = {
        "station_code": np.array([1, 0, 0, 1], dtype=np.float32),
        "day_index": np.array([5, 5, 5, 4], dtype=np.int64),
        "hour_of_day": np.array([0, 3, 1, 23], dtype=np.float32),
        "rentals": np.array([10, 20, 30, 40], dtype=np.float32),
    }
    result = frames.sort_frame(frame)
    assert result is frame  # 제자리 정렬(같은 사전을 돌려줌)
    assert frame["station_code"].tolist() == [0, 0, 1, 1]
    assert frame["day_index"].tolist() == [5, 5, 4, 5]
    assert frame["hour_of_day"].tolist() == [1, 3, 23, 0]
    assert frame["rentals"].tolist() == [30, 20, 40, 10]
    assert frame["day_index"].dtype == np.int64 and frame["rentals"].dtype == np.float32


# 지표 계산을 손으로 셀 수 있는 4행으로 확인한다. MAE (1+0+1+4)/4, 출퇴근(7시) MAE 4, WAPE = 오차 합
# 6 / 실제 합 10. 3시간 합 MAE는 대여소×날짜×(시간 // 3) 구간마다 실제·예측을 더한 뒤 비교하므로
# 0~2시 구간(6 대 6)과 6~8시 구간(4 대 0)의 평균 2다.
def test_evaluate_three_hour_blocks():
    actual = np.array([1, 2, 3, 4], dtype=np.float32)
    predicted = np.array([2, 2, 2, 0], dtype=np.float32)
    station = np.zeros(4, dtype=np.float32)
    day = np.zeros(4, dtype=np.int64)
    hour = np.array([0, 1, 2, 7], dtype=np.float32)  # 0~2시 한 구간, 7시는 6~8시 구간
    result = evaluate(actual, predicted, station, day, hour)
    assert result["mae"] == pytest.approx((1 + 0 + 1 + 4) / 4)
    assert result["peak_mae"] == pytest.approx(4)
    assert result["mae_3h"] == pytest.approx((abs(6 - 6) + abs(4 - 0)) / 2)
    assert result["wape"] == pytest.approx(6 / 10)
