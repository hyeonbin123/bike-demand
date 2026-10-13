"""v5 부족 경보 평가 하네스(shortage_eval)의 계산 규칙. DB 없이 합성 스냅샷으로 확인한다."""

# [무엇을 검증하나] v5 부족 경보 측정 하네스(model/shortage_eval.py)가 측정 전에 커밋한
# 계획(docs/experiments.md v5, 005b643)을 그대로 구현했는지 확인한다. 묶음은 소스의 구역과 같다.
# - 스냅샷과 정답: 기준 스냅샷(as_of 이전 30분 안), 평가 창 (as_of, as_of + 3시간], 10분 칸 15/18
#   규칙, 0대 도달 정답, 30·60분 전 변화
# - 겹침 비율과 입력: API와 같은 부분 시간 비례, 프로파일의 쉬는 날 규칙, 예보 비 옴 구분, 주 평가
#   행과 제외 사유
# - 후보 점수: S1b 순서, S3(Skellam·음이항 순유출 확률), S4(0 흡수 birth–death)를 독립 계산(scipy
#   분포, 오일러 적분)과 비교
# - 지표: 동점을 포함한 AUC·PR-AUC를 전수 계산과 비교, 가중치 = 행 반복, P@K 동점 규칙, API 목록
#   그대로 채점, 재배치 트럭 이벤트
# - 부트스트랩과 판정: 날짜 재표본의 재현성, 판정 규칙의 분기, φ·도전자 동점 규칙
# - 실행 안전장치: test는 평가 시각 뒤에 한 번만, 고정 파일(docs/v5)의 해시가 다르면 거부, 유효
#   발표가 없으면 시작 표시 전에 멈춤, 계획 상수
# [왜 중요한가] 판정은 test 14일을 한 번만 재서 내리는 것이라 코드가 계획과 다르면 되돌릴 수 없다.
# 그래서 DB 없이 합성 스냅샷으로 규칙 하나하나를 미리 고정해 둔다(v5 test는 실행하지 않았지만
# 하네스와 테스트는 남아 있다, docs/experiments.md 'v5 중단').

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# scipy.stats: Skellam·음이항·포아송 분포로 후보 확률을 독립적으로 다시 계산한다. expected_rentals:
# API의 부분 시간 비례(비교 기준).
import numpy as np
import pytest
from scipy import stats

from bike_demand.api.shortage import expected_rentals
from bike_demand.model import shortage_eval as se

KST = se.KST
# 기본 발표: 2026-10-06(화) 08시, as_of는 15분 뒤 08:15.
BASE = datetime(2026, 10, 6, 8, 0, tzinfo=KST)  # 화요일 08시 발표, as_of 08:15


# 대여소별 자전거 수 목록을 start부터 10분 간격 스냅샷으로 만든다. None을 넣으면 그 시각 스냅샷에 그
# 대여소가 빠진다(칸 부족을 만들 때 씀).
def snapshots(series: dict[str, list[int | None]], start: datetime, step_min: int = 10):
    """대여소별 자전거 수 목록 → Snapshots. None은 그 시각에 그 대여소가 없음."""
    rows = []
    for station, counts in series.items():
        for i, count in enumerate(counts):
            if count is not None:
                rows.append((station, start + timedelta(minutes=step_min * i), count))
    return se.Snapshots.from_rows(rows)


# 함수 series_fn(대여소, 순번)으로 여러 대여소의 긴 스냅샷 열을 만드는 도우미.
def full_day(series_fn, stations=("A", "B"), start=BASE - timedelta(hours=2), n=60):
    return snapshots({s: [series_fn(s, i) for i in range(n)] for s in stations}, start)


# 발표 입력(예측·예보 강수) 하나. 기본은 둘 다 비어 있다.
def issue(predictions=None, rain=None, base=BASE):
    return se.IssueInput(base, predictions or {}, rain or {})


# 빈 프로파일. 정답·칸 규칙만 볼 때 쓴다(R·Q는 결측).
PROFILE = se.Profile({})


# ---------------------------------------------------------------- 스냅샷과 정답


# (대여소, 시각, 자전거 수) 행을 순서 없이 넣어도 시각·대여소를 정렬한 행렬이 되고, 그 시각에 없는
# 대여소 칸은 −1이다.
def test_from_rows_builds_sorted_matrix_with_missing_cells():
    t0 = BASE
    snaps = se.Snapshots.from_rows(
        [("B", t0 + timedelta(minutes=10), 3), ("A", t0, 5), ("B", t0, 4)]
    )
    assert snaps.stations == ["A", "B"]
    assert snaps.times == [t0, t0 + timedelta(minutes=10)]
    assert snaps.bikes.tolist() == [[5, 4], [-1, 3]]


# 기준 스냅샷은 as_of 이전(같은 시각 포함) 30분 안의 가장 최근 것이다. 31분 전 것만 있으면 기준이
# 없고, 정확히 30분 전과 as_of 시각 스냅샷이 있으면 as_of 시각의 것이 기준이다(API의 30분 한도와
# 같은 경계).
def test_reference_is_latest_snapshot_within_30_minutes_before_as_of():
    as_of = BASE + se.SERVICE_DELAY
    snaps = se.Snapshots.from_rows(
        [("A", as_of - timedelta(minutes=31), 1), ("A", as_of - timedelta(minutes=5), 2)]
    )
    assert snaps.reference(as_of) == 1
    only_old = se.Snapshots.from_rows([("A", as_of - timedelta(minutes=31), 1)])
    assert only_old.reference(as_of) is None
    exactly = se.Snapshots.from_rows([("A", as_of - timedelta(minutes=30), 1), ("A", as_of, 7)])
    assert exactly.reference(as_of) == 1  # as_of와 같은 시각도 기준이 된다


# 발표 단위 규칙: 08:10 기준 스냅샷이 있고 창 (08:15, 11:15]의 10분 칸 18개가 다 있으면 ok, 4칸이
# 빠져 14칸이면 sparse(15칸 이상 필요), 첫 스냅샷이 08:20이면 기준이 없어 no_reference.
def test_issue_status_needs_reference_and_15_of_18_slots():
    start = BASE + timedelta(minutes=10)  # 08:10, 08:20, ..., 기준은 08:10
    dense = snapshots({"A": [3] * 19}, start)
    assert se.issue_status(dense, BASE + se.SERVICE_DELAY) == "ok"
    # 창 (08:15, 11:15]의 18칸 중 4칸을 지우면 14칸
    gappy = snapshots({"A": [3, 3, None, None, None, None] + [3] * 13}, start)
    assert se.issue_status(gappy, BASE + se.SERVICE_DELAY) == "sparse"
    late = snapshots({"A": [3] * 19}, BASE + timedelta(minutes=20))
    assert se.issue_status(late, BASE + se.SERVICE_DELAY) == "no_reference"


# 같은 10분 칸에 스냅샷이 두 번 있어도(90초 뒤 재수집) 한 칸으로 센다. 창 안 스냅샷은 28개지만 칸은
# 14개라 sparse다. 스냅샷 수로 세면 수집이 겹친 발표가 규칙을 잘못 통과한다.
def test_duplicate_snapshots_in_one_slot_count_once():
    start = BASE + timedelta(minutes=10)
    rows = [("A", start + timedelta(minutes=10 * i), 3) for i in range(15)]  # 칸 15개 = 기준 + 14
    rows += [("A", start + timedelta(minutes=10 * i, seconds=90), 3) for i in range(1, 15)]
    snaps = se.Snapshots.from_rows(rows)
    window = snaps.window(BASE + se.SERVICE_DELAY, 3)
    assert len(window) == 28
    assert se.distinct_slots(snaps, window) == 14
    assert se.issue_status(snaps, BASE + se.SERVICE_DELAY) == "sparse"


# 정답 y3: 창 안 스냅샷 중 하나라도 0대면 1. 창 끝 11:15의 0은 들어가고(끝 포함), 11:25의 0은 창
# 밖이다. 처음부터 0대인 대여소도 정답은 1이지만 b0 = 0이라 주 평가에서는 따로 센다. recovered3은 창
# 안에서 1대 이상이 된 적이 있는지(이미 0대인 곳의 보고용).
def test_label_counts_zero_up_to_and_including_the_window_end():
    start = BASE + timedelta(minutes=15)  # 08:15가 기준(as_of와 같은 시각), 창 끝은 11:15
    n = 40
    series = {
        "hit": [2] * 10 + [0] + [1] * (n - 11),
        "edge": [2] * 18 + [0] + [2] * (n - 19),  # 11:15 = 창 끝, 들어감
        "after": [2] * 19 + [0] + [2] * (n - 20),  # 11:25, 창 밖
        "empty": [0] * n,
    }
    snaps = snapshots(series, start)
    rows = se.issue_rows(snaps, issue(), PROFILE)
    got = dict(zip(rows["station"], rows["y3"], strict=True))
    assert got == {"after": 0, "edge": 1, "empty": 1, "hit": 1}
    assert dict(zip(rows["station"], rows["b0"], strict=True))["empty"] == 0
    assert rows["ok3"].all()
    recovered = dict(zip(rows["station"], rows["recovered3"], strict=True))
    assert not recovered["empty"] and recovered["hit"]


# 발표가 규칙을 통과해도, 그 대여소가 창 안 15칸 이상에 나오지 않으면(여기서는 13칸) 그 행만
# 채점에서 뺀다(ok3 False).
def test_station_with_too_few_slots_is_not_scored():
    start = BASE + timedelta(minutes=15)
    series = {"dense": [3] * 40, "sparse": [3] + [None] * 5 + [3] * 34}
    rows = se.issue_rows(snapshots(series, start), issue(), PROFILE)
    ok = dict(zip(rows["station"], rows["ok3"], strict=True))
    assert ok == {"dense": True, "sparse": False}  # 창 18칸 중 13칸


# 분류기 N의 특징 Δ30·Δ60 = b0 − (기준 스냅샷 30·60분 전 ±5분 안 가장 가까운 스냅샷의 자전거 수).
# 1대씩 줄어드는 A는 b0 14, Δ30 −3, Δ60 −6. 30분 전 스냅샷에서 B를 지우면 Δ30은 0, 빈 값 표시
# miss30은 1이 된다.
def test_nowcast_deltas_use_the_snapshot_30_and_60_minutes_before_the_reference():
    start = BASE - timedelta(hours=1) + timedelta(minutes=10)  # 07:10 ... 기준 08:10
    counts = list(range(20, 0, -1)) + [1] * 30  # 07:10 20대 → 08:10 14대
    series = {"A": counts, "B": [5] * 50}
    snaps = snapshots(series, start)
    snaps.bikes[snaps.times.index(BASE + timedelta(minutes=10) - timedelta(minutes=30)), 1] = -1
    rows = se.issue_rows(snaps, issue(), PROFILE)
    a, b = (list(rows["station"]).index(s) for s in ("A", "B"))
    assert rows["b0"][a] == 14
    assert rows["d30"][a] == -3 and rows["d60"][a] == -6
    assert rows["miss30"][b] == 1 and rows["d30"][b] == 0  # 30분 전 스냅샷에 B가 없음
    assert rows["miss60"][b] == 0


# ---------------------------------------------------------------- 겹침 비율과 입력


# as_of의 분(0·15·40)과 창 길이(1·3·6시간) 조합마다, 하네스의 겹침 가중 합이
# API(api/shortage.expected_rentals)의 합과 같다. 계획이 'E는 API와 같은 계산'이라고 정했기
# 때문이다.
@pytest.mark.parametrize("minute", [0, 15, 40])
@pytest.mark.parametrize("hours", [1, 3, 6])
def test_overlap_weights_match_the_api_sum(minute, hours):
    as_of = BASE.replace(minute=minute)
    first = as_of.replace(minute=0)
    rng = np.random.default_rng(minute + hours)
    hourly = {first + timedelta(hours=k): float(rng.uniform(0, 5)) for k in range(7)}
    values = np.array([[hourly[first + timedelta(hours=k)] for k in range(7)]])
    weights = se.overlap_weights(minute / 60, hours)
    assert se.horizon_sum(values, weights)[0] == pytest.approx(
        expected_rentals(as_of, hours, hourly)
    )


# 창에 쓰는 시간(as_of 08:15 + 3시간 → 0~3번째 시간)에 빈 값이 있으면 결측, 쓰지 않는 시간(5번째)의
# 빈 값은 무시한다.
def test_horizon_sum_is_missing_only_when_a_used_hour_is_missing():
    weights = se.overlap_weights(0.25, 3)  # 시간 0..3 사용, 4..6 안 씀
    values = np.ones((2, 7))
    values[0, 5] = np.nan
    values[1, 3] = np.nan
    out = se.horizon_sum(values, weights)
    assert out[0] == pytest.approx(3.0)
    assert np.isnan(out[1])


# 프로파일 R·Q는 대상 시각의 쉬는 날 여부를 서비스 예측과 같은 규칙(주말 또는 KR 공휴일)으로 고른다.
# 한글날(10-09 금) 22:15부터 3시간은 금요일 22·23시(공휴일)와 토요일 0·1시라 모두 쉬는 날 값(10,
# 5)을 0.75 + 1 + 1 + 0.25 비율로 더해 R 30, Q 15. 평일 화요일은 평일 값 1 × 3 = 3.
def test_profile_inputs_use_service_offday_rule_and_overlap_weights():
    profile = se.Profile(
        {("A", False, h): (1.0, 0.5) for h in range(24)}
        | {("A", True, h): (10.0, 5.0) for h in range(24)}
    )
    friday = datetime(2026, 10, 9, 22, 0, tzinfo=KST)  # 한글날(공휴일)
    snaps = snapshots({"A": [5] * 40}, friday + timedelta(minutes=10))
    rows = se.issue_rows(snaps, issue(base=friday), profile)
    se.add_horizon_inputs(rows)
    assert rows["offday"][0] == 1.0
    # 22:15~01:15: 9일 22·23시(공휴일) 0.75+1, 10일(토) 0·1시 1+0.25 → 모두 쉬는 날
    assert rows["R3"][0] == pytest.approx(30.0)
    assert rows["Q3"][0] == pytest.approx(15.0)
    tuesday = se.issue_rows(
        snapshots({"A": [5] * 40}, BASE + timedelta(minutes=10)), issue(), profile
    )
    se.add_horizon_inputs(tuesday)
    assert tuesday["R3"][0] == pytest.approx(3.0)


# 비 옴 구분(보고만)은 그 발표의 예보 강수로 정한다. 창과 겹치는 시간(08~11시, 11시는 15분만 겹침)에
# 강수가 하나라도 있으면 1, 모두 0이면 0, 예보가 없으면 −1(알 수 없음).
def test_rain_flag_comes_from_the_issue_forecast_over_the_window():
    snaps = snapshots({"A": [5] * 40}, BASE + timedelta(minutes=10))
    hours = [BASE + timedelta(hours=k) for k in range(4)]
    dry = {h: 0.0 for h in hours}
    assert se.issue_rows(snaps, issue(rain=dry), PROFILE)["wet"][0] == 0
    wet = dry | {hours[3]: 0.5}  # 11:00~11:15도 창과 겹친다
    assert se.issue_rows(snaps, issue(rain=wet), PROFILE)["wet"][0] == 1
    assert se.issue_rows(snaps, issue(rain={}), PROFILE)["wet"][0] == -1


# 주 평가 행 = b0 ≥ 1, 대여소 칸 규칙 통과, E·R·Q가 모두 있음. 빠진 행은 이유별로 한 번씩만
# 센다(이미 0대 → 칸 부족 → 예측 없음 → 프로파일 없음 순서).
def test_main_mask_and_exclusions():
    rows = {
        "b0": np.array([0, 1, 2, 3, 4]),
        "ok3": np.array([True, False, True, True, True]),
        "E3": np.array([1.0, 1.0, np.nan, 1.0, 1.0]),
        "R3": np.array([1.0, 1.0, 1.0, np.nan, 1.0]),
        "Q3": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    }
    assert se.main_mask(rows).tolist() == [False, False, False, False, True]
    assert se.exclusion_counts(rows) == {
        "rows": 5,
        "already_empty": 1,
        "station_sparse": 1,
        "no_prediction": 1,
        "no_profile": 1,
        "main": 1,
    }


# ---------------------------------------------------------------- 후보 점수


# S1b = −b0 + 0.001·tanh(R − Q). tanh가 −1~1이라 보정항이 0.001을 넘지 못해 b0 순서를 절대 뒤집지
# 않고, 같은 b0 안에서만 순유출이 큰 쪽을 앞에 둔다.
def test_s1b_orders_by_bikes_then_profile_net_outflow():
    b0 = np.array([1, 1, 2, 2, 3])
    rent = np.array([0.0, 5.0, 50.0, 0.0, 99.0])
    ret = np.zeros(5)
    order = np.argsort(-se.score_s1b(b0, rent, ret), kind="stable")
    assert order.tolist() == [1, 0, 2, 3, 4]


# S3에서 φ = 1이면 대여·반납이 독립 Poisson이라 순유출은 Skellam 분포다. P(순유출 ≥ b0)를 scipy의
# skellam.sf(b0 − 1)과 비교한다. 대여 기대값이 0이면 확률 0.
@pytest.mark.parametrize("b0,e,q", [(1, 0.3, 0.1), (3, 2.5, 1.0), (4, 1.0, 0.5), (2, 0.0, 0.0)])
def test_net_outflow_with_phi_one_is_skellam(b0, e, q):
    got = se.net_outflow_prob(np.array([b0]), np.array([e]), np.array([q]), 1.0)[0]
    if e == 0:
        assert got == pytest.approx(0.0, abs=1e-9)
    else:
        assert got == pytest.approx(stats.skellam.sf(b0 - 1, e, max(q, 1e-12)), rel=1e-6)


# φ > 1이면 대여·반납이 분산 φ × 평균인 음이항이다(n = μ/(φ − 1), p = 1/φ). 반납 수 k마다 P(반납 =
# k) × P(대여 ≥ b0 + k)를 200까지 직접 더한 값과 비교하고, 분산이 φ × 평균인지도 확인한다. φ가 크면
# 꼬리가 두꺼워져 먼 사건(b0 = 6) 확률이 커진다.
def test_net_outflow_with_phi_above_one_is_a_negative_binomial_difference():
    b0, e, q, phi = 2, 3.0, 1.5, 2.0
    out = stats.nbinom(e / (phi - 1), 1 / phi)
    inn = stats.nbinom(q / (phi - 1), 1 / phi)
    brute = sum(inn.pmf(k) * out.sf(b0 + k - 1) for k in range(200))
    got = se.net_outflow_prob(np.array([b0]), np.array([e]), np.array([q]), phi)[0]
    assert got == pytest.approx(brute, rel=1e-9)
    assert out.var() == pytest.approx(phi * e)
    wider = se.net_outflow_prob(np.array([6]), np.array([e]), np.array([q]), 3.0)[0]
    narrow = se.net_outflow_prob(np.array([6]), np.array([e]), np.array([q]), 1.0)[0]
    assert wider > narrow  # 큰 φ는 먼 꼬리를 키운다


# S4에서 반납이 없으면 자전거는 줄기만 하므로 '0에 닿을 확률' = P(구간별 대여 합 ~ Poisson(Σ λ·t) ≥
# b0). 균등화로 푼 값이 포아송 꼬리와 같아야 한다.
def test_absorb_without_returns_is_a_poisson_tail():
    lam = np.array([[4.0, 2.0, 6.0, 1.0]])
    dur = np.array([[0.75, 1.0, 1.0, 0.25]])
    for b0 in (1, 3, 8):
        got = se.absorb_prob(np.array([b0]), lam, np.zeros_like(lam), dur)[0]
        assert got == pytest.approx(stats.poisson.sf(b0 - 1, (lam * dur).sum()), rel=1e-9)


# 반납이 있으면 확률이 낮아지고, 자전거가 많을수록 낮아진다. 위쪽을 b0 + 30에서 자른 것(계획)이
# 결과에 영향이 없는지 b0 + 80과 비교하고, 대여가 0이면 확률 0이다.
def test_absorb_with_returns_is_lower_and_insensitive_to_headroom():
    lam = np.array([[3.0, 3.0, 3.0, 3.0]] * 3)
    mu = np.array([[2.0, 2.0, 2.0, 2.0]] * 3)
    dur = np.array([[0.75, 1.0, 1.0, 0.25]] * 3)
    b0 = np.array([1, 2, 4])
    with_returns = se.absorb_prob(b0, lam, mu, dur)
    no_returns = se.absorb_prob(b0, lam, np.zeros_like(mu), dur)
    assert (with_returns < no_returns).all()
    assert np.all(np.diff(with_returns) < 0)
    assert se.absorb_prob(b0, lam, mu, dur, headroom=80) == pytest.approx(with_returns, abs=1e-12)
    assert se.absorb_prob(b0, np.zeros_like(lam), mu, dur) == pytest.approx(0.0)


# 같은 birth–death 모델(0 흡수, 위 끝 b0 + 30)의 상태 확률 미분방정식을 구간마다 20,000단계
# 오일러법으로 직접 적분한 값과, 균등화 해의 0 상태 확률이 2e-4 안에서 같은지 본다. 구현과 독립된
# 방법으로 수식 자체를 확인한다.
def test_absorb_matches_a_fine_euler_solution():
    b0, lam, mu = 2, np.array([5.0, 1.0]), np.array([1.0, 4.0])
    dur = np.array([0.5, 1.0])
    top = b0 + 30
    p = np.zeros(top + 1)
    p[b0] = 1.0
    for rate_down, rate_up, t in zip(lam, mu, dur, strict=True):
        steps = 20000
        dt = t / steps
        for _ in range(steps):
            flow_down = rate_down * p[1:] * dt
            flow_up = rate_up * p[1:top] * dt
            p[1:] -= flow_down
            p[:-1] += flow_down
            p[1:top] -= flow_up
            p[2 : top + 1] += flow_up
    got = se.absorb_prob(np.array([b0]), lam[None, :], mu[None, :], dur[None, :])[0]
    assert got == pytest.approx(p[0], abs=2e-4)


# ---------------------------------------------------------------- 지표


# AUC의 정의 그대로: 모든 (양성, 음성) 쌍에서 양성 점수가 크면 1, 같으면 0.5, 작으면 0의 가중 평균.
# O(n²)라 테스트용.
def brute_auc(score, y, w=None):
    w = np.ones(len(y)) if w is None else w
    num = den = 0.0
    for i, j in itertools.product(range(len(y)), repeat=2):
        if y[i] == 1 and y[j] == 0:
            weight = w[i] * w[j]
            den += weight
            num += weight * (1.0 if score[i] > score[j] else 0.5 if score[i] == score[j] else 0.0)
    return num / den


# 평균 정밀도의 정의 그대로: 높은 점수부터 동점 묶음 단위로 문턱을 내리며, 그 묶음의 양성
# 비율(재현율 증가) × 그때의 정밀도를 더한다.
def brute_ap(score, y):
    total = y.sum()
    ap = 0.0
    for threshold in sorted(set(score), reverse=True):
        tie = score == threshold
        selected = score >= threshold
        ap += y[tie].sum() / total * (y[selected].sum() / selected.sum())
    return ap


# 점수가 0~5 정수라 동점이 많은 자료에서 RankIndex의 빠른 계산(동점 묶음 누적합)이 전수 계산과 같다.
# S1처럼 동점이 많은 후보의 AUC가 정의대로 계산되는지 확인한다.
def test_auc_and_average_precision_with_ties_match_brute_force():
    rng = np.random.default_rng(0)
    score = rng.integers(0, 6, 40).astype(float)
    y = (rng.uniform(size=40) < 0.3).astype(float)
    index = se.RankIndex.of(score)
    assert index.auc(y) == pytest.approx(brute_auc(score, y))
    assert index.average_precision(y) == pytest.approx(brute_ap(score, y))


# 가중치 w는 '그 행을 w번 반복'과 같아야 한다. 날짜 블록 부트스트랩이 뽑힌 횟수를 가중치로 넣어
# 계산하므로, AUC·PR-AUC·Brier·log loss 모두 실제로 행을 반복한 결과와 같은지 본다(가중치 0은 행을
# 빼는 것).
def test_weights_are_row_repetitions():
    rng = np.random.default_rng(1)
    score = rng.normal(size=30).round(1)
    y = (rng.uniform(size=30) < 0.4).astype(float)
    w = rng.integers(0, 4, 30).astype(float)
    repeated = np.repeat(np.arange(30), w.astype(int))
    index = se.RankIndex.of(score)
    assert index.auc(y, w) == pytest.approx(se.RankIndex.of(score[repeated]).auc(y[repeated]))
    assert index.average_precision(y, w) == pytest.approx(
        se.RankIndex.of(score[repeated]).average_precision(y[repeated])
    )
    p = rng.uniform(size=30)
    assert se.brier(p, y, w) == pytest.approx(se.brier(p[repeated], y[repeated]))
    assert se.log_loss(p, y, w) == pytest.approx(se.log_loss(p[repeated], y[repeated]))


# P@K는 같은 점수면 station_id 순으로 자른다(계획). a·b·c가 동점이면 a, b, c 순이라 상위 2곳은 모두
# 양성, 3곳이면 2/3.
def test_precision_at_k_breaks_ties_by_station_id():
    score = np.array([1.0, 1.0, 1.0, 0.0])
    y = np.array([0, 1, 1, 1])
    station = np.array(["c", "a", "b", "d"], dtype=object)
    issue_ids = np.zeros(4, dtype=int)
    got = se.precision_at_k(score, y, issue_ids, station, ks=(1, 2, 3))
    assert got == {1: 1.0, 2: 1.0, 3: pytest.approx(2 / 3)}


# API 그대로 채점(S0): 이미 0대인 곳도 포함해 부족 대수 > 0인 곳만 큰 순으로 limit곳을 고르고(목록
# 길이 3), 칸이 부족한 대여소(d)는 목록에는 남되 채점에서만 뺀다. 그래서 채점은 b(음성)·a(양성, 이미
# 0대) 두 곳으로 정밀도 0.5, 이미 0대 비율 0.5.
def test_api_list_includes_already_empty_stations_and_only_positive_shortfall():
    rows = {
        "issue": np.zeros(5, dtype=int),
        "station": np.array(["a", "b", "c", "d", "e"], dtype=object),
        "b0": np.array([0, 1, 5, 2, 0]),
        "E3": np.array([0.5, 3.0, 1.0, 2.5, np.nan]),
        "ok3": np.array([True, True, True, False, True]),
        "y3": np.array([1, 0, 0, 1, 1]),
    }
    got = se.api_list_precision(rows, limit=3)
    # 부족 대수: b 2.0, a 0.5, d 0.5 (c는 음수, e는 예측 없음). d는 칸 부족이라 채점에서 뺌
    assert got["mean_list_length"] == 3
    assert got["precision"] == pytest.approx(0.5)  # b 0, a 1
    assert got["already_empty_share"] == pytest.approx(0.5)


# 재배치 트럭 보고(판정에 안 씀): 이웃한 스냅샷 사이 +5대 이상은 회복, −5대 이하는 수거로 본다. 0이
# 된 뒤 +7은 '0 이후 회복', 0이 되기 전 −7은 '0 이전 수거', 변화가 작은 곳은 이벤트 없음.
def test_truck_events_mark_jumps_around_the_first_zero():
    series = {
        "refill": [3, 2, 0, 0, 7] + [7] * 20,  # 0 뒤 +7
        "pickup": [9, 2, 1, 0] + [0] * 21,  # 0 전 −7
        "calm": [3, 3, 2] + [2] * 22,
    }
    snaps = snapshots(series, BASE + timedelta(minutes=15))
    rows = se.issue_rows(snaps, issue(), PROFILE)
    by = {s: i for i, s in enumerate(rows["station"])}
    assert rows["truck_recovery_after_zero"][by["refill"]]
    assert rows["truck_pickup_before_zero"][by["pickup"]]
    assert not rows["truck_recovery"][by["calm"]]
    assert not rows["truck_pickup_before_zero"][by["refill"]]


# ---------------------------------------------------------------- 부트스트랩과 판정


# 날짜 재표본: 재표본마다 날짜 수만큼 복원 추출해 뽑힌 횟수의 합이 날짜 수와 같고, 같은 시드면
# 결과가 같다(계획의 시드 20261003으로 결과를 다시 만들 수 있게).
def test_bootstrap_is_reproducible_and_resamples_days():
    counts = se.day_counts(5, n_boot=50, seed=7)
    assert counts.shape == (50, 5)
    assert (counts.sum(axis=1) == 5).all()
    assert np.array_equal(counts, se.day_counts(5, n_boot=50, seed=7))


# 판정 분기를 시험할 합성 평가 결과. 정답은 자전거가 적을수록 잘 나오고, S2(strong='S2'일 때)와 N은
# 잡음이 작아 S1b보다 확실히 낫게, S0은 잡음이 커서 약하게 만든다. 12일이라 유효한 날 10일 조건을
# 넘는다.
def synthetic_eval(n_days=12, strong="S2"):
    rng = np.random.default_rng(3)
    n = 600 * n_days
    day = np.repeat(np.arange(n_days), 600)
    b0 = rng.integers(1, 8, n)
    y = (rng.uniform(size=n) < 1 / (1 + b0)).astype(float)
    noise = lambda s: rng.normal(scale=s, size=n)  # noqa: E731
    signal = y * 2.0 - b0 * 0.3
    scores = {
        "S0": signal + noise(3.0),
        "S1b": -b0 + noise(0.001),
        "S2": signal + noise(0.5) if strong == "S2" else signal + noise(5.0),
        "S3": 1 / (1 + np.exp(-(signal + noise(1.0)))),
        "S4": 1 / (1 + np.exp(-(signal + noise(1.0)))),
        "N": 1 / (1 + np.exp(-(signal * 1.5 + noise(0.2)))),
    }
    return se.evaluate(scores, y, b0, day, probabilities=("S3", "S4", "N"), n_boot=200), day


# 도전자 S2의 AUC − S1b 구간 아래 끝이 0보다 크면 P1 결과는 S2이고, 최종은 S2 또는 N(분류기 조건 네
# 개를 모두 통과할 때만 N). 판정 기록에 조건 네 개가 모두 남는지 보고, 유효한 날이 10일 미만이면
# 판정 자체를 하지 않는다.
def test_decide_switches_to_the_challenger_or_nowcast_only_on_clear_wins():
    ev, _ = synthetic_eval()
    result = se.decide(ev, "S2", valid_days=12)
    assert result["p1_result"] == "S2"
    assert result["p1_ci"]["challenger_vs_S1b"][0] > 0
    assert result["final"] in ("S2", "N")
    assert set(result["nowcast_conditions"]) == {
        "N_vs_S1b_b0ge2",
        "N_vs_P1_b0ge2",
        "brier_not_worse",
        "auc_not_lower",
    }
    assert se.decide(ev, "S2", valid_days=9) == {
        "decided": False,
        "reason": "유효한 날 9일 < 10일",
    }


# 다른 모든 후보를 S1b보다 확실히 낮게 만들면 P1 결과도 최종도 S1b(N도 낮으니 바꾸지 않음). 이어서
# S0 − S1b 구간이 0을 걸치게 만들면 S1b로 바꿀 근거가 없어 S0을 유지하고 할 일은 '/shortage-risk를
# 그대로'.
def test_decide_keeps_s0_or_moves_to_s1b_by_the_s0_interval():
    ev, _ = synthetic_eval(strong="none")
    for metric in ("auc", "pr_auc", "auc_b0ge2", "auc_b0ge3"):
        for name in ev.boot[metric]:
            if name != "S1b":
                ev.boot[metric][name] = ev.boot[metric]["S1b"] - 0.01  # 모두 S1b보다 확실히 낮게
                ev.point[metric][name] = ev.point[metric]["S1b"] - 0.01
    assert se.decide(ev, "S2", 12)["p1_result"] == "S1b"
    assert se.decide(ev, "S2", 12)["final"] == "S1b"
    ev.boot["auc"]["S0"] = ev.boot["auc"]["S1b"] + np.linspace(
        -0.01, 0.01, len(ev.boot["auc"]["S0"])
    )
    assert se.decide(ev, "S2", 12)["p1_result"] == "S0"
    assert se.decide(ev, "S2", 12)["action"].startswith("/shortage-risk를 그대로")


# 정답을 실제로 φ = 2인 음이항 차(numpy의 p = 0.5면 분산 = 2 × 평균)에서 뽑으면 고른 φ는 2 근처이고
# 그 log loss가 가장 낮다. 도전자는 AUC가 같으면 앞의 것(S2 → S3 → S4, 더 단순한 것)을 고른다.
def test_pick_phi_and_challenger_tie_rules():
    rng = np.random.default_rng(5)
    b0 = rng.integers(1, 6, 3000).astype(float)
    e = rng.uniform(0.2, 4, 3000)
    q = rng.uniform(0.2, 3, 3000)
    # 정답을 φ = 2 음이항 차에서 뽑으면 φ = 2 근처가 가장 낮은 log loss
    out = rng.negative_binomial(e / 1.0, 0.5)
    inn = rng.negative_binomial(q / 1.0, 0.5)
    y = (out - inn >= b0).astype(float)
    phi, losses = se.pick_phi(b0, e, q, y)
    assert phi in (1.5, 2.0, 3.0)
    assert losses[str(phi)] == min(losses.values())
    assert se.pick_challenger({"S2": 0.9, "S3": 0.9, "S4": 0.8}) == "S2"
    assert se.pick_challenger({"S2": 0.8, "S3": 0.85, "S4": 0.85}) == "S3"


# 발표마다 상태를 기록한다: 규칙을 지키면 ok(주 평가 행 2개), 예측이 없으면 no_predictions, 스냅샷이
# 끝난 뒤의 발표는 no_reference. ok 발표의 E3은 시간당 1.0 × 3시간. 유효한 날 수는 as_of가 끝 시각
# 전이고 주 평가 행이 있는 발표의 날짜로 센다.
def test_build_rows_logs_each_issue_and_counts_valid_days():
    start = BASE - timedelta(hours=1)
    snaps = snapshots({"A": [1] * 80, "B": [4] * 80}, start)
    hours = {BASE + timedelta(hours=k): 1.0 for k in range(7)}
    profile = se.Profile({(s, False, h): (0.5, 0.2) for s in "AB" for h in range(24)})
    later = BASE + timedelta(hours=13)  # 마지막 스냅샷(20:25)보다 50분 뒤
    inputs = [
        se.IssueInput(BASE, {"A": hours, "B": hours}, {}, "v-fcst1"),
        se.IssueInput(BASE + timedelta(hours=3), {}, {}, None),
        se.IssueInput(later, {"A": hours}, {}, "v-fcst3"),
    ]
    rows, log = se.build_rows(snaps, inputs, profile)
    assert [e["status"] for e in log] == ["ok", "no_predictions", "no_reference"]
    assert log[0]["main_rows"] == 2
    assert rows["E3"][0] == pytest.approx(3.0)
    assert se.valid_days(log, BASE + timedelta(days=1)) == 1
    assert se.valid_days(log, BASE) == 0


# test는 마지막 창이 끝난 뒤(TEST_END + 6시간 이후)에만 돌고, test.json이 이미 있으면 다시 재지
# 않는다. 둘 다 DB를 읽기 전에 멈춘다(engine None으로도 통과).
def test_test_run_refuses_before_the_earliest_time_and_runs_only_once(tmp_path):
    with pytest.raises(SystemExit, match="이후"):
        se.run_test(None, None, tmp_path, se.TEST_END + timedelta(hours=5))
    (tmp_path / "test.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="한 번만"):
        se.run_test(None, None, tmp_path, se.TEST_END + timedelta(days=3))


# 저장소에 바이트 그대로 커밋한 개발 고정 파일 폴더(dev_fit.json·dev_lgbm.txt).
FROZEN = Path(__file__).resolve().parents[1] / "docs" / "v5"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# test 실행 폴더에 고정 파일을 복사해 '정상적으로 준비된 test'를 만든다.
def frozen_out(tmp_path: Path) -> Path:
    """test 실행 폴더에 고정 파일 사본을 둔다."""
    for name in ("dev_fit.json", "dev_lgbm.txt"):
        shutil.copyfile(FROZEN / name, tmp_path / name)
    return tmp_path


# 해시 검사를 통과해 그 다음 단계(load_profile)까지 갔다는 표시로 쓰는 예외.
class PastHashCheck(Exception):
    pass


# 코드의 고정 해시 상수가 docs/experiments.md에 기록한 값과 같고, docs/v5 파일이 그 해시와 바이트
# 단위로 같으며, dev_fit 안의 값(φ 2, 도전자 S4, C 1, 표준화 특징 18개)이 문서와 같다. 줄 끝 변환
# 등으로 파일이 바뀌면 여기서 드러난다(de93428: LightGBM 파일의 CRLF 줄 때문에 docs/v5는 줄 끝
# 변환을 끔).
def test_frozen_dev_files_in_the_repo_match_the_recorded_hashes():
    # docs/experiments.md 'v5 개발 자료로 고정한 값'(8fb59c8)에 적은 SHA-256
    assert (
        se.FROZEN_FIT_SHA256 == "528be96c7e0d10fb674a79f11e25a325d3aef78eb80b2e6a7ee4de97ccdc7f6e"
    )
    assert (
        se.FROZEN_LGBM_SHA256 == "87b25afb0953826f362f799d60681fccd9a85a4e8aa164ba6906aafb4756de38"
    )
    assert _sha(FROZEN / "dev_fit.json") == se.FROZEN_FIT_SHA256
    assert _sha(FROZEN / "dev_lgbm.txt") == se.FROZEN_LGBM_SHA256
    fit = json.loads((FROZEN / "dev_fit.json").read_text("utf-8"))
    assert fit["lgbm"]["sha256"] == se.FROZEN_LGBM_SHA256
    assert (fit["phi"], fit["challenger"], fit["logistic"]["C"]) == (2.0, "S4", 1.0)
    assert len(fit["logistic"]["mean"]) == len(fit["logistic"]["std"]) == 18


# 출력 폴더에 dev_fit.json이나 dev_lgbm.txt가 하나라도 있으면 dev 실행은 덮어쓰지 않고 멈추고 파일도
# 그대로다. test 전에 커밋한 고정값이 다시 돌린 dev로 바뀌는 일을 막는다.
def test_dev_run_refuses_to_overwrite_the_frozen_values(tmp_path):
    for name in ("dev_fit.json", "dev_lgbm.txt"):
        out = tmp_path / name.split(".")[0]
        out.mkdir()
        (out / name).write_text("{}", encoding="utf-8")
        with pytest.raises(SystemExit, match="덮어쓰지"):
            se.run_dev(None, None, None, out)
        assert (out / name).read_text("utf-8") == "{}"


# test는 고정 해시와 같은 두 파일로만 돈다. 파일이 없을 때, dev를 다시 돌려 자기들끼리는 해시가 맞는
# 새 파일일 때, 고정 fit에 다른 LightGBM 파일일 때 모두 '고정한 파일'로 거부한다. 고정 파일 사본을
# 두면 검사를 통과해 다음 단계(가짜 load_profile)까지 가며, 어느 경우에도 test.started는 남지
# 않는다.
def test_test_run_refuses_fit_files_other_than_the_frozen_ones(tmp_path, monkeypatch):
    def past(*args, **kwargs):
        raise PastHashCheck

    monkeypatch.setattr(se, "load_profile", past)
    after = se.TEST_END + timedelta(days=8)
    with pytest.raises(SystemExit, match="고정한 파일"):
        se.run_test(None, None, tmp_path, after)  # 파일 없음
    # dev를 다시 돌려 자기 안에서는 맞는 새 파일이 생긴 경우
    lgbm = tmp_path / "dev_lgbm.txt"
    lgbm.write_text("tree\n", encoding="utf-8")
    fit = {"generation": "g", "phi": 2.0, "challenger": "S4", "lgbm": {"file": lgbm.name}}
    fit["lgbm"]["sha256"] = _sha(lgbm)
    (tmp_path / "dev_fit.json").write_text(json.dumps(fit), encoding="utf-8")
    with pytest.raises(SystemExit, match="고정한 파일"):
        se.run_test(None, None, tmp_path, after)
    # 고정 fit에 다른 LightGBM 파일
    shutil.copyfile(FROZEN / "dev_fit.json", tmp_path / "dev_fit.json")
    with pytest.raises(SystemExit, match="고정한 파일"):
        se.run_test(None, None, tmp_path, after)
    with pytest.raises(PastHashCheck):
        se.run_test(None, None, frozen_out(tmp_path), after)
    assert not (tmp_path / "test.started").exists()


# 저장 예측의 세대가 고정값의 세대와 다르면(예: 2027년 초 서비스 재학습 뒤) 그 발표는 쓰지 않는다.
# 그래서 유효한 발표가 하나도 없으면, 한 번뿐인 시작 표시(test.started)를 쓰기 전에 '다른 세대
# 1개'를 알리며 멈춘다(c1a367f 전에는 시작 표시를 쓴 뒤 빈 행에서 실패했다). DB 읽기 함수들은
# monkeypatch로 합성 자료를 돌려준다.
def test_test_run_stops_before_the_start_mark_when_no_issue_is_valid(tmp_path, monkeypatch):
    base = se.TEST_START + timedelta(hours=2)
    key = se.issue_key(base)
    monkeypatch.setattr(se, "load_profile", lambda warehouse: PROFILE)
    monkeypatch.setattr(
        se, "load_snapshots", lambda *a: snapshots({"A": [3] * 60}, base - timedelta(hours=1))
    )
    # 2027년 초 서비스 재학습 뒤처럼 저장 예측이 다른 세대
    monkeypatch.setattr(se, "stored_versions", lambda *a: {key: f"v9-new-fcst{key}"})
    monkeypatch.setattr(se, "issue_bases", lambda *a: [base])
    monkeypatch.setattr(se, "load_forecast", lambda *a: {})
    out = frozen_out(tmp_path)
    with pytest.raises(SystemExit, match="유효한 발표가 없음.*다른 세대 1개"):
        se.run_test(None, None, out, se.TEST_MAX_END + timedelta(days=1))
    assert not (out / "test.started").exists()
    assert not (out / "test.json").exists()


# 계획에 적은 상수를 고정한다: test 창 2026-10-04부터 14일, 평가 가능 시각 10-18 06:00, 창 연장 한도
# 10-25, 프로파일 기간 2025-09~10, 부트스트랩 2,000번·시드 20261003, φ 후보 {1, 1.5, 2, 3}, 3시간
# 창의 최소 칸 15.
def test_plan_constants_match_the_registered_rules():
    assert se.TEST_START.isoformat() == "2026-10-04T00:00:00+09:00"
    assert se.TEST_END - se.TEST_START == timedelta(days=14)
    assert (se.TEST_END + se.EVAL_DELAY).isoformat() == "2026-10-18T06:00:00+09:00"
    assert se.TEST_MAX_END.isoformat() == "2026-10-25T00:00:00+09:00"
    assert se.PROFILE_WINDOW == ("2025-09-01", "2025-11-01")
    assert (se.BOOT_N, se.BOOT_SEED, se.PHI_GRID) == (2000, 20261003, (1.0, 1.5, 2.0, 3.0))
    assert se.SLOTS_PER_HOUR * se.MAIN_HOURS == 15


# 결과 JSON 저장이 numpy 스칼라·배열과 datetime을 표준 JSON 값으로 바꾸고, 없는 폴더도 만든다.
def test_write_json_handles_numpy_and_dates(tmp_path):
    path = tmp_path / "x" / "r.json"
    se._write_json(path, {"a": np.float64(0.5), "b": np.arange(2), "c": BASE, "d": math.pi})
    data = json.loads(path.read_text("utf-8"))
    assert data["a"] == 0.5 and data["b"] == [0, 1] and data["c"].startswith("2026-10-06")
