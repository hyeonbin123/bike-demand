"""v5 부족 경보 평가 하네스(shortage_eval)의 계산 규칙. DB 없이 합성 스냅샷으로 확인한다."""

from __future__ import annotations

import itertools
import json
import math
from datetime import datetime, timedelta

import numpy as np
import pytest
from scipy import stats

from bike_demand.api.shortage import expected_rentals
from bike_demand.model import shortage_eval as se

KST = se.KST
BASE = datetime(2026, 10, 6, 8, 0, tzinfo=KST)  # 화요일 08시 발표, as_of 08:15


def snapshots(series: dict[str, list[int | None]], start: datetime, step_min: int = 10):
    """대여소별 자전거 수 목록 → Snapshots. None은 그 시각에 그 대여소가 없음."""
    rows = []
    for station, counts in series.items():
        for i, count in enumerate(counts):
            if count is not None:
                rows.append((station, start + timedelta(minutes=step_min * i), count))
    return se.Snapshots.from_rows(rows)


def full_day(series_fn, stations=("A", "B"), start=BASE - timedelta(hours=2), n=60):
    return snapshots({s: [series_fn(s, i) for i in range(n)] for s in stations}, start)


def issue(predictions=None, rain=None, base=BASE):
    return se.IssueInput(base, predictions or {}, rain or {})


PROFILE = se.Profile({})


# ---------------------------------------------------------------- 스냅샷과 정답


def test_from_rows_builds_sorted_matrix_with_missing_cells():
    t0 = BASE
    snaps = se.Snapshots.from_rows(
        [("B", t0 + timedelta(minutes=10), 3), ("A", t0, 5), ("B", t0, 4)]
    )
    assert snaps.stations == ["A", "B"]
    assert snaps.times == [t0, t0 + timedelta(minutes=10)]
    assert snaps.bikes.tolist() == [[5, 4], [-1, 3]]


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


def test_issue_status_needs_reference_and_15_of_18_slots():
    start = BASE + timedelta(minutes=10)  # 08:10, 08:20, ..., 기준은 08:10
    dense = snapshots({"A": [3] * 19}, start)
    assert se.issue_status(dense, BASE + se.SERVICE_DELAY) == "ok"
    # 창 (08:15, 11:15]의 18칸 중 4칸을 지우면 14칸
    gappy = snapshots({"A": [3, 3, None, None, None, None] + [3] * 13}, start)
    assert se.issue_status(gappy, BASE + se.SERVICE_DELAY) == "sparse"
    late = snapshots({"A": [3] * 19}, BASE + timedelta(minutes=20))
    assert se.issue_status(late, BASE + se.SERVICE_DELAY) == "no_reference"


def test_duplicate_snapshots_in_one_slot_count_once():
    start = BASE + timedelta(minutes=10)
    rows = [("A", start + timedelta(minutes=10 * i), 3) for i in range(15)]  # 칸 15개 = 기준 + 14
    rows += [("A", start + timedelta(minutes=10 * i, seconds=90), 3) for i in range(1, 15)]
    snaps = se.Snapshots.from_rows(rows)
    window = snaps.window(BASE + se.SERVICE_DELAY, 3)
    assert len(window) == 28
    assert se.distinct_slots(snaps, window) == 14
    assert se.issue_status(snaps, BASE + se.SERVICE_DELAY) == "sparse"


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


def test_station_with_too_few_slots_is_not_scored():
    start = BASE + timedelta(minutes=15)
    series = {"dense": [3] * 40, "sparse": [3] + [None] * 5 + [3] * 34}
    rows = se.issue_rows(snapshots(series, start), issue(), PROFILE)
    ok = dict(zip(rows["station"], rows["ok3"], strict=True))
    assert ok == {"dense": True, "sparse": False}  # 창 18칸 중 13칸


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


def test_horizon_sum_is_missing_only_when_a_used_hour_is_missing():
    weights = se.overlap_weights(0.25, 3)  # 시간 0..3 사용, 4..6 안 씀
    values = np.ones((2, 7))
    values[0, 5] = np.nan
    values[1, 3] = np.nan
    out = se.horizon_sum(values, weights)
    assert out[0] == pytest.approx(3.0)
    assert np.isnan(out[1])


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


def test_rain_flag_comes_from_the_issue_forecast_over_the_window():
    snaps = snapshots({"A": [5] * 40}, BASE + timedelta(minutes=10))
    hours = [BASE + timedelta(hours=k) for k in range(4)]
    dry = {h: 0.0 for h in hours}
    assert se.issue_rows(snaps, issue(rain=dry), PROFILE)["wet"][0] == 0
    wet = dry | {hours[3]: 0.5}  # 11:00~11:15도 창과 겹친다
    assert se.issue_rows(snaps, issue(rain=wet), PROFILE)["wet"][0] == 1
    assert se.issue_rows(snaps, issue(rain={}), PROFILE)["wet"][0] == -1


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


def test_s1b_orders_by_bikes_then_profile_net_outflow():
    b0 = np.array([1, 1, 2, 2, 3])
    rent = np.array([0.0, 5.0, 50.0, 0.0, 99.0])
    ret = np.zeros(5)
    order = np.argsort(-se.score_s1b(b0, rent, ret), kind="stable")
    assert order.tolist() == [1, 0, 2, 3, 4]


@pytest.mark.parametrize("b0,e,q", [(1, 0.3, 0.1), (3, 2.5, 1.0), (4, 1.0, 0.5), (2, 0.0, 0.0)])
def test_net_outflow_with_phi_one_is_skellam(b0, e, q):
    got = se.net_outflow_prob(np.array([b0]), np.array([e]), np.array([q]), 1.0)[0]
    if e == 0:
        assert got == pytest.approx(0.0, abs=1e-9)
    else:
        assert got == pytest.approx(stats.skellam.sf(b0 - 1, e, max(q, 1e-12)), rel=1e-6)


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


def test_absorb_without_returns_is_a_poisson_tail():
    lam = np.array([[4.0, 2.0, 6.0, 1.0]])
    dur = np.array([[0.75, 1.0, 1.0, 0.25]])
    for b0 in (1, 3, 8):
        got = se.absorb_prob(np.array([b0]), lam, np.zeros_like(lam), dur)[0]
        assert got == pytest.approx(stats.poisson.sf(b0 - 1, (lam * dur).sum()), rel=1e-9)


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


def brute_auc(score, y, w=None):
    w = np.ones(len(y)) if w is None else w
    num = den = 0.0
    for i, j in itertools.product(range(len(y)), repeat=2):
        if y[i] == 1 and y[j] == 0:
            weight = w[i] * w[j]
            den += weight
            num += weight * (1.0 if score[i] > score[j] else 0.5 if score[i] == score[j] else 0.0)
    return num / den


def brute_ap(score, y):
    total = y.sum()
    ap = 0.0
    for threshold in sorted(set(score), reverse=True):
        tie = score == threshold
        selected = score >= threshold
        ap += y[tie].sum() / total * (y[selected].sum() / selected.sum())
    return ap


def test_auc_and_average_precision_with_ties_match_brute_force():
    rng = np.random.default_rng(0)
    score = rng.integers(0, 6, 40).astype(float)
    y = (rng.uniform(size=40) < 0.3).astype(float)
    index = se.RankIndex.of(score)
    assert index.auc(y) == pytest.approx(brute_auc(score, y))
    assert index.average_precision(y) == pytest.approx(brute_ap(score, y))


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


def test_precision_at_k_breaks_ties_by_station_id():
    score = np.array([1.0, 1.0, 1.0, 0.0])
    y = np.array([0, 1, 1, 1])
    station = np.array(["c", "a", "b", "d"], dtype=object)
    issue_ids = np.zeros(4, dtype=int)
    got = se.precision_at_k(score, y, issue_ids, station, ks=(1, 2, 3))
    assert got == {1: 1.0, 2: 1.0, 3: pytest.approx(2 / 3)}


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


def test_bootstrap_is_reproducible_and_resamples_days():
    counts = se.day_counts(5, n_boot=50, seed=7)
    assert counts.shape == (50, 5)
    assert (counts.sum(axis=1) == 5).all()
    assert np.array_equal(counts, se.day_counts(5, n_boot=50, seed=7))


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


def test_test_run_refuses_before_the_earliest_time_and_runs_only_once(tmp_path):
    with pytest.raises(SystemExit, match="이후"):
        se.run_test(None, None, tmp_path, se.TEST_END + timedelta(hours=5))
    (tmp_path / "test.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="한 번만"):
        se.run_test(None, None, tmp_path, se.TEST_END + timedelta(days=3))


def test_plan_constants_match_the_registered_rules():
    assert se.TEST_START.isoformat() == "2026-10-04T00:00:00+09:00"
    assert se.TEST_END - se.TEST_START == timedelta(days=14)
    assert (se.TEST_END + se.EVAL_DELAY).isoformat() == "2026-10-18T06:00:00+09:00"
    assert se.TEST_MAX_END.isoformat() == "2026-10-25T00:00:00+09:00"
    assert se.PROFILE_WINDOW == ("2025-09-01", "2025-11-01")
    assert (se.BOOT_N, se.BOOT_SEED, se.PHI_GRID) == (2000, 20261003, (1.0, 1.5, 2.0, 3.0))
    assert se.SLOTS_PER_HOUR * se.MAIN_HOURS == 15


def test_write_json_handles_numpy_and_dates(tmp_path):
    path = tmp_path / "x" / "r.json"
    se._write_json(path, {"a": np.float64(0.5), "b": np.arange(2), "c": BASE, "d": math.pi})
    data = json.loads(path.read_text("utf-8"))
    assert data["a"] == 0.5 and data["b"] == [0, 1] and data["c"].startswith("2026-10-06")
