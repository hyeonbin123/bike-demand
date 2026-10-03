"""v5(T60·T61) 측정: 부족 경보 점수를 실시간 스냅샷의 0대 도달로 채점한다.

계획과 판정 규칙은 docs/experiments.md v5 절(측정 전 커밋 `005b643`).

- 정답: 발표 시각 + 15분(`as_of`)의 기준 스냅샷에서 1대 이상인 대여소가 `(as_of, as_of + 3시간]`의
  스냅샷에서 0대가 되면 양성. 창이 10분 칸 15개 이상으로 덮인 발표만 쓴다
- 후보: S0(E - b0, 서비스가 저장한 예측), S1b(적은 순 + 프로파일 순유출로 동점), S2, S3(순유출 확률,
  Poisson/음이항), S4(0 흡수 birth–death), N(nowcast 분류기, `shortage_nowcast`)
- 지표: 전체·층별 AUC, PR-AUC, Brier, log loss, 신뢰도 곡선, P@K. 날짜 블록 부트스트랩 구간
- 실행(`__main__`): `dev`(개발 6일로 φ·도전자·분류기를 정해 고정), `test`(14일 test를 한 번)

계산 함수는 DB와 무관하다. DB·warehouse·모델은 `load_*`·`stored_*`·`issue_bases`와
`run_dev`·`run_test`만 읽는다.
"""

from __future__ import annotations

import bisect
import math
from array import array
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np
from scipy import stats

KST = timezone(timedelta(hours=9))
HOUR = timedelta(hours=1)
SLOT_SECONDS = 600  # 10분 칸. KST 오프셋(9시간)이 10분의 배수라 UTC로 내려도 같은 칸
SERVICE_DELAY = timedelta(minutes=15)  # as_of = 발표 + 15분 (v4와 같음)
REF_MAX_AGE = timedelta(minutes=30)  # 기준 스냅샷은 as_of 이전 30분 안 (API 한도)
SLOTS_PER_HOUR = 5  # 창 h시간에 10분 칸 5h개 이상 (3시간이면 18칸 중 15칸)
MAIN_HOURS = 3
HORIZONS = (1, 2, 3, 4, 5, 6)
HOUR_COLUMNS = max(HORIZONS) + 1  # as_of가 든 시간부터 7개 시간
DELTA_TOLERANCE = timedelta(minutes=5)
PROFILE_WINDOW = ("2025-09-01", "2025-11-01")
API_LIMIT = 50
P_AT_K = (10, 20, 50, 100)
PHI_GRID = (1.0, 1.5, 2.0, 3.0)
BD_HEADROOM = 30
BOOT_N = 2000
BOOT_SEED = 20261003
RECOVERY_JUMP = 5
ISSUE_HOURS = (2, 5, 8, 11, 14, 17, 20, 23)
CHALLENGERS = ("S2", "S3", "S4")  # 같으면 앞의 것(단순한 것)
EPS = 1e-12

DEV_WINDOW = (datetime(2026, 9, 16, tzinfo=KST), datetime(2026, 9, 23, tzinfo=KST))
TEST_START = datetime(2026, 10, 4, tzinfo=KST)
TEST_END = datetime(2026, 10, 18, tzinfo=KST)
TEST_MAX_END = datetime(2026, 10, 25, tzinfo=KST)
EVAL_DELAY = timedelta(hours=6)  # 마지막 발표(23:15)의 6시간 창이 끝난 뒤
MIN_VALID_DAYS = 10


# ---------------------------------------------------------------- 스냅샷과 정답


@dataclass
class Snapshots:
    """스냅샷 행렬. bikes[t, s]는 시각 t의 대여소 s 자전거 수, 없으면 -1."""

    times: list[datetime]
    stations: list[str]
    bikes: np.ndarray
    seconds: np.ndarray = field(init=False)
    slots: np.ndarray = field(init=False)
    column: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.seconds = np.array([t.timestamp() for t in self.times], dtype=np.float64)
        self.slots = np.floor(self.seconds / SLOT_SECONDS).astype(np.int64)
        self.column = {s: i for i, s in enumerate(self.stations)}

    @classmethod
    def from_rows(cls, rows: Iterable[tuple[str, datetime, int]]) -> Snapshots:
        """(station_id, fetched_at, bike_count) 행에서. 행을 한 번만 훑고 정수 배열로 모은다."""
        t_index: dict[datetime, int] = {}
        s_index: dict[str, int] = {}
        ti, si, counts = array("i"), array("i"), array("i")
        for station, fetched_at, count in rows:
            ti.append(t_index.setdefault(fetched_at, len(t_index)))
            si.append(s_index.setdefault(station, len(s_index)))
            counts.append(int(count))
        times = sorted(t_index)
        stations = sorted(s_index)
        t_new = np.empty(len(times), dtype=np.int64)
        t_new[[t_index[t] for t in times]] = np.arange(len(times))
        s_new = np.empty(len(stations), dtype=np.int64)
        s_new[[s_index[s] for s in stations]] = np.arange(len(stations))
        bikes = np.full((len(times), len(stations)), -1, dtype=np.int32)
        rows_t = t_new[np.asarray(ti, dtype=np.int64)]
        rows_s = s_new[np.asarray(si, dtype=np.int64)]
        bikes[rows_t, rows_s] = np.asarray(counts, dtype=np.int32)
        return cls(times, stations, bikes)

    def reference(self, as_of: datetime) -> int | None:
        """as_of 이전(포함) REF_MAX_AGE 안의 가장 최근 스냅샷 번호."""
        i = bisect.bisect_right(self.seconds, as_of.timestamp()) - 1
        if i < 0 or self.seconds[i] < (as_of - REF_MAX_AGE).timestamp():
            return None
        return i

    def window(self, as_of: datetime, hours: float) -> np.ndarray:
        """(as_of, as_of + hours] 안의 스냅샷 번호."""
        lo = bisect.bisect_right(self.seconds, as_of.timestamp())
        hi = bisect.bisect_right(self.seconds, (as_of + hours * HOUR).timestamp())
        return np.arange(lo, hi)

    def nearest(self, target: datetime, tolerance: timedelta = DELTA_TOLERANCE) -> int | None:
        """target ± tolerance 안에서 가장 가까운 스냅샷 번호(같으면 이른 것)."""
        x = target.timestamp()
        i = bisect.bisect_left(self.seconds, x)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.seconds) and abs(self.seconds[j] - x) <= tolerance.total_seconds():
                if best is None or abs(self.seconds[j] - x) < abs(self.seconds[best] - x):
                    best = j
        return best


def distinct_slots(snaps: Snapshots, indices: np.ndarray) -> int:
    return len(set(snaps.slots[indices].tolist()))


def station_slots(snaps: Snapshots, indices: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """대여소마다 창 안에서 나온 10분 칸 수(한 칸에 스냅샷이 여럿이어도 한 번)."""
    present = snaps.bikes[np.ix_(indices, columns)] >= 0
    slots = snaps.slots[indices]
    counts = np.zeros(len(columns), dtype=np.int64)
    for slot in np.unique(slots):
        counts += present[slots == slot].any(axis=0)
    return counts


def window_hits_zero(snaps: Snapshots, indices: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """창 안 스냅샷 중 하나라도 0대였는가(빠진 스냅샷은 보지 않음)."""
    if len(indices) == 0:
        return np.zeros(len(columns), dtype=bool)
    return (snaps.bikes[np.ix_(indices, columns)] == 0).any(axis=0)


def truck_events(
    snaps: Snapshots, ref: int, indices: np.ndarray, columns: np.ndarray, jump: int = RECOVERY_JUMP
) -> dict[str, np.ndarray]:
    """재배치 트럭 보고용: 이웃한 스냅샷 사이 +jump 이상(회복), -jump 이하(수거).

    반환: recovery(창 안 회복 있음), recovery_after_zero(처음 0 이후 회복), pickup_before_zero
    (처음 0 이전 수거). 처음 0이 없으면 뒤의 두 값은 False.
    """
    seq = snaps.bikes[np.ix_(np.concatenate([[ref], indices]).astype(int), columns)]
    present = seq >= 0
    n = len(columns)
    if seq.shape[0] < 2:
        false = np.zeros(n, dtype=bool)
        return {"recovery": false, "recovery_after_zero": false, "pickup_before_zero": false}
    diff = seq[1:] - seq[:-1]
    valid = present[1:] & present[:-1]
    up = valid & (diff >= jump)
    down = valid & (diff <= -jump)
    zero = (seq[1:] == 0) & present[1:]  # 위치 j+1이 0
    has_zero = zero.any(axis=0)
    first = np.where(has_zero, zero.argmax(axis=0) + 1, seq.shape[0])  # seq 안의 위치
    j = np.arange(diff.shape[0])[:, None]  # 전이 j: 위치 j → j+1
    return {
        "recovery": up.any(axis=0),
        "recovery_after_zero": has_zero & (up & (j >= first)).any(axis=0),
        "pickup_before_zero": has_zero & (down & (j + 1 <= first)).any(axis=0),
    }


# ---------------------------------------------------------------- 겹침 비율과 입력


def overlap_weights(
    frac: np.ndarray | float, hours: int, columns: int = HOUR_COLUMNS
) -> np.ndarray:
    """[as_of, as_of + hours)와 시간 k([first + k, first + k + 1))의 겹친 비율.

    frac = as_of가 든 시간 안에서 지난 비율(분/60). `api.shortage.expected_rentals`와 같은 비율.
    """
    frac = np.atleast_1d(np.asarray(frac, dtype=np.float64))
    k = np.arange(columns, dtype=np.float64)[None, :]
    start = frac[:, None]
    return np.clip(np.minimum(k + 1, start + hours) - np.maximum(k, start), 0.0, 1.0)


def horizon_sum(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """시간별 값(n × 시간)의 겹침 가중 합. 쓰는 시간에 빈 값이 있으면 NaN."""
    used = weights > 0
    missing = (np.isnan(values) & used).any(axis=1)
    total = np.where(used, np.nan_to_num(values) * weights, 0.0).sum(axis=1)
    return np.where(missing, np.nan, total)


@dataclass
class Profile:
    """작년 같은 계절의 대여소 × 쉬는 날 × 시각 시간당 평균 (대여, 반납)."""

    values: dict[tuple[str, bool, int], tuple[float, float]]
    window: tuple[str, str] = PROFILE_WINDOW

    def hourly(
        self, stations: list[str], hours: list[datetime], offday: list[bool]
    ) -> tuple[np.ndarray, np.ndarray]:
        rent = np.full((len(stations), len(hours)), np.nan)
        ret = np.full((len(stations), len(hours)), np.nan)
        for k, (hour, off) in enumerate(zip(hours, offday, strict=True)):
            key_hour = hour.astimezone(KST).hour
            for i, station in enumerate(stations):
                value = self.values.get((station, off, key_hour))
                if value is not None:
                    rent[i, k], ret[i, k] = value
        return rent, ret


def is_offday(hour: datetime) -> bool:
    """서비스 예측과 같은 규칙(주말 또는 KR 공휴일)."""
    from bike_demand.model.predict import calendar_features

    return bool(calendar_features(hour)["is_offday"])


# ---------------------------------------------------------------- 발표 하나의 행


@dataclass
class IssueInput:
    base: datetime  # 발표 시각(KST)
    predictions: Mapping[str, Mapping[datetime, float]]  # 대여소 → hour_start → 예상 대여
    rain: Mapping[datetime, float] = field(default_factory=dict)  # hour_start → 예보 강수(mm)
    version: str | None = None

    @property
    def as_of(self) -> datetime:
        return self.base + SERVICE_DELAY


def issue_status(snaps: Snapshots, as_of: datetime, hours: int = MAIN_HOURS) -> str:
    """발표 단위 규칙: ok, no_reference(30분 안 스냅샷 없음), sparse(10분 칸 부족)."""
    if snaps.reference(as_of) is None:
        return "no_reference"
    if distinct_slots(snaps, snaps.window(as_of, hours)) < SLOTS_PER_HOUR * hours:
        return "sparse"
    return "ok"


def issue_rows(snaps: Snapshots, issue: IssueInput, profile: Profile) -> dict[str, np.ndarray]:
    """기준 스냅샷에 나온 모든 대여소(b0 = 0 포함)의 행. 열 이름은 아래 키.

    발표 단위 규칙(기준 스냅샷, 10분 칸)은 호출하는 쪽이 `issue_status`로 먼저 본다.
    여기서는 기준 스냅샷이 있어야 한다. 시간 단위 창의 칸 수가 모자라면 그 h의 ok가 모두 False.
    """
    as_of = issue.as_of.astimezone(KST)
    ref = snaps.reference(as_of)
    if ref is None:
        raise ValueError(f"기준 스냅샷 없음: {as_of.isoformat()}")
    columns = np.flatnonzero(snaps.bikes[ref] >= 0)
    stations = [snaps.stations[c] for c in columns]
    n = len(columns)
    b0 = snaps.bikes[ref, columns].astype(np.int64)
    first = as_of.replace(minute=0, second=0, microsecond=0)
    hours = [first + k * HOUR for k in range(HOUR_COLUMNS)]
    frac = (as_of - first) / HOUR
    offday = [is_offday(h) for h in hours]

    pred = np.full((n, HOUR_COLUMNS), np.nan)
    for i, station in enumerate(stations):
        by_hour = issue.predictions.get(station)
        if by_hour:
            for k, hour in enumerate(hours):
                value = by_hour.get(hour)
                if value is not None:
                    pred[i, k] = value
    rent, ret = profile.hourly(stations, hours, offday)

    out: dict[str, np.ndarray] = {
        "station": np.array(stations, dtype=object),
        "b0": b0,
        "frac": np.full(n, frac),
        "issue_hour": np.full(n, first.hour),
        "offday": np.full(n, float(is_offday(first))),
        "pred": pred,
        "rent": rent,
        "ret": ret,
    }
    for h in HORIZONS:
        win = snaps.window(as_of, h)
        issue_ok = distinct_slots(snaps, win) >= SLOTS_PER_HOUR * h
        enough = station_slots(snaps, win, columns) >= SLOTS_PER_HOUR * h
        out[f"y{h}"] = window_hits_zero(snaps, win, columns).astype(np.int64)
        out[f"ok{h}"] = enough & issue_ok
    main_window = snaps.window(as_of, MAIN_HOURS)
    out[f"recovered{MAIN_HOURS}"] = (snaps.bikes[np.ix_(main_window, columns)] >= 1).any(axis=0)
    events = truck_events(snaps, ref, main_window, columns)
    out.update({f"truck_{k}": v for k, v in events.items()})

    for minutes in (30, 60):
        j = snaps.nearest(snaps.times[ref] - timedelta(minutes=minutes))
        before = snaps.bikes[j, columns] if j is not None else np.full(n, -1)
        missing = before < 0
        out[f"d{minutes}"] = np.where(missing, 0, b0 - before).astype(np.float64)
        out[f"miss{minutes}"] = missing.astype(np.float64)

    rain_hours = [h for h, w in zip(hours, overlap_weights(frac, MAIN_HOURS)[0], strict=True) if w]
    rain = [issue.rain.get(h) for h in rain_hours]
    if any(r is not None and r > 0 for r in rain):
        wet = 1
    elif any(r is None or math.isnan(r) for r in rain):
        wet = -1
    else:
        wet = 0
    out["wet"] = np.full(n, wet)
    return out


def concat_rows(parts: list[dict[str, np.ndarray]], extra: list[dict[str, int]]) -> dict:
    """발표별 행을 합친다. extra[i]의 정수 값(발표 번호, 날짜 서수)은 그 발표의 모든 행에 붙인다."""
    keys = parts[0].keys()
    rows = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    for name in extra[0]:
        rows[name] = np.concatenate(
            [np.full(len(p["b0"]), int(e[name])) for p, e in zip(parts, extra, strict=True)]
        )
    return rows


def add_horizon_inputs(rows: dict) -> None:
    """E·R·Q를 1~6시간 합으로 더한다(E{h}, R{h}, Q{h})."""
    for h in HORIZONS:
        w = overlap_weights(rows["frac"], h)
        rows[f"E{h}"] = horizon_sum(rows["pred"], w)
        rows[f"R{h}"] = horizon_sum(rows["rent"], w)
        rows[f"Q{h}"] = horizon_sum(rows["ret"], w)


def main_mask(rows: dict, h: int = MAIN_HOURS) -> np.ndarray:
    """주 평가 행: b0 ≥ 1, 창 규칙 통과, E·R·Q가 모두 있음."""
    return (
        (rows["b0"] >= 1)
        & rows[f"ok{h}"]
        & ~np.isnan(rows[f"E{h}"])
        & ~np.isnan(rows[f"R{h}"])
        & ~np.isnan(rows[f"Q{h}"])
    )


def exclusion_counts(rows: dict, h: int = MAIN_HOURS) -> dict[str, int]:
    nonempty = rows["b0"] >= 1
    ok = rows[f"ok{h}"]
    no_e = np.isnan(rows[f"E{h}"])
    no_p = np.isnan(rows[f"R{h}"]) | np.isnan(rows[f"Q{h}"])
    return {
        "rows": int(len(nonempty)),
        "already_empty": int((~nonempty).sum()),
        "station_sparse": int((nonempty & ~ok).sum()),
        "no_prediction": int((nonempty & ok & no_e).sum()),
        "no_profile": int((nonempty & ok & ~no_e & no_p).sum()),
        "main": int(main_mask(rows, h).sum()),
    }


# ---------------------------------------------------------------- 후보 점수


def score_s1b(b0: np.ndarray, rent: np.ndarray, ret: np.ndarray) -> np.ndarray:
    """자전거 적은 순, 같은 b0 안에서는 프로파일 순유출(R - Q)이 큰 순."""
    return -np.asarray(b0, dtype=np.float64) + 0.001 * np.tanh(rent - ret)


def _counts(mean: np.ndarray, phi: float):
    mean = np.maximum(np.asarray(mean, dtype=np.float64), EPS)
    if phi == 1.0:
        return stats.poisson(mean)
    return stats.nbinom(mean / (phi - 1.0), 1.0 / phi)


def net_outflow_prob(
    b0: np.ndarray, mean_out: np.ndarray, mean_in: np.ndarray, phi: float = 1.0, chunk: int = 20000
) -> np.ndarray:
    """P(N_out − N_in ≥ b0). N_out, N_in 독립, 평균 mean_out·mean_in, 분산 φ × 평균.

    φ = 1이면 Poisson(차는 Skellam), φ > 1이면 음이항(n = μ/(φ−1), p = 1/φ).
    """
    if phi < 1.0:
        raise ValueError("phi는 1 이상")
    b0 = np.asarray(b0, dtype=np.float64)
    mean_out = np.asarray(mean_out, dtype=np.float64)
    mean_in = np.asarray(mean_in, dtype=np.float64)
    out = np.empty(len(b0))
    for start in range(0, len(b0), chunk):
        sl = slice(start, start + chunk)
        top = float(mean_in[sl].max()) if len(mean_in[sl]) else 0.0
        kmax = int(math.ceil(top + 15 * math.sqrt(phi * top + 1) + 20))
        k = np.arange(kmax + 1, dtype=np.float64)[None, :]
        pmf_in = _counts(mean_in[sl][:, None], phi).pmf(k)
        sf_out = _counts(mean_out[sl][:, None], phi).sf(b0[sl][:, None] + k - 1)
        out[sl] = (pmf_in * sf_out).sum(axis=1)
    return np.clip(out, 0.0, 1.0)


def _bd_step(p: np.ndarray, down: np.ndarray, up: np.ndarray, top: np.ndarray) -> np.ndarray:
    """균등화한 이산 단계 하나. 0은 흡수, top에서는 위로 가지 못하고 머문다."""
    m, d = p.shape
    new = np.zeros_like(p)
    new[:, :-1] += down[:, None] * p[:, 1:]
    states = np.arange(d)[None, :]
    can_up = (states >= 1) & (states < top[:, None])
    new[:, 1:] += (up[:, None] * p * can_up)[:, :-1]
    new[:, 0] += p[:, 0]
    rows = np.arange(m)
    new[rows, top] += up * p[rows, top]
    return new


def absorb_prob(
    b0: np.ndarray,
    rent_rate: np.ndarray,
    return_rate: np.ndarray,
    duration: np.ndarray,
    headroom: int = BD_HEADROOM,
    chunk: int = 512,
) -> np.ndarray:
    """0을 흡수 상태로 둔 시간비균질 birth–death 모델에서 끝까지 0에 닿을 확률.

    구간 k(길이 duration[:, k]시간) 동안 감소 비율 rent_rate[:, k], 증가 비율 return_rate[:, k]
    (시간당)가 일정하다. 상태는 0..b0+headroom, 위 끝에서는 증가가 막힌다. 구간마다 균등화
    (uniformization)로 정확히 푼다(포아송 가중치 꼬리 1e-12 밖은 버림).
    """
    b0 = np.asarray(b0, dtype=np.int64)
    rent_rate = np.atleast_2d(np.asarray(rent_rate, dtype=np.float64))
    return_rate = np.atleast_2d(np.asarray(return_rate, dtype=np.float64))
    duration = np.atleast_2d(np.asarray(duration, dtype=np.float64))
    out = np.empty(len(b0))
    order = np.argsort(b0, kind="stable")
    for start in range(0, len(b0), chunk):
        idx = order[start : start + chunk]
        top = b0[idx] + headroom
        p = np.zeros((len(idx), int(top.max()) + 1))
        p[np.arange(len(idx)), b0[idx]] = 1.0
        for k in range(duration.shape[1]):
            lam, mu, t = rent_rate[idx, k], return_rate[idx, k], duration[idx, k]
            total = lam + mu
            mass = total * t
            if not np.any(mass > 0):
                continue
            safe = np.where(total > 0, total, 1.0)
            down, up = np.where(total > 0, lam / safe, 0.0), np.where(total > 0, mu / safe, 0.0)
            n_max = int(math.ceil(mass.max() + 12 * math.sqrt(mass.max()) + 15))
            weight = np.exp(-mass)
            acc = weight[:, None] * p
            current = p
            for n in range(1, n_max + 1):
                current = _bd_step(current, down, up, top)
                weight = weight * mass / n
                acc += weight[:, None] * current
            p = acc
        out[idx] = p[:, 0]
    return np.clip(out, 0.0, 1.0)


def candidate_scores(rows: dict, mask: np.ndarray, phi: float, h: int = MAIN_HOURS) -> dict:
    """S0~S4와 참고 S1의 점수(mask 행만). S3·S4는 확률."""
    b0 = rows["b0"][mask].astype(np.float64)
    e, r, q = rows[f"E{h}"][mask], rows[f"R{h}"][mask], rows[f"Q{h}"][mask]
    weights = overlap_weights(rows["frac"][mask], h)
    used = weights > 0  # 창 밖 시간의 빈 값이 계산에 섞이지 않게 0으로
    rent_rate = np.where(used, rows["pred"][mask], 0.0)
    return_rate = np.where(used, rows["ret"][mask], 0.0)
    return {
        "S0": e - b0,
        "S1": -b0,
        "S1b": score_s1b(b0, r, q),
        "S2": (e - q) - b0,
        "S3": net_outflow_prob(b0, e, q, phi),
        "S4": absorb_prob(rows["b0"][mask], rent_rate, return_rate, weights),
    }


# ---------------------------------------------------------------- 지표


@dataclass
class RankIndex:
    """점수의 오름차순 동점 묶음. 같은 행에서 가중치만 바꿔 여러 번 잴 때 다시 정렬하지 않는다."""

    group: np.ndarray
    n_groups: int

    @classmethod
    def of(cls, score: np.ndarray) -> RankIndex:
        score = np.asarray(score, dtype=np.float64)
        if np.isnan(score).any():
            raise ValueError("점수에 NaN")
        _, group = np.unique(score, return_inverse=True)
        return cls(group.ravel(), int(group.max()) + 1 if len(group) else 0)

    def _sums(self, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos = np.bincount(self.group, weights=w * y, minlength=self.n_groups)
        neg = np.bincount(self.group, weights=w * (1 - y), minlength=self.n_groups)
        return pos, neg

    def auc(self, y: np.ndarray, w: np.ndarray | None = None) -> float:
        """P(양성 점수 > 음성 점수) + 0.5 P(같음). 가중치는 행의 반복 횟수."""
        w = np.ones(len(y)) if w is None else w
        pos, neg = self._sums(y, w)
        above = np.cumsum(pos[::-1])[::-1] - pos
        denom = pos.sum() * neg.sum()
        return float((neg * (above + 0.5 * pos)).sum() / denom) if denom else math.nan

    def average_precision(self, y: np.ndarray, w: np.ndarray | None = None) -> float:
        """높은 점수부터 동점 묶음 단위로 문턱을 내릴 때의 정밀도를 재현율 증가로 가중 평균."""
        w = np.ones(len(y)) if w is None else w
        pos, neg = self._sums(y, w)
        cum_pos = np.cumsum(pos[::-1])[::-1]
        cum_all = np.cumsum((pos + neg)[::-1])[::-1]
        precision = np.divide(cum_pos, cum_all, out=np.zeros_like(cum_pos), where=cum_all > 0)
        total = pos.sum()
        return float((pos * precision).sum() / total) if total else math.nan


def brier(p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    w = np.ones(len(y)) if w is None else w
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def log_loss(p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    w = np.ones(len(y)) if w is None else w
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    table = []
    for b in range(bins):
        m = which == b
        table.append(
            {
                "bin": [round(float(edges[b]), 1), round(float(edges[b + 1]), 1)],
                "rows": int(m.sum()),
                "mean_pred": float(p[m].mean()) if m.any() else None,
                "observed": float(y[m].mean()) if m.any() else None,
            }
        )
    return table


def precision_at_k(
    score: np.ndarray, y: np.ndarray, issue: np.ndarray, station: np.ndarray, ks=P_AT_K
) -> dict[int, float]:
    """발표마다 점수 큰 순(같으면 station_id 순) 상위 K곳의 양성 비율, 발표 평균."""
    result: dict[int, list[float]] = {k: [] for k in ks}
    for value in np.unique(issue):
        m = np.flatnonzero(issue == value)
        order = m[np.lexsort((station[m].astype(str), -score[m]))]
        for k in ks:
            result[k].append(float(y[order[:k]].mean()))
    return {k: float(np.mean(v)) if v else math.nan for k, v in result.items()}


def api_list_precision(rows: dict, h: int = MAIN_HOURS, limit: int = API_LIMIT) -> dict:
    """API 그대로(S0): b0 = 0 포함, E − b0 > 0을 큰 순으로 limit곳. 칸 부족 대여소는 채점에서 뺌."""
    precisions, empties, lengths = [], [], []
    for value in np.unique(rows["issue"]):
        m = np.flatnonzero((rows["issue"] == value) & ~np.isnan(rows[f"E{h}"]))
        shortfall = rows[f"E{h}"][m] - rows["b0"][m]
        m, shortfall = m[shortfall > 0], shortfall[shortfall > 0]
        listed = m[np.lexsort((rows["station"][m].astype(str), -shortfall))][:limit]
        scored = listed[rows[f"ok{h}"][listed]]
        lengths.append(len(listed))
        if len(scored):
            precisions.append(float(rows[f"y{h}"][scored].mean()))
            empties.append(float((rows["b0"][scored] == 0).mean()))
    return {
        "issues": len(precisions),
        "precision": float(np.mean(precisions)) if precisions else None,
        "already_empty_share": float(np.mean(empties)) if empties else None,
        "mean_list_length": float(np.mean(lengths)) if lengths else None,
    }


def already_empty_report(rows: dict, h: int = MAIN_HOURS) -> dict:
    """b0 = 0인 대여소: 수와 비율, 창 안에서 1대 이상이 된 비율, 창 내내 0대였던 비율."""
    empty = (rows["b0"] == 0) & rows[f"ok{h}"]
    counted = (rows["b0"] >= 0) & rows[f"ok{h}"]
    n = int(empty.sum())
    return {
        "rows": n,
        "share_of_rows": float(n / counted.sum()) if counted.any() else None,
        "recovered_share": float(rows[f"recovered{h}"][empty].mean()) if n else None,
        "always_empty_share": float(1 - rows[f"recovered{h}"][empty].mean()) if n else None,
    }


# ---------------------------------------------------------------- 블록 부트스트랩과 판정


def day_counts(n_days: int, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> np.ndarray:
    """(n_boot × n_days) 날짜별 뽑힌 횟수."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_days, size=(n_boot, n_days))
    return np.stack([np.bincount(d, minlength=n_days) for d in draws])


@dataclass
class Evaluation:
    """같은 행 집합에서 후보별 지표의 점 추정과 부트스트랩 값."""

    point: dict[str, dict[str, float]]
    boot: dict[str, dict[str, np.ndarray]]

    def ci(self, metric: str, a: str, b: str) -> list[float]:
        diff = self.boot[metric][a] - self.boot[metric][b]
        return [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]


def evaluate(
    scores: dict[str, np.ndarray],
    y: np.ndarray,
    b0: np.ndarray,
    day: np.ndarray,
    probabilities: Iterable[str] = (),
    n_boot: int = BOOT_N,
    seed: int = BOOT_SEED,
    only_auc: bool = False,
) -> Evaluation:
    """전체 AUC, PR-AUC, b0≥2·b0≥3 AUC(모든 후보)와 Brier·log loss(확률 후보).

    only_auc: 전체 AUC만(1~6시간 보조 보고용).
    """
    days, day_id = np.unique(day, return_inverse=True)
    counts = day_counts(len(days), n_boot, seed)
    y = np.asarray(y, dtype=np.float64)
    strata = {"auc_b0ge2": b0 >= 2, "auc_b0ge3": b0 >= 3}
    point: dict[str, dict[str, float]] = {}
    boot: dict[str, dict[str, np.ndarray]] = {}

    def put(metric: str, name: str, fn) -> None:
        """fn(w): w는 행 가중치(None이면 모두 1). 재표본 i의 가중치는 그 행의 날짜가 뽑힌 횟수."""
        point.setdefault(metric, {})[name] = fn(None)
        boot.setdefault(metric, {})[name] = np.array([fn(c[day_id]) for c in counts])

    for name, score in scores.items():
        index = RankIndex.of(score)
        put("auc", name, lambda w, ix=index: ix.auc(y, w))
        if only_auc:
            continue
        put("pr_auc", name, lambda w, ix=index: ix.average_precision(y, w))
        for metric, mask in strata.items():
            sub, ys = RankIndex.of(score[mask]), y[mask]
            put(
                metric,
                name,
                lambda w, ix=sub, ys=ys, m=mask: ix.auc(ys, None if w is None else w[m]),
            )
    for name in probabilities:
        p = scores[name]
        put("brier", name, lambda w, p=p: brier(p, y, w))
        put("log_loss", name, lambda w, p=p: log_loss(p, y, w))
    return Evaluation(point, boot)


def decide(ev: Evaluation, challenger: str, valid_days: int) -> dict:
    """docs/experiments.md v5 판정. 결과마다 할 일 하나."""
    if valid_days < MIN_VALID_DAYS:
        return {"decided": False, "reason": f"유효한 날 {valid_days}일 < {MIN_VALID_DAYS}일"}
    ci = {
        "challenger_vs_S1b": ev.ci("auc", challenger, "S1b"),
        "S0_vs_S1b": ev.ci("auc", "S0", "S1b"),
    }
    if ci["challenger_vs_S1b"][0] > 0:
        p1 = challenger
    elif ci["S0_vs_S1b"][1] < 0:
        p1 = "S1b"
    else:
        p1 = "S0"
    checks = {"N_vs_S1b_b0ge2": ev.ci("auc_b0ge2", "N", "S1b")}
    if p1 != "S1b":
        checks["N_vs_P1_b0ge2"] = ev.ci("auc_b0ge2", "N", p1)
    brier_ref = min(ev.point["brier"]["S3"], ev.point["brier"]["S4"])
    conditions = {
        **{name: value[0] > 0 for name, value in checks.items()},
        "brier_not_worse": ev.point["brier"]["N"] <= brier_ref,
        "auc_not_lower": ev.point["auc"]["N"] >= ev.point["auc"][p1],
    }
    final = "N" if all(conditions.values()) else p1
    actions = {
        "S0": "/shortage-risk를 그대로 두고 README 한계에 수치를 적는다",
        "S1b": "/shortage-risk 순위를 S1b로 바꾸고 이미 0대인 곳은 따로 보인다",
    }
    action = actions.get(
        final, f"/shortage-risk 순위를 {final}로 바꾸고 이미 0대인 곳은 따로 보인다"
    )
    return {
        "decided": True,
        "p1_result": p1,
        "p1_ci": ci,
        "nowcast_ci": checks,
        "nowcast_conditions": conditions,
        "final": final,
        "action": action,
    }


def pick_phi(b0, e, q, y, grid=PHI_GRID) -> tuple[float, dict[str, float]]:
    """개발 자료 log loss가 가장 낮은 φ(같으면 작은 것)."""
    losses = {str(phi): log_loss(net_outflow_prob(b0, e, q, phi), y) for phi in grid}
    best = min(grid, key=lambda phi: (losses[str(phi)], phi))
    return best, losses


def pick_challenger(auc: Mapping[str, float]) -> str:
    """S2·S3·S4 중 개발 자료 전체 AUC가 가장 높은 것, 같으면 앞의 것."""
    return max(CHALLENGERS, key=lambda name: (auc[name], -CHALLENGERS.index(name)))


# ---------------------------------------------------------------- 보고


def _round(value, digits: int = 4):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), digits)


def metric_table(ev: Evaluation, baseline: str = "S1b") -> dict:
    """후보별 점 추정과 기준선과의 짝지은 차이 95% 구간."""
    table: dict = {}
    for metric, by_name in ev.point.items():
        for name, value in by_name.items():
            entry = table.setdefault(name, {})
            entry[metric] = _round(value)
            if name != baseline and baseline in by_name:
                entry[f"{metric}_minus_{baseline}_ci"] = [
                    _round(v) for v in ev.ci(metric, name, baseline)
                ]
    return table


def stratum_report(rows: dict, mask: np.ndarray, scores: dict) -> dict:
    """비 옴 구분(예보 기준)별 발표 수·행 수·전체 AUC(보고만)."""
    wet = rows["wet"][mask]
    issue = rows["issue"][mask]
    y = rows[f"y{MAIN_HOURS}"][mask].astype(np.float64)
    out = {}
    for name, value in (("wet", 1), ("dry", 0), ("unknown", -1)):
        m = wet == value
        group = {"issues": int(len(np.unique(issue[m]))), "rows": int(m.sum())}
        if m.any() and 0 < y[m].sum() < m.sum():
            group["auc"] = {k: _round(RankIndex.of(s[m]).auc(y[m])) for k, s in scores.items()}
        out[name] = group
    out["reference_only"] = out["wet"]["issues"] < 10
    return out


def truck_report(rows: dict, mask: np.ndarray) -> dict:
    y = rows[f"y{MAIN_HOURS}"][mask].astype(bool)
    rec = rows["truck_recovery"][mask]
    return {
        "rows_with_recovery": _round(rec.mean()),
        "positives_recovery_after_zero": _round(rows["truck_recovery_after_zero"][mask][y].mean())
        if y.any()
        else None,
        "positives_pickup_before_zero": _round(rows["truck_pickup_before_zero"][mask][y].mean())
        if y.any()
        else None,
    }


def horizon_report(rows: dict, phi: float, challenger: str, n_boot: int) -> dict:
    """1~6시간 보조 보고: S0~S4 전체 AUC와 (도전자 − S1b) 구간."""
    out = {}
    for h in HORIZONS:
        mask = main_mask(rows, h)
        y = rows[f"y{h}"][mask]
        if not mask.any() or y.min() == y.max():
            out[str(h)] = {"rows": int(mask.sum())}
            continue
        scores = candidate_scores(rows, mask, phi, h)
        ev = evaluate(scores, y, rows["b0"][mask], rows["day"][mask], n_boot=n_boot, only_auc=True)
        out[str(h)] = {
            "rows": int(mask.sum()),
            "issues": int(len(np.unique(rows["issue"][mask]))),
            "positive_rate": _round(y.mean()),
            "auc": {k: _round(v) for k, v in ev.point["auc"].items()},
            f"{challenger}_minus_S1b_ci": [_round(v) for v in ev.ci("auc", challenger, "S1b")],
        }
    return out


def full_report(rows: dict, scores: dict, mask: np.ndarray, probabilities, n_boot: int) -> tuple:
    """주 평가 행의 지표 전부. (Evaluation, 보고 dict)."""
    y = rows[f"y{MAIN_HOURS}"][mask]
    b0 = rows["b0"][mask]
    ev = evaluate(scores, y, b0, rows["day"][mask], probabilities=probabilities, n_boot=n_boot)
    report = {
        "rows": int(mask.sum()),
        "issues": int(len(np.unique(rows["issue"][mask]))),
        "days": int(len(np.unique(rows["day"][mask]))),
        "positive_rate": _round(y.mean()),
        "b0_ge2_rows": int((b0 >= 2).sum()),
        "b0_ge3_rows": int((b0 >= 3).sum()),
        "metrics": metric_table(ev),
        "reliability": {name: reliability(scores[name], y) for name in probabilities},
        "precision_at_k": {
            name: {
                str(k): _round(v)
                for k, v in precision_at_k(
                    score, y, rows["issue"][mask], rows["station"][mask]
                ).items()
            }
            for name, score in scores.items()
        },
        "rain": stratum_report(rows, mask, scores),
        "truck": truck_report(rows, mask),
    }
    return ev, report


# ---------------------------------------------------------------- 읽기 (DB, warehouse, 모델)


def load_snapshots(engine, start: datetime, end: datetime) -> Snapshots:
    from sqlalchemy import select

    from bike_demand.serving.models import RealtimeSnapshot as RS

    with engine.connect() as conn:
        result = conn.execution_options(stream_results=True, yield_per=50000).execute(
            select(RS.station_id, RS.fetched_at, RS.bike_count).where(
                RS.fetched_at >= start, RS.fetched_at <= end
            )
        )
        return Snapshots.from_rows((s, t.astimezone(KST), c) for s, t, c in result)


def _grid():
    from bike_demand.serving.models import WeatherForecast as WF

    return (WF.nx == 60) & (WF.ny == 127)


def issue_bases(engine, start: datetime, end: datetime) -> list[datetime]:
    """as_of가 [start, end)인 발표 시각(격자 60·127)."""
    from sqlalchemy import select

    from bike_demand.serving.models import WeatherForecast as WF

    with engine.connect() as conn:
        bases = conn.execute(
            select(WF.base_datetime)
            .where(_grid(), WF.base_datetime >= start - SERVICE_DELAY)
            .where(WF.base_datetime < end - SERVICE_DELAY)
            .distinct()
            .order_by(WF.base_datetime)
        ).scalars()
        return [b.astimezone(KST) for b in bases]


def load_forecast(engine, base: datetime) -> dict[datetime, dict[str, float]]:
    from sqlalchemy import select

    from bike_demand.model.forecast_weather import hourly_weather
    from bike_demand.serving.models import WeatherForecast as WF

    with engine.connect() as conn:
        rows = conn.execute(
            select(WF.fcst_datetime, WF.category, WF.value).where(_grid(), WF.base_datetime == base)
        ).all()
    return hourly_weather((r[0], r[1], r[2]) for r in rows)


def issue_key(base: datetime) -> str:
    return f"{base.astimezone(KST):%Y%m%d%H%M}"


def stored_versions(engine, since: datetime) -> dict[str, str]:
    """발표 키(YYYYMMDDHHMM) → 그 발표로 저장된 버전 중 created_at이 가장 늦은 것."""
    from sqlalchemy import func, select

    from bike_demand.serving.models import Prediction

    with engine.connect() as conn:
        rows = conn.execute(
            select(Prediction.model_version, func.max(Prediction.created_at))
            .where(Prediction.created_at >= since)
            .group_by(Prediction.model_version)
        ).all()
    latest: dict[str, tuple[datetime, str]] = {}
    for version, created in rows:
        if "-fcst" not in version:
            continue
        key = version.rsplit("-fcst", 1)[1]
        if key not in latest or (created, version) > latest[key]:
            latest[key] = (created, version)
    return {key: version for key, (_, version) in latest.items()}


def stored_predictions(engine, version: str, first: datetime) -> dict[str, dict[datetime, float]]:
    from sqlalchemy import select

    from bike_demand.serving.models import Prediction

    out: dict[str, dict[datetime, float]] = {}
    with engine.connect() as conn:
        rows = conn.execute(
            select(Prediction.station_id, Prediction.hour_start, Prediction.predicted_rentals)
            .where(Prediction.model_version == version)
            .where(Prediction.hour_start >= first)
            .where(Prediction.hour_start < first + HOUR_COLUMNS * HOUR)
        ).all()
    for station, hour, value in rows:
        out.setdefault(station, {})[hour.astimezone(KST)] = float(value)
    return out


def load_profile(warehouse) -> Profile:
    """작년 같은 계절(PROFILE_WINDOW)의 대여소 × 쉬는 날 × 시각 시간당 평균 대여·반납."""
    import duckdb

    con = duckdb.connect(str(warehouse), read_only=True)
    try:
        rows = con.execute(
            """
            select g.station_id, d.is_offday, d.hour_of_day,
                   sum(coalesce(g.rentals, 0)) / count(*) as rentals,
                   sum(coalesce(h.returns, 0)) / count(*) as returns
            from int_station_hour_grid as g
            join dim_hours as d using (hour_start)
            left join fct_station_hourly as h using (station_id, hour_start)
            where g.hour_start >= cast(? as timestamp) and g.hour_start < cast(? as timestamp)
            group by all
            """,
            list(PROFILE_WINDOW),
        ).fetchall()
    finally:
        con.close()
    return Profile({(s, bool(off), int(hr)): (float(r), float(q)) for s, off, hr, r, q in rows})


def recompute_predictions(
    booster,
    artifacts,
    features,
    meta: Mapping[str, dict],
    stations: list[str],
    first: datetime,
    weather: Mapping[datetime, dict[str, float]],
) -> dict[str, dict[datetime, float]]:
    """서비스 예측 작업과 같은 특징으로 그 발표의 예보를 넣어 다시 예측한다(개발 자료용)."""
    from bike_demand.model.predict import feature_rows

    hours = [first + k * HOUR for k in range(HOUR_COLUMNS)]
    hours = [h for h in hours if h in weather]
    listed = [meta[s] for s in stations if s in meta]
    if not hours or not listed:
        return {}
    matrix, keys = feature_rows(listed, hours, weather, artifacts, features)
    values = booster.predict(matrix)
    out: dict[str, dict[datetime, float]] = {}
    for (station, hour), value in zip(keys, values, strict=True):
        out.setdefault(station, {})[hour] = float(value)
    return out


def build_rows(snaps: Snapshots, inputs: Iterable[IssueInput], profile: Profile) -> tuple:
    """발표마다 규칙을 보고 행을 만든다. (행 dict 또는 None, 발표 기록 목록)."""
    parts, extra, log = [], [], []
    for issue in inputs:
        as_of = issue.as_of.astimezone(KST)
        entry = {"base": issue.base.isoformat(), "version": issue.version}
        status = issue_status(snaps, as_of)
        if status == "ok" and not issue.predictions:
            status = "no_predictions"
        entry["status"] = status
        log.append(entry)
        if status != "ok":
            continue
        part = issue_rows(snaps, issue, profile)
        parts.append(part)
        extra.append({"issue": len(log) - 1, "day": as_of.date().toordinal()})
    if not parts:
        return None, log
    rows = concat_rows(parts, extra)
    add_horizon_inputs(rows)
    main = main_mask(rows)
    for i, entry in enumerate(log):
        if entry["status"] == "ok":
            entry["main_rows"] = int((main & (rows["issue"] == i)).sum())
    return rows, log


def valid_days(log: list[dict], end: datetime) -> int:
    """as_of가 end 전인 발표 중 주 평가 행이 하나라도 있는 발표의 날짜 수."""
    days = set()
    for entry in log:
        as_of = datetime.fromisoformat(entry["base"]) + SERVICE_DELAY
        if as_of < end and entry.get("main_rows", 0) > 0:
            days.add(as_of.astimezone(KST).date())
    return len(days)


def _sha256(path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, data) -> None:
    import json

    def default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, datetime | date):
            return value.isoformat()
        raise TypeError(type(value))

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, default=default)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


# ---------------------------------------------------------------- 개발과 test 실행


def run_dev(engine, warehouse, models_dir, out_dir, n_boot: int = BOOT_N) -> dict:
    """개발 6일: 행을 만들고 φ·도전자·분류기를 정해 out_dir에 고정한다(test 전에 커밋할 값)."""
    import json

    from bike_demand.model import shortage_nowcast as nc
    from bike_demand.model.final import load_serving_model
    from bike_demand.serving.models import Station

    start, end = DEV_WINDOW
    generation, booster, artifacts = load_serving_model(models_dir)
    features = booster.feature_name()
    with engine.connect() as conn:
        from sqlalchemy import select

        meta = {
            r["station_id"]: dict(r)
            for r in conn.execute(
                select(
                    Station.station_id, Station.district, Station.docks, Station.lat, Station.lon
                )
            ).mappings()
        }
    profile = load_profile(warehouse)
    snaps = load_snapshots(engine, start - 2 * HOUR, end + (HOUR_COLUMNS + 1) * HOUR)
    versions = stored_versions(engine, start - timedelta(days=1))
    checks = []

    def inputs():
        for base in issue_bases(engine, start, end):
            as_of = base + SERVICE_DELAY
            ref = snaps.reference(as_of)
            weather = load_forecast(engine, base)
            first = as_of.replace(minute=0, second=0, microsecond=0)
            predictions = {}
            if ref is not None:
                stations = [snaps.stations[c] for c in np.flatnonzero(snaps.bikes[ref] >= 0)]
                predictions = recompute_predictions(
                    booster, artifacts, features, meta, stations, first, weather
                )
            stored = versions.get(issue_key(base))
            if predictions and stored and stored.startswith(f"{generation}-fcst"):
                saved = stored_predictions(engine, stored, first)
                diffs = [
                    abs(v - predictions[s][h])
                    for s, by_hour in saved.items()
                    for h, v in by_hour.items()
                    if s in predictions and h in predictions[s]
                ]
                checks.append(
                    {"base": base.isoformat(), "version": stored, "pairs": len(diffs),
                     "max_abs_diff": max(diffs) if diffs else None}
                )  # fmt: skip
            rain = {h: w["rain_mm"] for h, w in weather.items()}
            yield IssueInput(
                base, predictions, rain, f"{generation}-fcst{issue_key(base)}(다시 계산)"
            )

    rows, log = build_rows(snaps, inputs(), profile)
    mask = main_mask(rows)
    y = rows["y3"][mask]
    b0 = rows["b0"][mask].astype(np.float64)
    day = rows["day"][mask]
    phi, phi_losses = pick_phi(b0, rows["E3"][mask], rows["Q3"][mask], y)
    scores = candidate_scores(rows, mask, phi)
    challenger = pick_challenger({k: RankIndex.of(scores[k]).auc(y) for k in CHALLENGERS})

    x = nc.logit_matrix(rows, mask)
    logit_sel = nc.select_logistic(x, y, day)
    logistic = nc.fit_logistic(x, y, logit_sel["C"])
    xl = nc.lgbm_matrix(rows, mask)
    lgbm_sel = nc.select_lgbm(xl, y, day)
    lgbm_model = nc.fit_lgbm(xl, y, lgbm_sel["num_leaves"], lgbm_sel["rounds"])
    scores["N"] = logistic.predict(x)
    scores["N_lgbm"] = lgbm_model.predict(xl)

    out_dir.mkdir(parents=True, exist_ok=True)
    lgbm_path = out_dir / "dev_lgbm.txt"
    lgbm_model.save_model(str(lgbm_path))
    fit = {
        "plan": "docs/experiments.md v5 (005b643)",
        "generation": generation,
        "dev_window": [start.isoformat(), end.isoformat()],
        "phi": phi,
        "challenger": challenger,
        "logistic": logistic.to_dict(),
        "lgbm": {
            "file": lgbm_path.name,
            "sha256": _sha256(lgbm_path),
            "num_leaves": lgbm_sel["num_leaves"],
            "rounds": lgbm_sel["rounds"],
            "features": list(nc.LGBM_FEATURES),
        },
    }
    fit_path = out_dir / "dev_fit.json"
    _write_json(fit_path, fit)

    oof_mask = ~np.isnan(logit_sel["oof"])
    _, main_report = full_report(rows, scores, mask, ("S3", "S4", "N", "N_lgbm"), n_boot)
    report = {
        "note": "개발 자료(이미 본 09-16~22). N·N_lgbm 값은 학습 자료 안 값이라 낙관적이다",
        "fit_sha256": _sha256(fit_path),
        "fit": fit,
        "issues": log,
        "issues_used": sum(1 for e in log if e.get("main_rows", 0) > 0),
        "recompute_vs_stored": checks,
        "exclusions": exclusion_counts(rows),
        "already_empty": already_empty_report(rows),
        "api_list_S0": api_list_precision(rows),
        "phi_log_loss": phi_losses,
        "challenger_auc": {k: _round(RankIndex.of(scores[k]).auc(y)) for k in CHALLENGERS},
        "logistic_cv": logit_sel["cv"],
        "lgbm_cv": lgbm_sel["cv"],
        "cv_out_of_fold": {
            "rows": int(oof_mask.sum()),
            "N_log_loss": _round(log_loss(logit_sel["oof"][oof_mask], y[oof_mask])),
            "N_brier": _round(brier(logit_sel["oof"][oof_mask], y[oof_mask])),
            "N_reliability": reliability(logit_sel["oof"][oof_mask], y[oof_mask]),
            "N_lgbm_log_loss": _round(log_loss(lgbm_sel["oof"][oof_mask], y[oof_mask])),
            "S3_log_loss": _round(log_loss(scores["S3"][oof_mask], y[oof_mask])),
            "S4_log_loss": _round(log_loss(scores["S4"][oof_mask], y[oof_mask])),
            "auc_b0ge2": {
                name: _round(RankIndex.of(s[oof_mask & (b0 >= 2)]).auc(y[oof_mask & (b0 >= 2)]))
                for name, s in {
                    "N": logit_sel["oof"],
                    "N_lgbm": lgbm_sel["oof"],
                    "S1b": scores["S1b"],
                    challenger: scores[challenger],
                }.items()
            },  # fmt: skip
        },
        "main": main_report,
        "horizons": horizon_report(rows, phi, challenger, n_boot),
    }
    _write_json(out_dir / "dev_report.json", report)
    print(json.dumps({k: report[k] for k in ("fit_sha256", "issues_used", "exclusions")}))
    return report


def run_test(engine, warehouse, out_dir, now: datetime, n_boot: int = BOOT_N) -> dict:
    """14일 test를 한 번 잰다. 평가 가능 시각 전이나 이미 잰 뒤에는 SystemExit."""
    import json

    from bike_demand.model import shortage_nowcast as nc
    from bike_demand.model.final import reserve_once

    result_path = out_dir / "test.json"
    if result_path.exists():
        raise SystemExit(f"test는 한 번만 잰다: {result_path}가 이미 있음")
    if now < TEST_END + EVAL_DELAY:
        raise SystemExit(f"test 평가는 {(TEST_END + EVAL_DELAY).isoformat()} 이후")
    fit_path = out_dir / "dev_fit.json"
    fit = json.loads(fit_path.read_text("utf-8"))
    lgbm_path = out_dir / fit["lgbm"]["file"]
    if _sha256(lgbm_path) != fit["lgbm"]["sha256"]:
        raise SystemExit("고정한 LightGBM 파일이 바뀜")

    profile = load_profile(warehouse)
    snaps = load_snapshots(engine, TEST_START - 2 * HOUR, TEST_MAX_END + (HOUR_COLUMNS + 1) * HOUR)
    versions = stored_versions(engine, TEST_START - timedelta(days=1))
    skipped_generation = []

    def inputs():
        for base in issue_bases(engine, TEST_START, TEST_MAX_END):
            version = versions.get(issue_key(base))
            predictions = {}
            if version and not version.startswith(f"{fit['generation']}-fcst"):
                skipped_generation.append({"base": base.isoformat(), "version": version})
                version = None
            if version:
                first = (base + SERVICE_DELAY).replace(minute=0, second=0, microsecond=0)
                predictions = stored_predictions(engine, version, first)
            weather = load_forecast(engine, base)
            rain = {h: w["rain_mm"] for h, w in weather.items()}
            yield IssueInput(base, predictions, rain, version)

    rows, log = build_rows(snaps, inputs(), profile)
    end = TEST_END
    while True:
        if now < end + EVAL_DELAY:
            raise SystemExit(
                f"유효한 날이 {valid_days(log, end - timedelta(days=1))}일이라 창을 늘림: "
                f"{(end + EVAL_DELAY).isoformat()} 이후에 다시 실행"
            )
        days = valid_days(log, end)
        if days >= MIN_VALID_DAYS or end >= TEST_MAX_END:
            break
        end += timedelta(days=1)
    reserve_once(result_path.with_suffix(".started"))

    in_window = np.array([
        datetime.fromisoformat(log[i]["base"]) + SERVICE_DELAY < end for i in rows["issue"]
    ])  # fmt: skip
    for key in list(rows):
        rows[key] = rows[key][in_window]
    mask = main_mask(rows)
    phi, challenger = float(fit["phi"]), fit["challenger"]
    scores = candidate_scores(rows, mask, phi)
    logistic = nc.Logistic.from_dict(fit["logistic"])
    scores["N"] = logistic.predict(nc.logit_matrix(rows, mask))
    import lightgbm as lgb

    scores["N_lgbm"] = lgb.Booster(model_file=str(lgbm_path)).predict(nc.lgbm_matrix(rows, mask))
    ev, main_report = full_report(rows, scores, mask, ("S3", "S4", "N", "N_lgbm"), n_boot)
    report = {
        "plan": "docs/experiments.md v5 (005b643)",
        "fit_sha256": _sha256(fit_path),
        "window": [TEST_START.isoformat(), end.isoformat()],
        "valid_days": days,
        "generation_changed": skipped_generation,
        "issues": log,
        "exclusions": exclusion_counts(rows),
        "already_empty": already_empty_report(rows),
        "api_list_S0": api_list_precision(rows),
        "main": main_report,
        "decision": decide(ev, challenger, days),
        "horizons": horizon_report(rows, phi, challenger, n_boot),
    }
    _write_json(result_path, report)
    print(json.dumps(report["decision"], ensure_ascii=False))
    return report


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from bike_demand.serving.db import make_engine

    parser = argparse.ArgumentParser(description="v5 부족 경보 평가 (docs/experiments.md v5)")
    parser.add_argument("step", choices=["dev", "test"])
    parser.add_argument("--warehouse", type=Path, default=Path("data/warehouse/bike_demand.duckdb"))
    parser.add_argument("--models", type=Path, default=Path("data/models/v1"))
    parser.add_argument("--out", type=Path, default=Path("data/models/v5"))
    args = parser.parse_args()
    if args.step == "dev":
        run_dev(make_engine(), args.warehouse, args.models, args.out)
    else:
        run_test(make_engine(), args.warehouse, args.out, datetime.now(KST))
