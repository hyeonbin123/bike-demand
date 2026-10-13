# [무엇을 검증하나] Airflow collect_realtime 작업의 본체(pipelines.collect_realtime)가 실시간
# 스냅샷을 빠짐없이, 중복 없이 적재하는지 확인한다.
# - 적재 날짜는 수집 시각(KST)의 날짜다(23:59:59.9 수집분도 그날로)
# - 적재가 실패한 지난 날은 loaded.txt 기록과 원본 목록을 비교해 다음 정상 실행이 회수한다(자정 넘어
#   재시도, 0시대 내내 장애). 한 날이 계속 실패해도 다른 날(오늘 포함)은 같은 실행에서 적재한다
# - 겹친 실행은 잠금으로 막고, 오래된 잠금은 가져가며, 잠금을 빼앗긴 실행도 실제로 적재한 것만
#   기록한다
# - latest-trip-month 명령의 최신 달·말일 계산
# [왜 중요한가] 실시간 스냅샷은 지나가면 다시 받을 수 없다. 적재가 조용히 빠지면 부족 위험 화면과 v5
# 측정의 정답 자료에 구멍이 생긴다.
# [방식] 실시간 API는 MockTransport로 대여소 2곳을 돌려주고, DB 적재 함수(load.load_realtime_file)는
# monkeypatch로 실패·지연을 흉내 낸다. DB가 필요한 테스트는 pg_engine(임시 PostgreSQL)을 쓴다.

from datetime import date, datetime

# httpx: 가짜 실시간 API. sqlalchemy: 적재된 스냅샷 수·시각 확인.
import httpx
from sqlalchemy import func, select

from bike_demand.ingest import realtime
from bike_demand.pipelines import collect_realtime
from bike_demand.serving.models import RealtimeSnapshot

SECRET = "fake-key"


# 적재 함수가 받은 Parquet 경로에서 날짜를 읽어, 어느 날을 적재하려 했는지 기록하는 데 쓴다.
def day_of(parquet_path):
    """bronze/date=YYYY-MM-DD/파일 → 그 날짜."""
    return date.fromisoformat(parquet_path.parent.name.removeprefix("date="))


# 가짜 bikeList 응답: 대여소 2곳(ST-1, ST-2), 각 자전거 3대. 한 번 수집할 때마다 스냅샷 2행이
# 생긴다.
def bike_list(request: httpx.Request) -> httpx.Response:
    rows = [
        {"rackTotCnt": "10", "stationName": f"{n}. 대여소", "parkingBikeTotCnt": "3",
         "shared": "30", "stationLatitude": "37.5", "stationLongitude": "127.0",
         "stationId": f"ST-{n}"}
        for n in (1, 2)
    ]  # fmt: skip
    body = {
        "rentBikeStatus": {
            "list_total_count": 2,
            "RESULT": {"CODE": "INFO-000", "MESSAGE": "정상"},
            "row": rows,
        }
    }
    return httpx.Response(200, json=body)


# 23:59:59.9에 수집한 스냅샷은 적재가 자정을 넘겨 끝나도 09-16의 Parquet과 DB에 들어간다. 수집과
# 적재가 따로 '오늘'을 계산하던 때 생길 수 있던 날짜 어긋남을 막는다. 처음 보는 대여소 2곳도 함께
# 추가된다.
def test_snapshot_taken_just_before_midnight_is_loaded_under_its_own_day(pg_engine, tmp_path):
    raw, bronze = tmp_path / "raw", tmp_path / "bronze"
    just_before_midnight = datetime(2026, 9, 16, 23, 59, 59, 900000, tzinfo=realtime.KST)
    client = httpx.Client(transport=httpx.MockTransport(bike_list))

    result = collect_realtime(pg_engine, raw, bronze, SECRET, just_before_midnight, client)

    assert result["day"] == "2026-09-16"
    assert result["snapshots_inserted"] == 2 and result["new_stations"] == 2
    assert (bronze / "date=2026-09-16" / "snapshots.parquet").exists()
    with pg_engine.connect() as conn:
        fetched = conn.execute(select(func.max(RealtimeSnapshot.fetched_at))).scalar_one()
    assert fetched.astimezone(realtime.KST).date() == date(2026, 9, 16)


# 원본 파일 이름 순서와 상관없이 가장 최근 달(2026-12)을 고르고, 말일은 윤년 2월(2028-02-29)까지
# 맞게 계산한다. refresh_history DAG가 날씨 수집 끝 달과 dbt calendar_end에 넘기는 값이다.
def test_latest_trip_month_and_last_day(tmp_path):
    from bike_demand.pipelines import last_day, latest_trip_month

    raw = tmp_path / "trips"
    raw.mkdir()
    for name in ("대여이력_2606.csv", "대여이력_2612.csv", "대여이력_2609.csv"):
        (raw / name).write_bytes(b"")
    assert latest_trip_month(raw) == "2026-12"
    assert last_day("2026-12") == "2026-12-31"
    assert last_day("2028-02") == "2028-02-29"


# 23:59:59 실행은 원본 저장 뒤 적재가 한 번 실패한다. 00:02 실행은 기록이 없는 전날(09-16)을 먼저
# 다시 적재하고 오늘을 적재해 스냅샷 4행이 된다. 01:00 실행은 이미 기록이 원본과 같은 전날을
# 건드리지 않고 오늘만 적재한다(불필요한 재적재 없음).
def test_retry_after_midnight_recovers_the_previous_day(pg_engine, tmp_path, monkeypatch):
    """23:59:59 수집 성공 → 적재 실패 → 00:02 다음 실행에서 전날 스냅샷도 적재된다(T35)."""
    from bike_demand.serving import load

    raw, bronze = tmp_path / "raw", tmp_path / "bronze"
    client = httpx.Client(transport=httpx.MockTransport(bike_list))
    real_load = load.load_realtime_file
    calls = []
    failures = [RuntimeError("DB 잠깐 끊김")]

    def flaky_load(engine, path):
        calls.append(day_of(path))
        if failures:
            raise failures.pop()
        return real_load(engine, path)

    monkeypatch.setattr(load, "load_realtime_file", flaky_load)
    before = datetime(2026, 9, 16, 23, 59, 59, tzinfo=realtime.KST)
    after = datetime(2026, 9, 17, 0, 2, tzinfo=realtime.KST)
    try:
        collect_realtime(pg_engine, raw, bronze, SECRET, before, client)
    except RuntimeError:
        pass
    result = collect_realtime(pg_engine, raw, bronze, SECRET, after, client)

    assert calls == [date(2026, 9, 16), date(2026, 9, 16), date(2026, 9, 17)]
    assert result["days"]["2026-09-16"]["snapshots_inserted"] == 2
    assert result["days"]["2026-09-17"]["snapshots_inserted"] == 2
    with pg_engine.connect() as conn:
        count = conn.execute(select(func.count()).select_from(RealtimeSnapshot)).scalar_one()
    assert count == 4

    # 이미 다 적재한 전날은 다시 건드리지 않는다
    calls.clear()
    one_am = datetime(2026, 9, 17, 1, 0, tzinfo=realtime.KST)
    collect_realtime(pg_engine, raw, bronze, SECRET, one_am, client)
    assert calls == [date(2026, 9, 17)]


# DB 장애가 23:59 실행과 00:02 실행 모두를 실패시키면 전날 loaded.txt가 생기지 않는다. 03:10에 DB가
# 돌아오면 그 실행이 전날(1개 원본)과 오늘(00:02·03:10 원본 2개)을 모두 적재하고, 전날 기록에는 실제
# 적재한 원본 이름만 남는다.
# 다음 실행은 오늘만 다시 적재하는데, 이미 있는 스냅샷은 건너뛰므로 행은 오늘 새 수집분만큼만 늘어
# 8행이다(멱등).
def test_failures_through_the_midnight_hour_are_recovered_later(pg_engine, tmp_path, monkeypatch):
    """23:59와 0시대 재시도가 모두 적재에 실패해도 1시대 이후 정상 실행이 전날을 회수한다(T39)."""
    from bike_demand.serving import load

    raw, bronze = tmp_path / "raw", tmp_path / "bronze"
    client = httpx.Client(transport=httpx.MockTransport(bike_list))
    real_load = load.load_realtime_file
    calls = []
    db_down = {"value": True}

    def flaky_load(engine, path):
        calls.append(day_of(path))
        if db_down["value"]:
            raise RuntimeError("DB 장애")
        return real_load(engine, path)

    monkeypatch.setattr(load, "load_realtime_file", flaky_load)
    for at in (datetime(2026, 9, 16, 23, 59, 59), datetime(2026, 9, 17, 0, 2)):
        try:
            collect_realtime(
                pg_engine, raw, bronze, SECRET, at.replace(tzinfo=realtime.KST), client
            )
        except RuntimeError:
            pass
    assert not (bronze / "date=2026-09-16" / "loaded.txt").exists()

    db_down["value"] = False
    calls.clear()
    late = datetime(2026, 9, 17, 3, 10, tzinfo=realtime.KST)
    result = collect_realtime(pg_engine, raw, bronze, SECRET, late, client)
    assert calls == [date(2026, 9, 16), date(2026, 9, 17)]
    assert result["days"]["2026-09-16"]["snapshots_inserted"] == 2
    assert result["days"]["2026-09-17"]["snapshots_inserted"] == 4  # 00:02와 03:10
    loaded = (bronze / "date=2026-09-16" / "loaded.txt").read_text(encoding="utf-8").split()
    assert loaded == ["235959.json"]

    # 다음 실행은 오늘만 적재하고, 같은 원본을 다시 넣어도 행이 늘지 않는다
    calls.clear()
    again = datetime(2026, 9, 17, 3, 20, tzinfo=realtime.KST)
    collect_realtime(pg_engine, raw, bronze, SECRET, again, client)
    assert calls == [date(2026, 9, 17)]
    with pg_engine.connect() as conn:
        count = conn.execute(select(func.count()).select_from(RealtimeSnapshot)).scalar_one()
    assert count == 8


# 새 잠금 파일이 있으면 원본 JSON은 저장하되 적재는 하지 않고 LoadBusy를 낸다(적재 함수가 불리면
# 실패). 잠금 시각을 STALE_LOCK_SECONDS보다 오래되게 바꾸면 다음 실행이 그 잠금을 지우고 들어가,
# 쌓인 원본 2개(4행)를 적재·기록한 뒤 자기 잠금을 푼다. engine은 적재를 가짜로 바꿔 None으로 둔다.
def test_overlapping_runs_do_not_rebuild_and_load_at_the_same_time(tmp_path, monkeypatch):
    """적재 구간은 한 실행만: 잠금이 있으면 원본만 남기고 LoadBusy, 오래된 잠금은 지운다(T42)."""
    import os
    import time

    import pytest

    from bike_demand import pipelines

    raw, bronze = tmp_path / "raw", tmp_path / "bronze"
    client = httpx.Client(transport=httpx.MockTransport(bike_list))
    at = datetime(2026, 9, 17, 12, 0, tzinfo=realtime.KST)
    lock = bronze / pipelines.LOCK_NAME
    bronze.mkdir(parents=True)
    lock.touch()
    monkeypatch.setattr(
        pipelines.load, "load_realtime_file", lambda *a: pytest.fail("적재하면 안 됨")
    )
    with pytest.raises(pipelines.LoadBusy):
        pipelines.collect_realtime(None, raw, bronze, SECRET, at, client)
    assert len(list((raw / "date=2026-09-17").glob("*.json"))) == 1  # 수집분은 남음
    assert lock.exists() and not (bronze / "date=2026-09-17" / "loaded.txt").exists()

    old = time.time() - pipelines.STALE_LOCK_SECONDS - 60
    os.utime(lock, (old, old))
    monkeypatch.setattr(pipelines.load, "load_realtime_file", lambda *a: (2, 0))
    later = datetime(2026, 9, 17, 12, 10, tzinfo=realtime.KST)
    result = pipelines.collect_realtime(None, raw, bronze, SECRET, later, client)
    assert result["days"]["2026-09-17"]["parquet_rows"] == 4
    assert not lock.exists()
    loaded = (bronze / "date=2026-09-17" / "loaded.txt").read_text(encoding="utf-8").split()
    assert loaded == ["120000.json", "121000.json"]


# 가장 까다로운 겹침을 재현한다. 실행 A(23:40)가 적재하는 도중 잠금이 30분 넘게 묵어 실행 B(23:50)가
# 잠금을 가져가 끝까지 돌고, B가 적재하기 직전에는 늦게 끝난 A의 재생성이 공용 snapshots.parquet을
# 옛 내용으로 덮어쓴다.
# 각 실행이 자기 이름의 임시 Parquet을 적재하므로 공용 파일이 덮여도 B는 23:40·23:50을 모두 적재하고
# 그만큼만 기록한다. A는 Parquet을 만들기 전에 본 원본(23:40)만 기록하므로 기록이 적재를 앞서지
# 않는다(빠진 23:50은 기록에 없어 다음 실행이 다시 확인). A는 끝날 때 B의 잠금을 지우지 않고(토큰
# 비교), 임시 .part 파일도 남지 않는다.
def test_taken_over_lock_cannot_mark_snapshots_that_were_not_loaded(tmp_path, monkeypatch):
    """느린 실행 A의 잠금을 30분 뒤 B가 가져가도, 각자 만든 파일을 적재하므로 기록이 적재를 앞서지
    않는다. A는 끝날 때 B의 잠금을 지우지 않는다(T44, codex 라운드 6 재현)."""
    import io
    import os
    import time

    import pyarrow.parquet as pq

    from bike_demand import pipelines

    raw, bronze = tmp_path / "raw", tmp_path / "bronze"
    client = httpx.Client(transport=httpx.MockTransport(bike_list))
    lock = bronze / pipelines.LOCK_NAME
    day_dir = bronze / "date=2026-09-17"
    a_run = datetime(2026, 9, 17, 23, 40, tzinfo=realtime.KST)
    b_run = datetime(2026, 9, 17, 23, 50, tzinfo=realtime.KST)
    a_old: list[bytes] = []
    loaded_by_run: list[set[str]] = []

    # 파일을 바이트로 먼저 읽어 메모리에서 연다. pq.read_table(path)는 파일 핸들(메모리 맵)을
    # 바로 닫지 않을 때가 있어, 이어지는 os.replace가 Windows에서 가끔 WinError 32로
    # 실패했다(테스트만의 흔들림).
    def names_in(path):
        rows = pq.read_table(io.BytesIO(path.read_bytes())).to_pylist()
        return {
            datetime.fromisoformat(r["fetched_at"]).astimezone(realtime.KST).strftime("%H%M%S")
            + ".json"
            for r in rows
        }

    def load_file(engine, path):
        if not a_old:
            # A 적재 도중: A가 만든 옛 내용(23:40분)을 기억하고, 잠금을 30분 넘게 만든 뒤 B 실행
            a_old.append(path.read_bytes())
            old = time.time() - pipelines.STALE_LOCK_SECONDS - 60
            os.utime(lock, (old, old))
            pipelines.collect_realtime(None, raw, bronze, SECRET, b_run, client)
            b_marked = set((day_dir / "loaded.txt").read_text(encoding="utf-8").split())
            assert b_marked == {"234000.json", "235000.json"}
            assert b_marked <= loaded_by_run[-1]  # B가 기록한 원본은 B가 실제로 적재했다
        elif not loaded_by_run:
            # B 적재 직전: 늦게 끝난 A의 재생성이 공용 Parquet을 옛 내용으로 덮는다
            (day_dir / "snapshots.parquet").write_bytes(a_old[0])
        loaded_by_run.append(names_in(path))
        return 0, 0

    monkeypatch.setattr(pipelines.load, "load_realtime_file", load_file)
    pipelines.collect_realtime(None, raw, bronze, SECRET, a_run, client)

    a_marked = set((day_dir / "loaded.txt").read_text(encoding="utf-8").split())
    assert a_marked <= loaded_by_run[-1]  # A의 기록(옛 목록)도 A가 적재한 것뿐 → 다음 실행이 회수
    assert not lock.exists()  # B가 가져간 잠금을 A가 지우지 않았고, B는 자기 잠금을 풀었다
    assert not list(day_dir.glob("*.part"))


# 잠금 안에서 다른 실행이 잠금 파일 내용(토큰)을 바꿔 놓으면, 끝날 때 그 잠금을 지우지 않는다. 자기
# 토큰 그대로면 끝날 때 지운다.
def test_lock_release_keeps_a_lock_taken_over_by_another_run(tmp_path):
    from bike_demand import pipelines

    lock = tmp_path / pipelines.LOCK_NAME
    with pipelines._load_lock(tmp_path):
        lock.write_text("someone-else", encoding="utf-8")
    assert lock.read_text(encoding="utf-8") == "someone-else"
    lock.unlink()
    with pipelines._load_lock(tmp_path):
        assert lock.exists()
    assert not lock.exists()


# 회귀 테스트: 덜 적재된 지난 날 하나가 계속 실패해도(그날 원본에 적재할 수 없는 행이 있는 경우
# 등) 다른 지난 날과 오늘은 같은 실행에서 적재·기록된다. 전에는 오래된 날부터 차례로 적재하다 첫
# 실패에서 멈춰, 그 날이 회수 범위(7일)를 벗어날 때까지 매 실행에서 오늘 적재가 진행되지 않았다.
# 실패한 날은 기록이 남지 않아 다음 실행이 다시 시도하고, 실행은 실패한 날짜를 담은 LoadFailed로
# 끝난다(Airflow 작업 실패로 보임). DB 없이 적재 함수를 가짜로 바꿔 돈다.
def test_a_failing_past_day_does_not_block_other_days(tmp_path, monkeypatch):
    import pytest

    from bike_demand import pipelines

    raw, bronze = tmp_path / "raw", tmp_path / "bronze"
    client = httpx.Client(transport=httpx.MockTransport(bike_list))
    for at in (datetime(2026, 9, 15, 12, 0), datetime(2026, 9, 16, 12, 0)):
        realtime.download(raw, SECRET, at.replace(tzinfo=realtime.KST), client=client)
    calls = []

    def load_file(engine, path):
        calls.append(day_of(path))
        if day_of(path) == date(2026, 9, 15):
            raise ValueError("적재할 수 없는 행")
        return 2, 0

    monkeypatch.setattr(pipelines.load, "load_realtime_file", load_file)
    now = datetime(2026, 9, 17, 0, 10, tzinfo=realtime.KST)
    with pytest.raises(pipelines.LoadFailed, match="2026-09-15") as failed:
        pipelines.collect_realtime(None, raw, bronze, SECRET, now, client)

    assert isinstance(failed.value.__cause__, ValueError)
    assert calls == [date(2026, 9, 15), date(2026, 9, 16), date(2026, 9, 17)]
    assert not (bronze / "date=2026-09-15" / "loaded.txt").exists()
    for day in ("2026-09-16", "2026-09-17"):
        assert (bronze / f"date={day}" / "loaded.txt").exists()
        assert (bronze / f"date={day}" / "snapshots.parquet").exists()
    assert not list(bronze.glob("date=*/*.part"))

    # 다음 실행은 실패한 날만 다시 시도하고 오늘을 적재한다
    calls.clear()
    with pytest.raises(pipelines.LoadFailed):
        pipelines.collect_realtime(None, raw, bronze, SECRET, now.replace(minute=20), client)
    assert calls == [date(2026, 9, 15), date(2026, 9, 17)]
