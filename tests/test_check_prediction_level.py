"""scripts/check_prediction_level.py의 조회 범위 고정(T55).

예측이 더 쌓여도 문서와 같은 범위를 읽는다.
"""

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bike_demand.serving.models import Prediction

KST = timezone(timedelta(hours=9))
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_prediction_level.py"


def load_script():
    spec = importlib.util.spec_from_file_location("check_prediction_level", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_predicted_cells_stop_at_until_and_last_issue(pg_engine):
    script = load_script()
    hours = [datetime(2026, 9, 23, 18 + i, tzinfo=KST) for i in range(3)]  # 18, 19, 20시
    old, new = "v3-M3p-X-fcst202609212000", "v3-M3p-X-fcst202609220200"
    rows = [
        {"station_id": "ST-1", "hour_start": h, "model_version": old, "predicted_rentals": 1.0,
         "created_at": hours[0]}
        for h in hours
    ] + [
        {"station_id": "ST-1", "hour_start": hours[1], "model_version": new,
         "predicted_rentals": 5.0, "created_at": hours[1]},
        {"station_id": "ST-1", "hour_start": hours[0], "model_version": "v2-M3-Y-fcst202609212000",
         "predicted_rentals": 9.0, "created_at": hours[0]},
    ]  # fmt: skip
    with pg_engine.begin() as conn:
        conn.execute(Prediction.__table__.insert(), rows)

    since, until = hours[0], hours[1]
    predicted, weights, seen = script.predicted_cells(
        pg_engine, "v3-M3p-", since, until, datetime(2026, 9, 21, 20, tzinfo=KST)
    )
    assert seen == hours[:2]  # 20시(until 뒤)는 빠지고 19시(until)는 포함
    assert predicted == {("ST-1", False, 18): 1.0, ("ST-1", False, 19): 1.0}  # 22일 발표는 무시
    assert weights[("ST-1", False, 19)] == 1

    later, _, _ = script.predicted_cells(
        pg_engine, "v3-M3p-", since, until, datetime(2026, 9, 22, 2, tzinfo=KST)
    )
    assert later[("ST-1", False, 19)] == 5.0  # 발표 한도를 늦추면 새 발표의 값


def test_predicted_cells_drop_holidays_like_the_actual_side(pg_engine):
    """실제 쪽은 공휴일을 뺀다. 예측 쪽도 빼야 추석 예측(쉬는 날 수준)이 평일 칸에 섞이지 않는다."""
    script = load_script()
    wednesday = datetime(2026, 9, 23, 20, tzinfo=KST)
    chuseok_eve = datetime(2026, 9, 24, 8, tzinfo=KST)  # 목요일, 추석 연휴
    saturday = datetime(2026, 9, 26, 8, tzinfo=KST)  # 토요일, 추석 연휴
    version = "v3-M3p-X-fcst202609232000"
    rows = [
        {"station_id": "ST-1", "hour_start": h, "model_version": version,
         "predicted_rentals": value, "created_at": wednesday}
        for h, value in ((wednesday, 3.0), (chuseok_eve, 0.5), (saturday, 0.7))
    ]  # fmt: skip
    with pg_engine.begin() as conn:
        conn.execute(Prediction.__table__.insert(), rows)

    predicted, weights, seen = script.predicted_cells(
        pg_engine, "v3-M3p-", wednesday, datetime(2026, 9, 26, 23, tzinfo=KST), wednesday
    )
    assert predicted == {("ST-1", False, 20): 3.0}
    assert weights == {("ST-1", False, 20): 1}
    assert seen == [wednesday]
