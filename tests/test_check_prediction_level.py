"""scripts/check_prediction_level.py의 조회 범위 고정(T55).

예측이 더 쌓여도 문서와 같은 범위를 읽는다.
"""

# [무엇을 검증하나] 서비스 예측 수준 진단 스크립트(scripts/check_prediction_level.py)의
# predicted_cells가 정해진 범위만 읽는지 확인한다: 시각은 since~until(끝 포함), 발표는 last_issue
# 이하, 버전 접두어가 맞는 것만, 시각마다 가장 최근 발표의 값 하나, 공휴일 시간 제외.
# [왜 중요한가] 이 진단의 수치(407,847행·149시간, 예측 ÷ 실제 0.98·1.02·1.15)는
# docs/experiments.md에 적혀 있다. 예측이 계속 쌓이는 DB에서 범위를 고정하지 않으면 다시 돌릴 때마다
# 결과가 바뀌어 문서 값을 재현할 수 없다(4eb819c). 공휴일 제외는 실제 쪽과 규칙을 맞추려고
# 넣었다(d5b8025).

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 임시 DB(pg_engine)에 예측 행을 넣는 데 쓴다.
from bike_demand.serving.models import Prediction

KST = timezone(timedelta(hours=9))
# scripts/는 패키지가 아니라 import할 수 없으므로 파일 경로로 불러온다.
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_prediction_level.py"


# importlib로 스크립트 파일을 모듈로 읽는다. if __name__ == '__main__' 아래 실행 부분은 돌지 않는다.
def load_script():
    spec = importlib.util.spec_from_file_location("check_prediction_level", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 18·19·20시에 옛 발표(09-21 20시) 값 1.0, 19시에만 새 발표(09-22 02시) 값 5.0, 18시에 접두어가 다른
# v2 버전 값 9.0을 넣는다.
# - until(19시)은 포함, 그 뒤 20시는 빠진다
# - last_issue가 09-21 20시면 22일 발표는 무시되고 19시도 옛 값 1.0이다
# - 접두어 'v3-M3p-'가 아닌 v2 버전은 시각이 같아도 섞이지 않는다
# - last_issue를 22일 02시로 늦추면 같은 시각의 값이 가장 최근 발표(5.0)로 바뀐다
# 키는 (대여소, 쉬는 날 여부, 시각)이고 9월 23일은 수요일이라 쉬는 날이 아니다.
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


# 2026년 추석 연휴(9월 24일 목요일, 26일 토요일)의 예측 시간이 빠지고 평일 23일 20시만 남는지
# 확인한다. 공휴일인 목요일을 그대로 두면 '평일' 칸에 쉬는 날 수준의 낮은 예측이 섞여 평일 비교가
# 틀어진다(고치기 전에 이 테스트가 실패했다).
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
