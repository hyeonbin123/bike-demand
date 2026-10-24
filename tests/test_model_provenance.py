# [무엇을 검증하나] 서비스 세대의 provenance(무엇으로 만들었는지) 수집과 세대
# 검증(model/provenance.py), 그리고 산출물 번호표를 지금 warehouse의 번호와 대조하는
# artifacts.warehouse_codes를 확인한다.
# - 번호 밀림 회귀: 새 반기를 넣어 다시 빌드한 warehouse에 정렬 중간에 끼는 새 대여소(예: 'ST-15'는
#   'ST-1'과 'ST-2' 사이)가 들어오면 frames.load_frame의 station_code는 밀리지만, 산출물
#   경로(predict.feature_rows)는 학습 때 고정한 번호와 값을 그대로 쓰고 새 대여소만
#   max_station_code+1을 받는다. verify_generation(con=...)은 번호가 달라진 곳 수를 보고하고
#   strict면 멈춘다
# - history 끝 전에 끊겼다가 뒤에 다시 나온 대여소(다시 빌드한 격자가 그 사이를 0건으로 채움)도
#   산출물 경로에서는 산출물 값을 그대로 받는다
# - 특징 순서: 모델의 feature_name()과 meta.json features가 다르면 provenance 없는 옛 세대는 경고,
#   provenance가 있는 세대는 오류
# - 파일 지문: 기록(serving.json의 provenance.files 또는 넘긴 사전)과 다른 파일·빠진 파일을 보고
# - git 상태·warehouse 지문·수집 결과의 모양. git 명령은 주입한 함수로 흉내 내 개발 중 작업 트리에
#   흔들리지 않는다
# - scripts/generation_manifest.py: 세대 폴더의 파일(serving.json 포함) 지문·검사 보고·메모를
#   폴더 밖에 남기는 읽기 전용 스크립트. 출력 경로가 세대 폴더 안이면 거부한다
# [왜 중요한가] 고정한 세대로 나중의 warehouse에서 채점·예측할 때 입력이 학습 때와 같은지 코드가
# 확인하는 관문이다. 범주형 특징 station_code는 'warehouse의 대여소ID 문자열 정렬 순위'라 새
# 대여소가 들어오면 기존 대여소의 번호가 밀려 다른 대여소의 트리 분기를 타게 된다. 또 세대가 어느
# 코드 커밋·어느 라이브러리·어느 자료로 만들어졌는지 남아야 결과를 증명할 수 있다.

import hashlib
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from bike_demand.model import artifacts, frames, predict, provenance

KST = timezone(timedelta(hours=9))
# two_year_warehouse(2023-01 ~ 2025-06)에서 쓰는 학습 기간. 그 뒤(2025-01~)는 '새 반기' 역할이다.
HISTORY = ("2023-01-01", "2025-01-01")
# 수준 특징까지 포함한 서비스 모델의 특징 순서(M3'). 게시된 세대 model.txt의 feature_names와 같다.
LEVELED = [*frames.FEATURES, *frames.LEVEL_FEATURES]
# 2025-03-12은 수요일(평일). 그 달에 공개된 반기는 2024 하반기다(published_half).
DAY_HOURS = [datetime(2025, 3, 12, h, tzinfo=KST) for h in range(24)]
# dim_hours 픽스처의 날씨 값과 같게 두어 날씨 출처 차이가 비교에 끼지 않게 한다.
WEATHER_ROW = {"temp_c": 10.0, "rain_mm": 0.0, "wind_ms": 1.0, "humidity_pct": 50.0, "is_snow": 0.0}
WEATHER = dict.fromkeys(DAY_HOURS, WEATHER_ROW)
COMMIT = "0123456789abcdef0123456789abcdef01234567"
FAKE_DIFF = "diff --git a/src/x.py b/src/x.py\n+changed\n"


# feature_name()만 있는 가짜 모델. verify_generation은 모델에서 특징 이름만 읽는다.
class StubBooster:
    def __init__(self, names):
        self._names = list(names)

    def feature_name(self):
        return list(self._names)


# git 명령을 흉내 내는 실행 함수. status가 비어 있으면 깨끗한 작업 트리다. calls에 불린 명령을
# 모은다.
def fake_git(status: str = "", root: Path | None = None, calls: list | None = None):
    def run(args, cwd):
        if calls is not None:
            calls.append(list(args))
        if args == ["rev-parse", "--show-toplevel"]:
            return f"{(root or cwd).as_posix()}\n"
        if args == ["rev-parse", "HEAD"]:
            return COMMIT + "\n"
        if args == ["status", "--porcelain"]:
            return status
        if args == ["diff", "HEAD"]:
            return FAKE_DIFF
        raise AssertionError(f"예상하지 않은 git 명령: {args}")

    return run


# warehouse의 history 기간으로 산출물을 세대 폴더(gen/artifacts)에 내보내고 다시 읽는다.
def export_generation(db, directory: Path) -> artifacts.Artifacts:
    artifacts.export(db, HISTORY, directory / "artifacts", with_levels=True)
    return artifacts.load(directory / "artifacts")


# 특징 이름만 서비스 모델과 같은, 무작위 자료로 3라운드 학습한 작은 LightGBM(저장·읽기 경로 확인용).
def tiny_booster(names):
    rng = np.random.default_rng(0)
    x = rng.random((200, len(names)), dtype=np.float32)
    y = rng.poisson(2, 200).astype(np.float32)
    data = lgb.Dataset(x, y, feature_name=names)
    return lgb.train({"objective": "poisson", "verbose": -1, "num_leaves": 4}, data, 3)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# scripts/는 패키지가 아니라 import할 수 없으므로 파일 경로로 불러온다
# (tests/test_check_prediction_level.py와 같은 방식).
MANIFEST_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generation_manifest.py"


# importlib로 스크립트 파일을 모듈로 읽는다. if __name__ == '__main__' 아래 실행 부분은 돌지
# 않는다.
def load_manifest_script():
    spec = importlib.util.spec_from_file_location("generation_manifest", MANIFEST_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- 번호 밀림 회귀 ---------------------------------------------------------------------------


# 세대를 만든 warehouse에 ST-1(0번)·ST-2(1번)가 있고, 새 반기에 처음 나온 ST-15(history 끝 뒤에만
# 행)가 들어온 뒤 다시 빌드했다고 하자. 문자열 정렬에서 'ST-1' < 'ST-15' < 'ST-2'라 load_frame의
# 번호는 ST-15가 1, ST-2가 2로 밀리지만, 산출물 경로는 기존 두 곳의 행이 바이트까지 그대로이고
# ST-15만 max_station_code + 1 = 2를 받는다. verify_generation은 세대를 만든 warehouse에서는
# 통과하고(번호가 다른 곳 0), 다시 빌드한 warehouse에서는 달라진 곳 1·새 대여소 1(기존 번호를
# 받음)을 보고하며 strict면 멈춘다.
def test_new_station_inside_the_sort_order_shifts_load_frame_codes_but_not_artifact_codes(
    two_year_warehouse, tmp_path
):
    db = two_year_warehouse
    db.execute("insert into dim_stations values ('ST-2', 2, '마포구', 12, 37.6, 126.9)")
    db.execute("insert into int_station_hour_grid select 'ST-2', hour_start, 1 from dim_hours")
    gen = tmp_path / "gen"
    loaded = export_generation(db, gen)
    assert {s: r["station_code"] for s, r in loaded.stations.items()} == {"ST-1": 0, "ST-2": 1}
    stations = [{"station_id": "ST-1"}, {"station_id": "ST-2"}]
    before, _ = predict.feature_rows(stations, DAY_HOURS, WEATHER, loaded, LEVELED)

    # 세대를 만든 warehouse 그대로면 관문이 통과한다
    report = provenance.verify_generation(gen, StubBooster(LEVELED), con=db, strict=True)
    assert report["ok"] and report["codes"]["moved"] == 0 and report["codes"]["same"] == 2

    # 새 반기를 넣어 다시 빌드한 warehouse: ST-15는 history 끝 뒤에만 행이 있다
    db.execute("insert into dim_stations values ('ST-15', 15, '강남구', 8, 37.55, 127.05)")
    db.execute(
        """insert into int_station_hour_grid
           select 'ST-15', hour_start, 2 from dim_hours where hour_start >= '2025-01-01'"""
    )
    # (1) 산출물 경로: 기존 두 곳은 그대로, 새 대여소는 학습 최대 번호 + 1
    new = {"station_id": "ST-15", "district": "강남구", "docks": 8, "lat": 37.55, "lon": 127.05}
    after, keys = predict.feature_rows([*stations, new], DAY_HOURS, WEATHER, loaded, LEVELED)
    np.testing.assert_array_equal(after[:48], before)
    code = LEVELED.index("station_code")
    assert set(after[48:, code].tolist()) == {float(loaded.max_station_code + 1)} == {2.0}
    assert keys[48] == ("ST-15", DAY_HOURS[0])
    # (2) load_frame 경로(특성 기록): ST-15가 기존 ST-2의 번호 1을 받고 ST-2는 2로 밀린다. 번호별
    # 거치대 수로 어느 대여소가 어느 번호를 받았는지 본다(ST-1 10, ST-15 8, ST-2 12)
    assert artifacts.warehouse_codes(db) == {"ST-1": 0, "ST-15": 1, "ST-2": 2}
    window = frames.Window(("2025-03-12", "2025-03-13"), HISTORY)
    frame = frames.load_frame(db, window, False, with_levels=True)
    docks_by_code = {
        int(c): set(frame["docks"][frame["station_code"] == c].tolist())
        for c in np.unique(frame["station_code"])
    }
    assert docks_by_code == {0: {10.0}, 1: {8.0}, 2: {12.0}}
    # (3) verify_generation: 번호가 달라진 곳 수를 보고하고 strict면 멈춘다
    report = provenance.verify_generation(gen, StubBooster(LEVELED), con=db)
    assert not report["ok"]
    assert report["codes"] == {
        "checked": True,
        "artifact_stations": 2,
        "warehouse_stations": 3,
        "same": 1,
        "moved": 1,
        "new": 1,
        "missing": 0,
        "new_with_reused_code": 1,
        "moved_examples": [["ST-2", 1, 2]],
        "new_examples": ["ST-15"],
        "missing_examples": [],
    }
    assert any("번호" in problem for problem in report["problems"])
    with pytest.raises(provenance.GenerationMismatch, match="번호"):
        provenance.verify_generation(gen, StubBooster(LEVELED), con=db, strict=True)


# 대여소가 warehouse에서 사라져도(dim_stations에서 빠짐) 뒤 번호가 당겨져 밀린다. 사라진 수와 밀린
# 수를 따로 보고한다.
def test_verify_generation_reports_stations_missing_from_the_warehouse(
    two_year_warehouse, tmp_path
):
    db = two_year_warehouse
    db.execute("insert into dim_stations values ('ST-2', 2, '마포구', 12, 37.6, 126.9)")
    db.execute("insert into int_station_hour_grid select 'ST-2', hour_start, 1 from dim_hours")
    gen = tmp_path / "gen"
    export_generation(db, gen)
    db.execute("delete from dim_stations where station_id = 'ST-1'")
    report = provenance.verify_generation(gen, StubBooster(LEVELED), con=db)
    codes = report["codes"]
    assert (codes["missing"], codes["moved"], codes["new"], codes["same"]) == (1, 1, 0, 0)
    assert codes["missing_examples"] == ["ST-1"] and codes["moved_examples"] == [["ST-2", 1, 0]]
    assert not report["ok"]


# ST-2는 2024-06-30까지만 운영돼 격자가 거기서 끝났다(history 끝 2025-01-01 전에 끊김). 뒤에 다시
# 나타나면 다시 빌드한 격자(int_station_hour_grid)는 첫·마지막 활동일 사이를 0건으로 채우므로
# 2024-07~2025-02에 0건 행이 생긴다. 산출물 경로의 특징은 그래도 산출물 값 그대로(profile_mean 2,
# 2024 하반기 수준 결측)이고, load_frame 경로는 0건 행이 섞여 profile_mean이 2보다 작아지고 2024
# 하반기 수준이 0이 된다.
def test_reappearing_station_keeps_artifact_values_on_the_artifact_path(
    two_year_warehouse, tmp_path
):
    db = two_year_warehouse
    db.execute("insert into dim_stations values ('ST-2', 2, '마포구', 12, 37.6, 126.9)")
    db.execute(
        """insert into int_station_hour_grid
           select 'ST-2', hour_start, 2 from dim_hours where hour_start < '2024-07-01'"""
    )
    gen = tmp_path / "gen"
    loaded = export_generation(db, gen)
    hour = DAY_HOURS[8]
    station = [{"station_id": "ST-2"}]
    before = predict.feature_rows(station, [hour], {hour: WEATHER_ROW}, loaded, LEVELED)[0][0]
    row = dict(zip(LEVELED, before.tolist(), strict=True))
    assert row["profile_mean"] == 2.0 and np.isnan(row["station_recent_mean"])

    # 다시 나타남(2025-03부터 시간당 3대) + 다시 빌드한 격자의 0건 채우기(2024-07 ~ 2025-02)
    db.execute(
        """insert into int_station_hour_grid
           select 'ST-2', hour_start, case when hour_start >= '2025-03-01' then 3 else 0 end
           from dim_hours where hour_start >= '2024-07-01'"""
    )
    after = predict.feature_rows(station, [hour], {hour: WEATHER_ROW}, loaded, LEVELED)[0][0]
    np.testing.assert_array_equal(after, before)
    window = frames.Window(("2025-03-12", "2025-03-13"), HISTORY)
    frame = frames.load_frame(db, window, False, with_levels=True)
    st2_8h = (frame["station_code"] == 1) & (frame["hour_of_day"] == 8)
    assert st2_8h.sum() == 1
    assert 0.0 < float(frame["profile_mean"][st2_8h][0]) < 2.0
    assert float(frame["station_recent_mean"][st2_8h][0]) == 0.0


# warehouse_codes는 frames.history_ctes의 codes CTE와 같은 규칙(대여소ID 문자열 정렬 순위,
# 0부터)이다. load_frame이 준 번호와 거치대 수의 짝으로 같은 번호 규칙인지 확인한다.
def test_warehouse_codes_follow_the_load_frame_numbering(small_warehouse):
    assert artifacts.warehouse_codes(small_warehouse) == {"ST-1": 0, "ST-2": 1}
    window = frames.Window(("2024-01-16", "2024-01-18"), ("2024-01-01", "2024-01-16"))
    frame = frames.load_frame(small_warehouse, window, False)
    assert set(frame["docks"][frame["station_code"] == 0].tolist()) == {10.0}
    assert set(frame["docks"][frame["station_code"] == 1].tolist()) == {12.0}


# --- 특징 순서·파일 지문 -------------------------------------------------------------------


# 게시된 세대 v3-M3p-20260917T0904처럼 meta.json features가 수준 특징 3개를 빠뜨린 20개이고 모델은
# 23개인 경우: provenance가 없는 옛 세대면 경고(ok 유지, strict도 멈추지 않음), serving.json에
# provenance가 있는 세대면 오류(strict에서 멈춤). provenance에 적힌 feature_names가 모델과 다른 것도
# 오류다.
def test_feature_order_mismatch_is_a_warning_for_old_generations_and_an_error_with_provenance(
    two_year_warehouse, tmp_path
):
    gen = tmp_path / "gen"
    export_generation(two_year_warehouse, gen)
    meta_path = gen / "artifacts" / "meta.json"
    meta = json.loads(meta_path.read_text("utf-8"))
    assert meta["features"] == LEVELED
    meta["features"] = frames.FEATURES  # 옛 내보내기 코드의 기록(수준 특징 3개 누락)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")

    report = provenance.verify_generation(gen, StubBooster(LEVELED))
    assert report["ok"] and not report["has_provenance"]
    assert report["features"] == {"model": LEVELED, "meta": frames.FEATURES, "match": False}
    assert report["problems"] == [] and any("특징" in w for w in report["warnings"])
    assert report["codes"] == {"checked": False} and report["files"] == {"checked": False}
    assert provenance.verify_generation(gen, StubBooster(LEVELED), strict=True)["ok"]

    (gen / "serving.json").write_text(
        json.dumps({"provenance": {"feature_names": LEVELED}}), "utf-8"
    )
    report = provenance.verify_generation(gen, StubBooster(LEVELED))
    assert report["has_provenance"] and not report["ok"]
    assert any("특징" in problem for problem in report["problems"])
    with pytest.raises(provenance.GenerationMismatch, match="특징"):
        provenance.verify_generation(gen, StubBooster(LEVELED), strict=True)

    # 기록된 feature_names와 모델이 다른 것도 오류
    meta["features"] = LEVELED
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    (gen / "serving.json").write_text(
        json.dumps({"provenance": {"feature_names": frames.FEATURES}}), "utf-8"
    )
    report = provenance.verify_generation(gen, StubBooster(LEVELED))
    assert report["features"]["match"] and not report["ok"]
    assert any("feature_names" in problem for problem in report["problems"])


# 파일 지문: 넘긴 기대값 또는 serving.json의 provenance.files와 대조한다. 지문을 담는 serving.json
# 자신은 대상이 아니다. 바뀐 파일과 빠진 파일을 이름으로 보고하고 strict면 멈춘다. 기록에 없는 새
# 파일은 경고만 한다.
def test_file_digests_are_compared_with_the_record(two_year_warehouse, tmp_path):
    gen = tmp_path / "gen"
    export_generation(two_year_warehouse, gen)
    (gen / "model.txt").write_text("fake", "utf-8")
    expected = provenance.file_digests(gen)
    assert set(expected) == {
        "model.txt",
        "artifacts/halves.parquet",
        "artifacts/meta.json",
        "artifacts/profile.parquet",
        "artifacts/stations.parquet",
        "artifacts/system_halves.parquet",
        "artifacts/trend.parquet",
    }
    assert expected["model.txt"] == sha256(b"fake")
    assert provenance.sha256_file(gen / "model.txt") == sha256(b"fake")

    report = provenance.verify_generation(gen, StubBooster(LEVELED), expected=expected)
    assert report["ok"]
    assert report["files"] == {
        "checked": True,
        "compared": 7,
        "mismatched": [],
        "missing": [],
        "unexpected": [],
    }
    # 기록이 serving.json에 있으면 expected 없이도 대조한다. serving.json 자신은 지문 대상이 아니다
    (gen / "serving.json").write_text(
        json.dumps({"provenance": {"feature_names": LEVELED, "files": expected}}), "utf-8"
    )
    assert "serving.json" not in provenance.file_digests(gen, exclude=("serving.json",))
    assert "serving.json" in provenance.file_digests(gen)
    assert provenance.verify_generation(gen, StubBooster(LEVELED))["files"]["compared"] == 7

    # 파일이 바뀌거나 빠지면 이름으로 보고하고 strict면 멈춘다. 기록에 없는 파일은 경고
    meta_path = gen / "artifacts" / "meta.json"
    meta_path.write_text(json.dumps(json.loads(meta_path.read_text("utf-8"))), "utf-8")
    (gen / "model.txt").unlink()
    (gen / "extra.txt").write_text("x", "utf-8")
    report = provenance.verify_generation(gen, StubBooster(LEVELED))
    assert not report["ok"]
    assert report["files"]["mismatched"] == ["artifacts/meta.json"]
    assert report["files"]["missing"] == ["model.txt"]
    assert report["files"]["unexpected"] == ["extra.txt"]
    assert any("extra.txt" in warning for warning in report["warnings"])
    with pytest.raises(provenance.GenerationMismatch, match="파일"):
        provenance.verify_generation(gen, StubBooster(LEVELED), strict=True)


# booster를 넘기지 않으면 세대 폴더의 model.txt를 읽는다(세대 폴더만으로 검증할 수 있게).
def test_verify_generation_loads_the_saved_model_when_no_booster_is_given(
    two_year_warehouse, tmp_path
):
    gen = tmp_path / "gen"
    export_generation(two_year_warehouse, gen)
    tiny_booster(LEVELED).save_model(str(gen / "model.txt"))
    report = provenance.verify_generation(gen)
    assert report["features"]["model"] == LEVELED and report["features"]["match"]
    assert report["ok"]


# --- git 상태·warehouse 지문·수집 ---------------------------------------------------------


# 깨끗한 작업 트리: 커밋과 저장소 루트를 적고 diff는 부르지 않는다. 변경이 있으면 변경 파일 목록과
# `git diff HEAD`의 SHA-256을 남긴다(어느 변경으로 학습했는지 두 실행을 구분하는 지문).
def test_git_state_records_commit_and_changes(tmp_path):
    calls: list = []
    clean = provenance.git_state(tmp_path, run=fake_git(root=Path("D:/repo"), calls=calls))
    assert clean == {
        "available": True,
        "root": "D:/repo",
        "commit": COMMIT,
        "dirty": False,
        "changed_files": [],
    }
    assert ["diff", "HEAD"] not in calls
    dirty = provenance.git_state(tmp_path, run=fake_git(" M a.py\n?? b.py\n", root=Path("D:/repo")))
    assert dirty["dirty"] and dirty["changed_files"] == [" M a.py", "?? b.py"]
    assert dirty["diff_sha256"] == sha256(FAKE_DIFF.encode("utf-8"))


# git이 없거나(컨테이너) 실패하면 값 대신 이유를 적고 예외를 내지 않는다.
def test_git_state_without_git_records_the_reason(tmp_path):
    def no_git(args, cwd):
        raise FileNotFoundError("git")

    assert provenance.git_state(tmp_path, run=no_git) == {
        "available": False,
        "reason": "FileNotFoundError: git",
    }


# warehouse 지문: 학습 기간 격자의 행 수·대여 합과 대여소 표(station_id 순)의 SHA-256. 2023~2024 두
# 해(윤년 포함 731일) × 24시간 = 17,544행, 반기마다 시간당 1·2·3·4대라 대여 합은
# 181·24·1 + 184·24·2 + 182·24·3 + 184·24·4 = 43,944. 대여소가 늘면 대여소 표 지문만 바뀐다.
def test_warehouse_fingerprint_counts_the_history_grid_and_hashes_the_stations(two_year_warehouse):
    first = provenance.warehouse_fingerprint(two_year_warehouse, HISTORY)
    assert first["history"] == list(HISTORY)
    assert (first["grid_rows"], first["grid_rentals"], first["dim_stations_rows"]) == (
        17544,
        43944,
        1,
    )
    assert len(first["dim_stations_sha256"]) == 64
    assert provenance.warehouse_fingerprint(two_year_warehouse, HISTORY) == first
    two_year_warehouse.execute(
        "insert into dim_stations values ('ST-2', 2, '마포구', 12, 37.6, 126.9)"
    )
    second = provenance.warehouse_fingerprint(two_year_warehouse, HISTORY)
    assert second["dim_stations_rows"] == 2
    assert second["dim_stations_sha256"] != first["dim_stations_sha256"]
    assert (second["grid_rows"], second["grid_rentals"]) == (17544, 43944)


# collect는 git 상태·저장소 파일 지문(uv.lock, 공휴일 시드)·라이브러리 버전·학습 설정·특징 순서·
# warehouse 지문·세대 파일 지문을 한 사전에 모은다. 저장소 파일이 없으면(컨테이너에 uv.lock이 없는
# 경우 등) None으로 둔다. 지문 대상에서 serving.json은 뺀다(기록을 담는 파일이 자기 지문을 가질 수
# 없음).
def test_collect_gathers_versions_files_and_warehouse_fingerprint(two_year_warehouse, tmp_path):
    gen = tmp_path / "gen"
    export_generation(two_year_warehouse, gen)
    (gen / "model.txt").write_text("fake", "utf-8")
    (gen / "serving.json").write_text("{}", "utf-8")
    root = tmp_path / "repo"
    (root / "dbt" / "seeds").mkdir(parents=True)
    (root / "uv.lock").write_bytes(b"lock\n")  # write_text는 Windows에서 CRLF로 바꿔 쓴다
    git = provenance.git_state(root, run=fake_git(root=root))
    warehouse = provenance.warehouse_fingerprint(two_year_warehouse, HISTORY)
    params = {"objective": "poisson", "seed": 42}

    record = provenance.collect(gen, StubBooster(LEVELED), git, warehouse, params)
    assert record["git"] == git and record["warehouse"] == warehouse
    assert record["params"] == params and record["params"] is not params
    assert record["feature_names"] == LEVELED
    assert record["repo_root"] == root.as_posix()
    assert record["repo_files"] == {"uv.lock": sha256(b"lock\n"), "dbt/seeds/kr_holidays.csv": None}
    assert set(record["versions"]) == {
        "python",
        "lightgbm",
        "duckdb",
        "numpy",
        "pyarrow",
        "holidays",
    }
    assert record["versions"]["lightgbm"] == lgb.__version__
    assert set(record["files"]) == set(provenance.file_digests(gen, exclude=("serving.json",)))
    assert "serving.json" not in record["files"] and record["files"]["model.txt"] == sha256(b"fake")
    datetime.fromisoformat(record["recorded_at"])  # KST ISO 시각
    json.dumps(record)  # 그대로 JSON으로 쓸 수 있다


# --- scripts/generation_manifest.py: 세대 폴더 밖에 남기는 읽기 전용 지문 ---------------


# 세대 폴더의 모든 파일(serving.json 포함)마다 SHA-256과 바이트 수를 적고, model.txt가 있으면
# verify_generation 보고(warehouse 대조 없음)를 붙이고, 메모는 그대로 담는다. 세대 폴더 안에는
# 아무것도 쓰지 않아 지문이 그대로다. model.txt가 없으면 검사 없이 지문만 남기고, artifacts/가
# 없는 폴더는 세대 폴더가 아니라 멈춘다.
def test_manifest_lists_every_file_and_verifies_the_saved_model(two_year_warehouse, tmp_path):
    script = load_manifest_script()
    gen = tmp_path / "gen"
    export_generation(two_year_warehouse, gen)
    tiny_booster(LEVELED).save_model(str(gen / "model.txt"))
    (gen / "serving.json").write_text("{}", "utf-8")
    before = provenance.file_digests(gen)
    assert "serving.json" in before and len(before) == 8

    result = script.manifest(gen, ["메모 하나", "메모 둘"])
    assert result["generation"] == "gen" and result["directory"] == gen.resolve().as_posix()
    assert set(result["files"]) == set(before)
    for name, info in result["files"].items():
        assert info == {"sha256": before[name], "bytes": (gen / name).stat().st_size}
    check = result["verification"]
    assert check["ok"] and check["features"]["match"] and not check["has_provenance"]
    assert check["codes"] == {"checked": False} and check["files"] == {"checked": False}
    assert result["notes"] == ["메모 하나", "메모 둘"]
    datetime.fromisoformat(result["recorded_at"])
    json.dumps(result, ensure_ascii=False)  # 그대로 JSON으로 쓸 수 있다
    assert provenance.file_digests(gen) == before  # 세대 폴더에는 아무것도 쓰지 않았다

    (gen / "model.txt").unlink()
    assert script.manifest(gen)["verification"] is None
    with pytest.raises(SystemExit, match="세대 폴더가 아님"):
        script.manifest(tmp_path / "not-a-generation")


# 결과 파일은 세대 폴더 밖에만 쓴다: 폴더 바로 아래든 artifacts/ 아래든 안이면 거부하고, 이름이
# 같은 글자로 시작하는 다른 폴더(gen2)나 형제 폴더(provenance/)는 밖이라 그대로 돌려준다.
def test_manifest_output_inside_the_generation_folder_is_refused(tmp_path):
    script = load_manifest_script()
    gen = tmp_path / "generations" / "gen"
    (gen / "artifacts").mkdir(parents=True)
    for inside in (gen / "manifest.json", gen / "artifacts" / "manifest.json"):
        with pytest.raises(SystemExit, match="세대 폴더 안"):
            script.checked_out_path(inside, gen)
    for outside in (
        tmp_path / "generations" / "gen2" / "manifest.json",
        tmp_path / "provenance" / "gen.json",
    ):
        assert script.checked_out_path(outside, gen) == outside


# 메모 파일은 한 줄에 메모 하나이고 빈 줄과 앞뒤 공백은 뺀다(Windows 콘솔에서 한글 인자가
# 깨지는 일을 피해 파일로도 받는다).
def test_read_notes_keeps_one_note_per_line_and_drops_blank_lines(tmp_path):
    script = load_manifest_script()
    path = tmp_path / "notes.txt"
    path.write_bytes("첫 메모\n\n  둘째 메모  \n\n".encode())
    assert script.read_notes(path) == ["첫 메모", "둘째 메모"]
