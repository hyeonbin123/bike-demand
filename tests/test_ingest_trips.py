import zipfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from bike_demand.ingest import trips

HEADER_2025 = (
    '"자전거번호","대여일시","대여 대여소번호","대여 대여소명","대여거치대","반납일시",'
    '"반납대여소번호","반납대여소명","반납거치대","이용시간(분)","이용거리(M)","생년","성별",'
    '"이용자종류","대여대여소ID","반납대여소ID","자전거구분"'
)
ROW_2025 = (
    '"SPB-66002","2025-07-01 00:00:38","03812","KT관악지점","0","2025-07-01 00:02:46",'
    '"02169","봉천역 2번출구","0","2","500.00","2000",\\N,"내국인","ST-2842","ST-1264","일반자전거"'
)
HEADER_2026 = HEADER_2025.rsplit(",", 1)[0]
ROW_2026 = (
    '"SPB-65030","2026-01-01 00:00:13","01729","도봉한신아파트 버스정류장","0",'
    '"2026-01-01 00:02:00","01729","도봉한신아파트 버스정류장","0","1","0.00","1990","M",'
    '"내국인","ST-1","ST-1"'
)


def write_cp949(path: Path, *lines: str) -> None:
    path.write_bytes(("\n".join(lines) + "\n").encode("cp949"))


def make_raw(raw: Path) -> None:
    raw.mkdir()
    write_cp949(raw / "서울특별시 공공자전거 대여이력 정보_2601.csv", HEADER_2026, ROW_2026)
    member = "서울특별시 공공자전거 대여이력 정보_2507.csv"
    csv_bytes = ("\n".join([HEADER_2025, ROW_2025, ROW_2025]) + "\n").encode("cp949")
    zip_path = raw / "서울특별시 공공자전거 대여이력 정보_2025.zip"
    # 실제 원본처럼 UTF-8 플래그 없이 cp949 바이트로 이름을 적는다. zipfile은 ASCII가 아닌
    # 이름에 UTF-8 플래그를 붙이므로, 같은 길이의 ASCII 이름으로 쓴 뒤 바이트를 바꿔 넣는다.
    name_bytes = member.encode("cp949")
    placeholder = b"x" * len(name_bytes)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(placeholder.decode("ascii"), csv_bytes)
    zip_path.write_bytes(zip_path.read_bytes().replace(placeholder, name_bytes))


def test_month_of():
    assert trips.month_of("서울특별시 공공자전거 대여이력 정보_2601.csv") == "2026-01"
    with pytest.raises(ValueError):
        trips.month_of("대여이력_2613.csv")
    with pytest.raises(ValueError):
        trips.month_of("대여이력.csv")


def test_discover_reads_csv_and_zip_members(tmp_path):
    raw = tmp_path / "raw"
    make_raw(raw)
    sources = trips.discover_sources(raw)
    assert [s.month for s in sources] == ["2025-07", "2026-01"]
    assert sources[0].name.endswith("!서울특별시 공공자전거 대여이력 정보_2507.csv")


def test_duplicate_month_is_rejected(tmp_path):
    raw = tmp_path / "raw"
    make_raw(raw)
    write_cp949(raw / "다른이름_2601.csv", HEADER_2026, ROW_2026)
    with pytest.raises(ValueError, match="2026-01"):
        trips.discover_sources(raw)


def test_convert_normalizes_columns_and_nulls(tmp_path):
    raw, bronze = tmp_path / "raw", tmp_path / "bronze"
    make_raw(raw)
    results = {s.month: rows for s, _, rows in trips.iter_convert(raw, bronze)}
    assert results == {"2025-07": 2, "2026-01": 1}

    t2025 = pq.read_table(trips.output_path(bronze, "2025-07"))
    assert t2025.schema == trips.SCHEMA
    row = t2025.to_pylist()[0]
    assert row["rent_station_no"] == "03812"  # 앞의 0 유지
    assert row["gender"] is None  # \N -> null
    assert row["bike_type"] == "일반자전거"
    assert row["source_file"].endswith("_2507.csv")

    row2026 = pq.read_table(trips.output_path(bronze, "2026-01")).to_pylist()[0]
    assert row2026["bike_type"] is None  # 2026년 파일에는 컬럼이 없음
    assert row2026["rent_station_name"] == "도봉한신아파트 버스정류장"


def test_reconvert_replaces_instead_of_appending(tmp_path):
    raw, bronze = tmp_path / "raw", tmp_path / "bronze"
    make_raw(raw)
    list(trips.iter_convert(raw, bronze, {"2026-01"}))
    list(trips.iter_convert(raw, bronze, {"2026-01"}))
    files = list((bronze / "month=2026-01").iterdir())
    assert [f.name for f in files] == ["trips.parquet"]
    assert pq.read_table(files[0]).num_rows == 1


def test_unexpected_header_fails(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_cp949(raw / "x_2601.csv", HEADER_2026.replace("성별", "성"), ROW_2026)
    (source,) = trips.discover_sources(raw)
    with pytest.raises(ValueError, match="예상과 다른 헤더"):
        trips.read_source(source)
