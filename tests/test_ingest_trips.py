# [무엇을 검증하나] 대여이력 원본(월별 CSV, 연도별 ZIP 안의 월별 CSV)을 월 단위 문자열 Parquet로
# 옮기는 ingest/trips.py를 확인한다: 파일 이름에서 연월 읽기, CSV와 ZIP 안 파일을 함께 찾고 같은
# 달이 두 번이면 거부, cp949·'\N' 빈 값 처리, 2026년 파일에 없는 '자전거구분' 열, 같은 달을 다시
# 변환하면 덧붙이지 않고 바꾸기, 헤더가 바뀌면 실패.
# [왜 중요한가] 1억 4천만 건 전체가 이 변환을 거친다(README). 같은 달을 다시 넣어도 중복되지 않아야
# 하고(README 1단계), 대여소번호 앞자리 0처럼 값을 해석하면 잃는 정보가 없어야 dbt staging이 정확히
# 정제할 수 있다.
# [방식] 실제 원본과 같은 헤더·행 모양의 작은 CSV를 cp949로 쓰고, ZIP 안 파일 이름도 실제처럼 UTF-8
# 플래그 없는 cp949 바이트로 만든다.

import zipfile
from pathlib import Path

# 변환 결과 Parquet을 읽어 스키마와 값을 확인한다.
import pyarrow.parquet as pq
import pytest

from bike_demand.ingest import trips

# 2023~2025년 파일의 17열 헤더와 행. 성별 칸의 \N은 원본의 빈 값 표시이고, 대여소번호 '03812'의
# 앞자리 0이 남는지 확인하는 데 쓴다.
HEADER_2025 = (
    '"자전거번호","대여일시","대여 대여소번호","대여 대여소명","대여거치대","반납일시",'
    '"반납대여소번호","반납대여소명","반납거치대","이용시간(분)","이용거리(M)","생년","성별",'
    '"이용자종류","대여대여소ID","반납대여소ID","자전거구분"'
)
ROW_2025 = (
    '"SPB-66002","2025-07-01 00:00:38","03812","KT관악지점","0","2025-07-01 00:02:46",'
    '"02169","봉천역 2번출구","0","2","500.00","2000",\\N,"내국인","ST-2842","ST-1264","일반자전거"'
)
# 2026년 파일은 마지막 '자전거구분' 열이 없는 16열이다(docs/data.md 파일 형식).
HEADER_2026 = HEADER_2025.rsplit(",", 1)[0]
ROW_2026 = (
    '"SPB-65030","2026-01-01 00:00:13","01729","도봉한신아파트 버스정류장","0",'
    '"2026-01-01 00:02:00","01729","도봉한신아파트 버스정류장","0","1","0.00","1990","M",'
    '"내국인","ST-1","ST-1"'
)


# 원본과 같은 cp949 인코딩으로 CSV를 쓴다.
def write_cp949(path: Path, *lines: str) -> None:
    path.write_bytes(("\n".join(lines) + "\n").encode("cp949"))


# 2026-01 CSV 하나(1행)와, 2025-07 CSV(2행)를 담은 2025년 ZIP 하나로 원본 폴더를 만든다.
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


# '_2601.csv' → 2026-01. 13월 같은 잘못된 월이나 연월이 없는 이름은 오류.
def test_month_of():
    assert trips.month_of("서울특별시 공공자전거 대여이력 정보_2601.csv") == "2026-01"
    with pytest.raises(ValueError):
        trips.month_of("대여이력_2613.csv")
    with pytest.raises(ValueError):
        trips.month_of("대여이력.csv")


# ZIP 안 파일과 CSV를 함께 찾아 달 순서로 돌려주고, ZIP 안 파일은 'ZIP이름!안쪽이름'으로 표시하며
# 깨진 cp949 이름을 한글로 되살리는지 본다.
def test_discover_reads_csv_and_zip_members(tmp_path):
    raw = tmp_path / "raw"
    make_raw(raw)
    sources = trips.discover_sources(raw)
    assert [s.month for s in sources] == ["2025-07", "2026-01"]
    assert sources[0].name.endswith("!서울특별시 공공자전거 대여이력 정보_2507.csv")


# 같은 달(2026-01) 원본이 두 개면 어느 쪽을 쓸지 정하지 않고 그 달을 적은 오류로 멈춘다. 둘 다
# 변환하면 그 달 대여가 두 배가 된다.
def test_duplicate_month_is_rejected(tmp_path):
    raw = tmp_path / "raw"
    make_raw(raw)
    write_cp949(raw / "다른이름_2601.csv", HEADER_2026, ROW_2026)
    with pytest.raises(ValueError, match="2026-01"):
        trips.discover_sources(raw)


# 변환 결과: 달별 행 수, 스키마가 trips.SCHEMA(모든 열 문자열 + source_file)와 같음, 대여소번호
# 앞자리 0 유지, '\N' → null, 원본 파일 이름 기록. 2026년 파일은 없는 '자전거구분' 열이 null로
# 채워진다.
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


# 같은 달을 두 번 변환해도 그 달 폴더에는 trips.parquet 하나, 1행만 있다(임시 파일이 남거나 행이
# 덧붙지 않음). 반기 갱신 DAG를 다시 돌려도 안전해야 하기 때문이다.
def test_reconvert_replaces_instead_of_appending(tmp_path):
    raw, bronze = tmp_path / "raw", tmp_path / "bronze"
    make_raw(raw)
    list(trips.iter_convert(raw, bronze, {"2026-01"}))
    list(trips.iter_convert(raw, bronze, {"2026-01"}))
    files = list((bronze / "month=2026-01").iterdir())
    assert [f.name for f in files] == ["trips.parquet"]
    assert pq.read_table(files[0]).num_rows == 1


# 헤더의 열 이름이 하나라도 다르면('성별' → '성') 변환을 멈춘다. 공개 양식이 바뀌었을 때 열을 잘못
# 짝지어 저장하지 않게 한다.
def test_unexpected_header_fails(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_cp949(raw / "x_2601.csv", HEADER_2026.replace("성별", "성"), ROW_2026)
    (source,) = trips.discover_sources(raw)
    with pytest.raises(ValueError, match="예상과 다른 헤더"):
        trips.read_source(source)
