import datetime as dt

import openpyxl
import pyarrow.parquet as pq
import pytest

from bike_demand.ingest import stations

HEADER = [
    [*stations.EXPECTED_HEADER, None, None, None, "설치\n시기", "설치형태"],
    [None, None, None, None, None, None, None, "LCD", "QR"],
    [None, None, "자치구", "상세주소", "위도", "경도"],
    [None, None, None, None, None, None, None, "거치\n대수", "거치\n대수"],
    [],
]


def make_xlsx(path, rows, header=HEADER):
    workbook = openpyxl.Workbook()
    sheet = workbook.create_sheet("대여소현황", 0)
    del workbook[workbook.sheetnames[1]]
    for row in header + rows:
        sheet.append(row)
    workbook.save(path)


def test_snapshot_of():
    assert stations.snapshot_of("공공자전거 대여소 정보(24.6월 기준).xlsx") == "2024-06"
    assert stations.snapshot_of("공공자전거 대여소 정보(22.12월 기준).xlsx") == "2022-12"
    with pytest.raises(ValueError):
        stations.snapshot_of("대여소.xlsx")


def test_to_parquet_reads_rows_as_text(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    row = [301, " 경복궁역 7번출구 앞", "종로구", "주소", 37.5757, 126.9714,
           dt.datetime(2015, 10, 7, 12, 3, 46), None, 20, "QR", None]  # fmt: skip
    make_xlsx(raw / "공공자전거 대여소 정보(26.6월 기준).xlsx", [row, [None] * 10])
    make_xlsx(raw / "공공자전거 대여소 정보(25.12월 기준).xlsx", [row, row])

    out = tmp_path / "stations.parquet"
    assert stations.to_parquet(raw, out) == {"2025-12": 2, "2026-06": 1}
    first = [r for r in pq.read_table(out).to_pylist() if r["snapshot"] == "2026-06"][0]
    assert first["station_no"] == "301"
    assert first["station_name"] == "경복궁역 7번출구 앞"  # 앞 공백 제거
    assert first["installed_at"] == "2015-10-07 12:03:46"
    assert first["docks_lcd"] is None and first["docks_qr"] == "20"


def test_unexpected_header_fails(tmp_path):
    bad = [["번호", "이름", "위치"]] + HEADER[1:]
    path = tmp_path / "공공자전거 대여소 정보(26.6월 기준).xlsx"
    make_xlsx(path, [], header=bad)
    with pytest.raises(ValueError, match="머리글"):
        stations.read_workbook(path)
