"""따릉이 대여소 정보(반기별 xlsx)를 Parquet 하나로 모은다.

파일마다 머리글이 5줄이고 앞 10개 컬럼의 위치가 같다. 값은 문자열로 두고
(해석은 dbt에서), 어느 시점 파일인지를 ``snapshot``(YYYY-MM)으로 붙인다.
대여소번호는 xlsx에서 숫자(301)지만 대여이력에서는 앞자리 0이 붙은 문자열("00301")이다.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path

import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq

HEADER_ROWS = 5
COLUMNS = [
    "station_no",
    "station_name",
    "district",
    "address",
    "lat",
    "lon",
    "installed_at",
    "docks_lcd",
    "docks_qr",
    "operation_type",
]
SCHEMA = pa.schema(
    [(name, pa.string()) for name in COLUMNS]
    + [("snapshot", pa.string()), ("source_file", pa.string())]
)
_SNAPSHOT_RE = re.compile(r"\((\d{2})\.(\d{1,2})월 기준\)")
# 머리글 첫 줄 앞 세 칸. 양식이 바뀌면 조용히 잘못 읽지 않도록 확인한다.
EXPECTED_HEADER = ("대여소\n번호", "보관소(대여소)명", "소재지(위치)")


def snapshot_of(filename: str) -> str:
    match = _SNAPSHOT_RE.search(filename)
    if not match:
        raise ValueError(f"파일 이름에서 기준 시점(YY.M월 기준)을 찾지 못함: {filename}")
    yy, mm = match.groups()
    return f"20{yy}-{int(mm):02d}"


def _text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.isoformat(sep=" ")
    text = str(value).strip()
    return text or None


def read_workbook(path: Path) -> list[dict]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook[workbook.sheetnames[0]]
        rows = sheet.iter_rows(values_only=True)
        header = next(rows)
        if tuple(header[:3]) != EXPECTED_HEADER:
            raise ValueError(f"{path.name}: 예상과 다른 머리글 {header[:3]}")
        for _ in range(HEADER_ROWS - 1):
            next(rows)
        snapshot = snapshot_of(path.name)
        records = []
        for row in rows:
            values = [_text(v) for v in row[: len(COLUMNS)]]
            if values[0] is None:  # 빈 줄
                continue
            record = dict(zip(COLUMNS, values, strict=True))
            record.update(snapshot=snapshot, source_file=path.name)
            records.append(record)
        return records
    finally:
        workbook.close()


def to_parquet(raw_dir: Path, out_path: Path) -> dict[str, int]:
    """모든 xlsx를 읽어 Parquet 하나로 다시 만들고 시점별 행 수를 돌려준다."""
    records: list[dict] = []
    counts: dict[str, int] = {}
    for path in sorted(raw_dir.glob("*.xlsx")):
        rows = read_workbook(path)
        snapshot = snapshot_of(path.name)
        if snapshot in counts:
            raise ValueError(f"{snapshot} 시점 파일이 두 개")
        counts[snapshot] = len(rows)
        records.extend(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(records, schema=SCHEMA), tmp, compression="zstd")
    os.replace(tmp, out_path)
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="대여소 정보 xlsx를 Parquet로 모은다")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/stations"))
    parser.add_argument("--out", type=Path, default=Path("data/bronze/stations/stations.parquet"))
    args = parser.parse_args()
    for snapshot, rows in to_parquet(args.raw, args.out).items():
        print(snapshot, rows)
