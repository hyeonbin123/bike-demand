"""따릉이 대여이력 원본(월별 CSV, 연도별 ZIP 안의 월별 CSV)을 월 단위 Parquet로 옮긴다.

원본은 cp949, 빈 값은 ``\\N``이고, 2023~2025년 파일에만 ``자전거구분`` 컬럼이 있다.
이 단계는 값을 해석하지 않는다: 모든 컬럼을 문자열로 두고 컬럼 이름만 영어로 바꾼다.
타입 변환과 이상값 처리는 다음 단계(dbt staging)에서 한다.

같은 달을 다시 변환하면 그 달의 파일을 통째로 바꾸므로 행이 중복되지 않는다.
"""

from __future__ import annotations

import os
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import pyarrow as pa
import pyarrow.compute as pac
import pyarrow.csv as pc
import pyarrow.parquet as pq

SOURCE_ENCODING = "cp949"
NULL_TOKEN = "\\N"

# 원본 헤더 -> 저장할 컬럼 이름. 순서가 곧 Parquet 컬럼 순서다.
COLUMNS: dict[str, str] = {
    "자전거번호": "bike_no",
    "대여일시": "rented_at",
    "대여 대여소번호": "rent_station_no",
    "대여 대여소명": "rent_station_name",
    "대여거치대": "rent_dock",
    "반납일시": "returned_at",
    "반납대여소번호": "return_station_no",
    "반납대여소명": "return_station_name",
    "반납거치대": "return_dock",
    "이용시간(분)": "duration_min",
    "이용거리(M)": "distance_m",
    "생년": "birth_year",
    "성별": "gender",
    "이용자종류": "user_type",
    "대여대여소ID": "rent_station_id",
    "반납대여소ID": "return_station_id",
    "자전거구분": "bike_type",
}
OPTIONAL_COLUMNS = {"자전거구분"}

SCHEMA = pa.schema(
    [(name, pa.string()) for name in COLUMNS.values()] + [("source_file", pa.string())]
)

_MONTH_RE = re.compile(r"_(\d{2})(\d{2})\.csv$")


@dataclass(frozen=True)
class TripSource:
    """월 하나의 원본 CSV. ``member``가 있으면 ``path`` ZIP 안의 파일이다."""

    month: str  # "YYYY-MM"
    path: Path
    member: str | None = None
    member_label: str | None = None  # 깨진 한글을 되살린 ZIP 안 파일 이름

    @property
    def name(self) -> str:
        if self.member is None:
            return self.path.name
        return f"{self.path.name}!{self.member_label or self.member}"

    def open(self) -> IO[bytes]:
        if self.member is None:
            return open(self.path, "rb")
        archive = zipfile.ZipFile(self.path)
        stream = archive.open(self.member)
        original_close = stream.close

        def close() -> None:
            original_close()
            archive.close()

        stream.close = close  # type: ignore[method-assign]
        return stream


def _member_name(info: zipfile.ZipInfo) -> str:
    """ZIP이 UTF-8 플래그 없이 cp949로 이름을 적었으면 한글이 깨지므로 되살린다."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode(SOURCE_ENCODING)
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def month_of(filename: str) -> str:
    match = _MONTH_RE.search(filename)
    if not match:
        raise ValueError(f"파일 이름에서 연월(_YYMM.csv)을 찾지 못함: {filename}")
    yy, mm = match.groups()
    if not 1 <= int(mm) <= 12:
        raise ValueError(f"잘못된 월: {filename}")
    return f"20{yy}-{mm}"


def discover_sources(raw_dir: Path) -> list[TripSource]:
    """``raw_dir``의 CSV와 ZIP에서 월별 원본을 찾는다. 같은 달이 두 번 나오면 오류."""
    sources: list[TripSource] = []
    for path in sorted(raw_dir.iterdir()):
        if path.suffix.lower() == ".csv":
            sources.append(TripSource(month_of(path.name), path))
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    if info.is_dir() or not info.filename.lower().endswith(".csv"):
                        continue
                    label = _member_name(info)
                    sources.append(TripSource(month_of(label), path, info.filename, label))

    seen: dict[str, TripSource] = {}
    for source in sources:
        if source.month in seen:
            raise ValueError(
                f"{source.month}월 원본이 두 개: {seen[source.month].name}, {source.name}"
            )
        seen[source.month] = source
    return sorted(sources, key=lambda s: s.month)


def read_source(source: TripSource) -> pa.Table:
    with source.open() as stream:
        reader = pc.open_csv(
            stream,
            read_options=pc.ReadOptions(encoding=SOURCE_ENCODING, block_size=64 << 20),
            convert_options=pc.ConvertOptions(
                null_values=[NULL_TOKEN, ""],
                strings_can_be_null=True,
                quoted_strings_can_be_null=True,
                column_types={header: pa.string() for header in COLUMNS},
            ),
        )
        table = reader.read_all()

    headers = table.column_names
    unknown = [h for h in headers if h not in COLUMNS]
    missing = [h for h in COLUMNS if h not in headers and h not in OPTIONAL_COLUMNS]
    if unknown or missing:
        raise ValueError(f"{source.name}: 예상과 다른 헤더 (모름={unknown}, 없음={missing})")

    arrays = [
        table.column(header) if header in headers else pa.nulls(table.num_rows, pa.string())
        for header in COLUMNS
    ]
    arrays.append(pac.fill_null(pa.nulls(table.num_rows, pa.string()), source.name))
    return pa.Table.from_arrays(arrays, schema=SCHEMA)


def output_path(bronze_dir: Path, month: str) -> Path:
    return bronze_dir / f"month={month}" / "trips.parquet"


def convert(source: TripSource, bronze_dir: Path) -> tuple[Path, int]:
    """월 하나를 변환해 쓰고 (경로, 행 수)를 돌려준다. 임시 파일에 쓴 뒤 바꿔치기한다."""
    table = read_source(source)
    target = output_path(bronze_dir, source.month)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, target)
    return target, table.num_rows


def iter_convert(
    raw_dir: Path, bronze_dir: Path, months: set[str] | None = None
) -> Iterator[tuple[TripSource, Path, int]]:
    for source in discover_sources(raw_dir):
        if months is not None and source.month not in months:
            continue
        path, rows = convert(source, bronze_dir)
        yield source, path, rows


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="따릉이 대여이력 원본을 월 단위 Parquet로 변환")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/trips"))
    parser.add_argument("--bronze", type=Path, default=Path("data/bronze/trips"))
    parser.add_argument("--month", action="append", help="YYYY-MM, 여러 번 줄 수 있음")
    args = parser.parse_args()

    total = 0
    for source, _path, rows in iter_convert(
        args.raw, args.bronze, set(args.month) if args.month else None
    ):
        total += rows
        print(f"{time.strftime('%H:%M:%S')} {source.month} {rows:>10,} rows <- {source.name}")
    print(f"total {total:,} rows")
