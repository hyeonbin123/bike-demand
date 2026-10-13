"""따릉이 대여이력 원본(월별 CSV, 연도별 ZIP 안의 월별 CSV)을 월 단위 Parquet로 옮긴다.

원본은 cp949, 빈 값은 ``\\N``이고, 2023~2025년 파일에만 ``자전거구분`` 컬럼이 있다.
이 단계는 값을 해석하지 않는다: 모든 컬럼을 문자열로 두고 컬럼 이름만 영어로 바꾼다.
타입 변환과 이상값 처리는 다음 단계(dbt staging)에서 한다.

같은 달을 다시 변환하면 그 달의 파일을 통째로 바꾸므로 행이 중복되지 않는다.
"""

# [파일 역할] 파이프라인의 첫 단계(raw → bronze). 서울 열린데이터광장 대여이력(OA-15182, 2023-01 ~
# 2026-06, 약 1억 4,464만 건, 원본 CSV 약 24GB)을 월별 Parquet(zstd, 약 2.8GB)으로 옮긴다. 변환은 약
# 8분 걸린다(docs/data.md).
# [설계 이유] 원본 형식이 섞여 있다(2023~2025년은 연도별 ZIP 안의 월별 CSV, 2026년은 월별 CSV, 컬럼
# 수 17/16개). 여기서는 형식 차이만 흡수하고 값은 해석하지 않으며, 타입·중복·취소 처리는 dbt
# stg_trips가 한다. 원본을 Postgres에 올리지 않고 Parquet + DuckDB로 정제하는 이유는 README 설계
# 판단에 있다.
# [출력] data/bronze/trips/month=YYYY-MM/trips.parquet. dbt 소스가 폴더 이름의 month를 hive 분할
# 컬럼으로 읽는다.
# [연결] refresh_history DAG(convert_trips), pipelines.latest_trip_month(discover_sources로 원본
# 최신 달을 찾음).

from __future__ import annotations

import os
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO

# pyarrow.csv: 큰 CSV를 C++ 구현으로 빠르게 읽고 cp949를 디코딩한다. pyarrow.compute: 출처 파일 이름
# 컬럼을 채운다. pyarrow.parquet: zstd 압축 Parquet 쓰기.
import pyarrow as pa
import pyarrow.compute as pac
import pyarrow.csv as pc
import pyarrow.parquet as pq

# 원본 CSV 인코딩(cp949)과 빈 값 표기(\N). docs/data.md 파일 형식 확인 결과.
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
# 2026년 파일에는 자전거구분 컬럼이 없다. 없으면 빈 값 컬럼으로 채워 모든 달의 스키마를 같게 맞춘다.
OPTIONAL_COLUMNS = {"자전거구분"}

# 모든 컬럼을 문자열로 둔다. 대여소번호의 앞자리 0(예: 01653) 같은 원본 모양을 그대로 보존하고,
# 해석은 dbt에서 한 번만 한다. source_file은 어느 원본(ZIP 안 파일 포함)에서 온 행인지 남긴다.
SCHEMA = pa.schema(
    [(name, pa.string()) for name in COLUMNS.values()] + [("source_file", pa.string())]
)

# 파일 이름 끝의 _YYMM.csv에서 연·월을 뽑는다(예: _2601.csv → 2026-01).
_MONTH_RE = re.compile(r"_(\d{2})(\d{2})\.csv$")


# 월 하나의 원본 위치. ZIP 안 파일이면 member(ZIP 안 실제 이름)와 member_label(사람이 읽을 수 있게
# 되살린 한글 이름)을 함께 둔다. frozen이라 만든 뒤 바뀌지 않는다.
@dataclass(frozen=True)
class TripSource:
    """월 하나의 원본 CSV. ``member``가 있으면 ``path`` ZIP 안의 파일이다."""

    month: str  # "YYYY-MM"
    path: Path
    member: str | None = None
    member_label: str | None = None  # 깨진 한글을 되살린 ZIP 안 파일 이름

    @property
    # 로그·오류·source_file 컬럼에 쓰는 이름. ZIP 안 파일은 'ZIP이름!안쪽이름'.
    def name(self) -> str:
        if self.member is None:
            return self.path.name
        return f"{self.path.name}!{self.member_label or self.member}"

    # 원본을 바이트 스트림으로 연다. ZIP 안 파일이면 압축을 풀지 않고 스트림으로 읽는다.
    def open(self) -> IO[bytes]:
        if self.member is None:
            return open(self.path, "rb")
        archive = zipfile.ZipFile(self.path)
        stream = archive.open(self.member)
        # 스트림을 닫을 때 ZIP 파일도 함께 닫히게 close를 감싼다. 호출하는 쪽은 with 문으로 스트림만
        # 다루면 되고, ZIP 핸들이 남지 않는다.
        original_close = stream.close

        def close() -> None:
            original_close()
            archive.close()

        stream.close = close  # type: ignore[method-assign]
        return stream


# ZIP 형식은 UTF-8 플래그(0x800)가 없으면 이름을 cp437로 해석하므로, cp949로 적힌 한글 이름이 깨져
# 보인다. cp437로 되돌린 바이트를 cp949로 다시 읽어 원래 이름을 찾는다. 실패하면 받은 이름을 그대로
# 쓴다.
def _member_name(info: zipfile.ZipInfo) -> str:
    """ZIP이 UTF-8 플래그 없이 cp949로 이름을 적었으면 한글이 깨지므로 되살린다."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode(SOURCE_ENCODING)
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


# 파일 이름에서 'YYYY-MM'을 만든다. 연월이 없거나 월이 1~12가 아니면 멈춘다(달을 잘못 붙이면 다른 달
# 파일을 덮어쓰게 됨).
def month_of(filename: str) -> str:
    match = _MONTH_RE.search(filename)
    if not match:
        raise ValueError(f"파일 이름에서 연월(_YYMM.csv)을 찾지 못함: {filename}")
    yy, mm = match.groups()
    if not 1 <= int(mm) <= 12:
        raise ValueError(f"잘못된 월: {filename}")
    return f"20{yy}-{mm}"


# 호출 위치: iter_convert, pipelines.latest_trip_month.
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

    # 같은 달 원본이 둘이면(예: ZIP 안과 낱개 CSV에 같은 달) 어느 쪽을 쓸지 모르므로 멈춘다. 그대로
    # 두면 나중 것이 앞 것을 덮어써 결과가 파일 순서에 달라진다.
    seen: dict[str, TripSource] = {}
    for source in sources:
        if source.month in seen:
            raise ValueError(
                f"{source.month}월 원본이 두 개: {seen[source.month].name}, {source.name}"
            )
        seen[source.month] = source
    return sorted(sources, key=lambda s: s.month)


# 원본 CSV 하나를 문자열 테이블로 읽고 헤더를 검사한 뒤 COLUMNS 순서·이름으로 맞춘다. 호출 위치:
# convert.
def read_source(source: TripSource) -> pa.Table:
    with source.open() as stream:
        # 읽기 설정: cp949 디코딩, 64MB 단위 블록. 원본은 모든 값을 따옴표로 감싸므로 따옴표 안의
        # \N과 빈 문자열도 null로 본다. 컬럼 타입을 모두 string으로 지정해 pyarrow가 숫자로 추론하며
        # 앞자리 0을 없애지 않게 한다.
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

    # 모르는 헤더가 있거나 필수 헤더가 빠지면 원본 양식이 바뀐 것이므로 조용히 넘어가지 않고 멈춘다.
    headers = table.column_names
    unknown = [h for h in headers if h not in COLUMNS]
    missing = [h for h in COLUMNS if h not in headers and h not in OPTIONAL_COLUMNS]
    if unknown or missing:
        raise ValueError(f"{source.name}: 예상과 다른 헤더 (모름={unknown}, 없음={missing})")

    # COLUMNS 순서대로 열을 모으고, 없는 선택 컬럼은 null 열로 채운다. 마지막에 모든 행이 같은
    # source_file 값을 갖는 열을 붙인다.
    arrays = [
        table.column(header) if header in headers else pa.nulls(table.num_rows, pa.string())
        for header in COLUMNS
    ]
    arrays.append(pac.fill_null(pa.nulls(table.num_rows, pa.string()), source.name))
    return pa.Table.from_arrays(arrays, schema=SCHEMA)


# 월별 출력 경로(month=YYYY-MM 폴더). 한 달에 파일 하나라 같은 달을 다시 변환하면 그 파일만 통째로
# 바뀐다.
def output_path(bronze_dir: Path, month: str) -> Path:
    return bronze_dir / f"month={month}" / "trips.parquet"


def convert(source: TripSource, bronze_dir: Path) -> tuple[Path, int]:
    """월 하나를 변환해 쓰고 (경로, 행 수)를 돌려준다. 임시 파일에 쓴 뒤 바꿔치기한다."""
    table = read_source(source)
    target = output_path(bronze_dir, source.month)
    target.parent.mkdir(parents=True, exist_ok=True)
    # 임시 파일에 다 쓴 뒤 os.replace로 바꿔, 중간에 실패해도 dbt가 반쯤 쓴 파일을 읽지 않는다.
    tmp = target.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, target)
    return target, table.num_rows


# 원본을 달 순서로 변환하며 하나씩 결과를 내준다(진행 상황을 바로 출력하려고 생성기로 둠). months를
# 주면 그 달만 변환한다. 호출 위치: 아래 실행부.
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

    # 실행: python -m bike_demand.ingest.trips [--month YYYY-MM ...]. 달마다 시각·행 수·원본 이름을
    # 출력한다.
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
