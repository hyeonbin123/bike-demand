"""서비스 모델 세대 폴더의 파일 지문(SHA-256)을 JSON으로 남긴다(읽기 전용).

세대 폴더(`data/models/v1/serving/generations/<세대>/`)의 모든 파일(model.txt, serving.json,
artifacts/*)의 SHA-256과 크기, 그리고 `provenance.verify_generation`의 검사 결과(모델 특징 순서 대
산출물 meta, 기록이 있으면 파일 지문 대조)를 한 JSON으로 적는다. 세대 폴더 안에는 아무것도 쓰지
않는다(--out이 세대 폴더 안이면 거부).

실행 (저장소 루트에서, 콘솔이 cp949면 PYTHONUTF8=1):
  uv run python scripts/generation_manifest.py                  # CURRENT 세대, 표준 출력
  uv run python scripts/generation_manifest.py <세대 폴더> --out <파일> [--note 글]...
      [--notes-file 파일]
"""

# [파일 역할] provenance 기록 기능이 생기기 전에 학습된 세대(게시된 v3-M3p-20260917T0904)의 사후
# 기록을 세대 폴더 밖에 남기고, 새 세대도 같은 형식으로 지문을 뽑아 보관하기 위한 읽기 전용
# 스크립트다. 세대 폴더는 커밋되지 않는 data/ 아래 한 곳뿐이라 지문이 있어야 나중에
# '같은 파일인지'를 말할 수 있다.
# [흐름] 세대 폴더의 파일을 모두 읽어 SHA-256·바이트 수를 만들고 → model.txt가 있으면
# verify_generation(booster는 model.txt에서 읽음, warehouse 대조는 하지 않음)을 돌려 특징 순서·기록
# 대조 결과를 붙이고 → 사람이 넘긴 메모(--note, --notes-file)를 더해 JSON으로 쓴다.
# [연결] bike_demand.model.provenance(file_digests, verify_generation), bike_demand.model.final
# (CURRENT 세대 찾기). tests/test_model_provenance.py가 manifest·출력 경로 거부를 확인한다.

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bike_demand.model import final, provenance

KST = timezone(timedelta(hours=9))


# 세대 폴더의 지문 사전. files는 폴더 기준 슬래시 경로 → {"sha256", "bytes"}(serving.json 포함: 이
# 기록은 폴더 밖에 두므로 자기 지문 문제가 없다). verification은 model.txt가 있을 때만
# verify_generation 보고(없으면 None). notes는 사람이 넘긴 메모 그대로. 호출 위치: 아래 실행부,
# tests/test_model_provenance.py.
def manifest(directory: Path, notes: list[str] | None = None) -> dict:
    """세대 폴더의 파일 지문·검사 결과·메모를 담은 사전(JSON으로 바로 쓸 수 있음)."""
    directory = Path(directory)
    if not (directory / "artifacts").is_dir():
        raise SystemExit(f"세대 폴더가 아님(artifacts/ 없음): {directory}")
    digests = provenance.file_digests(directory)
    files = {
        name: {"sha256": digest, "bytes": (directory / name).stat().st_size}
        for name, digest in digests.items()
    }
    verification = None
    if (directory / "model.txt").is_file():
        verification = provenance.verify_generation(directory)
    return {
        "generation": directory.name,
        "directory": directory.resolve().as_posix(),
        "recorded_at": datetime.now(KST).isoformat(timespec="seconds"),
        "files": files,
        "verification": verification,
        "notes": list(notes or []),
    }


# 출력 파일이 세대 폴더 안이면 거부한다. 세대 폴더의 파일은 지문 대상이라 하나도 더하거나 고치지
# 않는다(지문 관문이 '기록에 없는 파일'을 보게 됨).
def checked_out_path(out: Path, directory: Path) -> Path:
    """out이 세대 폴더 밖이면 그대로 돌려주고, 안이면 SystemExit."""
    if out.resolve().is_relative_to(Path(directory).resolve()):
        raise SystemExit(f"세대 폴더 안에는 쓰지 않는다: {out}")
    return out


# 메모 파일(UTF-8, 한 줄에 메모 하나, 빈 줄 무시)을 읽는다. Windows 콘솔에서 한글 인자가 깨지는 일을
# 피하려고 파일로도 받는다.
def read_notes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text("utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="세대 폴더의 파일 SHA-256 지문과 검사 결과를 JSON으로 남긴다(읽기 전용)"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="세대 폴더. 생략하면 data/models/v1/serving/CURRENT가 가리키는 세대",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="결과 JSON을 쓸 파일(세대 폴더 밖). 생략하면 표준 출력에 JSON을 찍는다",
    )
    parser.add_argument(
        "--note", action="append", default=[], help="결과의 notes에 붙일 메모(여러 번 가능)"
    )
    parser.add_argument(
        "--notes-file", type=Path, help="메모 파일(UTF-8, 한 줄에 하나). --note 뒤에 이어 붙인다"
    )
    args = parser.parse_args()

    directory = args.directory
    if directory is None:
        _, directory = final.current_generation(Path("data/models/v1/serving"))
    notes = list(args.note)
    if args.notes_file is not None:
        notes.extend(read_notes(args.notes_file))
    result = manifest(directory, notes)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out is None:
        print(text)
    else:
        out = checked_out_path(args.out, directory)
        out.parent.mkdir(parents=True, exist_ok=True)
        # LF로 쓴다(.gitattributes와 같은 줄 끝, Windows 기본 줄 끝은 CRLF)
        with open(out, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text + "\n")
        check = result["verification"]
        status = "검사 없음(model.txt 없음)" if check is None else f"ok={check['ok']}"
        print(f"{result['generation']}: 파일 {len(result['files'])}개, {status} → {out}")
