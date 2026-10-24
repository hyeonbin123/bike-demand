"""서비스 모델 세대의 provenance(무엇으로 만들었는지) 수집과 세대 검증.

서비스 학습(`model.final serving`)이 끝나면 세대의 serving.json에 `provenance` 블록을 적는다: 코드
커밋과 작업 트리 변경, 저장소 파일(uv.lock, 공휴일 시드) SHA-256, 파이썬·라이브러리 버전, 학습 설정,
모델의 특징 순서, warehouse 지문, 세대 파일 SHA-256. `verify_generation`은 그 기록과 지금
상태(저장한 모델, 산출물 meta, 지금 warehouse의 대여소 번호, 파일)를 대조한다.
"""

# [파일 역할] 두 가지다. (1) collect·git_state·warehouse_fingerprint·package_versions·file_digests:
# 서비스 세대를 만들 때 '이 세대는 이 커밋·이 라이브러리·이 자료로 나왔다'를 남긴다. (2)
# verify_generation: 고정한 세대를 나중의 warehouse에서 예측·채점하기 전에 입력이 학습 때와 같은지
# 확인하는 관문이다.
# [왜 필요한가] 세대 폴더(data/ 아래, 커밋되지 않음)를 잃으면 같은 바이트로 다시 만들 수 없고(학습
# 행 순서가 실행마다 달라 재학습이 같은 모델이라는 보장이 없음), 남아 있어도 어느 코드로 학습했는지
# 기록이 없었다. 또 범주형 특징 station_code는 frames.history_ctes가 '지금 warehouse의 대여소ID
# 문자열 정렬 순위'로 매기므로, 새 반기를 넣어 warehouse를 다시 빌드하면 새 대여소가 정렬 중간에
# 끼어 기존 번호가 밀린다(예: 'ST-3435'는 'ST-343'과 'ST-344' 사이). 고정한 모델은 학습 때
# 번호표(artifacts의 stations.parquet)를 써야 하고, 번호가 밀렸는지는 코드가 확인해야
# 한다(tests/test_model_provenance.py의 회귀 테스트가 밀림과 대조를 재현한다).
# [연결] model/final.py fit_serving이 git_state → (학습) → collect를 부른다. verify_generation은
# scripts/generation_manifest.py와 고정 세대 채점 코드가 부른다. artifacts.warehouse_codes가 번호
# 대조의 warehouse 쪽이다.
# [주의] git 호출은 주입할 수 있는 함수(run)로 두어 테스트가 실제 작업 트리에 흔들리지 않는다. git이
# 없는 환경(컨테이너)에서는 값 대신 이유를 적고 멈추지 않는다.

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path

# duckdb: warehouse 지문(격자 행 수·대여 합, 대여소 표 해시)을 읽기 전용 연결로 뽑는다. lightgbm:
# booster를 넘기지 않았을 때 세대 폴더의 model.txt를 읽는다.
import duckdb
import lightgbm as lgb

# artifacts: 세대 산출물(번호표·meta)을 읽고 지금 warehouse의 번호를 같은 규칙으로 매긴다.
from bike_demand.model import artifacts

KST = timezone(timedelta(hours=9))
# 세대 안에서 지문 대상에서 빼는 파일: provenance를 담는 serving.json은 자기 지문을 가질 수 없다.
RECORD_FILE = "serving.json"
# 저장소 파일 중 학습 결과에 영향을 주는 입력으로 지문을 남기는 것: 의존성 잠금과 dbt 공휴일
# 시드(달력 특징 is_holiday의 원천).
REPO_FILES = ("uv.lock", "dbt/seeds/kr_holidays.csv")
# 버전을 기록하는 패키지(배포 이름). 학습·예측 값에 영향을 줄 수 있는 것만.
PACKAGES = ("lightgbm", "duckdb", "numpy", "pyarrow", "holidays")
# git 명령 실행 함수의 모양: (인자 목록, 작업 폴더) → 표준 출력.
GitRunner = Callable[[list[str], Path], str]


# 검증에서 '같아야 하는 것이 다름'을 strict 모드에서 알리는 예외. SystemExit가 아니라 일반 예외로
# 두어 부르는 쪽(채점 하네스)이 잡아 보고할 수 있게 한다.
class GenerationMismatch(Exception):
    pass


# 파일 하나의 SHA-256(16진수). 모델 파일이 60 MB대라 1 MiB씩 읽는다.
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 폴더 아래 모든 파일의 {폴더 기준 슬래시 경로: SHA-256}. exclude에 든 경로(폴더 기준)는 뺀다. 경로
# 순으로 정렬해 같은 폴더면 같은 사전이 나온다. 호출 위치: collect(세대 파일),
# verify_generation(지금 파일), scripts/generation_manifest.py.
def file_digests(directory: Path, exclude: Iterable[str] = ()) -> dict[str, str]:
    """directory 아래 파일들의 {상대 경로: SHA-256}. exclude는 뺄 상대 경로 모음."""
    directory = Path(directory)
    skipped = set(exclude)
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    return {
        p.relative_to(directory).as_posix(): sha256_file(p)
        for p in files
        if p.relative_to(directory).as_posix() not in skipped
    }


# 기본 git 실행 함수. 실패(명령 없음, 저장소 아님)는 예외로 올려 git_state가 이유를 적는다. 출력은
# UTF-8로 읽고 깨진 바이트는 대치 문자로 바꾼다(파일 이름에 한글이 있어도 멈추지 않게).
def run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return result.stdout


# 코드 상태: 저장소 루트, HEAD 커밋, 커밋하지 않은 변경(추적하지 않는 새 파일 포함)과 그 diff의
# SHA-256. 변경이 있으면 어느 변경으로 학습했는지 두 실행을 구분할 수 있게 `git diff HEAD`의
# SHA-256을 남긴다(새 파일 내용은 diff에 없으므로 목록으로만 남음). git을 쓸 수 없으면
# {"available": False, "reason": ...}. 호출 위치: final.fit_serving(세대 폴더를 만들기 전), tests.
def git_state(repo_root: Path | None = None, run: GitRunner | None = None) -> dict:
    """작업 트리 상태 사전. run은 테스트용 주입 함수(기본 run_git)."""
    run = run or run_git
    cwd = Path(repo_root) if repo_root is not None else Path.cwd()
    try:
        root = run(["rev-parse", "--show-toplevel"], cwd).strip()
        commit = run(["rev-parse", "HEAD"], cwd).strip()
        status = run(["status", "--porcelain"], cwd)
        changed = [line.rstrip("\r\n") for line in status.splitlines() if line.strip()]
        state: dict = {
            "available": True,
            "root": root,
            "commit": commit,
            "dirty": bool(changed),
            "changed_files": changed,
        }
        if changed:
            diff = run(["diff", "HEAD"], cwd)
            state["diff_sha256"] = hashlib.sha256(diff.encode("utf-8")).hexdigest()
        return state
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


# 파이썬과 PACKAGES의 설치 버전. 설치되지 않은 패키지는 None.
def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


# 학습 자료의 지문: 학습 기간(history, [시작, 끝)) 격자의 행 수와 대여 합, 대여소 표(dim_stations의
# 번호 규칙에 쓰이는 열)를 station_id 순으로 뽑아 행 repr을 이어 만든 SHA-256과 행 수. ORDER BY로 행
# 순서를 고정하므로 같은 자료면 실행마다 같다. DuckDB hash() 집계는 버전마다 구현이 달라질 수 있어
# 쓰지 않는다. 호출 위치: final.fit_serving(산출물 내보내기와 같은 연결에서), tests.
def warehouse_fingerprint(con: duckdb.DuckDBPyConnection, history: tuple[str, str]) -> dict:
    """{"history", "grid_rows", "grid_rentals", "dim_stations_rows", "dim_stations_sha256"}."""
    start, end = history
    rows, rentals = con.execute(
        """select count(*), coalesce(sum(rentals), 0) from int_station_hour_grid
           where hour_start >= cast(? as timestamp) and hour_start < cast(? as timestamp)""",
        [start, end],
    ).fetchone()
    stations = con.execute(
        """select station_id, station_no, district, docks, lat, lon
           from dim_stations order by station_id"""
    ).fetchall()
    digest = hashlib.sha256()
    for row in stations:
        digest.update(repr(row).encode("utf-8"))
        digest.update(b"\n")
    return {
        "history": list(history),
        "grid_rows": int(rows),
        "grid_rentals": int(rentals) if float(rentals).is_integer() else float(rentals),
        "dim_stations_rows": len(stations),
        "dim_stations_sha256": digest.hexdigest(),
    }


# 세대 폴더에 모델·산출물을 다 쓴 뒤 serving.json에 넣을 provenance 블록을 모은다. git·warehouse는
# 학습 전에 뽑아 둔 값을 받는다(학습이 몇 시간 걸리는 동안의 상태가 아니라 학습을 시작한 코드·자료의
# 상태를 적기 위해). 저장소 파일(REPO_FILES)은 git 루트(없으면 repo_root, 그것도 없으면 현재 폴더)
# 기준이고 없으면 None. 세대 파일 지문에서 serving.json은 뺀다. 호출 위치: final.fit_serving.
def collect(
    directory: Path,
    booster,
    git: dict,
    warehouse: dict,
    params: dict,
    repo_root: Path | None = None,
) -> dict:
    """serving.json의 provenance 블록(JSON으로 바로 쓸 수 있는 사전)."""
    root = Path(git.get("root") or repo_root or Path.cwd())
    repo_files = {
        name: sha256_file(root / name) if (root / name).is_file() else None for name in REPO_FILES
    }
    return {
        "recorded_at": datetime.now(KST).isoformat(timespec="seconds"),
        "repo_root": root.as_posix(),
        "git": git,
        "repo_files": repo_files,
        "versions": package_versions(),
        "params": dict(params),
        "feature_names": list(booster.feature_name()),
        "warehouse": warehouse,
        "files": file_digests(directory, exclude=(RECORD_FILE,)),
    }


# serving.json을 읽는다. 없으면(산출물만 있는 폴더, 테스트) 빈 사전.
def _read_record(directory: Path) -> dict:
    path = directory / RECORD_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text("utf-8"))


# 고정한 세대가 지금 쓰기에 맞는지 세 가지를 대조한다.
# (i) 특징 순서: 모델의 feature_name() 대 산출물 meta.json의 features. 다르면 provenance가 없는 옛
#     세대는 경고(게시된 v3-M3p-20260917T0904는 meta가 수준 특징 3개를 빠뜨린 20개, 모델은 23개 —
#     당시 내보내기 코드의 누락이고 예측은 모델의 순서를 쓰므로 영향 없음), provenance가 있는 세대는
#     오류. provenance에 적힌 feature_names가 모델과 다른 것도 오류.
# (ii) 번호표(con이 있을 때): 산출물 stations.parquet의 번호 대 지금 warehouse의 번호(같은 규칙,
#     artifacts.warehouse_codes). 같은 대여소인데 번호가 다른 곳(moved), 산출물에 없는 새
#     대여소(new), warehouse에서 사라진 대여소(missing), 새 대여소가 학습 때 번호 범위 안의 번호를
#     받은 수(new_with_reused_code: load_frame 경로였다면 남의 범주를 타는 곳). moved가 하나라도
#     있으면 오류 — frames.load_frame으로 특징을 다시 만들면 이 모델에 맞지 않는 입력이 된다는
#     뜻이다(산출물 경로 predict.feature_rows는 번호표를 쓰므로 영향 없음).
# (iii) 파일 지문: expected(또는 serving.json의 provenance.files)와 지금 파일의 SHA-256. 다른 파일·
#     빠진 파일은 오류, 기록에 없는 새 파일은 경고. expected는 {경로: SHA-256}이거나
#     scripts/generation_manifest.py가 남기는 {경로: {"sha256", "bytes"}}다. 기록을 담는
#     serving.json은 expected에 있을 때만(세대 폴더 밖 manifest) 대조하고, 없으면(세대 자신의
#     provenance.files는 자기 지문을 가질 수 없음) 새 파일로도 치지 않는다.
# strict=True면 오류가 하나라도 있을 때 GenerationMismatch(채점용: 멈추고 보고). 아니면 보고 사전만
# 돌려준다(ok, problems, warnings). 호출 위치: scripts/generation_manifest.py, 고정 세대 채점,
# tests.
def verify_generation(
    directory: Path,
    booster=None,
    con: duckdb.DuckDBPyConnection | None = None,
    expected: dict[str, str | dict] | None = None,
    strict: bool = False,
) -> dict:
    """세대 폴더의 특징 순서·번호표·파일 지문 검사 보고. booster가 없으면 model.txt를 읽는다."""
    directory = Path(directory)
    if booster is None:
        booster = lgb.Booster(model_file=str(directory / "model.txt"))
    record = _read_record(directory).get("provenance") or {}
    meta = json.loads((directory / "artifacts" / "meta.json").read_text("utf-8"))
    problems: list[str] = []
    warnings: list[str] = []

    # (i) 특징 순서
    model_features = list(booster.feature_name())
    meta_features = meta.get("features")
    features = {
        "model": model_features,
        "meta": meta_features,
        "match": model_features == meta_features,
    }
    if not features["match"]:
        count = len(meta_features) if meta_features is not None else 0
        message = (
            f"특징 순서가 다름: 모델 feature_name() {len(model_features)}개 대 "
            f"artifacts/meta.json features {count}개"
        )
        (problems if record else warnings).append(message)
    if record.get("feature_names") is not None and record["feature_names"] != model_features:
        problems.append("provenance.feature_names가 모델의 feature_name()과 다름")

    # (ii) 번호표 대 지금 warehouse
    codes: dict = {"checked": False}
    if con is not None:
        loaded = artifacts.load(directory / "artifacts")
        fixed = {sid: int(row["station_code"]) for sid, row in loaded.stations.items()}
        current = artifacts.warehouse_codes(con)
        shared = fixed.keys() & current.keys()
        moved = sorted(
            [sid, fixed[sid], current[sid]] for sid in shared if fixed[sid] != current[sid]
        )
        new = sorted(current.keys() - fixed.keys())
        missing = sorted(fixed.keys() - current.keys())
        reused = [sid for sid in new if current[sid] <= loaded.max_station_code]
        codes = {
            "checked": True,
            "artifact_stations": len(fixed),
            "warehouse_stations": len(current),
            "same": len(shared) - len(moved),
            "moved": len(moved),
            "new": len(new),
            "missing": len(missing),
            "new_with_reused_code": len(reused),
            "moved_examples": moved[:10],
            "new_examples": new[:10],
            "missing_examples": missing[:10],
        }
        if moved:
            problems.append(
                f"지금 warehouse에서 번호가 달라진 대여소 {len(moved)}곳"
                f"(새 대여소 {len(new)}, 사라진 대여소 {len(missing)}): "
                "frames.load_frame으로 만든 특징은 이 세대에 맞지 않음"
            )

    # (iii) 파일 지문
    files: dict = {"checked": False}
    expected = expected if expected is not None else record.get("files")
    if expected is not None:
        # scripts/generation_manifest.py가 남긴 {경로: {"sha256", "bytes"}} 형식도 받는다
        expected = {p: v["sha256"] if isinstance(v, dict) else v for p, v in expected.items()}
        # 기록을 담는 serving.json은 기대값에 있을 때만 대조하고(세대 폴더 밖 manifest), 없으면
        # 새 파일로도 치지 않는다(세대 자신의 provenance.files)
        actual = file_digests(directory)
        mismatched = sorted(p for p in expected if p in actual and actual[p] != expected[p])
        missing_files = sorted(p for p in expected if p not in actual)
        unexpected = sorted(p for p in actual if p not in expected and p != RECORD_FILE)
        files = {
            "checked": True,
            "compared": len(expected),
            "mismatched": mismatched,
            "missing": missing_files,
            "unexpected": unexpected,
        }
        if mismatched or missing_files:
            problems.append(
                f"파일 지문이 기록과 다름: 바뀐 파일 {mismatched}, 빠진 파일 {missing_files}"
            )
        if unexpected:
            warnings.append(f"기록에 없는 파일: {unexpected}")

    report = {
        "generation": directory.name,
        "has_provenance": bool(record),
        "features": features,
        "codes": codes,
        "files": files,
        "problems": problems,
        "warnings": warnings,
        "ok": not problems,
    }
    if strict and problems:
        raise GenerationMismatch(f"{directory.name}: " + "; ".join(problems))
    return report
