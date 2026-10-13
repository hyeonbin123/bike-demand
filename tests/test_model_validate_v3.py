"""validate_v3.main은 결과 폴더를 학습 전에 만든다.

data/는 커밋하지 않으므로 새 클론에는 data/models/v1이 없다. 폴더가 없다는 이유로 몇 시간 학습한 뒤
모델 파일을 쓰다 실패하면 안 된다. 실제 warehouse·학습 없이 작은 합성 행으로 돈다.
"""

# [무엇을 검증하나] validate_v3.main이 몇 시간 걸리는 학습을 시작하기 전에 결과 폴더를 만드는지,
# 그리고 두 후보(M1'·M3')의 모델 파일과 validation_v3.json을 끝까지 쓰는지 확인한다.
# [왜 중요한가] 회귀 테스트(e4d30ea): 새 클론에는 data/models/v1이 없어서, 학습을 다 마친 뒤 모델을
# 저장하는 순간 실패했다(LightGBMError). 이 테스트는 고치기 전에 실패했다.
# [방식] load_frame과 학습 함수를 monkeypatch로 바꿔 실제 warehouse·긴 학습 없이 240행짜리 합성
# 자료로 main 전체 흐름만 돌린다.

import json

# duckdb: main이 여는 빈 warehouse 파일을 만든다. lightgbm: 3라운드짜리 작은 모델.
import duckdb
import lightgbm as lgb
import numpy as np

from bike_demand.model import validate_v3
from bike_demand.model.frames import FEATURES, LEVEL_FEATURES


# load_frame 대신 돌려줄 합성 행. 모든 특징(수준 특징 포함)을 무작위로 채우고, 지표 계산에 쓰는
# 열(대여소 코드, 날짜, 시간, 대여 수)은 뜻이 맞게 넣는다.
def synthetic_frame(n: int = 240) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    frame = {name: rng.random(n).astype(np.float32) for name in [*FEATURES, *LEVEL_FEATURES]}
    frame.update(
        station_code=np.zeros(n, np.float32),
        day_index=np.arange(n, dtype=np.int64) // 24,
        hour_of_day=(np.arange(n) % 24).astype(np.float32),
        rentals=rng.poisson(2, n).astype(np.float32),
    )
    return frame


# fit_lightgbm_separate_stop과 같은 인자를 받는 가짜 학습. 합성 자료로 3라운드만 학습한 모델과
# 라운드 수를 돌려준다.
def tiny_fit(load_stop_frame, load_full_frame, features, log, early_stop_from):
    frame = synthetic_frame()
    data = lgb.Dataset(np.column_stack([frame[f] for f in features]), frame["rentals"])
    params = {"objective": "poisson", "verbose": -1, "num_leaves": 4, "num_threads": 1}
    return lgb.train(params, data, num_boost_round=3), 3


# 아직 없는 out 폴더(fresh/v1)를 주고, 학습 함수가 불릴 때마다 그 순간 폴더가 이미 있는지 기록한다.
# 두 후보 모두 학습 시점에 폴더가 있어야 하고, 끝나면 선택 결과와
# validation_v3_M1.txt·validation_v3_M3.txt(작은따옴표를 뗀 이름)가 있어야 한다.
def test_main_creates_out_dir_before_training(tmp_path, monkeypatch):
    warehouse = tmp_path / "wh.duckdb"
    duckdb.connect(str(warehouse)).close()
    out = tmp_path / "fresh" / "v1"
    folder_at_fit = []

    def fit(*args):
        folder_at_fit.append(out.is_dir())
        return tiny_fit(*args)

    monkeypatch.setattr(validate_v3, "load_frame", lambda *args, **kwargs: synthetic_frame())
    monkeypatch.setattr(validate_v3, "fit_lightgbm_separate_stop", fit)

    validate_v3.main(warehouse, out)
    assert folder_at_fit == [True, True]  # 첫 학습 전에 이미 있음
    report = json.loads((out / "validation_v3.json").read_text("utf-8"))
    assert report["selected"] in {"M1'", "M3'"}
    assert (out / "validation_v3_M1.txt").exists() and (out / "validation_v3_M3.txt").exists()
