"""validate_v3.main은 결과 폴더를 학습 전에 만든다.

data/는 커밋하지 않으므로 새 클론에는 data/models/v1이 없다. 폴더가 없다는 이유로 몇 시간 학습한 뒤
모델 파일을 쓰다 실패하면 안 된다. 실제 warehouse·학습 없이 작은 합성 행으로 돈다.
"""

import json

import duckdb
import lightgbm as lgb
import numpy as np

from bike_demand.model import validate_v3
from bike_demand.model.frames import FEATURES, LEVEL_FEATURES


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


def tiny_fit(load_stop_frame, load_full_frame, features, log, early_stop_from):
    frame = synthetic_frame()
    data = lgb.Dataset(np.column_stack([frame[f] for f in features]), frame["rentals"])
    params = {"objective": "poisson", "verbose": -1, "num_leaves": 4, "num_threads": 1}
    return lgb.train(params, data, num_boost_round=3), 3


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
