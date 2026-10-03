"""v5 nowcast 부족 분류기(shortage_nowcast): 특징, 로지스틱 회귀, 날짜순 교차검증."""

from __future__ import annotations

import numpy as np
import pytest

from bike_demand.model import shortage_eval as se
from bike_demand.model import shortage_nowcast as nc


def toy_rows(n=8):
    rng = np.random.default_rng(0)
    return {
        "b0": np.array([1, 2, 3, 4, 5, 6, 7, 0])[:n],
        "d30": rng.integers(-3, 3, n).astype(float),
        "d60": rng.integers(-5, 5, n).astype(float),
        "miss30": np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=float)[:n],
        "miss60": np.zeros(n),
        "E3": rng.uniform(0, 3, n),
        "R3": rng.uniform(0, 3, n),
        "Q3": rng.uniform(0, 3, n),
        "issue_hour": np.array([2, 5, 8, 11, 14, 17, 20, 23])[:n],
        "offday": np.array([0, 0, 1, 1, 0, 0, 0, 1], dtype=float)[:n],
    }


def test_logit_matrix_columns_and_hour_dummies():
    rows = toy_rows()
    mask = rows["b0"] >= 1
    x = nc.logit_matrix(rows, mask)
    assert x.shape == (7, len(nc.LOGIT_FEATURES))
    col = {name: x[:, i] for i, name in enumerate(nc.LOGIT_FEATURES)}
    assert col["log_b0"] == pytest.approx(np.log1p(rows["b0"][mask]))
    assert col["inv_b0"][0] == 1.0
    e, q, b0 = rows["E3"][mask], rows["Q3"][mask], rows["b0"][mask]
    assert col["z"] == pytest.approx((b0 - 0.5 - (e - q)) / np.sqrt(e + q + 0.1))
    assert "hour_02" not in nc.LOGIT_FEATURES  # 기준 범주
    assert col["hour_05"].tolist() == [0, 1, 0, 0, 0, 0, 0]
    assert col["hour_20"].tolist() == [0, 0, 0, 0, 0, 0, 1]
    with pytest.raises(ValueError):
        nc.logit_matrix(rows, np.ones(8, dtype=bool))  # b0 = 0 행은 받지 않는다


def test_lgbm_matrix_turns_missing_deltas_into_nan():
    rows = toy_rows()
    x = nc.lgbm_matrix(rows, rows["b0"] >= 1)
    d30 = x[:, nc.LGBM_FEATURES.index("d30")]
    assert np.isnan(d30[1]) and not np.isnan(d30[0])


def test_logistic_recovers_a_known_model_and_round_trips():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20000, 3)) * [1.0, 2.0, 0.5] + [0.0, 1.0, -1.0]
    true = np.array([1.5, -0.7, 2.0])
    margin = ((x - x.mean(axis=0)) / x.std(axis=0)) @ true - 1.0
    y = (rng.uniform(size=len(x)) < 1 / (1 + np.exp(-margin))).astype(float)
    model = nc.fit_logistic(x, y, c=100.0, names=("a", "b", "c"))
    assert model.coef == pytest.approx(true, abs=0.08)
    assert model.intercept == pytest.approx(-1.0, abs=0.06)
    strong = nc.fit_logistic(x, y, c=1e-4, names=("a", "b", "c"))
    assert np.abs(strong.coef).sum() < np.abs(model.coef).sum() / 2  # 작은 C는 강하게 줄인다
    again = nc.Logistic.from_dict(model.to_dict())
    assert again.predict(x[:5]) == pytest.approx(model.predict(x[:5]))


def test_forward_folds_validate_the_fourth_to_sixth_day_on_earlier_days():
    day = np.repeat(np.array([10, 11, 12, 13, 15, 16]), 3)
    folds = nc.forward_folds(day)
    assert len(folds) == 3
    for (train, valid), val_day in zip(folds, (13, 15, 16), strict=True):
        assert set(day[valid]) == {val_day}
        assert set(day[train]) == {d for d in (10, 11, 12, 13, 15) if d < val_day}


def test_select_logistic_uses_the_lowest_mean_validation_log_loss():
    rng = np.random.default_rng(2)
    day = np.repeat(np.arange(6), 2000)
    x = rng.normal(size=(len(day), 2))
    y = (rng.uniform(size=len(day)) < 1 / (1 + np.exp(-(1.2 * x[:, 0] - 0.5)))).astype(float)
    result = nc.select_logistic(x, y, day, grid=(0.0001, 1.0))
    by_hand = {}
    for c in (0.0001, 1.0):
        losses = []
        for train, valid in nc.forward_folds(day):
            model = nc.fit_logistic(x[train], y[train], c, names=("a", "b"))
            losses.append(se.log_loss(model.predict(x[valid]), y[valid]))
        by_hand[c] = np.mean(losses)
    assert result["C"] == min(by_hand, key=by_hand.get) == 1.0
    assert result["cv"]["1.0"]["mean_log_loss"] == pytest.approx(by_hand[1.0])
    assert np.isnan(result["oof"][day < 3]).all() and not np.isnan(result["oof"][day >= 3]).any()


def test_select_lgbm_picks_leaves_and_rounds_from_the_mean_curve(monkeypatch):
    monkeypatch.setattr(nc, "LGBM_MAX_ROUNDS", 30)
    rng = np.random.default_rng(3)
    day = np.repeat(np.arange(6), 1500)
    x = rng.normal(size=(len(day), 3))
    y = (rng.uniform(size=len(day)) < 1 / (1 + np.exp(-(x[:, 0] * 2)))).astype(float)
    result = nc.select_lgbm(x, y, day)
    assert result["num_leaves"] in nc.LGBM_LEAVES_GRID
    assert 1 <= result["rounds"] <= 30
    best = result["cv"][str(result["num_leaves"])]["mean_log_loss"]
    assert best == min(v["mean_log_loss"] for v in result["cv"].values())
    model = nc.fit_lgbm(x, y, result["num_leaves"], result["rounds"])
    assert model.num_trees() == result["rounds"]
