"""v5 nowcast 부족 분류기(shortage_nowcast): 특징, 로지스틱 회귀, 날짜순 교차검증."""

# [무엇을 검증하나] v5의 nowcast 분류기 N(model/shortage_nowcast.py)을 확인한다: 로지스틱 특징
# 행렬의 열·발표 시각 더미, LightGBM 특징의 결측 처리, SciPy로 직접 푼 L2 로지스틱 회귀가 알려진
# 계수를 되찾는지와 저장·복원, 날짜순 교차검증의 겹 구성, C와 (잎 수, 라운드) 선택 규칙.
# [왜 중요한가] scikit-learn 없이 목적 함수를 직접 썼으므로 수식(벌점, 절편, 기울기)이 틀려도 학습은
# '돌아간다'. 알려진 모델에서 계수를 되찾는지 봐야 구현이 맞는지 알 수 있다. 교차검증이 날짜 순서를
# 어기면 미래 날로 학습한 값으로 설정을 고르게 된다.

from __future__ import annotations

import numpy as np
import pytest

# se.log_loss: 선택 결과를 손으로 다시 계산할 때 하네스와 같은 손실 함수를 쓴다.
from bike_demand.model import shortage_eval as se
from bike_demand.model import shortage_nowcast as nc


# shortage_eval이 만드는 행 사전과 같은 열 이름의 작은 자료. 마지막 행은 b0 = 0(이미 0대), 둘째 행은
# 30분 전 스냅샷이 없는 경우(miss30 = 1), 발표 시각 8개가 한 번씩 나온다.
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


# b0 ≥ 1인 7행의 특징이 LOGIT_FEATURES 순서로 정의대로 계산된다(log_b0 = log(1 + b0), inv_b0, z =
# (b0 − 0.5 − (E − Q)) / √(E + Q + 0.1)). 02시는 기준 범주라 더미가 없고, 05시·20시 더미는 그 발표
# 행에만 1이다. b0 = 0 행이 섞이면 ValueError(inv_b0가 무한대가 되므로).
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


# LightGBM 특징에서는 30분 전 스냅샷이 없는 행의 Δ30이 0이 아니라 NaN(결측)이다. 로지스틱은 0과 빈
# 값 표시로 나눠 넣고, 트리는 결측을 직접 다룬다.
def test_lgbm_matrix_turns_missing_deltas_into_nan():
    rows = toy_rows()
    x = nc.lgbm_matrix(rows, rows["b0"] >= 1)
    d30 = x[:, nc.LGBM_FEATURES.index("d30")]
    assert np.isnan(d30[1]) and not np.isnan(d30[0])


# 표준화한 특징에 대해 계수 (1.5, −0.7, 2.0), 절편 −1.0인 모델에서 20,000행을 뽑아 학습하면, 벌점이
# 거의 없는 C = 100에서 그 계수와 절편을 되찾는다. C = 1e-4(강한 벌점)면 계수 크기 합이 절반 아래로
# 줄어든다(벌점 방향 확인). to_dict → from_dict로 되살린 모델의 예측도 같다(test 실행이
# dev_fit.json에서 모델을 되살리는 경로).
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


# 개발 6일(날짜 사이에 빈 날이 있어도)을 날짜 순서로 놓고 4·5·6번째 날을 차례로 검증하며, 학습은
# 항상 그 날보다 앞선 날만 쓴다. 계획의 '앞선 날로 학습해 다음 날을 맞히는 3겹'.
def test_forward_folds_validate_the_fourth_to_sixth_day_on_earlier_days():
    day = np.repeat(np.array([10, 11, 12, 13, 15, 16]), 3)
    folds = nc.forward_folds(day)
    assert len(folds) == 3
    for (train, valid), val_day in zip(folds, (13, 15, 16), strict=True):
        assert set(day[valid]) == {val_day}
        assert set(day[train]) == {d for d in (10, 11, 12, 13, 15) if d < val_day}


# select_logistic의 선택을 손으로 다시 계산한 겹 평균 log loss와 비교한다. 신호가 있는 자료라 벌점이
# 약한 C = 1이 이기고, 표의 평균 손실도 같다. 검증 예측(oof)은 검증에 쓴 날(4~6번째)에만 값이 있고
# 앞의 세 날은 NaN이다.
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


# 라운드 상한을 30으로 줄여 빠르게 돌린다. 고른 잎 수는 후보 안에 있고 라운드는 1~30, 고른 잎 수의
# 손실이 표에서 가장 낮다. fit_lgbm은 조기 종료 없이 정확히 고른 라운드 수만큼 트리를 만든다.
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
