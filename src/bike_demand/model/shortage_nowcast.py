"""v5(T61) nowcast 부족 분류기: 스냅샷 변화·프로파일·예상 대여로 3시간 안 0대 도달 확률을 낸다.

계획은 docs/experiments.md v5 절. 주 모델은 L2 로지스틱 회귀(판정 대상), 보조는 얕은 LightGBM
(보고만). 설정은 개발 6일의 날짜순 3겹 교차검증(앞선 날로 학습, 다음 날 검증)으로 고르고, 고른
설정으로 개발 6일 전체에 다시 학습해 고정한다. scikit-learn 없이 SciPy L-BFGS로 푼다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize

from bike_demand.model.shortage_eval import EPS, ISSUE_HOURS, MAIN_HOURS, log_loss

LOGIT_C_GRID = (0.01, 0.1, 1.0, 10.0)
LGBM_LEAVES_GRID = (7, 15)
LGBM_MAX_ROUNDS = 500
LGBM_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "min_data_in_leaf": 200,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "seed": 0,
    "verbose": -1,
}
# 기준 범주는 02시 발표(one-hot에서 뺌)
HOUR_DUMMIES = tuple(f"hour_{h:02d}" for h in ISSUE_HOURS[1:])
LOGIT_FEATURES = (
    "log_b0",
    "inv_b0",
    "d30",
    "d60",
    "miss30",
    "miss60",
    "log_e",
    "log_r",
    "log_q",
    "z",
    "offday",
    *HOUR_DUMMIES,
)
LGBM_FEATURES = ("b0", "d30", "d60", "e", "r", "q", "issue_hour", "offday")


def logit_matrix(rows: dict, mask: np.ndarray, h: int = MAIN_HOURS) -> np.ndarray:
    """로지스틱 회귀 특징(LOGIT_FEATURES 순서). b0 ≥ 1 행만 받는다."""
    b0 = rows["b0"][mask].astype(np.float64)
    if (b0 < 1).any():
        raise ValueError("분류기는 b0 ≥ 1 행만")
    e, r, q = rows[f"E{h}"][mask], rows[f"R{h}"][mask], rows[f"Q{h}"][mask]
    hour = rows["issue_hour"][mask]
    columns = {
        "log_b0": np.log1p(b0),
        "inv_b0": 1.0 / b0,
        "d30": rows["d30"][mask],
        "d60": rows["d60"][mask],
        "miss30": rows["miss30"][mask],
        "miss60": rows["miss60"][mask],
        "log_e": np.log1p(e),
        "log_r": np.log1p(r),
        "log_q": np.log1p(q),
        "z": (b0 - 0.5 - (e - q)) / np.sqrt(e + q + 0.1),
        "offday": rows["offday"][mask].astype(np.float64),
        **{name: (hour == int(name[-2:])).astype(np.float64) for name in HOUR_DUMMIES},
    }
    return np.column_stack([columns[name] for name in LOGIT_FEATURES])


def lgbm_matrix(rows: dict, mask: np.ndarray, h: int = MAIN_HOURS) -> np.ndarray:
    """LightGBM 특징(LGBM_FEATURES 순서). 스냅샷 변화가 없으면 결측."""
    columns = {
        "b0": rows["b0"][mask].astype(np.float64),
        "d30": np.where(rows["miss30"][mask] > 0, np.nan, rows["d30"][mask]),
        "d60": np.where(rows["miss60"][mask] > 0, np.nan, rows["d60"][mask]),
        "e": rows[f"E{h}"][mask],
        "r": rows[f"R{h}"][mask],
        "q": rows[f"Q{h}"][mask],
        "issue_hour": rows["issue_hour"][mask].astype(np.float64),
        "offday": rows["offday"][mask].astype(np.float64),
    }
    return np.column_stack([columns[name] for name in LGBM_FEATURES])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -z))


@dataclass
class Logistic:
    """표준화 + L2 로지스틱 회귀. 목적: 평균 log loss + |w|² / (2·C·n) (절편 벌점 없음)."""

    names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float
    c: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(((x - self.mean) / self.std) @ self.coef + self.intercept)

    def to_dict(self) -> dict:
        return {
            "names": list(self.names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "C": self.c,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Logistic:
        return cls(
            tuple(data["names"]),
            np.asarray(data["mean"]),
            np.asarray(data["std"]),
            np.asarray(data["coef"]),
            float(data["intercept"]),
            float(data["C"]),
        )


def fit_logistic(
    x: np.ndarray, y: np.ndarray, c: float, names: tuple[str, ...] = LOGIT_FEATURES
) -> Logistic:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    z = (x - mean) / std
    y = np.asarray(y, dtype=np.float64)
    n, d = z.shape
    penalty = 1.0 / (c * n)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        b, w = theta[0], theta[1:]
        margin = z @ w + b
        loss = np.mean(np.logaddexp(0.0, margin) - y * margin) + 0.5 * penalty * w @ w
        residual = (_sigmoid(margin) - y) / n
        grad = np.concatenate([[residual.sum()], z.T @ residual + penalty * w])
        return float(loss), grad

    start = np.zeros(d + 1)
    rate = float(np.clip(y.mean(), EPS, 1 - EPS))
    start[0] = math.log(rate / (1 - rate))
    result = minimize(
        objective, start, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-9}
    )
    if not result.success:
        raise RuntimeError(f"로지스틱 회귀가 수렴하지 않음: {result.message}")
    return Logistic(names, mean, std, result.x[1:], float(result.x[0]), float(c))


def forward_folds(
    day: np.ndarray, first_validation: int = 3
) -> list[tuple[np.ndarray, np.ndarray]]:
    """날짜순 (학습 마스크, 검증 마스크). 개발 6일이면 4·5·6번째 날을 차례로 검증."""
    days = np.unique(day)
    return [(day < days[k], day == days[k]) for k in range(first_validation, len(days))]


def select_logistic(x, y, day, grid=LOGIT_C_GRID) -> dict:
    """C마다 교차검증 검증 log loss(겹 평균). 가장 낮은 C(같으면 작은 C)와 검증 예측."""
    folds = forward_folds(day)
    table, oof = {}, {}
    for c in grid:
        losses, pred = [], np.full(len(y), np.nan)
        for train, valid in folds:
            model = fit_logistic(x[train], y[train], c)
            pred[valid] = model.predict(x[valid])
            losses.append(log_loss(pred[valid], y[valid]))
        table[str(c)] = {"fold_log_loss": losses, "mean_log_loss": float(np.mean(losses))}
        oof[c] = pred
    best = min(grid, key=lambda c: (table[str(c)]["mean_log_loss"], c))
    return {"C": best, "cv": table, "oof": oof[best], "folds": len(folds)}


def _lgbm_curve(x_train, y_train, x_valid, y_valid, leaves: int) -> np.ndarray:
    record: dict = {}
    params = {**LGBM_PARAMS, "num_leaves": leaves, "metric": "binary_logloss"}
    train = lgb.Dataset(x_train, y_train, free_raw_data=False)
    valid = lgb.Dataset(x_valid, y_valid, reference=train, free_raw_data=False)
    lgb.train(
        params,
        train,
        num_boost_round=LGBM_MAX_ROUNDS,
        valid_sets=[valid],
        valid_names=["valid"],
        callbacks=[lgb.record_evaluation(record)],
    )
    return np.asarray(record["valid"]["binary_logloss"])


def select_lgbm(x, y, day, grid=LGBM_LEAVES_GRID) -> dict:
    """잎 수마다 라운드별 교차검증 log loss(겹 평균)의 최소. 가장 낮은 (잎, 라운드)."""
    folds = forward_folds(day)
    table, best = {}, None
    for leaves in grid:
        curves = [_lgbm_curve(x[t], y[t], x[v], y[v], leaves) for t, v in folds]
        mean = np.mean(curves, axis=0)
        rounds = int(np.argmin(mean)) + 1
        table[str(leaves)] = {"rounds": rounds, "mean_log_loss": float(mean[rounds - 1])}
        key = (float(mean[rounds - 1]), leaves)
        if best is None or key < best[0]:
            best = (key, leaves, rounds)
    _, leaves, rounds = best
    oof = np.full(len(y), np.nan)
    for train, valid in folds:
        model = fit_lgbm(x[train], y[train], leaves, rounds)
        oof[valid] = model.predict(x[valid])
    return {"num_leaves": leaves, "rounds": rounds, "cv": table, "oof": oof}


def fit_lgbm(x: np.ndarray, y: np.ndarray, leaves: int, rounds: int) -> lgb.Booster:
    params = {**LGBM_PARAMS, "num_leaves": leaves}
    return lgb.train(params, lgb.Dataset(x, y), num_boost_round=rounds)
