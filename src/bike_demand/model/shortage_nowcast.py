"""v5(T61) nowcast 부족 분류기: 스냅샷 변화·프로파일·예상 대여로 3시간 안 0대 도달 확률을 낸다.

계획은 docs/experiments.md v5 절. 주 모델은 L2 로지스틱 회귀(판정 대상), 보조는 얕은 LightGBM
(보고만). 설정은 개발 6일의 날짜순 3겹 교차검증(앞선 날로 학습, 다음 날 검증)으로 고르고, 고른
설정으로 개발 6일 전체에 다시 학습해 고정한다. scikit-learn 없이 SciPy L-BFGS로 푼다.
"""

# [파일 역할] v5 부족 경보 측정(docs/experiments.md v5)의 후보 N, 즉 'as_of 시점에 알 수 있는
# 값만으로 앞으로 3시간 안에 대여소가 0대가 될 확률'을 내는 분류기를 만든다. 특징 행렬 만들기,
# 로지스틱 회귀 학습, 날짜순 교차검증으로 설정 고르기가 들어 있다.
# [왜 따로 두었나] 정답·후보 점수·지표·부트스트랩은 shortage_eval.py가 맡고, 이 파일은 학습되는
# 후보(N과 보조 LightGBM)만 맡는다. shortage_eval.py는 dev 실행에서 설정을 고르고 고정
# 파일(dev_fit.json·dev_lgbm.txt)을 쓰며, test 실행에서 그 고정값으로 점수만 다시 낸다.
# [설계 이유] scikit-learn을 새 의존성으로 넣지 않고 이미 쓰는 SciPy의 L-BFGS로 로지스틱 회귀를 직접
# 푼다(모듈 docstring). 목적 함수가 짧아 계수·표준화 값을 그대로 JSON으로 저장·복원하기도 쉽다.
# [흐름] logit_matrix/lgbm_matrix로 특징을 만든다 → select_logistic/select_lgbm이 forward_folds(앞선
# 날로 학습, 다음 날 검증)로 C와 (잎 수, 라운드)를 고른다 → fit_logistic/fit_lgbm으로 개발 6일
# 전체에 다시 학습한다.

from __future__ import annotations

import math
from dataclasses import dataclass

# lightgbm: 보고만 하는 보조 분류기(얕은 트리). numpy: 대여소-발표 행 전체를 배열로 한 번에
# 계산한다. scipy.optimize.minimize: L2 로지스틱 회귀의 목적 함수를 L-BFGS-B로 최소화한다.
import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize

# shortage_eval과 같은 상수를 써서 두 파일의 기준이 어긋나지 않게 한다. EPS는 log 계산의 0·1 확률을
# 막는 작은 값, ISSUE_HOURS는 단기예보 발표 시각 8개(02·05·…·23시), MAIN_HOURS는 주 평가 창 3시간,
# log_loss는 교차검증 점수에 쓰는 공통 함수다.
from bike_demand.model.shortage_eval import EPS, ISSUE_HOURS, MAIN_HOURS, log_loss

# 계획에 적은 탐색 범위(docs/experiments.md v5 'nowcast 분류기 N'). C는 L2 벌점의 역수라 작을수록
# 계수를 강하게 0 쪽으로 누른다. 보조 LightGBM은 잎 7·15개, 라운드 1~500 중에서 고른다.
LOGIT_C_GRID = (0.01, 0.1, 1.0, 10.0)
LGBM_LEAVES_GRID = (7, 15)
LGBM_MAX_ROUNDS = 500
# 보조 LightGBM의 고정 설정. learning_rate 0.05와 min_data_in_leaf 200은 계획에 적은 값이다.
# feature_fraction·bagging_fraction을 1.0으로 두고 deterministic·seed·num_threads를 고정해 같은
# 자료에서 다시 학습해도 같은 모델이 나오게 한다.
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
# 발표 시각 8개를 one-hot으로 넣되 첫 시각(02시)은 빼서 절편과 겹치지 않게 한다(다중공선성 방지).
# 결과는 hour_05 ~ hour_23 일곱 열이다.
HOUR_DUMMIES = tuple(f"hour_{h:02d}" for h in ISSUE_HOURS[1:])
# 로지스틱 회귀 특징 순서. 계획의 목록과 같다: 지금 자전거 수 b0의 변환(log_b0, inv_b0), 30·60분 전
# 대비 변화(d30, d60)와 그 빈 값 표시(miss30, miss60), 예상 대여 E·작년 같은 시기 대여 R·반납 Q의
# log, '자전거가 순유출로 바닥나는 정도'를 표준화한 z, 쉬는 날 여부, 발표 시각. 이 순서가 저장된
# 계수의 순서이므로 바꾸면 dev_fit.json과 맞지 않게 된다.
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
# 보조 LightGBM 특징. 트리는 단조 변환이 필요 없어 원값(b0, E, R, Q)을 그대로 넣고, 발표 시각도 숫자
# 하나로 넣는다.
LGBM_FEATURES = ("b0", "d30", "d60", "e", "r", "q", "issue_hour", "offday")


# 행 사전(shortage_eval.build_rows가 만든 열 배열들)에서 mask로 고른 행의 로지스틱 특징 행렬을
# 만든다. 출력은 (행 수, LOGIT_FEATURES 개수) 배열.
# h는 창 길이(시간)로, E{h}·R{h}·Q{h} 열을 고른다. 기본은 주 평가 창 3시간이다.
# 호출 위치: shortage_eval의 dev 실행(분류기 학습)과 test 실행(고정 계수로 점수 계산),
# tests/test_model_shortage_nowcast.py.
def logit_matrix(rows: dict, mask: np.ndarray, h: int = MAIN_HOURS) -> np.ndarray:
    """로지스틱 회귀 특징(LOGIT_FEATURES 순서). b0 ≥ 1 행만 받는다."""
    b0 = rows["b0"][mask].astype(np.float64)
    # 이미 0대인 대여소(b0 = 0)는 주 평가에서 빼는 행이고, inv_b0 = 1/b0이 무한대가 되므로 여기서
    # 받지 않는다.
    if (b0 < 1).any():
        raise ValueError("분류기는 b0 ≥ 1 행만")
    e, r, q = rows[f"E{h}"][mask], rows[f"R{h}"][mask], rows[f"Q{h}"][mask]
    hour = rows["issue_hour"][mask]
    columns = {
        # b0는 log와 역수 두 가지로 넣어, 자전거가 1~2대일 때 위험이 급히 커지는 비선형 모양을 선형
        # 모델이 따라갈 수 있게 한다.
        "log_b0": np.log1p(b0),
        "inv_b0": 1.0 / b0,
        "d30": rows["d30"][mask],
        "d60": rows["d60"][mask],
        "miss30": rows["miss30"][mask],
        "miss60": rows["miss60"][mask],
        # 대여·반납 기대값은 대여소마다 크기 차이가 커서 log1p로 눌러 넣는다(0도 받을 수 있게
        # log1p).
        "log_e": np.log1p(e),
        "log_r": np.log1p(r),
        "log_q": np.log1p(q),
        # z: 순유출 기대값(E − Q)이 지금 자전거 수를 넘는 정도를 표준편차(√(E + Q), Poisson 두 개의
        # 차라고 볼 때)로 나눈 값. −0.5는 정수 개수에 대한 연속성 보정, +0.1은 E·Q가 모두 0일 때
        # 0으로 나누지 않으려는 것이다. 계획의 정의와 같다.
        "z": (b0 - 0.5 - (e - q)) / np.sqrt(e + q + 0.1),
        "offday": rows["offday"][mask].astype(np.float64),
        # 열 이름 끝 두 자리(예: hour_05 → 5)와 발표 시각이 같으면 1인 더미 열.
        **{name: (hour == int(name[-2:])).astype(np.float64) for name in HOUR_DUMMIES},
    }
    return np.column_stack([columns[name] for name in LOGIT_FEATURES])


# 보조 LightGBM의 특징 행렬. 로지스틱과 달리 b0 = 0 행도 막지 않는다(호출하는 쪽이 같은 주 평가
# mask를 넘긴다).
# 호출 위치: shortage_eval의 dev·test 실행, tests/test_model_shortage_nowcast.py.
def lgbm_matrix(rows: dict, mask: np.ndarray, h: int = MAIN_HOURS) -> np.ndarray:
    """LightGBM 특징(LGBM_FEATURES 순서). 스냅샷 변화가 없으면 결측."""
    columns = {
        "b0": rows["b0"][mask].astype(np.float64),
        # 로지스틱에서는 빈 변화를 0과 빈 값 표시로 나눠 넣지만, LightGBM은 결측을 스스로 처리하므로
        # NaN으로 바꿔 넣는다(계획: '빈 값은 결측').
        "d30": np.where(rows["miss30"][mask] > 0, np.nan, rows["d30"][mask]),
        "d60": np.where(rows["miss60"][mask] > 0, np.nan, rows["d60"][mask]),
        "e": rows[f"E{h}"][mask],
        "r": rows[f"R{h}"][mask],
        "q": rows[f"Q{h}"][mask],
        "issue_hour": rows["issue_hour"][mask].astype(np.float64),
        "offday": rows["offday"][mask].astype(np.float64),
    }
    return np.column_stack([columns[name] for name in LGBM_FEATURES])


# 1 / (1 + e^(−z))를 exp(−log(1 + e^(−z)))로 계산한다. logaddexp를 쓰면 |z|가 아주 커도 overflow
# 없이 안정적으로 값이 나온다.
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -z))


# 학습된 로지스틱 회귀 하나. 표준화에 쓴 평균·표준편차와 계수·절편·C를 함께 들고 있어서, 새 행을
# 같은 척도로 바꾼 뒤 확률을 낸다.
# 왜 표준화 값을 모델에 넣나: test 실행은 dev 때 계산한 평균·표준편차를 그대로 써야 하므로(test
# 자료로 다시 계산하면 계수의 뜻이 바뀜) 함께 저장·복원한다.
@dataclass
class Logistic:
    """표준화 + L2 로지스틱 회귀. 목적: 평균 log loss + |w|² / (2·C·n) (절편 벌점 없음)."""

    names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float
    c: float

    # 특징 행렬을 받아 0~1 확률을 돌려준다.
    def predict(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(((x - self.mean) / self.std) @ self.coef + self.intercept)

    # dev_fit.json의 logistic 항목으로 쓰는 JSON 직렬화. from_dict와 짝이다.
    def to_dict(self) -> dict:
        return {
            "names": list(self.names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "C": self.c,
        }

    # dev_fit.json에서 고정 계수를 읽어 모델을 되살린다. 호출 위치: shortage_eval의 test 실행.
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


# 특징 행렬 x와 0/1 정답 y로 L2 로지스틱 회귀를 학습한다. c는 벌점의 역수(클수록 벌점 약함), names는
# 저장용 특징 이름.
# 목적 함수: 평균 log loss + |w|² / (2·C·n). 벌점을 n으로 나눠 행 수가 달라도 C의 뜻이 같게
# 하고(scikit-learn의 C 정의와 같은 꼴), 절편에는 벌점을 주지 않는다.
# 수렴하지 않으면 RuntimeError를 내서, 덜 풀린 계수가 고정값으로 저장되는 일을 막는다.
def fit_logistic(
    x: np.ndarray, y: np.ndarray, c: float, names: tuple[str, ...] = LOGIT_FEATURES
) -> Logistic:
    # 특징을 학습 자료의 평균·표준편차로 표준화한다(계획). 값이 하나뿐인 열(예: 한 발표 시각만 있는
    # 자료의 더미)은 표준편차 0이라 1로 바꿔 0으로 나누지 않게 한다.
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    z = (x - mean) / std
    y = np.asarray(y, dtype=np.float64)
    n, d = z.shape
    penalty = 1.0 / (c * n)

    # theta = [절편, 계수…]. 손실과 기울기를 함께 돌려줘 L-BFGS가 수치 미분 없이 정확한 기울기를
    # 쓴다(jac=True).
    # log(1 + e^m) − y·m은 로지스틱 log loss를 margin으로 쓴 꼴이고, logaddexp로 overflow를 막는다.
    # 기울기는 (σ(m) − y)/n을 절편과 계수 방향으로 모은 것에 벌점 기울기 penalty·w를 더한다.
    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        b, w = theta[0], theta[1:]
        margin = z @ w + b
        loss = np.mean(np.logaddexp(0.0, margin) - y * margin) + 0.5 * penalty * w @ w
        residual = (_sigmoid(margin) - y) / n
        grad = np.concatenate([[residual.sum()], z.T @ residual + penalty * w])
        return float(loss), grad

    # 시작점: 계수 0, 절편은 양성 비율의 logit. 처음부터 평균 확률을 맞춘 상태에서 출발해 수렴이
    # 빠르다. EPS로 잘라 양성 비율이 0이나 1이어도 log가 무한대가 되지 않게 한다.
    start = np.zeros(d + 1)
    rate = float(np.clip(y.mean(), EPS, 1 - EPS))
    start[0] = math.log(rate / (1 - rate))
    # 기울기 허용 오차를 작게(1e-9), 반복 상한을 넉넉히(5,000) 둬서 최적점 가까이까지 푼다.
    result = minimize(
        objective, start, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-9}
    )
    if not result.success:
        raise RuntimeError(f"로지스틱 회귀가 수렴하지 않음: {result.message}")
    return Logistic(names, mean, std, result.x[1:], float(result.x[0]), float(c))


# 날짜 서수 배열(행마다 as_of의 KST 날짜)로 시간 순서 교차검증의 (학습, 검증) 불리언 마스크 목록을
# 만든다. k번째 날을 검증할 때 그보다 앞선 날만 학습에 쓴다.
# 왜 날짜순인가: 무작위로 나누면 같은 날의 다른 발표가 학습과 검증에 함께 들어가 날씨·요일 같은
# 그날의 공통 상황이 새어 검증이 낙관적이 된다. 실제 운영처럼 '과거로 학습해 다음 날을 맞히는' 꼴로
# 잰다.
# first_validation = 3이면 처음 세 날은 학습만 하고 4·5·6번째 날을 차례로 검증한다(계획의 3겹).
def forward_folds(
    day: np.ndarray, first_validation: int = 3
) -> list[tuple[np.ndarray, np.ndarray]]:
    """날짜순 (학습 마스크, 검증 마스크). 개발 6일이면 4·5·6번째 날을 차례로 검증."""
    days = np.unique(day)
    return [(day < days[k], day == days[k]) for k in range(first_validation, len(days))]


# C 후보마다 forward_folds의 각 겹에서 학습·검증해 검증 log loss를 겹 평균으로 비교하고, 가장 낮은
# C를 고른다. 같으면 더 작은 C(더 강한 벌점, 더 단순한 모델)를 고른다.
# 돌려주는 것: 고른 C, C별 겹 손실 표(dev_fit에 남길 근거), 고른 C의 검증 예측(oof: 검증 날 행에만
# 값, 나머지는 NaN), 겹 수. oof는 dev 보고서의 신뢰도 곡선(보정)에 쓴다.
# 호출 위치: shortage_eval의 dev 실행.
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
    # 정렬 키 (평균 손실, C): 손실이 같으면 작은 C가 앞선다.
    best = min(grid, key=lambda c: (table[str(c)]["mean_log_loss"], c))
    return {"C": best, "cv": table, "oof": oof[best], "folds": len(folds)}


# 한 겹에서 LightGBM을 상한(500) 라운드까지 학습하며 라운드마다 검증 log loss를 기록한 곡선을
# 돌려준다. 조기 종료를 쓰지 않고 곡선 전체를 남기는 이유는, 겹마다 곡선을 평균한 뒤 그 평균에서
# 가장 낮은 라운드를 고르기 위해서다(겹마다 따로 멈추면 라운드 수가 겹마다 달라 하나로 고정하기
# 어렵다).
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


# 잎 수 후보마다 겹별 곡선의 평균에서 최소 라운드를 찾고, 그 손실이 가장 낮은 (잎 수, 라운드)를
# 고른다. 같으면 작은 잎 수가 앞선다.
# 고른 설정으로 겹마다 다시 학습해 검증 예측(oof)도 만든다(신뢰도 곡선용). 호출 위치:
# shortage_eval의 dev 실행.
# 주의: 고른 라운드가 상한 500과 같으면 더 학습할 여지가 있다는 뜻이다. v5 개발 실행에서 실제로
# 500에 닿았고, 보고만 하는 모델이라 그대로 고정했다(docs/experiments.md v5 고정값 절).
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


# 고른 잎 수와 라운드로 조기 종료 없이 정확히 rounds번 학습한다. dev 실행은 이 모델을 dev_lgbm.txt로
# 저장하고 해시를 고정값으로 남긴다.
def fit_lgbm(x: np.ndarray, y: np.ndarray, leaves: int, rounds: int) -> lgb.Booster:
    params = {**LGBM_PARAMS, "num_leaves": leaves}
    return lgb.train(params, lgb.Dataset(x, y), num_boost_round=rounds)
