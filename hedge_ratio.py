"""
헤지비율 및 최적 포트폴리오 투자비중 모듈

1. Kroner & Sultan (1993) 최적 헤지비율
2. Kroner & Ng (1998) 최적 포트폴리오 가중치
3. Ederington (1979) 헤징 효율성

References:
- Kroner & Sultan (1993), "Time-Varying Distributions and Dynamic Hedging
  with Foreign Currency Futures", JFQA, 28(4), 535-551.
- Kroner & Ng (1998), "Modeling Asymmetric Comovements of Asset Returns",
  Review of Financial Studies, 11(4), 817-844.
- Ederington (1979), "The Hedging Performance of the New Futures Markets",
  Journal of Finance, 34(1), 157-170.
"""

import numpy as np
import pandas as pd


def compute_hedge_ratios(Sigma_series, columns=None):
    """TVP-VAR에서 추정된 시변 공분산으로 Kroner & Sultan (1993) 헤지비율을 계산한다.

    β_ij,t = h_ij,t / h_jj,t

    자산 i의 1달러 롱 포지션을 헤지하기 위해 자산 j에서 β 달러만큼 숏 포지션

    Args:
        Sigma_series: T × N × N 시변 공분산 행렬 배열
        columns: 변수 이름 리스트

    Returns:
        hedge_ratios: dict of DataFrames, 키 = "i_j" (자산i를 자산j로 헤지)
        hr_summary: 요약 통계표
    """
    T, N, _ = Sigma_series.shape
    if columns is None:
        columns = [f"Var{i+1}" for i in range(N)]

    hedge_ratios = {}
    hr_means = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # β_ij,t = h_ij,t / h_jj,t
            h_ij = Sigma_series[:, i, j]
            h_jj = Sigma_series[:, j, j]
            beta_t = h_ij / (h_jj + 1e-10)
            key = f"{columns[i]}_{columns[j]}"
            hedge_ratios[key] = beta_t
            hr_means[i, j] = np.mean(beta_t)

    # 요약 테이블
    hr_summary = pd.DataFrame(hr_means, index=columns, columns=columns)
    for i in range(N):
        hr_summary.iloc[i, i] = np.nan

    return hedge_ratios, hr_summary


def compute_portfolio_weights(Sigma_series, columns=None):
    """Kroner & Ng (1998) 최적 포트폴리오 투자비중을 계산한다.

    w_ij,t = (h_jj,t - h_ij,t) / (h_ii,t - 2*h_ij,t + h_jj,t)

    제약 조건: 0 ≤ w_ij,t ≤ 1
    자산i와 자산j의 2자산 포트폴리오에서 자산i의 비중

    Args:
        Sigma_series: T × N × N 시변 공분산 행렬 배열
        columns: 변수 이름 리스트

    Returns:
        weights: dict of arrays
        weight_summary: 요약 통계표
    """
    T, N, _ = Sigma_series.shape
    if columns is None:
        columns = [f"Var{i+1}" for i in range(N)]

    weights = {}
    wt_means = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i >= j:
                continue
            h_ii = Sigma_series[:, i, i]
            h_jj = Sigma_series[:, j, j]
            h_ij = Sigma_series[:, i, j]

            denom = h_ii - 2 * h_ij + h_jj
            w_t = (h_jj - h_ij) / (denom + 1e-10)

            # 제약: 0 ≤ w ≤ 1
            w_t = np.clip(w_t, 0, 1)

            key = f"{columns[i]}_{columns[j]}"
            weights[key] = w_t
            wt_means[i, j] = np.mean(w_t)
            wt_means[j, i] = 1 - np.mean(w_t)

    weight_summary = pd.DataFrame(wt_means, index=columns, columns=columns)
    for i in range(N):
        weight_summary.iloc[i, i] = np.nan

    return weights, weight_summary


def compute_hedging_effectiveness(returns, hedge_ratios, columns=None):
    """Ederington (1979) 헤징 효율성을 계산한다.

    HE = 1 - Var(hedged) / Var(unhedged)

    헤지된 포트폴리오 수익률: r_hedged = r_i - β_ij * r_j
    헤지되지 않은 포트폴리오: r_unhedged = r_i

    Args:
        returns: T × N 수익률 DataFrame
        hedge_ratios: dict of hedge ratio arrays
        columns: 변수 이름

    Returns:
        he_table: 헤징 효율성 테이블
    """
    if columns is None:
        columns = returns.columns.tolist() if isinstance(returns, pd.DataFrame) else \
            [f"Var{i+1}" for i in range(returns.shape[1])]

    returns_np = returns.values if isinstance(returns, pd.DataFrame) else returns
    N = returns_np.shape[1]
    T_hr = len(next(iter(hedge_ratios.values())))
    # 수익률 길이를 헤지비율 길이에 맞춤
    returns_aligned = returns_np[-T_hr:]

    he_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            key = f"{columns[i]}_{columns[j]}"
            if key not in hedge_ratios:
                continue

            beta_t = hedge_ratios[key]
            r_i = returns_aligned[:, i]
            r_j = returns_aligned[:, j]

            # 헤지된 수익률
            r_hedged = r_i - beta_t * r_j

            var_unhedged = np.var(r_i)
            var_hedged = np.var(r_hedged)

            he = 1 - var_hedged / (var_unhedged + 1e-10)
            he_matrix[i, j] = he * 100  # 퍼센트

    he_table = pd.DataFrame(he_matrix, index=columns, columns=columns)
    for i in range(N):
        he_table.iloc[i, i] = np.nan

    return he_table


def compute_portfolio_hedging_effectiveness(returns, weights, columns=None):
    """포트폴리오 가중치 기반 헤징 효율성을 계산한다.

    포트폴리오 수익률: r_p = w_ij * r_i + (1 - w_ij) * r_j
    HE = 1 - Var(r_p) / Var(r_i)

    Args:
        returns: T × N 수익률 DataFrame
        weights: dict of weight arrays
        columns: 변수 이름

    Returns:
        he_table: 포트폴리오 헤징 효율성 테이블
    """
    if columns is None:
        columns = returns.columns.tolist() if isinstance(returns, pd.DataFrame) else \
            [f"Var{i+1}" for i in range(returns.shape[1])]

    returns_np = returns.values if isinstance(returns, pd.DataFrame) else returns
    N = returns_np.shape[1]
    T_w = len(next(iter(weights.values())))
    returns_aligned = returns_np[-T_w:]

    he_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            key = f"{columns[i]}_{columns[j]}"
            if key not in weights:
                continue

            w_t = weights[key]
            r_i = returns_aligned[:, i]
            r_j = returns_aligned[:, j]

            r_portfolio = w_t * r_i + (1 - w_t) * r_j

            var_i = np.var(r_i)
            var_j = np.var(r_j)
            var_p = np.var(r_portfolio)

            he_i = (1 - var_p / (var_i + 1e-10)) * 100
            he_j = (1 - var_p / (var_j + 1e-10)) * 100

            he_matrix[i, j] = he_i
            he_matrix[j, i] = he_j

    he_table = pd.DataFrame(he_matrix, index=columns, columns=columns)
    for i in range(N):
        he_table.iloc[i, i] = np.nan

    return he_table


if __name__ == "__main__":
    np.random.seed(42)
    T, N = 500, 3
    columns = ["A", "B", "C"]

    # 가상 공분산
    Sigma = np.zeros((T, N, N))
    for t in range(T):
        A = np.random.randn(N, N) * 0.1
        Sigma[t] = A @ A.T + np.eye(N) * 0.5

    hr, hr_summary = compute_hedge_ratios(Sigma, columns)
    print("Hedge Ratio Summary:")
    print(hr_summary.round(4))

    w, w_summary = compute_portfolio_weights(Sigma, columns)
    print("\nPortfolio Weight Summary:")
    print(w_summary.round(4))
