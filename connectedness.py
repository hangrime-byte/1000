"""
Diebold-Yilmaz 동적 연결성 분석 모듈

Diebold & Yilmaz (2012, 2014) GFEVD 기반 연결성 측정
TVP-VAR (Antonakakis et al., 2020)과 결합하여 동적 연결성 산출

References:
- Diebold & Yilmaz (2012), "Better to give than to receive: Predictive
  directional measurement of volatility spillovers", IJOF 28(1), 57-66.
- Diebold & Yilmaz (2014), "On the network topology of variance
  decompositions", Journal of Econometrics, 182(1), 119-134.
- Pesaran & Shin (1998), "Generalized impulse response analysis in
  linear multivariate models", Economics Letters, 58(1), 17-29.
"""

import numpy as np
import pandas as pd
from tvp_var import TVPVAR


def compute_gfevd(Psi, Sigma, H):
    """일반화 예측오차 분산분해 (GFEVD)를 계산한다.

    Pesaran & Shin (1998) 방법에 기반하며, 변수 순서에 영향을 받지 않는다.

    θ_ij(H) = σ_jj^{-1} * Σ_{h=0}^{H-1} (e_i' Ψ_h Σ e_j)^2
              / Σ_{h=0}^{H-1} (e_i' Ψ_h Σ Ψ_h' e_i)

    정규화:
    θ̃_ij(H) = θ_ij(H) / Σ_j θ_ij(H)

    Args:
        Psi: list of H matrices (N×N), VMA 계수
        Sigma: N×N 오차 공분산 행렬
        H: 예측 수평선

    Returns:
        theta_norm: N×N 정규화된 GFEVD 행렬 (행합 = 1)
    """
    N = Sigma.shape[0]
    sigma_diag = np.diag(Sigma)  # σ_jj

    # 분자: σ_jj^{-1} * Σ_h (e_i' Ψ_h Σ e_j)^2   ← 각 h에서 제곱 후 합산
    numerator = np.zeros((N, N))
    # 분모: Σ_h (e_i' Ψ_h Σ Ψ_h' e_i)
    denominator = np.zeros(N)

    for h in range(H):
        Psi_h = Psi[h]

        for i in range(N):
            denominator[i] += Psi_h[i, :] @ Sigma @ Psi_h[i, :]
            for j in range(N):
                # 각 h-step 기여를 제곱하여 합산 (Pesaran & Shin, 1998)
                val = Psi_h[i, :] @ Sigma[:, j]
                numerator[i, j] += val ** 2

    # θ_ij(H) = σ_jj^{-1} * numerator_ij / denominator_i
    theta = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            theta[i, j] = numerator[i, j] / (sigma_diag[j] * denominator[i])

    # 정규화: 행합 = 1
    row_sums = theta.sum(axis=1, keepdims=True)
    theta_norm = theta / row_sums

    return theta_norm


def compute_connectedness_measures(theta_norm):
    """GFEVD 행렬로부터 연결성 지표를 계산한다.

    Args:
        theta_norm: N×N 정규화된 GFEVD 행렬

    Returns:
        dict: 연결성 지표 딕셔너리
            - TCI: Total Connectedness Index
            - FROM: 각 변수로의 타 변수 기여 (N,)
            - TO: 각 변수에서 타 변수로의 기여 (N,)
            - NET: 순연결성 = TO - FROM (N,)
            - NPDC: N×N 순쌍별 방향 연결성
    """
    N = theta_norm.shape[0]

    # 대각 원소 제거 (자기 자신에 대한 기여)
    off_diag = theta_norm.copy()
    np.fill_diagonal(off_diag, 0)

    # FROM: 변수 i가 다른 변수들로부터 받는 연결성
    FROM = off_diag.sum(axis=1)  # 행합 (대각 제외)

    # TO: 변수 j가 다른 변수들에게 주는 연결성
    TO = off_diag.sum(axis=0)  # 열합 (대각 제외)

    # NET: 순방향 연결성
    NET = TO - FROM

    # TCI: Total Connectedness Index = (1/N) * Σ FROM_i
    TCI = FROM.sum() / N

    # NPDC: Net Pairwise Directional Connectedness
    NPDC = theta_norm - theta_norm.T

    return {
        "TCI": TCI,
        "FROM": FROM,
        "TO": TO,
        "NET": NET,
        "NPDC": NPDC,
        "GFEVD": theta_norm,
    }


def dynamic_connectedness(data, nlag=1, nfore=10, kappa1=0.99, kappa2=0.96):
    """TVP-VAR 기반 동적 연결성을 계산한다.

    Args:
        data: T×N DataFrame 또는 numpy array
        nlag: VAR 시차
        nfore: GFEVD 예측 수평선
        kappa1: forgetting factor (계수)
        kappa2: decay factor (공분산)

    Returns:
        results: dict
            - TCI: T×1 시계열 (Total Connectedness Index)
            - FROM: T×N 시계열
            - TO: T×N 시계열
            - NET: T×N 시계열
            - NPDC: T×N×N 시계열
            - GFEVD_avg: N×N 평균 GFEVD 테이블
            - model: 추정된 TVP-VAR 모델
    """
    if isinstance(data, pd.DataFrame):
        columns = data.columns.tolist()
        index = data.index[nlag:]
        data_np = data.values
    else:
        columns = [f"Var{i+1}" for i in range(data.shape[1])]
        index = np.arange(data.shape[0] - nlag)
        data_np = data

    # 1. TVP-VAR 추정
    model = TVPVAR(nlag=nlag, kappa1=kappa1, kappa2=kappa2)
    model.fit(data_np)

    T = model.T
    N = model.N

    # 2. 각 시점에서 GFEVD 및 연결성 계산
    TCI_series = np.zeros(T)
    FROM_series = np.zeros((T, N))
    TO_series = np.zeros((T, N))
    NET_series = np.zeros((T, N))
    NPDC_series = np.zeros((T, N, N))
    GFEVD_sum = np.zeros((N, N))

    for t in range(T):
        Psi = model.compute_vma_coefficients(t, nfore)
        Sigma_t = model.Sigma[t]

        # GFEVD 계산
        theta_norm = compute_gfevd(Psi, Sigma_t, nfore)

        # 연결성 지표
        measures = compute_connectedness_measures(theta_norm)

        TCI_series[t] = measures["TCI"]
        FROM_series[t] = measures["FROM"]
        TO_series[t] = measures["TO"]
        NET_series[t] = measures["NET"]
        NPDC_series[t] = measures["NPDC"]
        GFEVD_sum += theta_norm

    # 평균 GFEVD 테이블
    GFEVD_avg = GFEVD_sum / T * 100  # 퍼센트 단위

    # DataFrame으로 변환
    TCI_df = pd.Series(TCI_series * 100, index=index, name="TCI")
    FROM_df = pd.DataFrame(FROM_series * 100, index=index, columns=columns)
    TO_df = pd.DataFrame(TO_series * 100, index=index, columns=columns)
    NET_df = pd.DataFrame(NET_series * 100, index=index, columns=columns)

    GFEVD_table = pd.DataFrame(GFEVD_avg, index=columns, columns=columns)

    # FROM/TO/NET 행 추가 (단순화)
    diag = np.diag(GFEVD_avg)
    from_values = GFEVD_avg.sum(axis=1) - diag       # 행합 - 대각
    to_values = GFEVD_avg.sum(axis=0) - diag          # 열합 - 대각
    net_values = to_values - from_values               # NET = TO - FROM
    tci = from_values.sum() / N                        # TCI

    gfevd_with_to = pd.DataFrame(GFEVD_avg, index=columns, columns=columns)
    gfevd_with_to["FROM"] = from_values
    to_row = pd.Series(np.append(to_values, tci), index=list(columns) + ["FROM"])
    net_row = pd.Series(np.append(net_values, np.nan), index=list(columns) + ["FROM"])
    gfevd_with_to.loc["TO"] = to_row
    gfevd_with_to.loc["NET"] = net_row

    results = {
        "TCI": TCI_df,
        "FROM": FROM_df,
        "TO": TO_df,
        "NET": NET_df,
        "NPDC": NPDC_series,
        "GFEVD_table": gfevd_with_to,
        "model": model,
        "index": index,
        "columns": columns,
    }

    avg_tci = TCI_df.mean()
    print(f"\nAverage Total Connectedness Index (TCI): {avg_tci:.2f}%")
    print("\n--- Average Connectedness Table ---")
    print(gfevd_with_to.round(2))

    return results


if __name__ == "__main__":
    np.random.seed(42)
    T, N = 500, 5
    data = np.random.randn(T, N) * 0.5
    # 약간의 상관관계 부여
    data[:, 1] += 0.3 * data[:, 0]
    data[:, 2] += 0.2 * data[:, 0] + 0.2 * data[:, 1]

    results = dynamic_connectedness(data, nlag=1, nfore=10)
