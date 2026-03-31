"""
포트폴리오 최적화 모듈

1. MVP (Minimum Variance Portfolio) - 최소분산 포트폴리오
2. MCP (Minimum Correlation Portfolio) - 최소상관 포트폴리오
3. MCoP (Minimum Connectedness Portfolio) - 최소연결성 포트폴리오

References:
- Markowitz (1952), "Portfolio Selection", Journal of Finance.
- Broadstock, Chatziantoniou & Gabauer (2022), "Minimum connectedness
  portfolios and the market for green bonds", Finance Research Letters.
- Christoffersen et al. (2014), "Correlation Structure, pp. 268-288.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def minimum_variance_portfolio(Sigma):
    """최소분산 포트폴리오 (MVP) 가중치를 계산한다.

    min  w' Σ w
    s.t. w'1 = 1, w_i >= 0

    Args:
        Sigma: N × N 공분산 행렬

    Returns:
        w: N × 1 최적 가중치
    """
    N = Sigma.shape[0]

    # 해석적 해 (공매도 허용 시): w = Σ^{-1} 1 / (1' Σ^{-1} 1)
    try:
        Sigma_inv = np.linalg.inv(Sigma + 1e-8 * np.eye(N))
        ones = np.ones(N)
        w = Sigma_inv @ ones / (ones @ Sigma_inv @ ones)

        # 음수 가중치가 있으면 제약 최적화로 전환
        if np.any(w < -0.01):
            raise ValueError("Negative weights")

        w = np.maximum(w, 0)
        w /= w.sum()
        return w
    except (np.linalg.LinAlgError, ValueError):
        pass

    # 수치 최적화 (공매도 금지)
    def objective(w):
        return w @ Sigma @ w

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * N
    w0 = np.ones(N) / N

    result = minimize(objective, w0, bounds=bounds, constraints=constraints,
                      method="SLSQP")
    return result.x


def minimum_correlation_portfolio(Sigma):
    """최소상관 포트폴리오 (MCP) 가중치를 계산한다.

    상관행렬 기반으로 MVP를 구한 뒤, 변동성으로 스케일링:
    1. D = diag(σ_1, ..., σ_N)
    2. R = D^{-1} Σ D^{-1} (상관행렬)
    3. w_R = MVP(R) (상관행렬의 최소분산)
    4. w_MCP = D^{-1} w_R / (1' D^{-1} w_R)

    Args:
        Sigma: N × N 공분산 행렬

    Returns:
        w: N × 1 최적 가중치
    """
    N = Sigma.shape[0]
    sigma = np.sqrt(np.diag(Sigma))
    sigma = np.maximum(sigma, 1e-10)

    D_inv = np.diag(1.0 / sigma)
    R = D_inv @ Sigma @ D_inv
    np.fill_diagonal(R, 1.0)

    # 상관행렬의 MVP
    w_R = minimum_variance_portfolio(R)

    # 변동성 역수로 스케일링
    w_scaled = D_inv @ w_R
    w = w_scaled / w_scaled.sum()
    w = np.maximum(w, 0)
    w /= w.sum()

    return w


def minimum_connectedness_portfolio(pci, method="Fisher"):
    """최소연결성 포트폴리오 (MCoP) 가중치를 계산한다.

    Broadstock et al. (2022) 방법론:
    GFEVD의 off-diagonal 쌍별 연결성(PCI)을 공분산 유사 행렬로 변환 후
    MVP와 동일 구조로 최적화. 연결성이 높은 자산의 비중을 줄인다.

    Args:
        pci: N × N 대칭 쌍별 연결성 행렬 (GFEVD off-diagonal, 대칭화됨)
        method: "Fisher" (Fisher 변환) 또는 "raw" (원래 값)

    Returns:
        w: N × 1 최적 가중치
    """
    N = pci.shape[0]
    C = pci.copy()

    # 대각 원소: 각 변수의 총 연결성 (FROM 값) → 자기 "분산" 역할
    row_sums = C.sum(axis=1)
    for i in range(N):
        C[i, i] = row_sums[i] if row_sums[i] > 0 else 1.0

    # 연결성 행렬을 직접 "공분산 유사 행렬"로 사용
    # Fisher 변환 없이 원시 연결성 값 사용 (inf 문제 회피)
    # 양정치 보장
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-6)
    C_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    C_pd = (C_pd + C_pd.T) / 2

    w = minimum_variance_portfolio(C_pd)
    return w


def compute_dynamic_portfolios(returns, Sigma_series, PCI_series=None, columns=None):
    """동적 포트폴리오 가중치 및 성과를 계산한다.

    Args:
        returns: T_total × N 수익률 배열
        Sigma_series: T × N × N 시변 공분산 행렬
        NPDC_series: T × N × N 순쌍별 연결성 행렬
        columns: 변수 이름

    Returns:
        results: dict with portfolio weights and performance
    """
    T_sigma = Sigma_series.shape[0]
    N = Sigma_series.shape[1]

    # PCI와 Sigma 길이가 다를 수 있음 (TVP-VAR vs DCC-GARCH)
    if PCI_series is not None:
        T_pci = PCI_series.shape[0]
        T = min(T_sigma, T_pci)
    else:
        T = T_sigma

    if columns is None:
        columns = [f"Var{i+1}" for i in range(N)]

    returns_np = returns.values if isinstance(returns, pd.DataFrame) else returns
    returns_aligned = returns_np[-T:]

    # Sigma와 PCI도 뒤에서 T개만 사용 (정렬)
    Sigma_aligned = Sigma_series[-T:]
    PCI_aligned = PCI_series[-T:] if PCI_series is not None else None

    # 가중치 저장
    w_mvp = np.zeros((T, N))
    w_mcp = np.zeros((T, N))
    w_mcop = np.zeros((T, N))
    w_equal = np.ones((T, N)) / N

    for t in range(T):
        Sigma_t = Sigma_aligned[t]

        # 양정치 보장
        eigvals, eigvecs = np.linalg.eigh(Sigma_t)
        eigvals = np.maximum(eigvals, 1e-8)
        Sigma_t = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # MVP
        w_mvp[t] = minimum_variance_portfolio(Sigma_t)

        # MCP
        w_mcp[t] = minimum_correlation_portfolio(Sigma_t)

        # MCoP
        if PCI_aligned is not None:
            pci_t = PCI_aligned[t]
            w_mcop[t] = minimum_connectedness_portfolio(pci_t, method="Fisher")
        else:
            w_mcop[t] = w_equal[t]

    # 포트폴리오 수익률 계산
    r_mvp = np.sum(w_mvp * returns_aligned, axis=1)
    r_mcp = np.sum(w_mcp * returns_aligned, axis=1)
    r_mcop = np.sum(w_mcop * returns_aligned, axis=1)
    r_equal = np.sum(w_equal * returns_aligned, axis=1)

    # 성과 지표
    def portfolio_stats(r, name):
        return {
            "Portfolio": name,
            "Mean Return (%)": np.mean(r),
            "Std Dev (%)": np.std(r),
            "Sharpe Ratio": np.mean(r) / (np.std(r) + 1e-10),
            "Min (%)": np.min(r),
            "Max (%)": np.max(r),
            "Skewness": float(pd.Series(r).skew()),
            "Kurtosis": float(pd.Series(r).kurtosis()),
        }

    stats = pd.DataFrame([
        portfolio_stats(r_equal, "Equal Weight (1/N)"),
        portfolio_stats(r_mvp, "MVP"),
        portfolio_stats(r_mcp, "MCP"),
        portfolio_stats(r_mcop, "MCoP"),
    ])

    # 가중치 평균
    avg_weights = pd.DataFrame({
        "Equal Weight": np.mean(w_equal, axis=0),
        "MVP": np.mean(w_mvp, axis=0),
        "MCP": np.mean(w_mcp, axis=0),
        "MCoP": np.mean(w_mcop, axis=0),
    }, index=columns)

    results = {
        "w_mvp": w_mvp,
        "w_mcp": w_mcp,
        "w_mcop": w_mcop,
        "w_equal": w_equal,
        "r_mvp": r_mvp,
        "r_mcp": r_mcp,
        "r_mcop": r_mcop,
        "r_equal": r_equal,
        "stats": stats,
        "avg_weights": avg_weights,
    }

    print("\n--- Portfolio Performance ---")
    print(stats.to_string(index=False))
    print("\n--- Average Portfolio Weights ---")
    print(avg_weights.round(4))

    return results


if __name__ == "__main__":
    np.random.seed(42)
    N = 5
    A = np.random.randn(N, N) * 0.1
    Sigma = A @ A.T + np.eye(N) * 0.5

    print("MVP:", minimum_variance_portfolio(Sigma).round(4))
    print("MCP:", minimum_correlation_portfolio(Sigma).round(4))
