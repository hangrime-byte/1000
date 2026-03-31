"""
DCC-GARCH 모형 모듈

논문의 2단계 접근법:
1단계: TVP-VAR → 연결성 분석
2단계: DCC-GARCH → 조건부 공분산 → 헤지비율/포트폴리오 가중치

DCC (Dynamic Conditional Correlation) - GARCH:
- Engle (2002), "Dynamic Conditional Correlation"
- 1단계: 각 변수에 개별 GARCH(1,1) 적합 → 조건부 분산
- 2단계: 표준화 잔차로 동적 조건부 상관행렬 추정

References:
- Engle, R. (2002), "Dynamic Conditional Correlation: A Simple Class of
  Multivariate Generalized Autoregressive Conditional Heteroskedasticity Models",
  Journal of Business & Economic Statistics, 20(3), 339-350.
"""

import numpy as np
from scipy.optimize import minimize


class DCC_GARCH:
    """DCC-GARCH(1,1) 모형.

    각 시계열에 GARCH(1,1)을 적합하고, 표준화 잔차의 동적 상관행렬을
    DCC 방법으로 추정하여 시변 공분산 행렬을 산출한다.

    GARCH(1,1): σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}
    DCC:        Q_t = (1-a-b)*Q̄ + a*z_{t-1}*z'_{t-1} + b*Q_{t-1}
                R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}
                H_t = D_t * R_t * D_t
    """

    def __init__(self):
        self.garch_params = None
        self.dcc_params = None

    def _fit_garch11(self, y):
        """단변량 GARCH(1,1) 추정.

        Args:
            y: T×1 수익률 시계열

        Returns:
            omega, alpha, beta: GARCH 파라미터
            sigma2: T×1 조건부 분산 시계열
        """
        T = len(y)
        var_y = np.var(y)

        def neg_loglik(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10

            sigma2 = np.zeros(T)
            sigma2[0] = var_y

            for t in range(1, T):
                sigma2[t] = omega + alpha * y[t-1]**2 + beta * sigma2[t-1]
                if sigma2[t] <= 0:
                    return 1e10

            ll = -0.5 * np.sum(np.log(sigma2) + y**2 / sigma2)
            return -ll

        # 초기값: 표본 분산 기반
        omega0 = var_y * 0.05
        alpha0 = 0.05
        beta0 = 0.90
        x0 = [omega0, alpha0, beta0]

        bounds = [(1e-8, None), (1e-8, 0.5), (0.5, 0.9999)]
        constraints = {"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]}

        result = minimize(neg_loglik, x0, bounds=bounds, constraints=constraints,
                          method="SLSQP", options={"maxiter": 500})

        omega, alpha, beta = result.x

        # 조건부 분산 재계산
        sigma2 = np.zeros(T)
        sigma2[0] = var_y
        for t in range(1, T):
            sigma2[t] = omega + alpha * y[t-1]**2 + beta * sigma2[t-1]

        return omega, alpha, beta, sigma2

    def fit(self, data):
        """DCC-GARCH(1,1) 모형을 추정한다.

        Args:
            data: T×N numpy array (수익률 행렬)

        Returns:
            self
        """
        if hasattr(data, 'values'):
            data = data.values

        T, N = data.shape
        self.T = T
        self.N = N

        # 1단계: 각 변수에 GARCH(1,1) 적합
        sigma2_all = np.zeros((T, N))
        self.garch_params = []
        z = np.zeros((T, N))  # 표준화 잔차

        for i in range(N):
            omega, alpha, beta, sigma2 = self._fit_garch11(data[:, i])
            self.garch_params.append((omega, alpha, beta))
            sigma2_all[:, i] = sigma2
            z[:, i] = data[:, i] / np.sqrt(sigma2 + 1e-10)

        self.sigma2 = sigma2_all
        self.z = z

        # 2단계: DCC 추정
        # Q̄ = 표준화 잔차의 무조건부 상관행렬
        Q_bar = np.corrcoef(z.T)  # N × N

        def dcc_neg_loglik(params):
            a, b = params
            if a < 0 or b < 0 or a + b >= 1:
                return 1e10

            Q_t = Q_bar.copy()
            ll = 0

            for t in range(1, T):
                z_t = z[t-1].reshape(-1, 1)
                Q_t = (1 - a - b) * Q_bar + a * (z_t @ z_t.T) + b * Q_t

                # R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
                Q_diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(Q_t) + 1e-10))
                R_t = Q_diag_inv_sqrt @ Q_t @ Q_diag_inv_sqrt

                # 로그우도 (상관 부분만)
                det_R = np.linalg.det(R_t)
                if det_R <= 0:
                    return 1e10
                z_curr = z[t].reshape(-1, 1)
                quad = z_curr.T @ np.linalg.inv(R_t) @ z_curr - z_curr.T @ z_curr
                ll += np.log(det_R) + quad.item()

            return 0.5 * ll

        result = minimize(dcc_neg_loglik, [0.01, 0.95],
                          bounds=[(1e-6, 0.3), (0.7, 0.9999)],
                          constraints={"type": "ineq", "fun": lambda p: 0.9999 - p[0] - p[1]},
                          method="SLSQP", options={"maxiter": 300})

        a, b = result.x
        self.dcc_params = (a, b)

        # 전체 시변 공분산 행렬 계산
        H_series = np.zeros((T, N, N))
        Q_t = Q_bar.copy()

        for t in range(T):
            if t > 0:
                z_t = z[t-1].reshape(-1, 1)
                Q_t = (1 - a - b) * Q_bar + a * (z_t @ z_t.T) + b * Q_t

            Q_diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(Q_t) + 1e-10))
            R_t = Q_diag_inv_sqrt @ Q_t @ Q_diag_inv_sqrt

            # H_t = D_t R_t D_t (D_t = diag(σ_1t, ..., σ_Nt))
            D_t = np.diag(np.sqrt(sigma2_all[t]))
            H_series[t] = D_t @ R_t @ D_t

        self.H = H_series
        self.Q_bar = Q_bar

        print(f"DCC-GARCH(1,1) estimated: T={T}, N={N}")
        print(f"  DCC params: a={a:.4f}, b={b:.4f}")
        for i in range(N):
            o, al, be = self.garch_params[i]
            print(f"  GARCH[{i}]: omega={o:.6f}, alpha={al:.4f}, beta={be:.4f}")

        return self


if __name__ == "__main__":
    np.random.seed(42)
    T, N = 1000, 3

    # GARCH 효과가 있는 데이터 생성
    data = np.zeros((T, N))
    sigma = np.ones(N)
    for t in range(1, T):
        sigma = np.sqrt(0.01 + 0.05 * data[t-1]**2 + 0.9 * sigma**2)
        data[t] = sigma * np.random.randn(N)

    model = DCC_GARCH()
    model.fit(data)
    print(f"H shape: {model.H.shape}")
    print(f"H[500] diagonal: {np.diag(model.H[500]).round(4)}")
