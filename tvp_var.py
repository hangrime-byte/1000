"""
TVP-VAR 모형 (Time-Varying Parameter VAR)

Antonakakis, Chatziantoniou & Gabauer (2020) 방법론 기반
Koop & Korobilis (2013, 2014)의 Forgetting Factor Kalman Filter 사용

References:
- Antonakakis et al. (2020), "Refined Measures of Dynamic Connectedness
  based on Time-Varying Parameter Vector Autoregressions", JRFM 13(4), 84.
- Koop & Korobilis (2013), "Large Time-Varying Parameter VARs",
  Journal of Econometrics, 177(2), 185-198.
"""

import numpy as np
from scipy.linalg import block_diag


class TVPVAR:
    """TVP-VAR 모형 with Forgetting Factor Kalman Filter.

    TVP-VAR(p) 모형:
        y_t = B_t * z_t + e_t,  e_t ~ N(0, Σ_t)
        B_t = B_{t-1} + u_t,    u_t ~ N(0, Q_t)

    여기서:
        y_t: N×1 관측 벡터
        z_t = [y_{t-1}', ..., y_{t-p}']' ⊗ I_N: 회귀자 행렬
        B_t: 시변 계수 벡터 (vec 형태)
        Σ_t: 시변 관측 오차 공분산 (decay factor κ로 EWMA 추정)
        Q_t: 상태 방정식 공분산 (forgetting factor λ로 대체)

    Parameters:
        nlag: VAR 시차 (p)
        kappa1: forgetting factor for VAR coefficients (λ), default 0.99
        kappa2: decay factor for error covariance (κ), default 0.96
    """

    def __init__(self, nlag=1, kappa1=0.99, kappa2=0.96):
        self.nlag = nlag
        self.kappa1 = kappa1  # λ: forgetting factor for coefficients
        self.kappa2 = kappa2  # κ: decay factor for error covariance

    def fit(self, data):
        """TVP-VAR 모형을 추정한다.

        Args:
            data: T×N numpy array (관측치 행렬)

        Returns:
            self: 추정 결과가 저장된 객체
        """
        Y = np.array(data)
        T_total, N = Y.shape
        p = self.nlag

        # 종속변수와 시차 변수 구성
        T = T_total - p
        y = Y[p:]  # T × N
        Z = np.zeros((T, N * p))  # T × (N*p)
        for lag in range(1, p + 1):
            Z[:, (lag - 1) * N: lag * N] = Y[p - lag: T_total - lag]

        self.N = N
        self.T = T
        self.y = y
        self.Z = Z

        # 저장용 배열
        # B_t: 각 시점의 VAR 계수 (vec 형태)
        # Sigma_t: 각 시점의 오차 공분산
        ncoef = N * N * p  # 계수 개수
        B_store = np.zeros((T, ncoef))
        Sigma_store = np.zeros((T, N, N))
        Phi_store = np.zeros((T, N, N * p))  # 원래 행렬 형태 계수

        # === 초기화 (OLS by first 200 obs or all available) ===
        init_end = min(200, T)
        y_init = y[:init_end]
        Z_init = Z[:init_end]

        # OLS: vec(B_0) = (Z'Z)^{-1} Z'y
        ZtZ = Z_init.T @ Z_init
        ZtZ_inv = np.linalg.inv(ZtZ + 1e-8 * np.eye(ZtZ.shape[0]))
        B_ols = ZtZ_inv @ (Z_init.T @ y_init)  # (N*p) × N
        b0 = B_ols.T.flatten()  # vec(B_0'), N*(N*p) 형태

        # 초기 상태 공분산
        resid_init = y_init - Z_init @ B_ols
        S0 = (resid_init.T @ resid_init) / init_end
        P0 = np.eye(ncoef) * 10  # 넓은 사전분포

        # === Kalman Filter with Forgetting Factors ===
        b_tt = b0.copy()
        P_tt = P0.copy()
        H_t = S0.copy()

        for t in range(T):
            z_t = Z[t]  # 1 × (N*p)
            y_t = y[t]  # 1 × N

            # 회귀자 행렬: z_t ⊗ I_N → N × (N*N*p)
            # y_t = (z_t ⊗ I_N) * vec(B_t) + e_t
            X_t = np.kron(z_t, np.eye(N))  # N × ncoef

            # 1. Prediction step
            b_t1t = b_tt  # 상태 예측 (random walk)
            P_t1t = P_tt / self.kappa1  # 공분산 인플레이션 (forgetting factor)

            # 2. Forecast error
            y_hat = X_t @ b_t1t  # N × 1
            e_t = y_t - y_hat  # 예측 오차

            # 3. EWMA update for error covariance
            H_t = self.kappa2 * H_t + (1 - self.kappa2) * np.outer(e_t, e_t)

            # 4. Kalman gain
            F_t = X_t @ P_t1t @ X_t.T + H_t  # N × N
            F_t_inv = np.linalg.inv(F_t + 1e-10 * np.eye(N))
            K_t = P_t1t @ X_t.T @ F_t_inv  # ncoef × N

            # 5. Update step
            b_tt = b_t1t + K_t @ e_t
            P_tt = (np.eye(ncoef) - K_t @ X_t) @ P_t1t

            # 대칭성 보장
            P_tt = (P_tt + P_tt.T) / 2

            # 저장
            B_store[t] = b_tt
            Sigma_store[t] = H_t

            # B_t를 N × (N*p) 행렬로 복원
            B_mat = b_tt.reshape(N, N * p)
            Phi_store[t] = B_mat

        self.B = B_store
        self.Sigma = Sigma_store
        self.Phi = Phi_store

        print(f"TVP-VAR({p}) estimated: T={T}, N={N}, kappa1={self.kappa1}, kappa2={self.kappa2}")
        return self

    def get_companion_matrix(self, t):
        """시점 t에서의 companion matrix를 반환한다.

        VAR(p)의 companion form:
            Y_t = A_t * Y_{t-1} + E_t

        A_t = | Φ_1,t  Φ_2,t  ...  Φ_p,t |
              | I_N     0      ...  0      |
              | 0       I_N    ...  0      |
              | ...                  ...    |
              | 0       0      ...  I_N  0 |

        Returns:
            A_t: (N*p) × (N*p) companion matrix
        """
        N = self.N
        p = self.nlag

        Phi_t = self.Phi[t]  # N × (N*p)

        if p == 1:
            return Phi_t  # N × N

        A = np.zeros((N * p, N * p))
        A[:N, :] = Phi_t  # 첫 번째 블록 행
        A[N:, :N * (p - 1)] = np.eye(N * (p - 1))  # 단위행렬 블록

        return A

    def compute_vma_coefficients(self, t, H):
        """시점 t에서의 VMA(H) 계수 Ψ_0, Ψ_1, ..., Ψ_{H-1}를 계산한다.

        TVP-VMA 표현: y_t = Σ_{h=0}^{∞} Ψ_h * e_{t-h}
        여기서 Ψ_0 = I_N, Ψ_h = J * A^h * J' (companion form 이용)

        Args:
            t: 시점 인덱스
            H: 예측 수평선

        Returns:
            Psi: list of H matrices, 각각 N × N
        """
        N = self.N
        p = self.nlag

        A_t = self.get_companion_matrix(t)
        dim = A_t.shape[0]

        # Selection matrix J: N × (N*p), [I_N, 0, ..., 0]
        J = np.zeros((N, dim))
        J[:N, :N] = np.eye(N)

        Psi = []
        A_power = np.eye(dim)
        for h in range(H):
            Psi_h = J @ A_power @ J.T
            Psi.append(Psi_h)
            A_power = A_power @ A_t

        return Psi


if __name__ == "__main__":
    # 테스트
    np.random.seed(42)
    T, N = 500, 3
    data = np.random.randn(T, N)
    model = TVPVAR(nlag=1, kappa1=0.99, kappa2=0.96)
    model.fit(data)
    print(f"Phi shape: {model.Phi.shape}")
    print(f"Sigma shape: {model.Sigma.shape}")

    Psi = model.compute_vma_coefficients(t=400, H=10)
    print(f"VMA coefficients: {len(Psi)} matrices of shape {Psi[0].shape}")
