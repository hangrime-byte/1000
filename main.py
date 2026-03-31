"""
ETF를 활용한 글로벌 대체투자 자산들의 동적 연결성과 포트폴리오 최적화에 관한 연구

윤병조 (2024), 금융공학연구, 제23권 제1호, pp. 69-92

메인 분석 스크립트:
1. 데이터 로드 및 전처리
2. TVP-VAR 기반 동적 연결성 분석
3. 헤지비율 및 포트폴리오 가중치 추정
4. 포트폴리오 최적화 (MVP, MCP, MCoP)
5. 헤징 효율성 분석
6. 결과 시각화 및 저장
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_data, TICKERS, ASSET_NAMES
from connectedness import dynamic_connectedness
from hedge_ratio import (
    compute_hedge_ratios,
    compute_portfolio_weights,
    compute_hedging_effectiveness,
    compute_portfolio_hedging_effectiveness,
)
from portfolio import compute_dynamic_portfolios
from visualization import generate_all_figures

# ============================================================
# 분석 파라미터
# ============================================================
NLAG = 1          # VAR 시차
NFORE = 10        # GFEVD 예측 수평선 (H-step ahead)
KAPPA1 = 0.99     # Forgetting factor for VAR coefficients
KAPPA2 = 0.96     # Decay factor for error covariance
USE_VOLATILITY = True   # True: 변동성 시계열, False: 수익률 시계열
VOL_WINDOW = 5          # 변동성 롤링 윈도우

OUTPUT_DIR = "results"


def save_results(conn_results, hr_summary, wt_summary, he_hr, he_wt,
                 port_results):
    """분석 결과를 CSV로 저장한다."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 연결성 테이블
    conn_results["GFEVD_table"].to_csv(
        os.path.join(OUTPUT_DIR, "connectedness_table.csv"))

    # TCI 시계열
    conn_results["TCI"].to_csv(
        os.path.join(OUTPUT_DIR, "tci_timeseries.csv"))

    # FROM/TO/NET 시계열
    conn_results["FROM"].to_csv(
        os.path.join(OUTPUT_DIR, "from_connectedness.csv"))
    conn_results["TO"].to_csv(
        os.path.join(OUTPUT_DIR, "to_connectedness.csv"))
    conn_results["NET"].to_csv(
        os.path.join(OUTPUT_DIR, "net_connectedness.csv"))

    # 헤지비율 요약
    hr_summary.to_csv(
        os.path.join(OUTPUT_DIR, "hedge_ratio_summary.csv"))

    # 포트폴리오 가중치 요약
    wt_summary.to_csv(
        os.path.join(OUTPUT_DIR, "portfolio_weight_summary.csv"))

    # 헤징 효율성
    he_hr.to_csv(
        os.path.join(OUTPUT_DIR, "hedging_effectiveness_hr.csv"))
    he_wt.to_csv(
        os.path.join(OUTPUT_DIR, "hedging_effectiveness_wt.csv"))

    # 포트폴리오 성과
    port_results["stats"].to_csv(
        os.path.join(OUTPUT_DIR, "portfolio_performance.csv"), index=False)
    port_results["avg_weights"].to_csv(
        os.path.join(OUTPUT_DIR, "portfolio_avg_weights.csv"))

    print(f"\nAll results saved to '{OUTPUT_DIR}/' directory.")


def main():
    """메인 분석 파이프라인."""
    print("=" * 70)
    print("ETF를 활용한 글로벌 대체투자 자산들의 동적 연결성과")
    print("포트폴리오 최적화에 관한 연구 (윤병조, 2024)")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. 데이터 로드
    # --------------------------------------------------------
    print("\n[Step 1] Loading data...")
    prices, returns, data = load_data(
        use_volatility=USE_VOLATILITY, vol_window=VOL_WINDOW
    )

    print(f"\nETFs: {TICKERS}")
    print(f"Assets: {ASSET_NAMES}")
    print(f"Observations: {len(data)}")

    # 기술 통계
    print("\n--- Descriptive Statistics ---")
    desc = data.describe()
    print(desc.round(4))

    # --------------------------------------------------------
    # 2. TVP-VAR 기반 동적 연결성 분석
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Step 2] TVP-VAR Dynamic Connectedness Analysis")
    print(f"  nlag={NLAG}, nfore={NFORE}, kappa1={KAPPA1}, kappa2={KAPPA2}")
    print("=" * 70)

    conn_results = dynamic_connectedness(
        data, nlag=NLAG, nfore=NFORE, kappa1=KAPPA1, kappa2=KAPPA2
    )

    # --------------------------------------------------------
    # 3. 헤지비율 및 포트폴리오 가중치
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Step 3] Hedge Ratios & Portfolio Weights")
    print("=" * 70)

    Sigma_series = conn_results["model"].Sigma
    columns = conn_results["columns"]

    # Kroner & Sultan (1993) 헤지비율
    hedge_ratios, hr_summary = compute_hedge_ratios(Sigma_series, columns)
    print("\n--- Average Hedge Ratios (Kroner & Sultan, 1993) ---")
    print(hr_summary.round(4))

    # Kroner & Ng (1998) 포트폴리오 가중치
    weights, wt_summary = compute_portfolio_weights(Sigma_series, columns)
    print("\n--- Average Portfolio Weights (Kroner & Ng, 1998) ---")
    print(wt_summary.round(4))

    # --------------------------------------------------------
    # 4. 헤징 효율성 (Ederington, 1979)
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Step 4] Hedging Effectiveness (Ederington, 1979)")
    print("=" * 70)

    he_hr = compute_hedging_effectiveness(returns, hedge_ratios, columns)
    print("\n--- Hedging Effectiveness (Hedge Ratio Strategy) ---")
    print(he_hr.round(2))

    he_wt = compute_portfolio_hedging_effectiveness(returns, weights, columns)
    print("\n--- Hedging Effectiveness (Portfolio Weight Strategy) ---")
    print(he_wt.round(2))

    # --------------------------------------------------------
    # 5. 포트폴리오 최적화 (MVP, MCP, MCoP)
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Step 5] Portfolio Optimization (MVP, MCP, MCoP)")
    print("=" * 70)

    port_results = compute_dynamic_portfolios(
        returns, Sigma_series, conn_results["NPDC"], columns
    )

    # --------------------------------------------------------
    # 6. 결과 저장 및 시각화
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Step 6] Saving Results & Generating Figures")
    print("=" * 70)

    save_results(conn_results, hr_summary, wt_summary, he_hr, he_wt,
                 port_results)

    hr_results = {
        "hedge_ratios": hedge_ratios,
        "hr_summary": hr_summary,
    }

    generate_all_figures(conn_results, hr_results, port_results, returns)

    # --------------------------------------------------------
    # 최종 요약
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    avg_tci = conn_results["TCI"].mean()
    print(f"\n  Average TCI: {avg_tci:.2f}%")
    print(f"  논문 보고치: 36.8% (참고)")
    print(f"\n  주요 결과:")
    print(f"    - 총 연결성(TCI)은 분석기간 동안 평균 {avg_tci:.1f}% 수준")

    net_means = conn_results["NET"].mean()
    top_transmitter = net_means.idxmax()
    top_receiver = net_means.idxmin()
    print(f"    - 최대 순전출자: {top_transmitter} (NET={net_means[top_transmitter]:.2f}%)")
    print(f"    - 최대 순유입자: {top_receiver} (NET={net_means[top_receiver]:.2f}%)")

    print(f"\n  Output directories:")
    print(f"    - Results: {OUTPUT_DIR}/")
    print(f"    - Figures: figures/")


if __name__ == "__main__":
    main()
