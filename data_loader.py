"""
ETF 데이터 다운로드 및 전처리 모듈

분석 대상: USO(원유), VNQ(부동산), IGF(인프라), ICLN(친환경에너지), SPY(주식)
분석 기간: 2011-01-02 ~ 2023-09-30
"""

import pandas as pd
import numpy as np
import os

try:
    import yfinance as yf
except ImportError:
    yf = None

TICKERS = ["USO", "VNQ", "IGF", "ICLN", "SPY"]
ASSET_NAMES = ["Crude Oil", "Real Estate", "Infrastructure", "Clean Energy", "Equity"]
TICKER_NAME_MAP = dict(zip(TICKERS, ASSET_NAMES))

START_DATE = "2011-01-02"
END_DATE = "2023-09-30"
DATA_DIR = "data"


def download_etf_data(tickers=TICKERS, start=START_DATE, end=END_DATE, save=True):
    """Yahoo Finance에서 ETF 종가 데이터를 다운로드한다."""
    print(f"Downloading ETF data: {tickers}")
    print(f"Period: {start} ~ {end}")

    if yf is None:
        raise ImportError("yfinance is not installed. Please install it or provide cached data.")

    # yfinance 버전에 따라 auto_adjust 동작이 다름
    # auto_adjust=False로 다운로드 후 Adj Close 사용 (가장 안정적)
    try:
        data = yf.download(tickers, start=start, end=end, auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(0):
                prices = data["Adj Close"][tickers]
            else:
                prices = data["Close"][tickers]
        else:
            if "Adj Close" in data.columns:
                prices = data[["Adj Close"]]
            else:
                prices = data[["Close"]]
            prices.columns = tickers
    except TypeError:
        # 일부 yfinance 버전에서 auto_adjust 파라미터 미지원
        data = yf.download(tickers, start=start, end=end)
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"][tickers]
        else:
            prices = data[["Close"]]
            prices.columns = tickers

    prices = prices.dropna()
    prices.index = pd.to_datetime(prices.index)

    # 데이터 검증: 이상치 제거 (배당/분할 미조정 감지)
    returns_check = prices.pct_change().dropna()
    extreme = (returns_check.abs() > 0.5).any(axis=1)  # 50% 이상 일일 변동
    if extreme.sum() > 0:
        print(f"Warning: {extreme.sum()} extreme return days detected, check data quality")

    print(f"Downloaded {len(prices)} observations")

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        prices.to_csv(os.path.join(DATA_DIR, "etf_prices.csv"))
        print(f"Saved to {DATA_DIR}/etf_prices.csv")

    return prices


def compute_log_returns(prices):
    """로그 수익률을 계산한다."""
    returns = np.log(prices / prices.shift(1)).dropna() * 100  # 퍼센트 단위
    return returns


def compute_volatility(returns, window=5):
    """롤링 변동성(표준편차)을 계산한다. 논문에서 변동성 전이 분석에 사용."""
    volatility = returns.rolling(window=window).std().dropna()
    return volatility


def load_data(use_volatility=True, vol_window=5):
    """데이터를 로드하고 전처리한다.

    Args:
        use_volatility: True이면 변동성 시계열 사용, False이면 수익률 사용
        vol_window: 변동성 계산 롤링 윈도우 크기

    Returns:
        prices: 종가 데이터
        returns: 로그 수익률
        data: 분석에 사용할 데이터 (변동성 또는 수익률)
    """
    csv_path = os.path.join(DATA_DIR, "etf_prices.csv")

    if os.path.exists(csv_path):
        print("Loading cached data...")
        prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        prices = prices[TICKERS]
    else:
        prices = download_etf_data()

    returns = compute_log_returns(prices)

    if use_volatility:
        data = compute_volatility(returns, window=vol_window)
        print(f"Using volatility series (window={vol_window})")
    else:
        data = returns
        print("Using log returns")

    print(f"Data shape: {data.shape}")
    print(f"Period: {data.index[0].date()} ~ {data.index[-1].date()}")

    return prices, returns, data


if __name__ == "__main__":
    prices, returns, data = load_data()
    print("\n--- Descriptive Statistics (Volatility) ---")
    print(data.describe())
