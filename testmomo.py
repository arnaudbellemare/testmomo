import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import adfuller
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# ------------------ Exchange Initialization and Market Filtering ------------------

# Initialize Kraken exchange
exchange = ccxt.kraken({
    'enableRateLimit': True,  # Respect rate limits
})

# Define stablecoins and other symbols to exclude
STABLECOINS = {
    'USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'PAX', 'GUSD',
    'USDK', 'UST', 'SUSD', 'FRAX', 'LUSD', 'MIM', 'USDQ', 'TBTC', 'WBTC',
    'EUL', 'EUR', 'EURT', 'USDS', 'USTS', 'USTC', 'USDG', 'MSOL', 'BTC/USDT',
    'TREMP/USDT', 'BSX', 'SBR', 'AUD', 'IDEX', 'FIS', 'CSM', 'GBP', 'POWR',
    'ATLAS', 'XCN', 'BOBA', 'OXY', 'BNC', 'POLIS', 'AIR', 'C98', 'EURT',
    'BTC/PYUSD', 'ETH/PYUSD', 'STRD'
}

def fetch_usd_pairs(exchange_instance):
    """Fetch USD-denominated trading pairs excluding stablecoin pairs."""
    try:
        markets = exchange_instance.load_markets()
        usd_pairs = [
            symbol.upper() for symbol in markets.keys()
            if '/USD' in symbol.upper()
            and all(asset.upper() not in STABLECOINS for asset in symbol.split('/'))
        ]
        usd_pairs = list(set(usd_pairs))
        logging.info(f"Fetched {len(usd_pairs)} USD-denominated trading pairs from Kraken.")
        return usd_pairs
    except Exception as e:
        logging.error(f"Error fetching markets from {exchange_instance.id}: {e}")
        return []

# ------------------ Fractional Differencing Functions ------------------

def fractional_difference_numpy(series, d, thresh=0.01):
    """Apply fractional differencing to a single time series."""
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thresh:
            break
        w.append(w_)
        k += 1
    w = np.array(w[::-1])
    len_w = len(w)
    diff_series_length = len(series) - len_w + 1
    if diff_series_length <= 0:
        return np.full(len(series), np.nan)
    shape = (diff_series_length, len_w)
    strides = (series.strides[0], series.strides[0])
    windowed = np.lib.stride_tricks.as_strided(series, shape=shape, strides=strides)
    diff_series = np.dot(windowed, w)
    nan_padding = np.full(len(series) - len(diff_series), np.nan)
    return np.concatenate((nan_padding, diff_series))

def apply_frac_diff_fast(df, d):
    """Apply fractional differencing to a dataframe of time series."""
    frac_diff_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        series = df[col].values
        if np.isnan(series).all():
            frac_diff_df[col] = np.nan
            logging.warning(f"All values are NaN for {col}.")
        else:
            non_na = ~np.isnan(series)
            frac_series = fractional_difference_numpy(series[non_na], d, thresh=0.01)
            full_series = np.full_like(series, np.nan, dtype=np.float64)
            full_series[non_na] = frac_series
            frac_diff_df[col] = full_series
    return frac_diff_df

def test_stationarity(series):
    """Test stationarity using the Augmented Dickey-Fuller (ADF) test."""
    result = adfuller(series, autolag='AIC', maxlag=10, regression="c")
    return result[1]  # Return the p-value

def find_optimal_d(df, d_values, stationarity_thresh=0.05):
    """
    Find the optimal differencing parameter `d` by testing stationarity.
    Choose the smallest `d` that achieves stationarity.
    """
    best_d = None
    for d in d_values:
        frac_diff_df = apply_frac_diff_fast(df, d)
        stationarity_results = frac_diff_df.apply(test_stationarity, axis=0)
        if (stationarity_results < stationarity_thresh).all():
            best_d = d
            break
    return best_d

# ------------------ Main Streamlit Workflow ------------------

st.sidebar.header("Data and Portfolio Parameters")

# Choose portfolio type: Mean Reversion or Momentum
portfolio_type = st.sidebar.selectbox("Portfolio Type", options=["Mean Reversion", "Momentum"])

# Timeframe and data limit
timeframe = st.sidebar.selectbox("Timeframe", options=['1m', '1h', '4h', '1d'], index=2)
limit = st.sidebar.number_input("Number of candles", min_value=24, max_value=1440, value=200, step=10)

# Portfolio constraints
k = st.sidebar.number_input("Max number of assets in portfolio (k)", min_value=2, max_value=20, value=10, step=1)
max_weight = st.sidebar.slider("Max weight per asset", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
min_weight = st.sidebar.slider("Min weight per asset", min_value=0.0, max_value=0.5, value=0.05, step=0.01)

# Asset selection
st.sidebar.header("Asset Selection")
all_symbols = fetch_usd_pairs(exchange)
selected_symbols = st.sidebar.multiselect("Select assets for portfolio formation", all_symbols, default=all_symbols[:5])

# Fetching data
data_dict = {}
for symbol in selected_symbols:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        data_dict[symbol] = df['close']
    except Exception as e:
        st.warning(f"Error fetching data for {symbol}: {e}")

if len(data_dict) < 2:
    st.error("Not enough asset data fetched to construct a portfolio.")
else:
    price_df = pd.concat(data_dict, axis=1).dropna()
    st.write("Aligned Price Data:")
    st.dataframe(price_df.head())

    # Find optimal d
    st.header("Finding Optimal Differencing Parameter (d)")
    d_values = np.linspace(0.1, 1.0, 10)
    optimal_d = find_optimal_d(price_df, d_values)
    if optimal_d is None:
        st.error("No optimal `d` found that ensures stationarity.")
    else:
        st.success(f"Optimal `d`: {optimal_d}")
        st.write(f"Applying fractional differencing with `d={optimal_d}`.")
        frac_diff_df = apply_frac_diff_fast(price_df, optimal_d)

        # Preprocess data and run portfolio construction
        S = frac_diff_df.values - np.mean(frac_diff_df.values, axis=0)
        selected_assets, weights, measure = greedy_search(
            S, list(price_df.columns), k, max_weight, min_weight, portfolio_type=portfolio_type.lower().replace(" ", "_")
        )
        if selected_assets is None:
            st.error("No valid portfolio found.")
        else:
            st.subheader("Selected Portfolio")
            portfolio_df = pd.DataFrame({"Asset": selected_assets, "Weight": weights})
            st.dataframe(portfolio_df)
            st.write(f"Predictability measure ({portfolio_type.lower()}): {measure:.4f}")
