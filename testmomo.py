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

# ------------------ Fractional Differencing ------------------

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
    """Find the optimal differencing parameter `d` by testing stationarity."""
    best_d = None
    for d in d_values:
        frac_diff_df = apply_frac_diff_fast(df, d)
        stationarity_results = frac_diff_df.apply(test_stationarity, axis=0)
        if (stationarity_results < stationarity_thresh).all():
            best_d = d
            break
    return best_d

# ------------------ Portfolio Construction Functions ------------------

def compute_predictability_for_subset(data, portfolio_type="mean_reversion"):
    """Compute the portfolio weights x that optimize predictability."""
    X = data[:-1, :]
    Y = data[1:, :]
    A, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    Gamma = np.cov(data.T)
    L = np.linalg.cholesky(Gamma)
    L_inv = np.linalg.inv(L)
    B = L_inv @ (A.T @ Gamma @ A) @ L_inv
    eigvals, eigvecs = np.linalg.eig(B)
    if portfolio_type == "momentum":
        idx = np.argmax(eigvals.real)
    else:
        idx = np.argmin(eigvals.real)
    z = eigvecs[:, idx].real
    x = L_inv @ z
    x = x / np.linalg.norm(x)
    predictability = eigvals[idx].real
    return x, predictability

def greedy_search(data, asset_names, k, max_weight=0.8, min_weight=None, portfolio_type="mean_reversion"):
    """Perform greedy search to select sparse portfolios."""
    if min_weight is None:
        min_weight = 1 / (4 * k)
    n_assets = data.shape[1]
    candidate_indices = list(range(n_assets))
    best_measure = -np.inf if portfolio_type == "momentum" else np.inf
    best_pair, best_weights = None, None

    for pair in combinations(candidate_indices, 2):
        sub_data = data[:, list(pair)]
        try:
            weights, measure = compute_predictability_for_subset(sub_data, portfolio_type)
        except Exception:
            continue
        if np.max(np.abs(weights)) > max_weight or np.min(np.abs(weights)) < min_weight:
            continue
        if (portfolio_type == "momentum" and measure > best_measure) or \
           (portfolio_type == "mean_reversion" and measure < best_measure):
            best_measure = measure
            best_pair = pair
            best_weights = weights

    if best_pair is None:
        return None, None, None

    current_subset = list(best_pair)
    current_measure = best_measure
    current_weights = best_weights

    while len(current_subset) < k:
        best_candidate, best_candidate_weights, best_candidate_measure = None, None, current_measure
        for i in candidate_indices:
            if i in current_subset:
                continue
            new_subset = current_subset + [i]
            sub_data = data[:, new_subset]
            try:
                weights, measure = compute_predictability_for_subset(sub_data, portfolio_type)
            except Exception:
                continue
            if np.max(np.abs(weights)) > max_weight or np.min(np.abs(weights)) < min_weight:
                continue
            if (portfolio_type == "momentum" and measure > best_candidate_measure) or \
               (portfolio_type == "mean_reversion" and measure < best_candidate_measure):
                best_candidate, best_candidate_weights, best_candidate_measure = i, weights, measure
        if best_candidate is not None:
            current_subset.append(best_candidate)
            current_measure = best_candidate_measure
            current_weights = best_candidate_weights
        else:
            break

    selected_assets = [asset_names[i] for i in current_subset]
    return selected_assets, current_weights, current_measure

# ------------------ Main Streamlit Workflow ------------------

st.sidebar.header("Data and Portfolio Parameters")

portfolio_type = st.sidebar.selectbox("Portfolio Type", options=["Mean Reversion", "Momentum"])
timeframe = st.sidebar.selectbox("Timeframe", options=['1m', '1h', '4h', '1d'], index=2)
limit = st.sidebar.number_input("Number of candles", min_value=24, max_value=1440, value=200, step=10)
k = st.sidebar.number_input("Max number of assets in portfolio (k)", min_value=2, max_value=20, value=10, step=1)
max_weight = st.sidebar.slider("Max weight per asset", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
min_weight = st.sidebar.slider("Min weight per asset", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
all_symbols = fetch_usd_pairs(exchange)
selected_symbols = st.sidebar.multiselect("Select assets for portfolio formation", all_symbols, default=all_symbols[:5])

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
    d_values = np.linspace(0.1, 1.0, 10)
    optimal_d = find_optimal_d(price_df, d_values)
    if optimal_d is None:
        st.error("No optimal `d` found.")
    else:
        frac_diff_df = apply_frac_diff_fast(price_df, optimal_d)
        S = frac_diff_df.values - np.mean(frac_diff_df.values, axis=0)
        selected_assets, weights, measure = greedy_search(S, list(price_df.columns), k, max_weight, min_weight, portfolio_type.lower())
        if selected_assets is None:
            st.error("No valid portfolio found.")
        else:
            portfolio_df = pd.DataFrame({"Asset": selected_assets, "Weight": weights})
            st.write("Selected Portfolio")
            st.dataframe(portfolio_df)
            st.write(f"Predictability measure ({portfolio_type}): {measure:.4f}")
