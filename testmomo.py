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
    'USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'PAX', 'GUSD', 'USDK', 'UST', 'SUSD',
    'FRAX', 'LUSD', 'MIM', 'USDQ', 'TBTC', 'WBTC', 'EUL', 'EUR', 'EURT', 'USDS',
    'USTS', 'USTC', 'USDG', 'MSOL', 'BTC/USDT', 'TREMP/USDT', 'BSX', 'SBR', 'AUD',
    'IDEX', 'FIS', 'CSM', 'GBP', 'POWR', 'ATLAS', 'XCN', 'BOBA', 'OXY', 'BNC',
    'POLIS', 'AIR', 'C98', 'EURT', 'BTC/PYUSD', 'ETH/PYUSD', 'STRD'
}

def fetch_usd_pairs(exchange_instance):
    """Fetch USD-denominated trading pairs excluding stablecoins."""
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

def fractional_difference(series, d, thresh=0.01):
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
    return np.concatenate((np.full(len(series) - len(diff_series), np.nan), diff_series))

def apply_frac_diff(df, d):
    """Apply fractional differencing to a dataframe."""
    return df.apply(lambda col: fractional_difference(col.dropna().values, d), axis=0)

def test_stationarity(series):
    """Test stationarity using the Augmented Dickey-Fuller (ADF) test."""
    clean_series = series.dropna()
    if len(clean_series) <= 1:
        return 1.0  # Return high p-value if not enough data
    try:
        result = adfuller(clean_series, autolag='AIC')
        return result[1]  # Return p-value
    except Exception:
        return 1.0  # If test fails, assume non-stationarity

def find_optimal_d(df, d_values, stationarity_thresh=0.05):
    """Find the optimal fractional differencing parameter `d` that makes the data stationary."""
    for d in d_values:
        frac_diff_df = apply_frac_diff(df, d)
        stationarity_results = frac_diff_df.apply(test_stationarity, axis=0)
        if (stationarity_results < stationarity_thresh).all():
            return d
    return None  # If no optimal `d` is found

# ------------------ Portfolio Construction Functions ------------------

def compute_predictability(data, portfolio_type="mean_reversion"):
    """Compute portfolio predictability and weights."""
    X, Y = data[:-1, :], data[1:, :]
    A, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    Gamma = np.cov(data.T)
    L = np.linalg.cholesky(Gamma)
    L_inv = np.linalg.inv(L)
    B = L_inv @ (A.T @ Gamma @ A) @ L_inv
    eigvals, eigvecs = np.linalg.eig(B)
    idx = np.argmax(eigvals.real) if portfolio_type == "momentum" else np.argmin(eigvals.real)
    x = L_inv @ eigvecs[:, idx].real
    return x / np.linalg.norm(x), eigvals[idx].real

def greedy_search(data, asset_names, k, max_weight=0.8, min_weight=0.05, portfolio_type="mean_reversion"):
    """Perform Greedy Search for Sparse Portfolio Selection."""
    n_assets = data.shape[1]
    best_measure = -np.inf if portfolio_type == "momentum" else np.inf
    best_pair, best_weights = None, None

    for pair in combinations(range(n_assets), 2):
        sub_data = data[:, list(pair)]
        weights, measure = compute_predictability(sub_data, portfolio_type)
        if np.max(np.abs(weights)) <= max_weight and np.min(np.abs(weights)) >= min_weight:
            if (portfolio_type == "momentum" and measure > best_measure) or (portfolio_type == "mean_reversion" and measure < best_measure):
                best_measure, best_pair, best_weights = measure, pair, weights

    if best_pair is None:
        return None, None, None

    current_subset = list(best_pair)
    current_measure, current_weights = best_measure, best_weights

    while len(current_subset) < k:
        best_candidate, best_candidate_measure, best_candidate_weights = None, current_measure, None
        for i in range(n_assets):
            if i in current_subset:
                continue
            new_subset = current_subset + [i]
            sub_data = data[:, new_subset]
            weights, measure = compute_predictability(sub_data, portfolio_type)
            if np.max(np.abs(weights)) <= max_weight and np.min(np.abs(weights)) >= min_weight:
                if (portfolio_type == "momentum" and measure > best_candidate_measure) or (portfolio_type == "mean_reversion" and measure < best_candidate_measure):
                    best_candidate, best_candidate_measure, best_candidate_weights = i, measure, weights
        if best_candidate is not None:
            current_subset.append(best_candidate)
            current_measure, current_weights = best_candidate_measure, best_candidate_weights
        else:
            break

    selected_assets = [asset_names[i] for i in current_subset]
    return selected_assets, current_weights, current_measure

# ------------------ Streamlit App ------------------

st.sidebar.header("Portfolio Parameters")
portfolio_type = st.sidebar.selectbox("Portfolio Type", ["Mean Reversion", "Momentum"])
timeframe = st.sidebar.selectbox("Timeframe", ['1m', '1h', '4h', '1d'], index=2)
limit = st.sidebar.number_input("Number of candles", 24, 1440, 200, step=10)
k = st.sidebar.number_input("Max number of assets (k)", 2, 20, 10, step=1)
max_weight = st.sidebar.slider("Max weight per asset", 0.5, 1.0, 0.8, step=0.05)
all_symbols = fetch_usd_pairs(exchange)
selected_symbols = st.sidebar.multiselect("Select assets", all_symbols, default=all_symbols[:5])

st.header("Fetching Data")
data_dict = {}
for symbol in selected_symbols:
    df = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).set_index('timestamp')
    data_dict[symbol] = df['close']

price_df = pd.concat(data_dict, axis=1).dropna()
optimal_d = find_optimal_d(price_df, np.linspace(0.1, 1.0, 10))
frac_diff_df = apply_frac_diff(price_df, optimal_d)
S = frac_diff_df.dropna().values
selected_assets, weights, measure = greedy_search(S, list(price_df.columns), k, max_weight, 0.05, portfolio_type.lower())
portfolio_df = pd.DataFrame({"Asset": selected_assets, "Weight": weights})
st.dataframe(portfolio_df)
st.write(f"Predictability measure ({portfolio_type}): {measure:.4f}")
