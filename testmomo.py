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
exchange = ccxt.kraken({'enableRateLimit': True})

# Define stablecoins and symbols to exclude
STABLECOINS = {'USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'PAX', 'GUSD', 'SUSD', 'FRAX', 
               'EUR', 'EURT', 'UST', 'LUSD', 'USDQ', 'USTS', 'WBTC', 'TBTC', 'BTC/USDT'}

def fetch_usd_pairs(exchange_instance):
    """Fetch USD trading pairs while excluding stablecoins."""
    try:
        markets = exchange_instance.load_markets()
        usd_pairs = [
            symbol for symbol in markets.keys()
            if '/USD' in symbol and all(asset not in STABLECOINS for asset in symbol.split('/'))
        ]
        return list(set(usd_pairs))
    except Exception as e:
        logging.error(f"Error fetching markets: {e}")
        return []

# ------------------ Fractional Differencing ------------------

def fractional_difference(series, d, thresh=0.01):
    """Apply fractional differencing to a single asset's time series."""
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thresh:
            break
        w.append(w_)
        k += 1
    w = np.array(w[::-1])
    
    diff_series = np.convolve(series, w, mode='valid')
    nan_padding = np.full(len(series) - len(diff_series), np.nan)
    return np.concatenate((nan_padding, diff_series))

def apply_frac_diff(df, d):
    """Apply fractional differencing to a dataframe of assets."""
    return df.apply(lambda col: fractional_difference(col.dropna().values, d), axis=0)

def test_stationarity(series):
    """Perform the Augmented Dickey-Fuller (ADF) test."""
    clean_series = series.dropna()
    if len(clean_series) <= 1:
        return 1.0  # High p-value if not enough data
    try:
        result = adfuller(clean_series, autolag='AIC')
        return result[1]  # Return p-value
    except Exception:
        return 1.0

def find_optimal_d(df, d_values=np.linspace(0.1, 1.0, 10), threshold=0.05):
    """Find the best differencing parameter `d` for stationarity."""
    for d in d_values:
        frac_diff_df = apply_frac_diff(df, d)
        p_values = frac_diff_df.apply(test_stationarity, axis=0)
        if (p_values < threshold).all():
            return d
    return None  # No optimal `d` found

# ------------------ Portfolio Optimization ------------------

def compute_predictability(data, portfolio_type="mean_reversion"):
    """Calculate predictability and optimal portfolio weights."""
    X, Y = data[:-1, :], data[1:, :]
    A, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    Gamma = np.cov(data.T)
    L_inv = np.linalg.inv(np.linalg.cholesky(Gamma))
    B = L_inv @ (A.T @ Gamma @ A) @ L_inv
    eigvals, eigvecs = np.linalg.eig(B)
    idx = np.argmax(eigvals.real) if portfolio_type == "momentum" else np.argmin(eigvals.real)
    x = L_inv @ eigvecs[:, idx].real
    return x / np.linalg.norm(x), eigvals[idx].real

def greedy_search(data, asset_names, k, max_weight=0.8, min_weight=0.05, portfolio_type="mean_reversion"):
    """Greedy Search for optimal sparse portfolio selection."""
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

price_df = pd.concat(data_dict, axis=1).dropna(how='all')

optimal_d = find_optimal_d(price_df)
frac_diff_df = apply_frac_diff(price_df, optimal_d).dropna()

S = frac_diff_df.values
selected_assets, weights, measure = greedy_search(S, list(price_df.columns), k, max_weight, 0.05, portfolio_type.lower())

portfolio_df = pd.DataFrame({"Asset": selected_assets, "Weight": weights})
st.dataframe(portfolio_df)
st.write(f"Predictability measure ({portfolio_type}): {measure:.4f}")
