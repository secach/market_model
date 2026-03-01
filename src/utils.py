import numpy as np

def normalize_weights(weights):
    weights = np.array(weights)
    return weights / weights.sum()

def compute_returns(prices):
    return prices.pct_change().dropna()