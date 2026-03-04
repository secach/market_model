# model.py (rolling window version)

import pandas as pd
import statsmodels.api as sm

def compute_signal(csv_path, ar_weight=0.5, weight_signal_weight=0.5, rolling_window=10):
    """
    Predict next-day index return using:
    1. Rolling-window AR(1) on index returns
    2. Rolling-window weighted sum of stock returns
    Automatically converts stock returns from percent to decimal.
    """

    # 1️⃣ Load CSV
    df = pd.read_csv(csv_path, sep=',')
    df.columns = df.columns.str.strip()  # remove spaces
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=False)
    df = df.sort_values("Date")

    # Index returns
    df["Index_Return"] = df["Index"].pct_change()
    df = df.dropna()

    # --- AR(1) signal using rolling window ---
    df["Lag_Return"] = df["Index_Return"].shift(1)
    ar_df = df.dropna()

    # Take last `rolling_window` rows for smoothing
    ar_df_rolling = ar_df.iloc[-rolling_window:]

    X_ar = sm.add_constant(ar_df_rolling["Lag_Return"])
    y_ar = ar_df_rolling["Index_Return"]
    model_ar = sm.OLS(y_ar, X_ar).fit()
    last_lag = ar_df_rolling["Lag_Return"].iloc[-1]
    predicted_return_ar = model_ar.predict([1, last_lag])[0]

    # --- Weighted stock signal using rolling window ---
    stock_return_cols = [col for col in df.columns 
                         if "_return" in col.lower() 
                         and col.lower() not in ["index_return", "lag_return"]]

    predicted_return_weighted = 0
    for ret_col in stock_return_cols:
        weight_col = ret_col.replace("_Return", "_Weight")
        # rolling average of stock returns
        avg_ret = df[ret_col].iloc[-rolling_window:].mean() / 100  # percent - decimal
        w = df[weight_col].iloc[-1] / 100  # latest weight
        predicted_return_weighted += avg_ret * w

    # --- Combine signals ---
    predicted_return = (
        ar_weight * predicted_return_ar +
        weight_signal_weight * predicted_return_weighted
    )

    # --- Predict next index price ---
    last_price = df["Index"].iloc[-1]
    predicted_price = last_price * (1 + predicted_return)

    return {
        "predicted_return": predicted_return,
        "predicted_price": predicted_price,
        "ar_prediction": predicted_return_ar,
        "weighted_prediction": predicted_return_weighted,
        "ar_model": model_ar
    }