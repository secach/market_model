import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def _to_decimal_return(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.abs().median(skipna=True) > 1:
        s = s / 100.0
    return s


def prepare_model_data(
    csv_path,
    use_sp500=False,
    use_vix=False,
    use_weighted_stocks=False,
    sp500_col="SP500_Return",
    vix_col="VIX_Return",
):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if use_sp500 and sp500_col not in df.columns:
        raise ValueError(
            f"Requested SP500 feature, but column '{sp500_col}' is missing from {csv_path}"
        )

    if use_vix and vix_col not in df.columns:
        raise ValueError(
            f"Requested VIX feature, but column '{vix_col}' is missing from {csv_path}"
        )

    if "Date" not in df.columns:
        raise ValueError("Missing required column: 'Date'")
    if "Open" not in df.columns:
        raise ValueError("Missing required column: 'Open'")
    if "Index" not in df.columns:
        raise ValueError("Missing required column: 'Index'")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Index"] = pd.to_numeric(df["Index"], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)

    # Previous close
    df["Prev_Close"] = df["Index"].shift(1)

    # Gap
    df["Gap"] = df["Open"] / df["Prev_Close"] - 1
    df["Lag_Gap"] = df["Gap"].shift(1)

    # Intraday return target
    df["Target_Return"] = df["Index"] / df["Open"] - 1
    df["Lag_Target_Return"] = df["Target_Return"].shift(1)

    # Interaction feature: gap * previous return
    df["Gap_x_LagReturn"] = df["Lag_Gap"] * df["Lag_Target_Return"]

    # Volatility regime
    df["Rolling_Vol"] = df["Target_Return"].rolling(5).std()
    df["Lag_Rolling_Vol"] = df["Rolling_Vol"].shift(1)

    # Start simple: Gap only
    feature_cols = ["Lag_Gap"]

    if use_weighted_stocks:
        stock_return_cols = [
            col
            for col in df.columns
            if col.endswith("_Return")
            and col not in {"Target_Return", sp500_col, vix_col}
            and col.replace("_Return", "_Weight") in df.columns
        ]

        weighted_stock_return = pd.Series(0.0, index=df.index)

        for ret_col in stock_return_cols:
            weight_col = ret_col.replace("_Return", "_Weight")

            stock_ret_decimal = _to_decimal_return(df[ret_col])
            weight_decimal = pd.to_numeric(df[weight_col], errors="coerce")

            if weight_decimal.abs().median(skipna=True) > 1:
                weight_decimal = weight_decimal / 100.0

            weighted_stock_return += stock_ret_decimal * weight_decimal

        df["Weighted_Stock_Return"] = weighted_stock_return
        df["Lag_Weighted_Stock_Return"] = df["Weighted_Stock_Return"].shift(1)
        feature_cols.append("Lag_Weighted_Stock_Return")

    if use_sp500:
        df["Lag_SP500_Return"] = _to_decimal_return(df[sp500_col]).shift(1)
        feature_cols.append("Lag_SP500_Return")

    if use_vix:
        df["Lag_VIX_Return"] = _to_decimal_return(df[vix_col]).shift(1)
        feature_cols.append("Lag_VIX_Return")

    model_df = df[["Date", "Target_Return"] + feature_cols].dropna().copy()

    return df, model_df, feature_cols


def compute_signal(
    csv_path,
    rolling_window=60,
    use_sp500=False,
    use_vix=False,
    use_weighted_stocks=False,
    sp500_col="SP500_Return",
    vix_col="VIX_Return",
    return_threshold=0.002,
):
    _, model_df, feature_cols = prepare_model_data(
        csv_path=csv_path,
        use_sp500=use_sp500,
        use_vix=use_vix,
        use_weighted_stocks=use_weighted_stocks,
        sp500_col=sp500_col,
        vix_col=vix_col,
    )

    if len(model_df) < rolling_window:
        raise ValueError(
            f"Not enough data after preprocessing. "
            f"Need at least {rolling_window} usable rows, got {len(model_df)}."
        )

    train_df = model_df.iloc[-rolling_window:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["Target_Return"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    latest_features = model_df[feature_cols].iloc[-1]
    X_next = pd.DataFrame([latest_features], columns=feature_cols)
    X_next_scaled = scaler.transform(X_next)

    predicted_return = float(model.predict(X_next_scaled)[0])

    if predicted_return > return_threshold:
        predicted_direction = 1
    elif predicted_return < -return_threshold:
        predicted_direction = -1
    else:
        predicted_direction = 0

    return {
        "predicted_return": predicted_return,
        "predicted_direction": predicted_direction,
        "features_used": feature_cols,
        "latest_feature_values": latest_features.to_dict(),
        "model_coefficients": dict(zip(feature_cols, model.coef_)),
        "model_intercept": float(model.intercept_),
        "return_threshold": return_threshold,
    }


def backtest_model(
    csv_path,
    rolling_window=60,
    use_sp500=False,
    use_vix=False,
    use_weighted_stocks=False,
    sp500_col="SP500_Return",
    vix_col="VIX_Return",
    return_threshold=0.002,
    long_only=False,
):
    _, model_df, feature_cols = prepare_model_data(
        csv_path=csv_path,
        use_sp500=use_sp500,
        use_vix=use_vix,
        use_weighted_stocks=use_weighted_stocks,
        sp500_col=sp500_col,
        vix_col=vix_col,
    )

    if len(model_df) <= rolling_window:
        raise ValueError(
            f"Not enough data after preprocessing. "
            f"Need more than {rolling_window} usable rows, got {len(model_df)}."
        )

    results = []

    print("Features used:", feature_cols)

    for i in range(rolling_window, len(model_df)):
        train = model_df.iloc[i - rolling_window:i].copy()
        test = model_df.iloc[i]

        X_train = train[feature_cols]
        y_train = train["Target_Return"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        X_test = pd.DataFrame([test[feature_cols]], columns=feature_cols)
        X_test_scaled = scaler.transform(X_test)

        predicted_return = float(model.predict(X_test_scaled)[0])

        if long_only:
            if predicted_return > return_threshold:
                position = 1
            else:
                position = 0
        else:
            if abs(predicted_return) > return_threshold:
                position = 1 if predicted_return > 0 else -1
            else:
                position = 0
        
        actual_return = float(test["Target_Return"])
        strategy_return = position * actual_return

        if position == 1:
            predicted_direction = 1
        elif position == -1:
            predicted_direction = 0
        else:
            predicted_direction = -1

        actual_direction = 1 if actual_return > 0 else 0

        results.append(
            {
                "Date": test["Date"],
                "Predicted_Return": predicted_return,
                "Predicted_Direction": predicted_direction,
                "Actual_Direction": actual_direction,
                "Position": position,
                "Actual_Return": actual_return,
                "Strategy_Return": strategy_return,
            }
        )

    results_df = pd.DataFrame(results)

    traded_mask = results_df["Position"] != 0
    if traded_mask.any():
        directional_accuracy = (
            results_df.loc[traded_mask, "Predicted_Direction"]
            == results_df.loc[traded_mask, "Actual_Direction"]
        ).mean()
    else:
        directional_accuracy = float("nan")

    always_up_accuracy = results_df["Actual_Direction"].mean()
    predicted_up_rate = (results_df["Predicted_Direction"] == 1).mean()
    trade_rate = traded_mask.mean()

    results_df["Cumulative_Strategy"] = (1 + results_df["Strategy_Return"]).cumprod()
    results_df["Cumulative_BuyHold"] = (1 + results_df["Actual_Return"]).cumprod()

    avg_strategy_return = results_df["Strategy_Return"].mean()
    avg_buyhold_return = results_df["Actual_Return"].mean()

    return {
        "results": results_df,
        "directional_accuracy": float(directional_accuracy)
        if pd.notna(directional_accuracy)
        else float("nan"),
        "always_up_accuracy": float(always_up_accuracy),
        "predicted_up_rate": float(predicted_up_rate),
        "trade_rate": float(trade_rate),
        "avg_strategy_return": float(avg_strategy_return),
        "avg_buyhold_return": float(avg_buyhold_return),
        "final_cumulative_strategy": float(results_df["Cumulative_Strategy"].iloc[-1]),
        "final_cumulative_buyhold": float(results_df["Cumulative_BuyHold"].iloc[-1]),
        "features_used": feature_cols,
        "return_threshold": return_threshold,
    }