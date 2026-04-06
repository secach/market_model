from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "ta35_model_data.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "output" / "backtest_results.csv"

DEFAULT_FEATURES = [
    "dual_us_weighted_return",
    "dual_us_fraction_up",
    "dual_us_normalized_return",
    "Lag1_Return",
    "RollingVol_5",
    "Open_Gap",
]


def load_backtest_data(csv_path: Path, features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_columns = {"Date", "TA35_Open", "TA35_Close", "Prev_Close"} | set(features)
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing_columns)}")

    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Found invalid dates in input data.")

    numeric_columns = ["TA35_Open", "TA35_Close", "Prev_Close", *features]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)
    df["Intraday_Return"] = df["TA35_Close"] / df["TA35_Open"] - 1
    df = df.dropna(subset=[*numeric_columns, "Intraday_Return"]).reset_index(drop=True)

    return df


def run_backtest(
    df: pd.DataFrame,
    features: list[str],
    rolling_window: int,
    return_threshold: float,
    long_only: bool,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    if len(df) <= rolling_window:
        raise ValueError(
            f"Not enough rows for rolling backtest. Need more than {rolling_window}, got {len(df)}."
        )

    results: list[dict[str, float | int | pd.Timestamp]] = []

    for i in range(rolling_window, len(df)):
        train = df.iloc[i - rolling_window : i]
        test = df.iloc[i]

        X_train = train[features]
        y_train = train["Intraday_Return"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        X_test = pd.DataFrame([test[features]], columns=features)
        predicted_return = float(model.predict(scaler.transform(X_test))[0])

        if long_only:
            position = 1 if predicted_return > return_threshold else 0
        else:
            if predicted_return > return_threshold:
                position = 1
            elif predicted_return < -return_threshold:
                position = -1
            else:
                position = 0

        actual_return = float(test["Intraday_Return"])
        strategy_return = position * actual_return

        results.append(
            {
                "Date": test["Date"],
                "Predicted_Return": predicted_return,
                "Actual_Return": actual_return,
                "Position": position,
                "Strategy_Return": strategy_return,
                "Predicted_Up": int(predicted_return > 0),
                "Actual_Up": int(actual_return > 0),
            }
        )

    results_df = pd.DataFrame(results)
    results_df["Cumulative_Strategy"] = (1 + results_df["Strategy_Return"]).cumprod()
    results_df["Cumulative_BuyHold"] = (1 + results_df["Actual_Return"]).cumprod()

    traded = results_df["Position"] != 0
    traded_df = results_df.loc[traded]

    summary: dict[str, float | int | str] = {
        "rows_used": int(len(df)),
        "trades": int(traded.sum()),
        "trade_rate": float(traded.mean()),
        "directional_accuracy_when_trading": float(
            (traded_df["Predicted_Up"] == traded_df["Actual_Up"]).mean()
        )
        if not traded_df.empty
        else float("nan"),
        "always_up_accuracy": float(results_df["Actual_Up"].mean()),
        "avg_strategy_return": float(results_df["Strategy_Return"].mean()),
        "avg_buy_hold_return": float(results_df["Actual_Return"].mean()),
        "final_cumulative_strategy": float(results_df["Cumulative_Strategy"].iloc[-1]),
        "final_cumulative_buy_hold": float(results_df["Cumulative_BuyHold"].iloc[-1]),
        "rolling_window": int(rolling_window),
        "return_threshold": float(return_threshold),
        "mode": "long_only" if long_only else "long_short",
    }

    return results_df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a rolling TA-35 intraday backtest on the model dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the input CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save the detailed backtest results CSV.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=60,
        help="Number of prior rows used for each rolling regression fit.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.002,
        help="Minimum absolute predicted return needed to open a position.",
    )
    parser.add_argument(
        "--long-only",
        action="store_true",
        help="Only take long trades. Otherwise trade long/short/flat.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURES,
        help="Feature columns to use from the dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_backtest_data(args.data, args.features)
    results_df, summary = run_backtest(
        df=df,
        features=args.features,
        rolling_window=args.rolling_window,
        return_threshold=args.threshold,
        long_only=args.long_only,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)

    print(f"Saved detailed results to: {args.output}")
    print(f"Features used: {', '.join(args.features)}")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
