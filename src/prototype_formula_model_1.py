# prototype_formula_model.py

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    rows_tested: int
    trades: int
    trade_rate: float
    model_accuracy: float
    baseline_always_up_accuracy: float
    baseline_open_gap_accuracy: float
    baseline_dual_return_accuracy: float
    avg_trade_return: float
    cumulative_strategy_return: float
    cumulative_buy_hold_return: float
    max_drawdown: float


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = [
        "Date",
        "TA35_Return",
        "Open_Gap",
        "dual_us_weighted_return",
        "dual_us_fraction_up",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Remove rows with missing values in required inputs/target
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    return df


def build_score(
    df: pd.DataFrame,
    w_gap: float,
    w_dual_ret: float,
    w_frac_up: float,
) -> pd.DataFrame:
    out = df.copy()

    # Center fraction_up around 0.5 so positive means "more names up than down"
    out["fraction_up_centered"] = out["dual_us_fraction_up"] - 0.5

    out["Score"] = (
        w_gap * out["Open_Gap"]
        + w_dual_ret * out["dual_us_weighted_return"]
        + w_frac_up * out["fraction_up_centered"]
    )

    out["Target_Up"] = (out["TA35_Return"] > 0).astype(int)
    return out


def expanding_walk_forward_backtest(
    df: pd.DataFrame,
    min_train_size: int,
    threshold_mode: str = "zero",
    threshold_quantile: float = 0.7,
) -> tuple[pd.DataFrame, BacktestResult]:
    """
    threshold_mode:
        - "zero": trade if Score > 0
        - "quantile": trade if Score > rolling train quantile
    """

    if len(df) <= min_train_size:
        raise ValueError(
            f"Not enough rows ({len(df)}) for min_train_size={min_train_size}"
        )

    rows = []

    for i in range(min_train_size, len(df)):
        train = df.iloc[:i].copy()
        test_row = df.iloc[i].copy()

        if threshold_mode == "zero":
            threshold = 0.0
        elif threshold_mode == "quantile":
            threshold = float(train["Score"].quantile(threshold_quantile))
        else:
            raise ValueError("threshold_mode must be 'zero' or 'quantile'")

        trade_signal = int(test_row["Score"] > threshold)
        predicted_up = int(test_row["Score"] > 0)

        always_up_pred = 1
        open_gap_pred = int(test_row["Open_Gap"] > 0)
        dual_ret_pred = int(test_row["dual_us_weighted_return"] > 0)

        realized_up = int(test_row["Target_Up"])
        realized_return = float(test_row["TA35_Return"])

        strategy_return = realized_return if trade_signal == 1 else 0.0

        rows.append(
            {
                "Date": test_row["Date"],
                "Score": float(test_row["Score"]),
                "Threshold": threshold,
                "Trade": trade_signal,
                "Predicted_Up": predicted_up,
                "Actual_Up": realized_up,
                "TA35_Return": realized_return,
                "Strategy_Return": strategy_return,
                "Always_Up_Pred": always_up_pred,
                "Open_Gap_Pred": open_gap_pred,
                "Dual_Return_Pred": dual_ret_pred,
            }
        )

    bt = pd.DataFrame(rows)

    if bt.empty:
        raise ValueError("Backtest produced no rows")

    model_accuracy = (bt["Predicted_Up"] == bt["Actual_Up"]).mean()
    baseline_always_up_accuracy = (bt["Always_Up_Pred"] == bt["Actual_Up"]).mean()
    baseline_open_gap_accuracy = (bt["Open_Gap_Pred"] == bt["Actual_Up"]).mean()
    baseline_dual_return_accuracy = (bt["Dual_Return_Pred"] == bt["Actual_Up"]).mean()

    trades = int(bt["Trade"].sum())
    trade_rate = float(bt["Trade"].mean())

    trade_returns = bt.loc[bt["Trade"] == 1, "Strategy_Return"]
    avg_trade_return = float(trade_returns.mean()) if len(trade_returns) > 0 else 0.0

    bt["Strategy_Equity"] = (1.0 + bt["Strategy_Return"]).cumprod()
    bt["BuyHold_Equity"] = (1.0 + bt["TA35_Return"]).cumprod()

    rolling_peak = bt["Strategy_Equity"].cummax()
    drawdown = bt["Strategy_Equity"] / rolling_peak - 1.0
    max_drawdown = float(drawdown.min())

    result = BacktestResult(
        rows_tested=len(bt),
        trades=trades,
        trade_rate=trade_rate,
        model_accuracy=float(model_accuracy),
        baseline_always_up_accuracy=float(baseline_always_up_accuracy),
        baseline_open_gap_accuracy=float(baseline_open_gap_accuracy),
        baseline_dual_return_accuracy=float(baseline_dual_return_accuracy),
        avg_trade_return=avg_trade_return,
        cumulative_strategy_return=float(bt["Strategy_Equity"].iloc[-1] - 1.0),
        cumulative_buy_hold_return=float(bt["BuyHold_Equity"].iloc[-1] - 1.0),
        max_drawdown=max_drawdown,
    )

    return bt, result


def print_summary(result: BacktestResult) -> None:
    print("\n=== Prototype Formula Backtest Summary ===")
    print(f"Rows tested:                  {result.rows_tested}")
    print(f"Trades:                       {result.trades}")
    print(f"Trade rate:                   {result.trade_rate:.2%}")
    print(f"Model directional accuracy:   {result.model_accuracy:.2%}")
    print(f"Always-up accuracy:           {result.baseline_always_up_accuracy:.2%}")
    print(f"Open-gap sign accuracy:       {result.baseline_open_gap_accuracy:.2%}")
    print(f"Dual-return sign accuracy:    {result.baseline_dual_return_accuracy:.2%}")
    print(f"Average return per trade:     {result.avg_trade_return:.4%}")
    print(f"Cumulative strategy return:   {result.cumulative_strategy_return:.2%}")
    print(f"Cumulative buy&hold return:   {result.cumulative_buy_hold_return:.2%}")
    print(f"Max drawdown:                 {result.max_drawdown:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype formula model for TA-35")
    parser.add_argument(
        "--csv",
        type=str,
        default="ta35_model_data.csv",
        help="Path to input CSV",
    )
    parser.add_argument("--w-gap", type=float, default=1.0, help="Weight for Open_Gap")
    parser.add_argument(
        "--w-dual-ret",
        type=float,
        default=0.3,
        help="Weight for dual_us_weighted_return",
    )
    parser.add_argument(
        "--w-frac-up",
        type=float,
        default=0.4,
        help="Weight for centered dual_us_fraction_up",
    )
    parser.add_argument(
        "--min-train-size",
        type=int,
        default=40,
        help="Minimum expanding train window size",
    )
    parser.add_argument(
        "--threshold-mode",
        type=str,
        default="zero",
        choices=["zero", "quantile"],
        help="Trade threshold mode",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.7,
        help="Used only if threshold-mode=quantile",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="prototype_formula_backtest.csv",
        help="Path to save backtest rows",
    )

    args = parser.parse_args()

    df = load_data(args.csv)
    df = build_score(
        df=df,
        w_gap=args.w_gap,
        w_dual_ret=args.w_dual_ret,
        w_frac_up=args.w_frac_up,
    )

    bt, result = expanding_walk_forward_backtest(
        df=df,
        min_train_size=args.min_train_size,
        threshold_mode=args.threshold_mode,
        threshold_quantile=args.threshold_quantile,
    )

    print_summary(result)

    output_path = Path(args.output)
    bt.to_csv(output_path, index=False)
    print(f"\nSaved backtest details to: {output_path.resolve()}")


if __name__ == "__main__":
    main()