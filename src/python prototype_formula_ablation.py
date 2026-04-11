# prototype_formula_ablation.py

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class BacktestResult:
    model_name: str
    rows_tested: int
    trades: int
    trade_rate: float
    model_accuracy: float
    baseline_always_up_accuracy: float
    baseline_open_gap_accuracy: float
    baseline_dual_return_accuracy: float
    avg_trade_return: float
    median_trade_return: float
    avg_skip_return: float
    median_skip_return: float
    traded_up_rate: float
    skipped_up_rate: float
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
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    df["Target_Up"] = (df["TA35_Return"] > 0).astype(int)
    df["fraction_up_centered"] = df["dual_us_fraction_up"] - 0.5

    return df


def build_score(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = df.copy()

    if model_name == "gap_only":
        out["Score"] = out["Open_Gap"]

    elif model_name == "gap_dual_ret":
        out["Score"] = (
            1.0 * out["Open_Gap"]
            + 0.3 * out["dual_us_weighted_return"]
        )

    elif model_name == "gap_frac_up":
        out["Score"] = (
            1.0 * out["Open_Gap"]
            + 0.4 * out["fraction_up_centered"]
        )

    elif model_name == "full_formula":
        out["Score"] = (
            1.0 * out["Open_Gap"]
            + 0.3 * out["dual_us_weighted_return"]
            + 0.4 * out["fraction_up_centered"]
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return out


def expanding_walk_forward_backtest(
    df: pd.DataFrame,
    min_train_size: int,
    threshold_mode: str = "zero",
    threshold_quantile: float = 0.7,
) -> tuple[pd.DataFrame, BacktestResult]:
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

    bt["Strategy_Equity"] = (1.0 + bt["Strategy_Return"]).cumprod()
    bt["BuyHold_Equity"] = (1.0 + bt["TA35_Return"]).cumprod()

    traded = bt[bt["Trade"] == 1].copy()
    skipped = bt[bt["Trade"] == 0].copy()

    rolling_peak = bt["Strategy_Equity"].cummax()
    drawdown = bt["Strategy_Equity"] / rolling_peak - 1.0

    result = BacktestResult(
        model_name="",
        rows_tested=len(bt),
        trades=int(bt["Trade"].sum()),
        trade_rate=float(bt["Trade"].mean()),
        model_accuracy=float((bt["Predicted_Up"] == bt["Actual_Up"]).mean()),
        baseline_always_up_accuracy=float((bt["Always_Up_Pred"] == bt["Actual_Up"]).mean()),
        baseline_open_gap_accuracy=float((bt["Open_Gap_Pred"] == bt["Actual_Up"]).mean()),
        baseline_dual_return_accuracy=float((bt["Dual_Return_Pred"] == bt["Actual_Up"]).mean()),
        avg_trade_return=float(traded["Strategy_Return"].mean()) if not traded.empty else 0.0,
        median_trade_return=float(traded["Strategy_Return"].median()) if not traded.empty else 0.0,
        avg_skip_return=float(skipped["TA35_Return"].mean()) if not skipped.empty else 0.0,
        median_skip_return=float(skipped["TA35_Return"].median()) if not skipped.empty else 0.0,
        traded_up_rate=float(traded["Actual_Up"].mean()) if not traded.empty else 0.0,
        skipped_up_rate=float(skipped["Actual_Up"].mean()) if not skipped.empty else 0.0,
        cumulative_strategy_return=float(bt["Strategy_Equity"].iloc[-1] - 1.0),
        cumulative_buy_hold_return=float(bt["BuyHold_Equity"].iloc[-1] - 1.0),
        max_drawdown=float(drawdown.min()),
    )

    return bt, result


def print_result(r: BacktestResult) -> None:
    print(f"\n=== {r.model_name} ===")
    print(f"Rows tested:                  {r.rows_tested}")
    print(f"Trades:                       {r.trades}")
    print(f"Trade rate:                   {r.trade_rate:.2%}")
    print(f"Model directional accuracy:   {r.model_accuracy:.2%}")
    print(f"Always-up accuracy:           {r.baseline_always_up_accuracy:.2%}")
    print(f"Open-gap sign accuracy:       {r.baseline_open_gap_accuracy:.2%}")
    print(f"Dual-return sign accuracy:    {r.baseline_dual_return_accuracy:.2%}")
    print(f"Traded days up-rate:          {r.traded_up_rate:.2%}")
    print(f"Skipped days up-rate:         {r.skipped_up_rate:.2%}")
    print(f"Average trade return:         {r.avg_trade_return:.4%}")
    print(f"Median trade return:          {r.median_trade_return:.4%}")
    print(f"Average skipped-day return:   {r.avg_skip_return:.4%}")
    print(f"Median skipped-day return:    {r.median_skip_return:.4%}")
    print(f"Cumulative strategy return:   {r.cumulative_strategy_return:.2%}")
    print(f"Cumulative buy&hold return:   {r.cumulative_buy_hold_return:.2%}")
    print(f"Max drawdown:                 {r.max_drawdown:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype TA-35 formula ablation")
    parser.add_argument("--csv", type=str, default="ta35_model_data.csv")
    parser.add_argument("--min-train-size", type=int, default=40)
    parser.add_argument(
        "--threshold-mode",
        type=str,
        default="zero",
        choices=["zero", "quantile"],
    )
    parser.add_argument("--threshold-quantile", type=float, default=0.7)
    parser.add_argument("--output-dir", type=str, default="output_ablation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv)

    model_names = [
        "gap_only",
        "gap_dual_ret",
        "gap_frac_up",
        "full_formula",
    ]

    summary_rows = []

    for model_name in model_names:
        df_model = build_score(df, model_name)

        bt, result = expanding_walk_forward_backtest(
            df=df_model,
            min_train_size=args.min_train_size,
            threshold_mode=args.threshold_mode,
            threshold_quantile=args.threshold_quantile,
        )

        result.model_name = model_name
        print_result(result)

        bt.to_csv(output_dir / f"{model_name}_backtest.csv", index=False)

        summary_rows.append(
            {
                "model_name": result.model_name,
                "rows_tested": result.rows_tested,
                "trades": result.trades,
                "trade_rate": result.trade_rate,
                "model_accuracy": result.model_accuracy,
                "always_up_accuracy": result.baseline_always_up_accuracy,
                "open_gap_accuracy": result.baseline_open_gap_accuracy,
                "dual_return_accuracy": result.baseline_dual_return_accuracy,
                "traded_up_rate": result.traded_up_rate,
                "skipped_up_rate": result.skipped_up_rate,
                "avg_trade_return": result.avg_trade_return,
                "median_trade_return": result.median_trade_return,
                "avg_skip_return": result.avg_skip_return,
                "median_skip_return": result.median_skip_return,
                "cumulative_strategy_return": result.cumulative_strategy_return,
                "cumulative_buy_hold_return": result.cumulative_buy_hold_return,
                "max_drawdown": result.max_drawdown,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "ablation_summary.csv", index=False)

    print(f"\nSaved summary to: {output_dir / 'ablation_summary.csv'}")


if __name__ == "__main__":
    main()