# monthly_expiry_spread_model.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = [
        "Date",
        "TA35_Close",
        "TA35_Return",
        "Lag1_Return",
        "RollingVol_5",
        "Open_Gap",
        "dual_us_weighted_return",
        "dual_us_fraction_up",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def add_medium_horizon_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ret_5d"] = out["TA35_Close"].pct_change(5)
    out["ret_10d"] = out["TA35_Close"].pct_change(10)

    out["ma_10"] = out["TA35_Close"].rolling(10).mean()

    out["dist_ma_10"] = out["TA35_Close"] / out["ma_10"] - 1.0

    out["vol_10"] = out["TA35_Return"].rolling(10).std()

    return out


def get_monthly_expiry_dates(expiry_csv_path: str) -> pd.DataFrame:
    expiry_df = pd.read_csv(expiry_csv_path)
    expiry_df["Expiry_Date"] = pd.to_datetime(expiry_df["Expiry_Date"])
    expiry_df["YearMonth"] = pd.PeriodIndex(expiry_df["Expiry_Month"], freq="M")
    return expiry_df[["YearMonth", "Expiry_Date"]]


def build_monthly_dataset(
    df: pd.DataFrame,
    expiry_csv_path: str,
    short_offset_pct: float = 0.01,
    spread_width_points: float = 10.0,
    min_days_to_expiry: int = 5,
) -> pd.DataFrame:
    """
    Build one row per entry date.
    Candidate spread is generated from spot:
      short strike = rounded spot * (1 + short_offset_pct)
      long strike = short strike + spread_width_points

    For a bullish version later, you'd mirror this differently.
    """
    df = add_medium_horizon_features(df)
    expiry_table = get_monthly_expiry_dates(expiry_csv_path)

    df = df.merge(
        expiry_table,
        left_on=df["Date"].dt.to_period("M"),
        right_on=expiry_table["YearMonth"],
        how="left",
    ).drop(columns=["key_0", "YearMonth"])

    # Look up expiry close before filtering out expiry-day rows.
    expiry_close_map = (
        df[["Date", "TA35_Close"]]
        .drop_duplicates()
        .rename(columns={"Date": "Expiry_Date", "TA35_Close": "Expiry_Close"})
    )

    df = df.merge(expiry_close_map, on="Expiry_Date", how="left")

    # Remove rows that are too close to expiry for entry.
    df["Days_To_Expiry"] = (df["Expiry_Date"] - df["Date"]).dt.days
    df = df[df["Days_To_Expiry"] >= min_days_to_expiry].copy()

    # Candidate bear call spread around current spot
    spot = df["TA35_Close"]

    raw_short = spot * (1.0 + short_offset_pct)

    # Round strike to nearest 10 points
    df["Short_Strike"] = (np.round(raw_short / 10.0) * 10.0).astype(float)
    df["Long_Strike"] = df["Short_Strike"] + spread_width_points

    # Distances / moneyness
    df["Dist_To_Short_Pct"] = df["Short_Strike"] / df["TA35_Close"] - 1.0
    df["Dist_To_Long_Pct"] = df["Long_Strike"] / df["TA35_Close"] - 1.0
    df["Spread_Width"] = df["Long_Strike"] - df["Short_Strike"]

    # Binary target:
    # 1 = spread finishes in best zone for bear call, below short strike
    # 0 = not in best zone
    df["Target_Binary"] = (df["Expiry_Close"] < df["Short_Strike"]).astype(int)

    df["Expiry_Return_From_Entry"] = df["Expiry_Close"] / df["TA35_Close"] - 1.0

    feature_cols = [
        "TA35_Close",
        "Days_To_Expiry",
        "Dist_To_Short_Pct",
        "Dist_To_Long_Pct",
        "Spread_Width",
        "Lag1_Return",
        "RollingVol_5",
        "Open_Gap",
        "dual_us_weighted_return",
        "dual_us_fraction_up",
        "ret_5d",
        "ret_10d",
        "dist_ma_10",
        "vol_10",
    ]

    df = df.dropna(subset=feature_cols + ["Target_Binary"]).reset_index(drop=True)

    return df


def time_split_by_expiry_month(
    df: pd.DataFrame,
    min_test_months: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["Expiry_Month"] = work["Expiry_Date"].dt.to_period("M")

    months = sorted(work["Expiry_Month"].unique())
    if len(months) < 2:
        raise ValueError(
            f"Need at least 2 distinct expiry months, got {len(months)}"
        )

    test_months = months[-min_test_months:]
    train_months = months[:-min_test_months]

    train = work[work["Expiry_Month"].isin(train_months)].copy()
    test = work[work["Expiry_Month"].isin(test_months)].copy()

    print("\n=== Expiry-month split ===")
    print(f"Train months: {[str(m) for m in train_months]}")
    print(f"Test months:  {[str(m) for m in test_months]}")
    print(f"Train rows: {len(train)} | Test rows: {len(test)}")

    train = train.drop(columns=["Expiry_Month"])
    test = test.drop(columns=["Expiry_Month"])

    return train, test


def print_class_balance(df: pd.DataFrame, name: str) -> None:
    label_map = {
        0: "Not_Best_Zone",
        1: "Below_Short",
    }

    counts = df["Target_Binary"].astype(int).value_counts().sort_index()
    print(f"\n=== Class balance: {name} ===")
    total = len(df)
    for k in [0, 1]:
        c = int(counts.get(k, 0))
        pct = c / total if total else 0.0
        print(f"{label_map[k]:>14}: {c:>4} ({pct:.2%})")


def train_binary_model(train_df: pd.DataFrame, feature_cols: list[str]):
    X_train = train_df[feature_cols]
    y_train = train_df["Target_Binary"].astype(int)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def baseline_threshold_predict(df: pd.DataFrame, threshold: float) -> pd.Series:
    return (df["Dist_To_Short_Pct"] > threshold).astype(int)


def baseline_accuracy(df: pd.DataFrame, threshold: float) -> float:
    preds = baseline_threshold_predict(df, threshold)
    return accuracy_score(df["Target_Binary"], preds)


def prob_bucket(p: float) -> str:
    if p >= 0.70:
        return "high"
    if p >= 0.55:
        return "medium"
    return "low"


def evaluate_binary_model(model, test_df: pd.DataFrame, feature_cols: list[str]) -> None:
    X_test = test_df[feature_cols]
    y_test = test_df["Target_Binary"].astype(int)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n=== Binary Classification Report ===")
    print(classification_report(y_test, preds, digits=4, zero_division=0))

    print("\n=== Binary Confusion Matrix ===")
    print(confusion_matrix(y_test, preds))

    out = test_df[["Date", "Expiry_Date", "TA35_Close", "Short_Strike", "Expiry_Close"]].copy()
    out["P_Below_Short"] = probs
    out["Rank_Bucket"] = out["P_Below_Short"].map(prob_bucket)
    print("\n=== Binary Sample Predictions ===")
    print(out.tail(10).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Monthly expiry spread model prototype")
    parser.add_argument("--csv", type=str, default="data/ta35_model_data.csv")
    parser.add_argument(
        "--expiry-csv",
        type=str,
        default="data/ta35_expiry_dates.csv",
        help="Path to CSV with actual monthly expiry dates",
    )
    parser.add_argument("--short-offset-pct", type=float, default=0.01)
    parser.add_argument("--spread-width-points", type=float, default=10.0)
    parser.add_argument("--min-days-to-expiry", type=int, default=5)
    parser.add_argument("--min-test-months", type=int, default=1)
    parser.add_argument("--output", type=str, default="data/monthly_expiry_dataset.csv")
    args = parser.parse_args()

    df = load_data(args.csv)
    offsets = [0.005, 0.01, 0.015]

    feature_cols = [
        "TA35_Close",
        "Days_To_Expiry",
        "Dist_To_Short_Pct",
        "Dist_To_Long_Pct",
        "Spread_Width",
        "Lag1_Return",
        "RollingVol_5",
        "Open_Gap",
        "dual_us_weighted_return",
        "dual_us_fraction_up",
        "ret_5d",
        "ret_10d",
        "dist_ma_10",
        "vol_10",
    ]

    summary_rows = []

    for offset in offsets:
        monthly_df = build_monthly_dataset(
            df=df,
            expiry_csv_path=args.expiry_csv,
            short_offset_pct=offset,
            spread_width_points=args.spread_width_points,
            min_days_to_expiry=args.min_days_to_expiry,
        )

        if len(monthly_df) < 20:
            print(
                f"Warning: only {len(monthly_df)} monthly rows were built. "
                "This is enough for prototype code, not enough for trust."
            )

        print(f"\n\n######## OFFSET = {offset:.3%} ########")
        print_class_balance(monthly_df, "full dataset")

        train_df, test_df = time_split_by_expiry_month(monthly_df, min_test_months=1)
        print_class_balance(train_df, "train")
        print_class_balance(test_df, "test")

        best_baseline = None
        for thr in [0.004, 0.007, 0.01]:
            acc = baseline_accuracy(test_df, thr)
            print(f"Baseline accuracy at {thr:.3%}: {acc:.4f}")
            if best_baseline is None or acc > best_baseline:
                best_baseline = acc

            preds = baseline_threshold_predict(test_df, thr)
            print(f"\n=== Baseline threshold {thr:.3%} ===")
            print(
                classification_report(
                    test_df["Target_Binary"],
                    preds,
                    digits=4,
                    zero_division=0,
                )
            )
            print(confusion_matrix(test_df["Target_Binary"], preds))

        model = train_binary_model(train_df, feature_cols)
        coef_df = pd.DataFrame({
            "feature": feature_cols,
            "coef": model.coef_[0],
        }).sort_values("coef", ascending=False)

        print("\n=== Binary model coefficients ===")
        print(coef_df.to_string(index=False))

        evaluate_binary_model(model, test_df, feature_cols)

        X_test = test_df[feature_cols]
        y_test = test_df["Target_Binary"].astype(int)
        preds = model.predict(X_test)
        model_acc = accuracy_score(y_test, preds)
        print(f"Model accuracy: {model_acc:.4f} | Best baseline: {best_baseline:.4f}")

        summary_rows.append({
            "offset": offset,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_pos_rate": train_df["Target_Binary"].mean(),
            "test_pos_rate": test_df["Target_Binary"].mean(),
            "pred_pos_rate": preds.mean(),
        })

        offset_tag = str(offset).replace(".", "p")
        output_path = Path(args.output).with_name(
            f"{Path(args.output).stem}_offset_{offset_tag}.csv"
        )
        monthly_df.to_csv(output_path, index=False)
        print(f"\nSaved monthly dataset to: {output_path.resolve()}")

    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Offset summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
