import pandas as pd


def build_ta35_model_data(
    ta35_path: str = "data/ta35_daily.csv",
    dual_path: str = "data/dual_daily_features.csv",
    output_path: str = "data/ta35_model_data.csv",
) -> pd.DataFrame:
    """
    Build model-ready TA-35 dataset for pre-open prediction.

    Expected input files:

    ta35_daily.csv columns:
        Date, TA35_Open, TA35_Close

    dual_daily_features.csv columns:
        Date,
        dual_us_weighted_return,
        dual_us_fraction_up,
        dual_us_weight_sum,
        dual_us_normalized_return
    """

    # ---------- Load TA-35 daily ----------
    ta35 = pd.read_csv(ta35_path)

    required_ta35 = {"Date", "TA35_Open", "TA35_Close"}
    missing_ta35 = required_ta35 - set(ta35.columns)
    if missing_ta35:
        raise ValueError(f"Missing TA-35 columns: {missing_ta35}")

    ta35["Date"] = pd.to_datetime(ta35["Date"])
    ta35 = ta35.sort_values("Date").reset_index(drop=True)

    # ---------- Load dual daily features ----------
    dual = pd.read_csv(dual_path)

    required_dual = {
        "Date",
        "dual_us_weighted_return",
        "dual_us_fraction_up",
        "dual_us_weight_sum",
        "dual_us_normalized_return",
    }
    missing_dual = required_dual - set(dual.columns)
    if missing_dual:
        raise ValueError(f"Missing dual feature columns: {missing_dual}")

    dual["Date"] = pd.to_datetime(dual["Date"])
    dual = dual.sort_values("Date").reset_index(drop=True)

    # ---------- Keep overlap only ----------
    start_date = max(ta35["Date"].min(), dual["Date"].min())
    end_date = min(ta35["Date"].max(), dual["Date"].max())

    ta35 = ta35[(ta35["Date"] >= start_date) & (ta35["Date"] <= end_date)].copy()
    dual = dual[(dual["Date"] >= start_date) & (dual["Date"] <= end_date)].copy()

    print(f"Overlap range: {start_date.date()} to {end_date.date()}")
    print(f"TA-35 rows in overlap: {len(ta35)}")
    print(f"Dual rows in overlap: {len(dual)}")

    # ---------- Merge ----------
    merged = ta35.merge(dual, on="Date", how="inner")

    print(f"Merged rows: {len(merged)}")

    # ---------- Create derived columns ----------
    merged["Prev_Close"] = merged["TA35_Close"].shift(1)
    merged["TA35_Return"] = merged["TA35_Close"] / merged["Prev_Close"] - 1
    merged["Lag1_Return"] = merged["TA35_Return"].shift(1)
    merged["RollingVol_5"] = merged["TA35_Return"].rolling(5).std()

    # Pre-open target
    merged["Open_Gap"] = merged["TA35_Open"] / merged["Prev_Close"] - 1
    merged["Open_Gap_Sign"] = (merged["Open_Gap"] > 0).astype("Int64")

    # ---------- Drop rows that cannot be used ----------
    model_df = merged.dropna(
        subset=[
            "Prev_Close",
            "TA35_Return",
            "Lag1_Return",
            "RollingVol_5",
            "Open_Gap",
        ]
    ).reset_index(drop=True)

    # ---------- Save ----------
    model_df.to_csv(output_path, index=False)
    print(f"Saved model dataset to: {output_path}")
    print(model_df.head(10))

    return model_df


if __name__ == "__main__":
    build_ta35_model_data()