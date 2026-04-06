import pandas as pd

# Load
df = pd.read_csv("data/dual_daily_input.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Convert weights from percent to decimal
df["Weight_Decimal"] = df["Weight"] / 100.0

# Weighted contribution
df["weighted_contribution"] = df["Weight_Decimal"] * df["US_Return"]

# Aggregate to one row per date
dual_features = df.groupby("Date").apply(
    lambda x: pd.Series({
        "dual_us_weighted_return": x["weighted_contribution"].sum(),
        "dual_us_fraction_up": (x["US_Return"] > 0).mean(),
        "dual_us_weight_sum": x["Weight_Decimal"].sum(),
    })
).reset_index()

# Normalized version
dual_features["dual_us_normalized_return"] = (
    dual_features["dual_us_weighted_return"] / dual_features["dual_us_weight_sum"]
)

# Save
dual_features.to_csv("data/dual_daily_features.csv", index=False)

print(dual_features.head(10))