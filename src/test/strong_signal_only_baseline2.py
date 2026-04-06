import pandas as pd

df = pd.read_csv("data/ta35_model_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["OpenToClose"] = df["TA35_Close"] / df["TA35_Open"] - 1

window_train = 40
window_test = 10

results = []

start = 0
while start + window_train + window_test <= len(df):
    train = df.iloc[start:start + window_train].copy()
    test = df.iloc[start + window_train:start + window_train + window_test].copy()

    threshold = train["dual_us_weighted_return"].quantile(0.2)

    test["trade_allowed"] = test["dual_us_weighted_return"] > threshold

    all_days_avg = test["OpenToClose"].mean()
    allowed_avg = test.loc[test["trade_allowed"], "OpenToClose"].mean()
    allowed_count = test["trade_allowed"].sum()

    results.append({
        "train_start": train["Date"].iloc[0],
        "train_end": train["Date"].iloc[-1],
        "test_start": test["Date"].iloc[0],
        "test_end": test["Date"].iloc[-1],
        "all_days_avg_otc": all_days_avg,
        "allowed_days_avg_otc": allowed_avg,
        "allowed_days_count": allowed_count,
    })

    start += window_test

res = pd.DataFrame(results)
print(res)
print("\nAverage across test blocks:")
print(res[["all_days_avg_otc", "allowed_days_avg_otc"]].mean())