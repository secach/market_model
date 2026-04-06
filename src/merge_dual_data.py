import pandas as pd

# -----------------------------
# Files
# -----------------------------
WEIGHTS_FILE = "data/dual_weights.csv"
YAHOO_FILE = "data/yahoo_dual_returns.csv"
OUTPUT_FILE = "data/dual_daily_input.csv"

weights = pd.read_csv(WEIGHTS_FILE)
yahoo_df = pd.read_csv(YAHOO_FILE)

required_weights = {"Effective_Date", "Security_ID", "TASE_Symbol", "US_Ticker", "Weight"}
required_yahoo = {"Date", "US_Ticker", "US_Return"}

missing_weights = required_weights - set(weights.columns)
missing_yahoo = required_yahoo - set(yahoo_df.columns)

if missing_weights:
    raise ValueError(f"Missing columns in {WEIGHTS_FILE}: {missing_weights}")

if missing_yahoo:
    raise ValueError(f"Missing columns in {YAHOO_FILE}: {missing_yahoo}")

weights["Effective_Date"] = pd.to_datetime(weights["Effective_Date"], errors="coerce")
yahoo_df["Date"] = pd.to_datetime(yahoo_df["Date"], dayfirst=True, errors="coerce")

if weights["Effective_Date"].isna().any():
    bad_rows = weights[weights["Effective_Date"].isna()]
    raise ValueError(f"Bad Effective_Date values in {WEIGHTS_FILE}:\n{bad_rows}")

if yahoo_df["Date"].isna().any():
    bad_rows = yahoo_df[yahoo_df["Date"].isna()]
    raise ValueError(f"Bad Date values in {YAHOO_FILE}:\n{bad_rows}")

weights["US_Ticker"] = weights["US_Ticker"].astype(str).str.strip()
weights["TASE_Symbol"] = weights["TASE_Symbol"].astype(str).str.strip()
yahoo_df["US_Ticker"] = yahoo_df["US_Ticker"].astype(str).str.strip()

weights["Weight"] = pd.to_numeric(weights["Weight"], errors="coerce")
yahoo_df["US_Return"] = pd.to_numeric(yahoo_df["US_Return"], errors="coerce")

weights = weights.dropna(subset=["Weight"]).copy()
yahoo_df = yahoo_df.dropna(subset=["US_Return"]).copy()

weights = weights.sort_values(["US_Ticker", "Effective_Date"]).reset_index(drop=True)
yahoo_df = yahoo_df.sort_values(["US_Ticker", "Date"]).reset_index(drop=True)

merged_parts = []

tickers = sorted(yahoo_df["US_Ticker"].dropna().unique())

for ticker in tickers:
    y = yahoo_df[yahoo_df["US_Ticker"] == ticker].copy()
    w = weights[weights["US_Ticker"] == ticker].copy()

    if w.empty:
        print(f"Warning: no weights found for {ticker}")
        continue

    # Drop duplicate ticker column from right side before merge
    w = w.drop(columns=["US_Ticker"])

    merged = pd.merge_asof(
        y.sort_values("Date"),
        w.sort_values("Effective_Date"),
        left_on="Date",
        right_on="Effective_Date",
        direction="backward"
    )

    merged_parts.append(merged)

if not merged_parts:
    raise ValueError("Nothing was merged. Check ticker names in both files.")

final_df = pd.concat(merged_parts, ignore_index=True)

final_df = final_df.dropna(subset=["Weight"]).copy()

final_df = final_df[
    ["Date", "Security_ID", "TASE_Symbol", "US_Ticker", "Weight", "US_Return"]
].copy()

final_df = final_df.sort_values(["Date", "TASE_Symbol"]).reset_index(drop=True)
final_df["Date"] = final_df["Date"].dt.strftime("%d/%m/%Y")

final_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(final_df)} rows to {OUTPUT_FILE}")
print(final_df.head(20))