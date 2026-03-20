import pandas as pd
import yfinance as yf

ta35 = pd.read_csv("data/index_data.csv")
ta35.columns = ta35.columns.str.strip()
ta35["Date"] = pd.to_datetime(ta35["Date"], errors="coerce")
ta35 = ta35.sort_values("Date").reset_index(drop=True)

sp500 = yf.download("^GSPC", start="2025-02-20", end="2026-03-25", auto_adjust=False)
sp500 = sp500[["Close"]].reset_index()
sp500["Date"] = pd.to_datetime(sp500["Date"], errors="coerce")
sp500["SP500_Return"] = sp500["Close"].pct_change()
sp500 = sp500[["Date", "SP500_Return"]].sort_values("Date").reset_index(drop=True)

merged = ta35.merge(sp500, on="Date", how="left")
merged["SP500_Return"] = merged["SP500_Return"].ffill()

merged.to_csv("data/index_data_with_sp500.csv", index=False)
print("Saved data/index_data_with_sp500.csv")
print(merged[["Date", "SP500_Return"]].head(10))
print(merged[["Date", "SP500_Return"]].tail(10))