import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import timedelta

OUTPUT_FILE = Path("data/yahoo_dual_returns.csv")
YFINANCE_CACHE_DIR = Path("data/.yfinance_cache")
TARGET_LAST_DATE = None   # set to "YYYY-MM-DD" to stop at a specific last date
ROLLBACK_DAYS = 7

tickers = ["TEVA", "NICE", "ESLT", "TSEM", "NVMI", "CAMT", "ENLT", "ORA", "ICL"]

YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))

if OUTPUT_FILE.exists():
    existing = pd.read_csv(OUTPUT_FILE)
    existing["Date"] = pd.to_datetime(existing["Date"], dayfirst=True, errors="coerce")

    if existing["Date"].notna().any():
        last_date = existing["Date"].max()
        start_date = (last_date - timedelta(days=ROLLBACK_DAYS)).strftime("%Y-%m-%d")
    else:
        existing = pd.DataFrame()
        start_date = "2025-05-10"
else:
    existing = pd.DataFrame()
    start_date = "2025-05-10"

if TARGET_LAST_DATE:
    end_date = (pd.to_datetime(TARGET_LAST_DATE) + timedelta(days=1)).strftime("%Y-%m-%d")
else:
    end_date = (pd.Timestamp.today().normalize() + timedelta(days=1)).strftime("%Y-%m-%d")

print(f"Downloading Yahoo data from {start_date} through before {end_date}")

raw = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=False,
    group_by="ticker",
    progress=False,
    threads=True,
)

frames = []

for ticker in tickers:
    try:
        df_t = raw[ticker].copy()
    except Exception:
        df_t = raw.copy()

    if "Adj Close" not in df_t.columns:
        continue

    s = df_t["Adj Close"].rename("Adj_Close").to_frame()
    s = s.dropna(subset=["Adj_Close"])
    s["US_Return"] = s["Adj_Close"].pct_change()
    s = s.reset_index()
    s["US_Ticker"] = ticker
    s = s[["Date", "US_Ticker", "Adj_Close", "US_Return"]]
    frames.append(s)

new_data = pd.concat(frames, ignore_index=True)
new_data["Date"] = pd.to_datetime(new_data["Date"], errors="coerce")
new_data = new_data.dropna(subset=["US_Return"])

if not existing.empty:
    combined = pd.concat([existing, new_data], ignore_index=True)
else:
    combined = new_data.copy()

combined = combined.drop_duplicates(subset=["Date", "US_Ticker"], keep="last")
combined = combined.sort_values(["US_Ticker", "Date"]).reset_index(drop=True)
combined["Date"] = combined["Date"].dt.strftime("%d/%m/%Y")

combined.to_csv(OUTPUT_FILE, index=False)
print(combined.tail(20))
print(f"Saved updated file to {OUTPUT_FILE}")
