from pathlib import Path

import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

TA35_SOURCE_PATH = DATA_DIR / "index_data.csv"
FULL_OUTPUT_PATH = DATA_DIR / "index_data_with_features_full.csv"
CLEAN_OUTPUT_PATH = DATA_DIR / "index_data_with_features_clean.csv"

INITIAL_START_DATE = pd.Timestamp("2025-02-20")
BUFFER_DAYS = 7


def normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_yf_history(symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )

    df = normalize_yf_columns(df).reset_index()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in ["Open", "Close", "High", "Low", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_update_start_date() -> pd.Timestamp:
    if not FULL_OUTPUT_PATH.exists():
        return INITIAL_START_DATE

    existing = pd.read_csv(FULL_OUTPUT_PATH, usecols=["Date"])
    existing["Date"] = pd.to_datetime(existing["Date"], errors="coerce")
    last_date = existing["Date"].max()

    if pd.isna(last_date):
        return INITIAL_START_DATE

    return last_date - pd.Timedelta(days=BUFFER_DAYS)


def main() -> None:
    if not TA35_SOURCE_PATH.exists():
        raise FileNotFoundError(f"TA-35 source file not found: {TA35_SOURCE_PATH}")

    ta35 = pd.read_csv(TA35_SOURCE_PATH)
    ta35.columns = ta35.columns.str.strip()
    ta35["Date"] = pd.to_datetime(ta35["Date"], errors="coerce")
    ta35 = ta35.sort_values("Date").reset_index(drop=True)

    update_start = get_update_start_date()
    update_end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)

    print(f"Downloading market data from {update_start.date()} to {update_end.date()}")

    ta35_yf = download_yf_history("TA35.TA", update_start, update_end)
    sp500 = download_yf_history("^GSPC", update_start, update_end)
    vix = download_yf_history("^VIX", update_start, update_end)

    ta35_open = ta35_yf[["Date", "Open"]].copy()
    ta35_open = ta35_open.sort_values("Date").reset_index(drop=True)

    sp500 = sp500[["Date", "Close"]].copy()
    sp500 = sp500.rename(columns={"Close": "SP500_Close"})
    sp500 = sp500.sort_values("Date").reset_index(drop=True)

    vix = vix[["Date", "Close"]].copy()
    vix = vix.rename(columns={"Close": "VIX_Close"})
    vix = vix.sort_values("Date").reset_index(drop=True)

    ta35["Date"] = pd.to_datetime(ta35["Date"], errors="coerce").dt.normalize()
    ta35_yf["Date"] = pd.to_datetime(ta35_yf["Date"], errors="coerce").dt.normalize()
    sp500["Date"] = pd.to_datetime(sp500["Date"], errors="coerce").dt.normalize()
    vix["Date"] = pd.to_datetime(vix["Date"], errors="coerce").dt.normalize()

    print("TA35/SP500 date overlap:", ta35["Date"].isin(sp500["Date"]).sum())
    print("TA35/VIX date overlap:", ta35["Date"].isin(vix["Date"]).sum())
    print("TA35/TA35_open overlap:", ta35["Date"].isin(ta35_open["Date"]).sum())

    merged_new = ta35.merge(ta35_open, on="Date", how="left")
    merged_new = merged_new.merge(sp500, on="Date", how="left")
    merged_new = merged_new.merge(vix, on="Date", how="left")

    if FULL_OUTPUT_PATH.exists():
        existing_full = pd.read_csv(FULL_OUTPUT_PATH)
        existing_full["Date"] = pd.to_datetime(existing_full["Date"], errors="coerce")

        merged = pd.concat([existing_full, merged_new], ignore_index=True)
        merged = (
            merged
            .sort_values("Date")
            .drop_duplicates(subset="Date", keep="last")
            .reset_index(drop=True)
        )
    else:
        merged = (
            merged_new
            .sort_values("Date")
            .drop_duplicates(subset="Date", keep="last")
            .reset_index(drop=True)
        )

    merged["SP500_Close"] = pd.to_numeric(merged["SP500_Close"], errors="coerce")
    merged["VIX_Close"] = pd.to_numeric(merged["VIX_Close"], errors="coerce")

    merged = merged.sort_values("Date").reset_index(drop=True)

    merged["SP500_Return"] = merged["SP500_Close"].pct_change()
    merged["VIX_Return"] = merged["VIX_Close"].pct_change()

    # --- CLEAN DATASETS ---

    # 1. Basic dataset (TA-35 only)
    clean_basic = merged.dropna(subset=["Open"]).copy()

    # 2. US-aligned dataset (TA-35 + SP500 + VIX)
    clean_us = merged.dropna(subset=["Open", "SP500_Return", "VIX_Return"]).copy()

    # --- SAVE ---
    FULL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(FULL_OUTPUT_PATH, index=False)

    clean_basic.to_csv("data/index_data_clean_basic.csv", index=False)
    clean_us.to_csv("data/index_data_clean_us.csv", index=False)

    print("Rows basic:", len(clean_basic))
    print("Rows US:", len(clean_us))

    print(f"Rows in full output: {len(merged)}")
    print(f"Rows in clean_basic output: {len(clean_basic)}")
    print(f"Rows in clean_us output: {len(clean_us)}")
    print("Missing Open values in full output:", merged["Open"].isna().sum())
    print("Missing SP500_Return values in full output:", merged["SP500_Return"].isna().sum())
    print("Missing VIX_Return values in full output:", merged["VIX_Return"].isna().sum())
    print(merged[["Date", "Open", "SP500_Return", "VIX_Return"]].tail(10))


if __name__ == "__main__":
    main()