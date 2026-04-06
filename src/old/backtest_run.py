from pathlib import Path
from model import backtest_model

BASE_DIR = Path(__file__).resolve().parent.parent

# --- MODEL 1: BASIC ---
print("\n--- BASIC MODEL (TA-35 only) ---")

csv_path = BASE_DIR / "data" / "index_data_clean_basic.csv"

for threshold in [0.001, 0.002, 0.003]:
    print(f"\n--- GAP MODEL return_threshold={threshold} ---")

    result = backtest_model(
        csv_path=csv_path,
        rolling_window=60,
        use_sp500=False,
        use_vix=False,
        use_weighted_stocks=False,
        return_threshold=threshold,
        long_only=True
    )

    print("Trade rate:", result["trade_rate"])
    print("Final strategy:", result["final_cumulative_strategy"])
    print("Buy & hold:", result["final_cumulative_buyhold"])
    print("Directional accuracy:", result["directional_accuracy"])
    print("Always-up accuracy:", result["always_up_accuracy"])


for threshold in [0.001, 0.002, 0.003]:
    print(f"\n--- GAP MODEL return_threshold={threshold} ---")

    result = backtest_model(
        csv_path=csv_path,
        rolling_window=60,
        use_sp500=False,
        use_vix=False,
        use_weighted_stocks=False,
        return_threshold=threshold,
        long_only=False
    )

    print("Trade rate:", result["trade_rate"])
    print("Final strategy:", result["final_cumulative_strategy"])
    print("Buy & hold:", result["final_cumulative_buyhold"])
    print("Directional accuracy:", result["directional_accuracy"])
    print("Always-up accuracy:", result["always_up_accuracy"])

# # --- MODEL 2: WITH US FEATURES ---
# print("\n--- MODEL WITH SP500 + VIX ---")

# csv_path = BASE_DIR / "data" / "index_data_clean_us.csv"

# result = backtest_model(
#     csv_path=csv_path,
#     rolling_window=60,
#     use_sp500=True,
#     use_vix=True,
#     use_weighted_stocks=False,
#     return_threshold=0.002,
#     long_only=True
# )

# print("Directional accuracy:", result["directional_accuracy"])
# print("Always-up accuracy:", result["always_up_accuracy"])
# print("Trade rate:", result["trade_rate"])
# print("Final strategy:", result["final_cumulative_strategy"])
# print("Buy & hold:", result["final_cumulative_buyhold"])