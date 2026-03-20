from model import backtest_model

result = backtest_model(
    "data/index_data.csv",
    rolling_window=60,
    use_sp500=False
)

print("Directional accuracy:", result["directional_accuracy"])
print("Always-up accuracy:", result["always_up_accuracy"])
print("Predicted up rate:", result["predicted_up_rate"])

print("\nLast 10 predictions:")
print(result["results"].tail(10))