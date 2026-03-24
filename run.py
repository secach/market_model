from src.model import compute_signal

result = backtest_model(
    "data/index_data_with_features.csv",
    rolling_window=60,
    use_sp500=True
)

print("Predicted return:", result["predicted_return"])
print("Predicted price:", result["predicted_price"])

print("\nModel parameters:")
for k, v in result["model_params"].items():
    print(f"{k}: {v}")

print("\nP-values:")
for k, v in result["model_pvalues"].items():
    print(f"{k}: {v}")

print("\nR-squared:", result["model_r_squared"])