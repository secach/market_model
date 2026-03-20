from src.model import compute_signal

result = compute_signal(
    "data/index_data.csv",
    rolling_window=60
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