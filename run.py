from src.model import compute_signal

result = compute_signal(
    "data/index_data.csv", 
    ar_weight=0.4, 
    weight_signal_weight=0.6,
    rolling_window=20
)

print("Predicted return:", result["predicted_return"])
print("Predicted price:", result["predicted_price"])
print("AR(1) signal:", result["ar_prediction"])
print("Weighted stock signal:", result["weighted_prediction"])