import pandas as pd
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/ta35_model_data.csv")
df["signal_abs"] = df["dual_us_weighted_return"].abs()
df["pred_sign_from_dual"] = (df["dual_us_weighted_return"] > 0).astype(int)

# use top 50% strongest signals
threshold = df["signal_abs"].median()
strong = df[df["signal_abs"] >= threshold].copy()

acc = accuracy_score(strong["Open_Gap_Sign"], strong["pred_sign_from_dual"])
always_up_acc = (strong["Open_Gap_Sign"] == 1).mean()

print("Strong-signal rows:", len(strong))
print("Dual-sign baseline accuracy:", round(acc, 4))
print("Always-up accuracy:", round(always_up_acc, 4))

print("Avg gap when signal > 0:",
      strong.loc[strong["dual_us_weighted_return"] > 0, "Open_Gap"].mean())
print("Avg gap when signal <= 0:",
      strong.loc[strong["dual_us_weighted_return"] <= 0, "Open_Gap"].mean())