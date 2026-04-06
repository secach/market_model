import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("data/ta35_model_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Simple baseline: signal sign predicts gap sign
df["pred_sign_from_dual"] = (df["dual_us_weighted_return"] > 0).astype(int)

acc = accuracy_score(df["Open_Gap_Sign"], df["pred_sign_from_dual"])
cm = confusion_matrix(df["Open_Gap_Sign"], df["pred_sign_from_dual"])

always_up_acc = (df["Open_Gap_Sign"] == 1).mean()

print("Rows:", len(df))
print("Dual-sign baseline accuracy:", round(acc, 4))
print("Always-up accuracy:", round(always_up_acc, 4))
print("Confusion matrix:")
print(cm)

print("\nAverage open gap when dual signal > 0:",
      df.loc[df["dual_us_weighted_return"] > 0, "Open_Gap"].mean())

print("Average open gap when dual signal <= 0:",
      df.loc[df["dual_us_weighted_return"] <= 0, "Open_Gap"].mean())