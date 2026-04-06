import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/ta35_model_data.csv")

features = [
    "dual_us_weighted_return",
    "dual_us_fraction_up",
    "dual_us_weight_sum",
    "RollingVol_5",
    "Lag1_Return",
]

df = df.dropna(subset=features + ["Open_Gap_Sign"]).copy()

split = int(len(df) * 0.7)
train = df.iloc[:split]
test = df.iloc[split:]

X_train = train[features]
y_train = train["Open_Gap_Sign"]
X_test = test[features]
y_test = test["Open_Gap_Sign"]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Test rows:", len(test))
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Always-up baseline:", round((y_test == 1).mean(), 4))