import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
import joblib

# === File path ===
csv_path = r"C:\Users\kuber\Desktop\PLL_Pull\PLL Mens Lacrosse.csv"

# === Load Data ===
df = pd.read_csv(csv_path)

# === Clean & Feature Engineering ===
features = [
    "goals", "assists", "points", "shotPct", "shotsOnGoalPct",
    "groundBalls", "causedTurnovers", "faceoffPct", "turnovers",
    "unassistedGoals", "powerPlayGoals", "shortHandedGoals"
]
target = "DSA Impact Factor"

# Drop rows missing target or any features
df = df.dropna(subset=[target] + features)

X = df[features]
y = df[target]

# === Train Model ===
model = Pipeline([
    ("scaler", StandardScaler()),
    ("elasticnetcv", ElasticNetCV(cv=5, random_state=42))
])
model.fit(X, y)

# === Save Model ===
joblib.dump({
    "model": model,
    "features": features
}, "trained_model_GLOBAL.joblib")
print("Model trained and saved as 'trained_model_GLOBAL.joblib'")

# === Global Feature Importance ===
coefs = model.named_steps["elasticnetcv"].coef_
coef_df = pd.Series(coefs, index=features).sort_values(ascending=False)
print("\nTop Global Positive Contributors:")
print(coef_df.head(5))
print("\nTop Global Negative Contributors:")
print(coef_df.tail(5))

# === Individualized Player Scouting Summary ===
print("\nIndividual Player Strengths/Weaknesses:\n")
for _, row in df.iterrows():
    name = row.get("Player", "Unknown")
    player_vals = pd.Series({f: row[f] for f in features})
    contribs = player_vals * coef_df  # Aligns with features

    top_pos = contribs.sort_values(ascending=False).head(3)
    top_neg = contribs.sort_values().head(3)

    print(f" {name}:")
    print("  Strengths:")
    for k, v in top_pos.items():
        print(f"    + {k}: {v:.3f}")
    print("  Weaknesses:")
    for k, v in top_neg.items():
        print(f"    - {k}: {v:.3f}")
    print("")
