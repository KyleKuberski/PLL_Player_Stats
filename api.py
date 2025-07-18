import pandas as pd
import joblib
import traceback
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# ✅ Define your request schema
class Player(BaseModel):
    Name: str
    goals: float
    assists: float
    points: float
    shotPct: float
    shotsOnGoalPct: float
    groundBalls: float
    causedTurnovers: float
    faceoffPct: float
    turnovers: float
    unassistedGoals: float
    powerPlayGoals: float
    shortHandedGoals: float
    DSA_Impact_Factor: float

# ✅ Wrap it in a list schema
class PlayerList(BaseModel):
    players: List[Player]

@app.get("/")
def root():
    return {"message": "PLL API is running!"}

def safe_fmt(val):
    return f"{val:.3f}" if isinstance(val, (int, float)) else "N/A"

@app.post("/summary")
def generate_summary(payload: PlayerList):
    players = payload.players
    df = pd.DataFrame([p.dict() for p in players])
    df = df.rename(columns={"DSA_Impact_Factor": "DSA Impact Factor"})

    # Load model
    try:
        model_bundle = joblib.load("trained_model_GLOBAL.joblib")
        model = model_bundle["model"]
        features = model_bundle["features"]
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Model loading failed: {e}"}

    summaries = []

    for _, row in df.iterrows():
        try:
            feature_vals = pd.Series({f: row.get(f, 0.0) for f in features})
            coefs = pd.Series(model.named_steps["elasticnetcv"].coef_, index=features)
            contributions = feature_vals * coefs
            top_pos = contributions.sort_values(ascending=False).head(3)
            top_neg = contributions.sort_values().head(3)

            missing = [f for f in features if pd.isna(row.get(f))]
            if missing:
                print("⚠️ Warning: Missing or NaN values for:", missing)

            llama_summary = f"""
This player's top contributing stats are:
- {top_pos.index[0]}: {safe_fmt(row.get(top_pos.index[0]))}
- {top_pos.index[1]}: {safe_fmt(row.get(top_pos.index[1]))}
- {top_pos.index[2]}: {safe_fmt(row.get(top_pos.index[2]))}

Areas with the lowest contribution were:
- {top_neg.index[0]}: {safe_fmt(row.get(top_neg.index[0]))}
- {top_neg.index[1]}: {safe_fmt(row.get(top_neg.index[1]))}
- {top_neg.index[2]}: {safe_fmt(row.get(top_neg.index[2]))}

These patterns are based on overall impact from the model. High contribution features may align with increased NIL value potential.
""".strip()

            summaries.append({
                "Name": row.get("Name", "Unnamed Player"),
                "Summary": llama_summary
            })

        except Exception as e:
            print("❌ Error processing row:")
            print(row.to_dict())
            traceback.print_exc()
            return {"error": f"{type(e).__name__}: {str(e)}"}

    return {"summaries": summaries}
