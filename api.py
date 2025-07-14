import pandas as pd
import joblib
import traceback
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

from llm_utils import get_cached_llama_response  # Make sure this is in the same folder or on your Python path

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

@app.get("/")
def root():
    return {"message": "PLL API is running!"}

@app.post("/summary")
def generate_summary(players: List[Player]):
    df = pd.DataFrame([p.dict() for p in players])

    # Rename to match training data column
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

            llama_prompt = f"""
You are an NIL analyst.

The player {row['Name']} has standout performance in the following metrics:
- {top_pos.index[0]}: {row[top_pos.index[0]]:.3f}
- {top_pos.index[1]}: {row[top_pos.index[1]]:.3f}
- {top_pos.index[2]}: {row[top_pos.index[2]]:.3f}

Their weaker metrics include:
- {top_neg.index[0]}: {row[top_neg.index[0]]:.3f}
- {top_neg.index[1]}: {row[top_neg.index[1]]:.3f}
- {top_neg.index[2]}: {row[top_neg.index[2]]:.3f}

Write a short NIL scouting summary explaining their strengths and weaknesses in 3–4 sentences.
""".strip()

            llama_summary = get_cached_llama_response(llama_prompt)

            summaries.append({
                "Name": row["Name"],
                "Summary": llama_summary
            })

        except Exception as e:
            print("❌ Error processing row:")
            print(row.to_dict())
            traceback.print_exc()
            return {"error": f"{type(e).__name__}: {str(e)}"}

    return {"summaries": summaries}
