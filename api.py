from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json

app = FastAPI(
    title="PLL Player API",
    description="NIL and stat-based player summary generator",
    version="1.0.0",
    docs_url="/docs"
)

@app.get("/")
def root():
    return {"message": "PLL API is running!"}

# Define what a player looks like
class Player(BaseModel):
    Name: str
    goals: float
    assists: float
    shotPct: float
    groundBalls: float
    causedTurnovers: float
    faceoffPct: float
    DSA_Impact_Factor: float

# Define the route that React will call
@app.post("/summary")
def generate_summary(players: List[Player]):
    # This is where you’ll plug in your LLM logic (or simple summary logic)
    top_goal = max(players, key=lambda p: p.goals)
    top_assist = max(players, key=lambda p: p.assists)

    summary = (
        f"{top_goal.Name} leads in goals with {top_goal.goals}. "
        f"{top_assist.Name} is the top assister with {top_assist.assists}."
    )

    return {"summary": summary}
