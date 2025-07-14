# PLL NIL Summary API

This API generates NIL scouting reports for lacrosse players based on performance data.

## Endpoints
- `GET /`: Health check
- `POST /summary`: Accepts player stats and returns a personalized NIL summary via LLM

## Setup
1. Upload player CSV to `/data/`
2. Run `train_model.py` to generate model
3. Run `api.py` via `uvicorn api:app --reload`
