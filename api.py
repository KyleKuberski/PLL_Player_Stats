@app.post("/summary")
def generate_summary(players: List[Player]):
    df = pd.DataFrame([p.dict() for p in players])

    # 🔁 Rename column to match model's expected column name
    df = df.rename(columns={"DSA_Impact_Factor": "DSA Impact Factor"})

    # Load your trained model
    try:
        model_bundle = joblib.load("trained_model_GLOBAL.joblib")
        model = model_bundle["model"]
        features = model_bundle["features"]
    except:
        return {"error": "Model loading failed."}
    
    summaries = []

    for _, row in df.iterrows():
        feature_vals = pd.Series({f: row.get(f, 0.0) for f in features})
        coefs = pd.Series(model.named_steps['elasticnetcv'].coef_, index=features)
        contributions = feature_vals * coefs
        top_pos = contributions.sort_values(ascending=False).head(3)
        top_neg = contributions.sort_values().head(3)

        llama_prompt = f"""
You are an NIL analyst.

The player {row['Name']} has standout performance in the following metrics:
- {top_pos.index[0]}: {row.get(top_pos.index[0]):.3f}
- {top_pos.index[1]}: {row.get(top_pos.index[1]):.3f}
- {top_pos.index[2]}: {row.get(top_pos.index[2]):.3f}

Their weaker metrics include:
- {top_neg.index[0]}: {row.get(top_neg.index[0]):.3f}
- {top_neg.index[1]}: {row.get(top_neg.index[1]):.3f}
- {top_neg.index[2]}: {row.get(top_neg.index[2]):.3f}

Write a short NIL scouting summary explaining their strengths and weaknesses in 3–4 sentences.
""".strip()

        llama_summary = get_cached_llama_response(llama_prompt)
        summaries.append({
            "Name": row["Name"],
            "Summary": llama_summary
        })

    return {"summaries": summaries}
