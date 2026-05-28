import sys 
import numpy as np
import joblib
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.config import MODELS_DIR,DATA_FEATURES,DATA_PROC

app = FastAPI(title="Anime top3 Recommender")

print("Loading model and data...")
model = joblib.load(MODELS_DIR / "als_model.pkl")
user_idx = joblib.load(MODELS_DIR / "user_idx.pkl") 
anime_idx = joblib.load(MODELS_DIR / "anime_idx.pkl")
matrix = joblib.load(DATA_FEATURES / "user_item_matrix.pkl")

anime_df = pd.read_parquet(DATA_PROC / "anime.parquet")
anime_lookup = anime_df.set_index("anime_id")[["name", "genre", "rating"]].to_dict("index")
print("Model and data loaded successfully!")


@app.get("/recommend/{user_id}")

def recommend(user_id: int, top_k: int = 3):
    if user_id not in user_idx:
        return {"error": f"User {user_id} not found"}

    u            = user_idx[user_id]
    idx_to_anime = {v: k for k, v in anime_idx.items()}

    # score all anime for this user
    scores = model.item_factors[u] @ model.user_factors.T
    
    # filter already rated
    rated_indices= matrix[u].nonzero()[1]
    for idx in rated_indices:
        scores[idx] = -999

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_indices, 1):
        anime_id = idx_to_anime.get(int(idx))
        if anime_id is None:
            continue
        info = anime_lookup.get(anime_id, {})
        results.append({
            "rank":     int(rank),
            "anime_id": int(anime_id),
            "name":     str(info.get("name", "Unknown")),
            "genre":    str(info.get("genre", "Unknown")),
            "rating":   float(info.get("rating", 0.0)),
            "score":    float(scores[idx])
        })

    return {"user_id": user_id, "top3": results}

@app.get("/health")
def health_check():
    return {"status": "running",
            "model": "ALS",
            "version": "1.0"}