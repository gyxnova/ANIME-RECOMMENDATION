import sys 
import numpy as np
import joblib
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.database.db import get_connection, init_db

# initialise DB on startup
init_db()

sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.config import MODELS_DIR,DATA_FEATURES,DATA_PROC

app = FastAPI(title="Anime top3 Recommender")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://gyxnova.github.io",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
# serve static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

print("Loading model and data...")
model = joblib.load(MODELS_DIR / "als_model.pkl")
user_idx = joblib.load(MODELS_DIR / "user_idx.pkl") 
anime_idx = joblib.load(MODELS_DIR / "anime_idx.pkl")
matrix = joblib.load(DATA_FEATURES / "user_item_matrix.pkl")

anime_df = pd.read_parquet(DATA_PROC / "anime.parquet")
anime_lookup = anime_df.set_index("anime_id")[["name", "genre", "rating"]].to_dict("index")
print("Model and data loaded successfully!")

# load images and descriptions
images_df  = pd.read_csv(DATA_PROC / "anime_images.csv")
fandom_df  = pd.read_csv(DATA_PROC / "fandom_clean.csv")

# merge everything
anime_full = anime_df.merge(
    images_df[['anime_id', 'image_url']], 
    on='anime_id', how='left'
).merge(
    fandom_df[['title', 'description_clean']],
    left_on='name', right_on='title', how='left'
)

anime_full['image_url']         = anime_full['image_url'].fillna('')
anime_full['description_clean'] = anime_full['description_clean'].fillna('')

anime_lookup = anime_full.set_index('anime_id')[[
    'name', 'genre', 'rating', 'image_url', 'description_clean'
]].to_dict('index')

from src.database.db import get_connection, init_db

def get_user_ratings_from_db(user_id: int):
    """Get ratings for SQLite users"""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT anime_id, rating FROM user_preferences WHERE user_id = ?",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return {row['anime_id']: row['rating'] for row in rows}

def find_similar_kaggle_users(rated_anime: dict, top_n: int = 10):
    """Find Kaggle users who rated same anime similarly"""
    if not rated_anime:
        return []

    anime_ids = list(rated_anime.keys())
    similarity_scores = {}

    for kaggle_user_id, u_idx in user_idx.items():
        score = 0
        for anime_id in anime_ids:
            if anime_id in anime_idx:
                a_idx = anime_idx[anime_id]
                if matrix[u_idx, a_idx] > 0:
                    score += 1
        if score > 0:
            similarity_scores[kaggle_user_id] = score

    # return top N similar users
    similar = sorted(
        similarity_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return [uid for uid, _ in similar]


@app.get("/recommend/{user_id}")

def recommend(user_id: int, top_k: int = 3):

    idx_to_anime = {v: k for k, v in anime_idx.items()}

    # check SQLite first
    db_ratings = get_user_ratings_from_db(user_id)

    if db_ratings:
        # NEW USER — find similar Kaggle users
        similar_users = find_similar_kaggle_users(db_ratings)

        if not similar_users:
            return {"error": "Not enough data to recommend. Rate more anime!"}

        # aggregate scores from similar users
        agg_scores = np.zeros(len(anime_idx))
        for sim_uid in similar_users:
            u = user_idx[sim_uid]
            agg_scores += model.item_factors[u] @ model.user_factors.T

        # filter already rated
        for anime_id in db_ratings:
            if anime_id in anime_idx:
                agg_scores[anime_idx[anime_id]] = -999

        top_indices = np.argsort(agg_scores)[::-1][:top_k]

    else:
        # KAGGLE USER — use ALS directly
        if user_id not in user_idx:
            return {"error": f"User {user_id} not found"}

        u      = user_idx[user_id]
        scores = model.item_factors[u] @ model.user_factors.T

        rated_indices = matrix[u].nonzero()[1]
        for idx in rated_indices:
            scores[idx] = -999

        top_indices   = np.argsort(scores)[::-1][:top_k]
        agg_scores    = scores

    # build response
    results = []
    for rank, idx in enumerate(top_indices, 1):
        anime_id = idx_to_anime.get(int(idx))
        if anime_id is None:
            continue
        info = anime_lookup.get(anime_id, {})
        results.append({
            "rank":        int(rank),
            "anime_id":    int(anime_id),
            "name":        str(info.get("name", "Unknown")),
            "genre":       str(info.get("genre", "Unknown")),
            "rating":      float(info.get("rating", 0.0)),
            "image_url":   str(info.get("image_url", "")),
            "description": str(info.get("description_clean", "")),
            "score":       float(agg_scores[idx])
        })

    return {"user_id": user_id, "top3": results}

@app.get("/featured")
def featured():
    sample = anime_full[
        anime_full['image_url'].ne('')
    ].nlargest(200, 'members').sample(20)

    return {
        "anime": [
            {
                "anime_id":    int(row['anime_id']),
                "name":        str(row['name']),
                "genre":       str(row['genre']),
                "rating":      float(row['rating']),
                "image_url":   str(row['image_url']),
                "description": str(row['description_clean'])
            }
            for _, row in sample.iterrows()
        ]
    }

@app.get("/health")
def health_check():
    return {"status": "running",
            "model": "ALS",
            "version": "1.0"}



@app.get("/")
def root():
    return FileResponse("src/static/index.html")

@app.post("/onboarding")
def onboarding(data: dict):
    conn   = get_connection()
    cursor = conn.cursor()

    # create new user
    cursor.execute("INSERT INTO users DEFAULT VALUES")
    new_user_id = cursor.lastrowid

    # save genre preferences
    for genre in data.get("genres", []):
        cursor.execute(
            "INSERT INTO user_genres (user_id, genre) VALUES (?, ?)",
            (new_user_id, genre)
        )

    # save anime ratings
    for item in data.get("anime_ratings", []):
        cursor.execute(
            "INSERT INTO user_preferences (user_id, anime_id, rating, source) VALUES (?, ?, ?, ?)",
            (new_user_id, item["anime_id"], item["rating"], "onboarding")
        )

    conn.commit()
    conn.close()

    return {"user_id": new_user_id, "message": "Profile created!"}


@app.get("/popular/{genre}")
def popular_by_genre(genre: str, limit: int = 20):
    filtered = anime_full[
        anime_full['genre'].str.contains(genre, case=False, na=False) &
        anime_full['image_url'].ne('')
    ].nlargest(limit, 'members')

    return {
        "anime": [
            {
                "anime_id": int(row['anime_id']),
                "name":     str(row['name']),
                "genre":    str(row['genre']),
                "rating":   float(row['rating']),
                "image_url": str(row['image_url'])
            }
            for _, row in filtered.iterrows()
        ]
    }


@app.get("/genres")
def get_genres():
    all_genres = anime_full['genre'].str.split(',').explode().str.strip().unique()
    genres = sorted([g for g in all_genres if g and g != 'nan'])
    return {"genres": genres}