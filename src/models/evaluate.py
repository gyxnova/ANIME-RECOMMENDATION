import json
import logging
import numpy as np
import joblib
from pathlib import Path
import sys
import pandas as pd


sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.config import DATA_PROC,METRICS_DIR,MODELS_DIR,DATA_FEATURES
METRICS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S", 
)
log = logging.getLogger(__name__)

def precision_at_k(recommended,actual,k=3):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(actual))
    return hits / k

def hit_rate_at_k(recommended,actual,k=3):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(actual))
    return 1 if hits > 0 else 0

def ndcg_at_k(recommended,actual,k=3):
    recommended_k = recommended[:k]
    dcg = sum([1/np.log2(i+2) for i, rec in enumerate(recommended_k) if rec in actual])
    idcg = sum([1/np.log2(i+2) for i in range(min(len(actual), k))])
    return dcg / idcg if idcg > 0 else 0


def evaluate_model(model, matrix, user_idx, anime_idx, test_df, top_k=3):
    log.info(f"Evaluating on {test_df['user_id'].nunique()} users...")

    precision_scores = []
    hit_rate_scores  = []
    ndcg_scores      = []

    idx_to_anime = {v: k for k, v in anime_idx.items()}

    for user_id, user_test in test_df.groupby("user_id"):
        if user_id not in user_idx:
            continue

        actual = user_test["anime_id"].tolist()

        u      = user_idx[user_id]
        scores = model.item_factors[u] @ model.user_factors.T
        top_indices  = np.argsort(scores)[::-1][:top_k]
        recommended  = [idx_to_anime.get(int(i)) for i in top_indices]
        recommended  = [r for r in recommended if r is not None]

        precision_scores.append(precision_at_k(recommended, actual, top_k))
        hit_rate_scores.append(hit_rate_at_k(recommended, actual, top_k))
        ndcg_scores.append(ndcg_at_k(recommended, actual, top_k))

    return {
        "precision_at_3": round(float(np.mean(precision_scores)), 4),
        "hitrate_at_3":   round(float(np.mean(hit_rate_scores)),  4),
        "ndcg_at_3":      round(float(np.mean(ndcg_scores)),      4),
        "n_users":        int(len(precision_scores))
    }

def main():
    log.info("=" * 50)
    log.info("STAGE 5 - Evaluation")
    log.info("=" * 50)

    log.info("\n[1/3] Loading model and data...")
    model     = joblib.load(MODELS_DIR / "als_model.pkl")
    user_idx  = joblib.load(MODELS_DIR / "user_idx.pkl")
    anime_idx = joblib.load(MODELS_DIR / "anime_idx.pkl")
    matrix    = joblib.load(DATA_FEATURES / "user_item_matrix.pkl")
    
    ratings_test  = pd.read_parquet(DATA_PROC / "rating_test.parquet")
    ratings_train = pd.read_parquet(DATA_PROC / "rating_train.parquet")
    log.info("Loaded successfully")

    log.info("\n[2/3] Running evaluation...")
    metrics = evaluate_model(model, matrix, user_idx, anime_idx, ratings_test)

    log.info("\n[3/3] Saving metrics...")
    with open(METRICS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("\n-- Results --")
    for k, v in metrics.items():
        log.info(f"  {k}: {v:.4f}")

    log.info("=" * 50)
    log.info("Stage 5 COMPLETE")
    log.info("=" * 50)

if __name__ == "__main__":
    main()