import sys
import numpy as np
import logging
import joblib
from pathlib import Path
import implicit


sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.config import DATA_FEATURES, MODELS_DIR
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def train_als(matrix):
    log.info('Training ALS model...')

    matrix_T = matrix.T.tocsr().astype('float32')


    model = implicit.als.AlternatingLeastSquares(
        factors = 50,
        iterations = 20,
        regularization= 0.01,
        random_state=42
    )

    model.fit(matrix_T)
    log.info('ALS training complete!!')
    return model
    

def get_recommendations(model, matrix, user_idx, anime_idx,
                        user_id, rated_anime=None, top_k=3):

    if user_id not in user_idx:
        log.warning(f"User {user_id} not found")
        return []

    u            = user_idx[user_id]
    idx_to_anime = {v: k for k, v in anime_idx.items()}

    # score all anime for this user
    scores = model.item_factors[u] @ model.user_factors.T
    
    # filter already rated
    if rated_anime:
        for anime_id in rated_anime:
            if anime_id in anime_idx:
                scores[anime_idx[anime_id]] = -999

    top_indices = np.argsort(scores)[::-1][:top_k]

    recommendations = []
    for rank, idx in enumerate(top_indices, 1):
        anime_id = idx_to_anime.get(int(idx))
        if anime_id is None:
            continue
        recommendations.append({
            "rank":     rank,
            "anime_id": int(anime_id),
            "score":    float(scores[idx])
        })

    return recommendations

def main():
    log.info("=" * 50)
    log.info("STAGE 4 - Model Training")
    log.info("=" * 50)

    log.info("\n[1/3] Loading features...")
    matrix   = joblib.load(DATA_FEATURES / "user_item_matrix.pkl")
    user_idx = joblib.load(DATA_FEATURES / "user_idx.pkl")
    anime_idx = joblib.load(DATA_FEATURES / "anime_idx.pkl")
    log.info(f"Matrix shape: {matrix.shape}")


    log.info("\n[2/3] Training ALS model...")
    model = train_als(matrix)

    log.info("\n[3/3] Saving model...")
    joblib.dump(model, MODELS_DIR / "als_model.pkl")
    joblib.dump(user_idx, MODELS_DIR / "user_idx.pkl")
    joblib.dump(anime_idx, MODELS_DIR / "anime_idx.pkl")

    # sanity check
    sample_user = list(user_idx.keys())[0]
    recs = get_recommendations(model, matrix, user_idx, anime_idx, sample_user)
    log.info(f"Sample recs for {sample_user}:")
    for r in recs:
        log.info(f"  Rank {r['rank']}: {r['anime_id']} (score: {r['score']:.4f})")
    

    log.info("=" * 50)
    log.info("Stage 4 COMPLETE")
    log.info("=" * 50)


if __name__ == "__main__":
    main()