import sys
import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from scipy.sparse import csr_matrix

sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.config import DATA_PROC, DATA_FEATURES
DATA_FEATURES.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def build_user_features(ratings):
    log.info('building user features...')

    user_features = ratings.groupby('user_id').agg(
        rating_count = ('rating','size'), #how active
        avg_rating = ('rating','mean'),#generous or strict
        min_rating = ('rating','min'),#rating range
        max_rating = ('rating','max'),
        unique_anime = ('anime_id','nunique'),#how much they do explore
    ).reset_index()

    log.info(f'User features shape : {user_features.shape}')
    return user_features

def build_anime_features(anime):
    log.info('building anime features...')
    #encodetype as no
    anime['type_encoded'] = anime['type'].astype('category').cat.codes
    anime['episodes_clean'] = pd.to_numeric(
        anime['episodes'],errors='coerce'
    ).fillna(0).astype(int)#encode episodes

    anime['genre_primary'] = anime['genre'].str.split(',').str[0].str.strip()
    anime['genre_encoded'] = anime['genre_primary'].astype('category').cat.codes#primary genre 1st genre in list

    #final col select
    anime_features = anime[[
        'anime_id',
        'name',
        'genre',
        'genre_primary',
        'genre_encoded',
        'type',
        'type_encoded',
        'episodes_clean',
        'rating',
        'rating_normalized',
        'members'
    ]].copy()

    log.info(f"Anime features shape: {anime_features.shape}")
    return anime_features

def build_user_item_matrix(ratings):
    log.info('building user-item matrix...')

    users = sorted(ratings['user_id'].unique())
    animes = sorted(ratings['anime_id'].unique())

    user_idx = {u: i for i,u in enumerate(users)}
    anime_idx = {r: i for i,r in enumerate(animes)}

    interaction_counts = (
        ratings.groupby(['user_id','anime_id']) #how many times each unique user rated specific anime
        .size().reset_index(name = 'rating_count')
    )
    rows = interaction_counts['user_id'].map(user_idx)
    cols = interaction_counts['anime_id'].map(anime_idx)
    data = interaction_counts['rating_count'].values

    matrix = csr_matrix(
        (data, (rows,cols)),
        shape=(len(users),len(animes))
    )

    log.info(f'Matrix shape : {matrix.shape}')
    log.info(f'Sparsity : {1 - matrix.nnz/(matrix.shape[0]*matrix.shape[1]): .2%}')
    return matrix,user_idx,anime_idx

def main():
    log.info('='*50)
    log.info('stage 3 - feature engineering')
    log.info('='*50)

    log.info('\n[1/4] loading processed data...')
    animes = pd.read_parquet(DATA_PROC /'anime.parquet')
    rating_train = pd.read_parquet(DATA_PROC /'rating_train.parquet')
    log.info(f'animes : {len(animes)}')
    log.info(f'train rating : {len(rating_train)}')

    log.info('\n [2/4] building user features...')
    user_features = build_user_features(rating_train)

    log.info('\n [3/4] building anime features ...')
    anime_features = build_anime_features(animes)

    log.info('\n [4/4] building user-item matrix...')
    matrix, user_idx ,anime_idx = build_user_item_matrix(rating_train)

    user_features.to_parquet(DATA_FEATURES /'user_features.parquet',index=False)
    anime_features.to_parquet(DATA_FEATURES /'anime_features.parquet',index=False)
    
    joblib.dump(matrix,   DATA_FEATURES / "user_item_matrix.pkl")
    joblib.dump(user_idx, DATA_FEATURES / "user_idx.pkl")
    joblib.dump(anime_idx, DATA_FEATURES / "anime_idx.pkl")

    log.info("\n-- Done --")
    log.info(f"User features     : {user_features.shape}")
    log.info(f"Restaurant features: {anime_features.shape}")
    log.info(f"Matrix shape      : {matrix.shape}")
    log.info("Output: data/features/")
    log.info("=" * 50)


if __name__ == '__main__':
    main()