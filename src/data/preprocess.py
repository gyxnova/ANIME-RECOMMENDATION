import pandas as pd
import numpy as np
import logging 
from pathlib import Path
import sys
import re
import html

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)
sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.config import ROOT ,DATA_DIR,DATA_EXT,DATA_PROC
DATA_PROC.mkdir(parents=True,exist_ok=True)



MIN_RATING_VALUE = 1 #FOR NOW WE REMOVE -1 SO TAKING MIN =1
TEST_SPLIT_RATIO = 0.2

def clean_anime(df):
    log.info(f'anime before cleaning : {len(df)}')
    df = df.dropna(subset=['rating','genre','type']) # as there is low na values so we can drop them 
    df = df[df['rating']>=MIN_RATING_VALUE].copy() #not including -1 values 
    log.info(f'after min rating: {len(df)}')
    r_min = df['rating'].min()
    r_max = df['rating'].max()
    df['rating_normalized'] = (df['rating']-r_min) /(r_max-r_min + 1e-9)

    # cleaning name  
    df['name'] = df['name'].apply(lambda x: html.unescape(str(x)))
    df['name'] = df['name'].str.lower()
    df['name'] = df['name'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df['name'] = df['name'].apply(lambda x: re.sub(r'\b2nd\b', 'second', x))
    df['name'] = df['name'].apply(lambda x: re.sub(r'\b3rd\b', 'third', x))
    df['name'] = df['name'].str.strip()

    log.info(f'Final anime : {len(df)}')
    return df.reset_index(drop=True)

def clean_rating(df,valid_anime_ids):
    log.info(f'rating before cleaning: {len(df)}')
    before = len(df)
    df = df[df['anime_id'].isin(valid_anime_ids)].copy()
    log.info(f'after removing orphan ratings : {len(df)} (dropped {before - len(df)})')

    df = df.sort_values('rating')
    df = df.drop_duplicates(subset=['user_id','anime_id'])
    df = df[df['rating'] != -1]

    return df.reset_index(drop=True)

def split_train_test(df):
    train_list = []
    test_list  = []

    for user_id, user_df in df.groupby("user_id"):
        user_df    = user_df.reset_index(drop=True)
        n          = len(user_df)
        split      = max(1, int(n * (1 - TEST_SPLIT_RATIO)))
        train_part = user_df.iloc[:split]
        test_part  = user_df.iloc[split:]

        if len(train_part) > 0:
            train_list.append(train_part)
        if len(test_part) > 0:
            test_list.append(test_part)

    log.info(f"Unique users : {df['user_id'].nunique()}")
    log.info(f"Total rating : {len(df)}")
    log.info(f"Avg per user : {len(df)/df['user_id'].nunique():.1f}")

    if not train_list:
        raise ValueError("Train list is empty — check your data")
    if not test_list:
        raise ValueError("Test list is empty — all users have only 1 order")

    train = pd.concat(train_list).reset_index(drop=True)
    test  = pd.concat(test_list).reset_index(drop=True)

    log.info(f"Train rating : {len(train)}")
    log.info(f"Test  rating : {len(test)}")
    return train, test

def main():
    log.info("=" * 50)
    log.info("STAGE 2 - Preprocessing")
    log.info("=" * 50)

    log.info("\n[1/4] Loading raw data...")
    anime = pd.read_csv(DATA_EXT / "anime.csv")
    rating      = pd.read_csv(DATA_EXT / "rating.csv")
    log.info(f"Loaded {len(anime)} anime")
    log.info(f"Loaded {len(rating)} rating")

    log.info("\n[2/4] Cleaning anime...")
    anime = clean_anime(anime)

    log.info("\n[3/4] Cleaning rating...")
    valid_ids = set(anime["anime_id"])
    rating    = clean_rating(rating, valid_ids)

    log.info("\n[4/4] Splitting train/test...")
    train, test = split_train_test(rating)
    log.info(f"\n-- Summary --")
    log.info(f"Anime    : {len(anime)}")
    log.info(f"Train    : {len(train)}")
    log.info(f"Test     : {len(test)}")
    log.info(f"Users    : {rating['user_id'].nunique()}")

    anime.to_parquet(DATA_PROC / "anime.parquet", index=False)
    train.to_parquet(DATA_PROC / "rating_train.parquet",      index=False)
    test.to_parquet(DATA_PROC / "rating_test.parquet",        index=False)


if __name__ == "__main__":
    main()



    
