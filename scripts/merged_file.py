import pandas as pd
import logging
from thefuzz import process, fuzz
from pathlib import Path

import html 
import re

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
DATA_EXT = ROOT / "data" / "external"
DATA_PROC = ROOT / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


def find_best_match(title, fandom_titles, threshold=70):
    result = process.extractOne(title, fandom_titles, scorer=fuzz.token_sort_ratio)
    if result is None:
        return None
    match_title, score = result
    return match_title if score >= threshold else None



def clean_title(title):
    title = str(title)
    title = html.unescape(title)           # fix &#039; → '
    title = title.lower()
    title = re.sub(r'[^\w\s]', '', title)  # remove punctuation
    title = re.sub(r'\b2nd\b', 'second', title)  # normalize seasons
    title = re.sub(r'\b3rd\b', 'third', title)
    title = re.sub(r'\b4th\b', 'fourth', title)
    title = re.sub(r'\s+', ' ', title)
    return title.strip()



def main():
    log.info("=" * 50)
    log.info("Merging Kaggle + Fandom datasets")
    log.info("=" * 50)

    # load data
    log.info("Loading data...")
    anime  = pd.read_csv(DATA_EXT / "anime.csv")
    fandom = pd.read_csv(DATA_DIR / "fandom.csv")
    log.info(f"Anime: {len(anime)} rows")
    log.info(f"Fandom: {len(fandom)} rows")

    # filter hentai
    before = len(anime)
    anime  = anime[~anime['genre'].str.contains('Hentai', na=False)]
    log.info(f"Removed hentai: {before - len(anime)} anime dropped")

    # fuzzy match
    
    fandom_titles_clean = [clean_title(t) for t in fandom['title'].tolist()]
    log.info(f"Finding fuzzy matches for {len(anime)} anime...")
    anime['match_key'] = anime['name'].apply(
        lambda x: find_best_match(clean_title(x), fandom_titles_clean)
    )
    matched = anime['match_key'].notna().sum()
    log.info(f"Matched: {matched}/{len(anime)} ({matched/len(anime)*100:.1f}%)")
    clean_to_original = dict(zip(fandom_titles_clean, fandom['title'].tolist()))
    anime['match_key'] = anime['match_key'].map(clean_to_original)

    # merge
    merged_anime = pd.merge(
        anime,
        fandom[['title', 'description']],
        left_on='match_key',
        right_on='title',
        how='left'
    )
    merged_anime = merged_anime.drop(columns=['match_key', 'title'])

    # check unmatched anime
    unmatched = anime[anime['match_key'].isna()]['name'].head(20).tolist()
    log.info(f"Sample unmatched: {unmatched}")

    # save
    out = DATA_PROC / "anime_merged.csv"
    merged_anime.to_csv(out, index=False)
    log.info(f"Saved {len(merged_anime)} anime → {out}")
    log.info(f"Columns: {merged_anime.columns.tolist()}")


if __name__ == "__main__":
    main()