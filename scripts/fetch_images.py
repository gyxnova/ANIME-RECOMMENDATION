import requests
import pandas as pd
import time
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[1]
anime_df  = pd.read_parquet(ROOT / "data/processed/anime.parquet")

JIKAN_URL = "https://api.jikan.moe/v4/anime/{}"

images = []

for i, row in anime_df.iterrows():
    mal_id = row['anime_id']
    try:
        r    = requests.get(JIKAN_URL.format(mal_id), timeout=5)
        data = r.json()
        img  = data['data']['images']['jpg']['image_url']
    except:
        img = ""

    images.append({
        "anime_id": mal_id,
        "image_url": img
    })

    time.sleep(0.5)  # Jikan rate limit = 3 requests/second

    if (i + 1) % 100 == 0:
        print(f"Progress: {i+1}/{len(anime_df)}")

images_df = pd.DataFrame(images)
images_df.to_csv(ROOT / "data/processed/anime_images.csv", index=False)
print(f"Done! Got {images_df['image_url'].ne('').sum()} images")