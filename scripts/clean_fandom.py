import re
import pandas as pd
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
fandom = pd.read_csv(ROOT / "data/raw/fandom.csv")

def clean_wiki_text(text):
    if not isinstance(text, str):
        return ""
    
    # skip redirect pages
    if text.strip().startswith('#REDIRECT'):
        return ""
    
    text = re.sub(r'\{\{[^}]*\}\}', '', text)  # remove {{templates}}
    text = re.sub(r'\[\[[^\]]*\]\]', '', text)  # remove [[links]]
    text = re.sub(r'\[http[^\]]*\]', '', text)  # remove [urls]
    text = re.sub(r"'{2,}", '', text)            # remove bold/italic
    text = re.sub(r'==.*?==', '', text)          # remove headers
    text = re.sub(r'<.*?>', '', text)            # remove html tags
    text = re.sub(r'\|[^|]*', '', text)          # remove | pipe content
    text = re.sub(r'\s+', ' ', text)             # normalize spaces
    return text.strip()[:300]


fandom['description_clean'] = fandom['description'].apply(clean_wiki_text)

# check how many are empty after cleaning
empty = fandom['description_clean'].eq('').sum()
print(f"Empty descriptions: {empty}/{len(fandom)}")

# keep only non-empty
fandom_clean = fandom[fandom['description_clean'] != ''].reset_index(drop=True)
print(f"Anime with descriptions: {len(fandom_clean)}")

fandom_clean.to_csv(ROOT / "data/processed/fandom_clean.csv", index=False)