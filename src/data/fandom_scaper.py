import requests
import pandas as pd
import time
import json 
import logging
import sys 
import os
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT/'data'/'raw'
DATA_RAW.mkdir(parents=True,exist_ok=True)

FANDOM_URL = 'https://anime.fandom.com/api.php'

def get_all_anime_pages(aplimit:int=500):
    all_pages=[]

    params = {
        'action':'query',
        'list':'allpages',
        'aplimit':aplimit,
        'format':'json'
    }

    while True:
        response = requests.get(url=FANDOM_URL,params=params).json()
        all_pages.extend(response['query']['allpages'])
        

        if 'continue' not in response:
            break

        params['apcontinue'] = response['continue']['apcontinue']

        time.sleep(1) 

    return all_pages

def get_anime_detail(pageid: int) -> dict:
    params = {
        'action':  'query',
        'pageids':  pageid,
        'prop':    'revisions',
        'rvprop':  'content',
        'rvslots': 'main',
        'format':  'json'
    }

    r     = requests.get(url=FANDOM_URL, params=params).json()
    pages = r['query']['pages']
    page  = list(pages.values())[0]

    try:

        content = page['revisions'][0]['slots']['main']['*']
        description = content[:500].strip()
    except:
        description = ''

    return {
        'pageid':      page.get('pageid'),
        'title':       page.get('title'),
        'description': description
    }


def main():
    log.info('='*50)
    log.info('Stage1 - Data_collection')
    log.info('='*50)

    #step1 : getting all pages 
    log.info('getting all anime pages...')
    pages = get_all_anime_pages()
    log.info(f'Found {len(pages)} anime pages')

    # step2 : getting all anime details
    log.info('getting all animes details')
    all_details = []
    for i, page in enumerate(pages):
        details = get_anime_detail(pageid=page['pageid'])
        all_details.append(details)
        time.sleep(0.2)
        
        # log every 100 anime
        if (i + 1) % 100 == 0:
            log.info(f"Progress: {i+1}/{len(pages)} anime done")

    # step3 : converting to csv file 
    fandom = DATA_RAW /'fandom.csv'
    pd.DataFrame(all_details).to_csv(fandom,index =False)


if __name__=='__main__':
    main()    
