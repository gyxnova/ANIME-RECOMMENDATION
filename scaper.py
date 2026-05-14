import argparse
import time
import logging
from pathlib import Path

import pandas as pd
import requests


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / 'data' / 'raw'
DATA_RAW.mkdir(parents=True, exist_ok=True)

FANDOM_URL = 'https://anime.fandom.com/api.php'
REQUEST_TIMEOUT = 30
DEFAULT_BATCH_SLEEP = 1.0
DEFAULT_LOG_EVERY = 100


def fetch_json(session: requests.Session, params: dict) -> dict:
    response = session.get(FANDOM_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def get_all_anime_pages(
    session: requests.Session,
    aplimit: int = 500,
    max_pages: int | None = None,
    batch_sleep: float = DEFAULT_BATCH_SLEEP,
) -> list[dict]:
    if not 1 <= aplimit <= 500:
        raise ValueError('aplimit must be between 1 and 500')
    if max_pages is not None and max_pages <= 0:
        raise ValueError('max_pages must be greater than 0')

    all_pages = []
    batch_number = 0

    params = {
        'action': 'query',
        'list': 'allpages',
        'aplimit': aplimit,
        'apnamespace': 0,
        'format': 'json',
    }

    while True:
        response = fetch_json(session, params)
        batch = response.get('query', {}).get('allpages', [])
        all_pages.extend(batch)
        batch_number += 1
        log.info('Fetched page batch %s (%s pages total)', batch_number, len(all_pages))

        if max_pages is not None and len(all_pages) >= max_pages:
            return all_pages[:max_pages]

        apcontinue = response.get('continue', {}).get('apcontinue')
        if not apcontinue:
            break

        params['apcontinue'] = apcontinue

        if batch_sleep > 0:
            time.sleep(batch_sleep)

    return all_pages


def get_anime_detail(session: requests.Session, pageid: int) -> dict:
    params1 = {
        'action': 'query',
        'pageids': pageid,
        'prop': 'extracts',
        'exintro': 'true',
        'format': 'json',
    }

    response = fetch_json(session, params1)
    pages = response.get('query', {}).get('pages', {})
    page = pages.get(str(pageid))

    if page is None:
        raise KeyError(f'No page returned for pageid={pageid}')

    return {
        'pageid': page.get('pageid'),
        'title': page.get('title'),
        'extract': page.get('extract'),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Scrape anime summaries from anime.fandom.com.')
    parser.add_argument(
        '--max-pages',
        type=int,
        default=None,
        help='Only scrape the first N pages. Useful for testing.',
    )
    parser.add_argument(
        '--batch-sleep',
        type=float,
        default=DEFAULT_BATCH_SLEEP,
        help='Seconds to sleep between page-list batches.',
    )
    parser.add_argument(
        '--log-every',
        type=int,
        default=DEFAULT_LOG_EVERY,
        help='Log progress every N detail pages.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.log_every <= 0:
        raise ValueError('--log-every must be greater than 0')

    log.info('=' * 50)
    log.info('Stage1 - Data_collection')
    log.info('=' * 50)

    log.info('getting all anime pages...')
    try:
        with requests.Session() as session:
            pages = get_all_anime_pages(
                session=session,
                max_pages=args.max_pages,
                batch_sleep=args.batch_sleep,
            )

            log.info(f'Found {len(pages)} anime pages')
            if args.max_pages is None:
                log.info('This can take a while because each page is fetched one by one.')

            log.info('getting all animes details')
            all_details = []
            for index, page in enumerate(pages, start=1):
                try:
                    details = get_anime_detail(session=session, pageid=page['pageid'])
                except (KeyError, requests.RequestException) as exc:
                    log.warning(
                        'Skipping page %s (%s): %s',
                        page.get('pageid'),
                        page.get('title', 'unknown'),
                        exc,
                    )
                    continue

                all_details.append(details)
                if index % args.log_every == 0 or index == len(pages):
                    log.info('Fetched %s/%s anime details', index, len(pages))
    except requests.RequestException as exc:
        log.error('Failed to fetch anime pages: %s', exc)
        raise SystemExit(1) from exc

    log.info(f'Found {len(all_details)} anime details')

    fandom = DATA_RAW / 'fandom.csv'
    pd.DataFrame(all_details).to_csv(fandom, index=False)


if __name__ == '__main__':
    main()

