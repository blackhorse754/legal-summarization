"""
src/scraper.py
--------------
Scrapes Indian legal case judgments from IndianKanoon.org.

Usage:
    python src/scraper.py --category murder --max_docs 200 --output data/raw/

Categories: murder, robbery, corruption, land_dispute
"""

import json
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://indiankanoon.org"
SEARCH_URL = f"{BASE_URL}/search/"

# Search queries per crime category — tuned to match Supreme Court judgments
CATEGORY_QUERIES = {
    "murder":       "murder Indian Penal Code Section 302 Supreme Court judgment",
    "robbery":      "robbery dacoity Section 392 394 IPC Supreme Court judgment",
    "corruption":   "corruption Prevention of Corruption Act Supreme Court judgment",
    "land_dispute": "land dispute property civil appeal Supreme Court judgment",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; LegalSummResearch/1.0; "
        "+https://github.com/YOUR_USERNAME/legal-summarization)"
    )
}

# Be polite — don't hammer the server
REQUEST_DELAY_SEC = 1.5


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Judgment:
    doc_id: str
    title: str
    url: str
    category: str
    court: str
    date: str
    full_text: str
    num_words: int


# ---------------------------------------------------------------------------
# Core scraping functions
# ---------------------------------------------------------------------------

def search_judgments(category: str, max_docs: int) -> list[dict]:
    """
    Search IndianKanoon and return a list of {title, url} results.
    Paginates through results until max_docs reached.
    """
    query = CATEGORY_QUERIES[category]
    results = []
    page = 0

    log.info(f"Searching for category='{category}' (query: {query!r})")

    while len(results) < max_docs:
        params = {"formInput": query, "pagenum": page}
        try:
            resp = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error(f"Search request failed on page {page}: {e}")
            break

        soup = BeautifulSoup(resp.text, "lxml")
        hits = soup.select("div.result_title a")

        if not hits:
            log.info(f"No more results at page {page}.")
            break

        for a in hits:
            href = a.get("href", "")
            if "/doc/" in href:
                results.append({
                    "title": a.get_text(strip=True),
                    "url": BASE_URL + href,
                })
            if len(results) >= max_docs:
                break

        log.info(f"  Page {page}: found {len(hits)} hits, total so far: {len(results)}")
        page += 1
        time.sleep(REQUEST_DELAY_SEC)

    return results[:max_docs]


def fetch_judgment_text(url: str) -> dict:
    """
    Fetch the full text and metadata of a single judgment page.
    Returns a dict with keys: court, date, full_text.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"Failed to fetch {url}: {e}")
        return {}

    soup = BeautifulSoup(resp.text, "lxml")

    # Extract main judgment text (inside #judgments div or <pre> blocks)
    judgment_div = soup.find("div", id="judgments") or soup.find("div", class_="judgment")
    if judgment_div:
        full_text = judgment_div.get_text(separator="\n", strip=True)
    else:
        # Fallback: grab all paragraph text
        paragraphs = soup.find_all("p")
        full_text = "\n".join(p.get_text(strip=True) for p in paragraphs)

    # Extract court name
    court_tag = soup.find("div", class_="docsource_main")
    court = court_tag.get_text(strip=True) if court_tag else "Unknown Court"

    # Extract date
    date_tag = soup.find("div", class_="doc_date")
    date = date_tag.get_text(strip=True) if date_tag else "Unknown Date"

    return {"court": court, "date": date, "full_text": full_text}


def make_doc_id(url: str) -> str:
    """Extract numeric doc ID from IndianKanoon URL."""
    parts = [p for p in url.split("/") if p]
    return parts[-1] if parts else url


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def scrape(category: str, max_docs: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    category_dir = output_dir / category
    category_dir.mkdir(exist_ok=True)

    log.info(f"Starting scrape: category={category}, max_docs={max_docs}")
    search_results = search_judgments(category, max_docs)

    scraped, skipped = 0, 0

    for item in tqdm(search_results, desc=f"Fetching {category}"):
        doc_id = make_doc_id(item["url"])
        out_path = category_dir / f"{doc_id}.json"

        if out_path.exists():
            log.debug(f"Already scraped {doc_id}, skipping.")
            skipped += 1
            continue

        details = fetch_judgment_text(item["url"])
        if not details.get("full_text"):
            log.warning(f"Empty text for {item['url']}, skipping.")
            skipped += 1
            continue

        judgment = Judgment(
            doc_id=doc_id,
            title=item["title"],
            url=item["url"],
            category=category,
            court=details["court"],
            date=details["date"],
            full_text=details["full_text"],
            num_words=len(details["full_text"].split()),
        )

        out_path.write_text(json.dumps(asdict(judgment), ensure_ascii=False, indent=2))
        scraped += 1
        time.sleep(REQUEST_DELAY_SEC)

    log.info(f"Done. Scraped: {scraped}, Skipped/Failed: {skipped}")
    log.info(f"Output directory: {category_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape IndianKanoon judgments")
    parser.add_argument(
        "--category",
        choices=list(CATEGORY_QUERIES.keys()) + ["all"],
        required=True,
        help="Crime category to scrape (or 'all')",
    )
    parser.add_argument("--max_docs", type=int, default=150)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    categories = list(CATEGORY_QUERIES.keys()) if args.category == "all" else [args.category]
    for cat in categories:
        scrape(cat, args.max_docs, args.output)


if __name__ == "__main__":
    main()
