"""
Flipkart price scraper — reads product list from products.csv,
scrapes each one, appends to flipkart_data_v2.csv.

Run this daily (or 3x/day) via a scheduler (cron / Task Scheduler).
Every run is independent — safe to re-run if it crashes partway.
"""

from playwright.sync_api import sync_playwright
import pandas as pd
from datetime import datetime
import os
import random
import re
import time
import json
import logging

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
PRODUCTS_CSV = "products.csv"
OUTPUT_CSV = "flipkart_data.csv"
LOG_FILE = "scrape_log.txt"

# ---------------------------------------------------------------------------
# CSS selectors — Flipkart auto-generates these hashed class names and they
# CAN change without warning on any redeploy. If the scraper starts failing
# on every product again, re-run debug_selectors.py, inspect an element in
# the browser, and update ONLY this block — nothing else needs to change.
# ---------------------------------------------------------------------------
SELECTORS = {
    "title": "h1.v1zwn21m.v1zwn26",
    "price": "div.v1zwn21m.v1zwn20",
    "mrp": "div.v1zwn21n.v1zwn21",
    "discount_percent": "div.v1zwn221.v1zwn20",
    "stock_status": "div.v1zwn220.v1zwn28",
    "bank_offer_price": "div.css-146c3p1.r-dnmrzs.r-1udh08x.r-1udbk01.r-3s2u2q.r-1iln25a",
}

# ---------------------------------------------------------------------------
# Scrape behaviour tuning
# ---------------------------------------------------------------------------
PAGE_LOAD_TIMEOUT_MS = 60_000       # how long to wait for a page to load
POST_LOAD_WAIT_MS = 5_000           # extra wait after load, for JS-rendered content
MAX_RETRIES_PER_PRODUCT = 2         # retry attempts before giving up on one product
RETRY_DELAY_RANGE_SEC = (5, 10)     # pause between retries of the same product
BETWEEN_PRODUCT_DELAY_RANGE_SEC = (3, 7)  # pause between different products
DEFAULT_STOCK_STATUS = "In stock / not shown"

# Fields every scraped row must have — keeps success/failure rows consistent
ROW_FIELDS = [
    "timestamp", "product_name", "sp", "mrp", "discount_percent", "rating",
    "stock_status_text", "bank_offer_price", "url",
]

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)


RATING_PATTERN = re.compile(r"^[0-5]\.\d$")


def get_text(page, selector, default=None):
    el = page.query_selector(selector)
    return el.inner_text() if el else default


def find_rating(page):
    """Scan all divs for a valid Flipkart rating: X.Y format, 0.0-5.0 range.
    Requiring the decimal + valid range (not just any lone digit) avoids
    matching unrelated numbers elsewhere on the page (e.g. quantity, count)."""
    for el in page.query_selector_all("div"):
        text = el.inner_text().strip()
        if RATING_PATTERN.match(text):
            return text
    return None


def get_valid_bank_offer(page, selector):
    """This selector can grab an 'Add to cart' button instead of real bank-offer
    text when no bank offer exists for a product. Only accept text that looks
    like an actual offer (contains ₹ or the word 'Buy')."""
    text = get_text(page, selector)
    if text and ("₹" in text or "buy" in text.lower()):
        return text.strip()
    return None


def extract_jsonld(page):
    """Flipkart embeds structured product data as JSON-LD (meant for Google
    Shopping/rich snippets), which is far more stable than CSS classes since
    it rarely changes structure. Returns the product dict if found, else None."""
    for script in page.query_selector_all('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.inner_text())
        except (json.JSONDecodeError, TypeError):
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict) and "offers" in item:
                return item
    return None


def empty_row(url, scrape_failed=False):
    """A row with every expected field set to None — used on total scrape failure."""
    row = {field: None for field in ROW_FIELDS}
    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row["url"] = url
    if scrape_failed:
        row["scrape_failed"] = True
    return row


def scrape_flipkart_product(page, url):
    page.goto(url, timeout=PAGE_LOAD_TIMEOUT_MS)
    page.wait_for_timeout(POST_LOAD_WAIT_MS)

    jsonld = extract_jsonld(page)

    # Title, price, and rating are pulled from JSON-LD when available — it's
    # structured data meant for search engines, so it's far less likely to
    # break than CSS classes. Fall back to CSS selectors if JSON-LD is missing
    # or a field isn't present in it.
    if jsonld:
        title = jsonld.get("name") or get_text(page, SELECTORS["title"])
        price = jsonld.get("offers", {}).get("price")
        price = f"₹ {price}" if price else get_text(page, SELECTORS["price"])
        rating_value = jsonld.get("aggregateRating", {}).get("ratingValue")
        rating = str(rating_value) if rating_value else find_rating(page)
    else:
        logging.warning(f"No JSON-LD found for {url[:60]}... falling back to CSS selectors")
        title = get_text(page, SELECTORS["title"])
        price = get_text(page, SELECTORS["price"])
        rating = find_rating(page)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "product_name": title,
        "sp": price,
        "mrp": get_text(page, SELECTORS["mrp"]),
        "discount_percent": get_text(page, SELECTORS["discount_percent"]),
        "rating": rating,
        "stock_status_text": get_text(page, SELECTORS["stock_status"], DEFAULT_STOCK_STATUS),
        "bank_offer_price": get_valid_bank_offer(page, SELECTORS["bank_offer_price"]),
        "url": url,
    }


def scrape_with_retry(page, url, max_retries=MAX_RETRIES_PER_PRODUCT):
    """Try scraping up to max_retries+1 times before giving up on this URL."""
    for attempt in range(max_retries + 1):
        try:
            data = scrape_flipkart_product(page, url)
            if not data["product_name"] and not data["sp"]:
                raise ValueError("Empty title and price — selectors may be broken")
            return data
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed for {url[:60]}... | {e}")
            if attempt < max_retries:
                time.sleep(random.uniform(*RETRY_DELAY_RANGE_SEC))
            else:
                logging.error(f"Giving up on {url[:60]}... after {max_retries+1} attempts")
                return empty_row(url, scrape_failed=True)


def main():
    if not os.path.exists(PRODUCTS_CSV):
        logging.error(f"{PRODUCTS_CSV} not found. Create it first with your 40 products.")
        return

    products = pd.read_csv(PRODUCTS_CSV, encoding="utf-8-sig", encoding_errors="replace")

    # Catch leftover TODO placeholders before wasting a scrape run on them
    todo_rows = products[products["url"].astype(str).str.contains("TODO", na=False)]
    if len(todo_rows) > 0:
        logging.warning(
            f"{len(todo_rows)} product(s) still have TODO_URL — skipping those, fill them in when ready."
        )
        products = products[~products["url"].astype(str).str.contains("TODO", na=False)]

    logging.info(f"Starting scrape run for {len(products)} products.")

    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for _, row in products.iterrows():
            logging.info(f"Scraping [{row['product_id']}] {row['name'][:40]}...")
            data = scrape_with_retry(page, row["url"])
            data["product_id"] = row["product_id"]
            data["category"] = row["category"]
            data["subcategory"] = row["subcategory"]
            results.append(data)
            # Random delay between products to reduce blocking risk
            time.sleep(random.uniform(*BETWEEN_PRODUCT_DELAY_RANGE_SEC))

        browser.close()

    df = pd.DataFrame(results)

    failed = df[df.get("scrape_failed", False) == True] if "scrape_failed" in df.columns else pd.DataFrame()
    if len(failed) > 0:
        logging.warning(f"{len(failed)} product(s) failed to scrape this run: "
                         f"{failed['product_id'].tolist()}")

    if os.path.exists(OUTPUT_CSV):
        df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(OUTPUT_CSV, mode="w", header=True, index=False, encoding="utf-8-sig")

    logging.info(f"Done. {len(df)} rows saved to {OUTPUT_CSV} "
                 f"({len(df) - len(failed)} succeeded, {len(failed)} failed).")


if __name__ == "__main__":
    main()