"""
Step 1-3: Load, clean, and quality-check the scraped Flipkart data.
Run this first, before any feature engineering or EDA.
"""
import pandas as pd

INPUT_CSV = "flipkart_data.csv"
OUTPUT_CSV = "flipkart_data_cleaned.csv"


def clean_price(value):
    """Strip ₹, commas, 'Buy at' — return a float, or None if not parseable/missing."""
    if pd.isna(value):
        return None
    value = str(value).replace("₹", "").replace(",", "").replace("Buy at", "").strip()
    try:
        return float(value)
    except ValueError:
        return None


def clean_percent(value):
    if pd.isna(value):
        return None
    value = str(value).replace("%", "").strip()
    try:
        return float(value)
    except ValueError:
        return None


def clean_rating(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_mixed_timestamp(series):
    """The CSV contains timestamps in two different formats — most rows use
    the scraper's original 'YYYY-MM-DD HH:MM:SS', but some rows appear to
    have been reformatted (likely by Excel being opened/saved at some point)
    to 'DD-MM-YYYY HH:MM' with no seconds. Try both, in order."""
    parsed = pd.to_datetime(series, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    mask = parsed.isna()
    if mask.any():
        parsed2 = pd.to_datetime(series[mask], format="%d-%m-%Y %H:%M", errors="coerce")
        parsed.loc[mask] = parsed2
    return parsed


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # --- scrape_failed: convert string TRUE/FALSE to real boolean ---
    if df["scrape_failed"].dtype == object:
        df["scrape_failed"] = df["scrape_failed"].astype(str).str.upper() == "TRUE"

    failed_count = df["scrape_failed"].sum()
    print(f"scrape_failed rows: {failed_count} ({failed_count/len(df)*100:.1f}%)")

    # --- clean product name (strip the "...more" truncation marker) ---
    df["product_name"] = (
        df["product_name"]
        .astype(str)
        .str.replace("...more", "", regex=False)
        .str.replace("…more", "", regex=False)
        .str.strip()
    )

    # --- clean numeric/price fields ---
    df["sp"] = df["sp"].apply(clean_price)
    df["mrp"] = df["mrp"].apply(clean_price)
    df["discount_percent"] = df["discount_percent"].apply(clean_percent)
    df["bank_offer_price"] = df["bank_offer_price"].apply(clean_price)
    df["rating"] = df["rating"].apply(clean_rating)

    # --- parse timestamp: handles mixed formats found in the real data ---
    df["timestamp"] = parse_mixed_timestamp(df["timestamp"])

    bad_timestamps = df["timestamp"].isna().sum()
    if bad_timestamps > 0:
        print(f"WARNING: {bad_timestamps} rows had a timestamp that couldn't be parsed")

    # --- check for exact duplicates (same product + same timestamp) ---
    dupes = df.duplicated(subset=["product_id", "timestamp"]).sum()
    if dupes > 0:
        print(f"WARNING: {dupes} duplicate (product_id, timestamp) rows found — "
              f"consider df.drop_duplicates(subset=['product_id','timestamp'])")

    # --- sanity range checks ---
    bad_sp = df[(df["sp"].notna()) & (df["sp"] <= 0)]
    bad_discount = df[(df["discount_percent"].notna()) &
                       ((df["discount_percent"] < 0) | (df["discount_percent"] > 100))]
    bad_rating = df[(df["rating"].notna()) & ((df["rating"] < 0) | (df["rating"] > 5))]
    if len(bad_sp) > 0:
        print(f"WARNING: {len(bad_sp)} rows have sp <= 0")
    if len(bad_discount) > 0:
        print(f"WARNING: {len(bad_discount)} rows have discount_percent outside 0-100")
    if len(bad_rating) > 0:
        print(f"WARNING: {len(bad_rating)} rows have rating outside 0-5")

    return df


def print_missing_report(df):
    print("\n--- Missing value report ---")
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    print(report[report["missing_count"] > 0])


def remove_failed_rows(df):
    """scrape_failed rows have every field empty except url/product_id — they
    carry no usable information, so drop them rather than carrying them
    through feature engineering and EDA as extra NaN rows."""
    before = len(df)
    df = df[~df["scrape_failed"]].copy()
    removed = before - len(df)
    print(f"Removed {removed} scrape_failed rows ({removed/before*100:.1f}%). "
          f"{len(df)} rows remain.")
    return df


def inspect_data(df):
    """Deeper checks beyond missing values — coverage per product, unexpected
    category values, and rough price outliers. Prints findings; doesn't
    modify the data (you decide what, if anything, needs handling)."""
    print("\n--- Per-product row coverage ---")
    counts = df.groupby("product_id").size()
    print(f"Products: {counts.shape[0]} | rows per product: "
          f"min={counts.min()}, max={counts.max()}, median={counts.median():.0f}")
    low_coverage = counts[counts < counts.median() * 0.5]
    if len(low_coverage) > 0:
        print(f"WARNING: {len(low_coverage)} product(s) have notably fewer rows "
              f"than the rest — check if their URL/scraping has been failing "
              f"disproportionately:\n{low_coverage}")

    print("\n--- stock_status_text unique values ---")
    print(df["stock_status_text"].value_counts())

    print("\n--- category / subcategory consistency check ---")
    inconsistent = df.groupby("product_id")[["category", "subcategory"]].nunique()
    bad = inconsistent[(inconsistent["category"] > 1) | (inconsistent["subcategory"] > 1)]
    if len(bad) > 0:
        print(f"WARNING: {len(bad)} product_id(s) have inconsistent category/"
              f"subcategory across rows — likely a products.csv edit mid-collection:\n{bad}")
    else:
        print("OK — every product_id has a single consistent category/subcategory.")


if __name__ == "__main__":
    df = load_and_clean(INPUT_CSV)
    print_missing_report(df)

    df = remove_failed_rows(df)
    inspect_data(df)

    print("\n--- Cleaned sample ---")
    print(df[["timestamp", "product_name", "sp", "mrp", "discount_percent",
              "rating", "bank_offer_price", "scrape_failed"]].head(10).to_string())

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved cleaned data to {OUTPUT_CSV}")
