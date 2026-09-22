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

    # --- parse timestamp: format is DD-MM-YYYY HH:MM ---
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")

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


if __name__ == "__main__":
    df = load_and_clean(INPUT_CSV)
    print_missing_report(df)

    print("\n--- Cleaned sample ---")
    print(df[["timestamp", "product_name", "sp", "mrp", "discount_percent",
              "rating", "bank_offer_price", "scrape_failed"]].head(10).to_string())

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved cleaned data to {OUTPUT_CSV}")
