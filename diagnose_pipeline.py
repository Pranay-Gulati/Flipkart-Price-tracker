"""
Diagnostic: check the date range of the data at each stage of the pipeline
(raw -> cleaned -> feature-engineered) to isolate exactly where rows are
being lost or dates are getting truncated.
"""
import pandas as pd

print("=" * 60)
print("STAGE 1: Raw scraped data (flipkart_data.csv)")
print("=" * 60)
try:
    raw = pd.read_csv("flipkart_data.csv")
    print(f"Rows: {len(raw)}")
    # raw timestamp is still text at this stage, mixed formats possible
    raw_ts = pd.to_datetime(raw["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    mask = raw_ts.isna()
    if mask.any():
        raw_ts2 = pd.to_datetime(raw.loc[mask, "timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
        raw_ts.loc[mask] = raw_ts2
    print(f"Date range: {raw_ts.min()} to {raw_ts.max()}")
    print(f"Unrecoverable timestamps: {raw_ts.isna().sum()}")
except FileNotFoundError:
    print("flipkart_data.csv not found in this folder")

print()
print("=" * 60)
print("STAGE 2: Cleaned data (flipkart_data_cleaned.csv)")
print("=" * 60)
try:
    cleaned = pd.read_csv("flipkart_data_cleaned.csv", parse_dates=["timestamp"])
    print(f"Rows: {len(cleaned)}")
    print(f"Date range: {cleaned['timestamp'].min()} to {cleaned['timestamp'].max()}")
    print(f"NaT timestamps: {cleaned['timestamp'].isna().sum()}")
except FileNotFoundError:
    print("flipkart_data_cleaned.csv not found in this folder")

print()
print("=" * 60)
print("STAGE 3: Feature-engineered data (flipkart_data_features.csv)")
print("=" * 60)
try:
    features = pd.read_csv("flipkart_data_features.csv")
    print(f"Rows: {len(features)}")
    print(f"Unique dates: {sorted(features['date'].unique())}")
except FileNotFoundError:
    print("flipkart_data_features.csv not found in this folder")
