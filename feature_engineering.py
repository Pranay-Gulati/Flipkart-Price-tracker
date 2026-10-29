"""
Step 4: Feature engineering — date-based features, price fluctuation
measures, and the computed-vs-shown discount cross-check.
Run this AFTER clean_data.py, on flipkart_data_cleaned.csv.
"""
import pandas as pd

INPUT_CSV = "flipkart_data_cleaned.csv"
OUTPUT_CSV = "flipkart_data_features.csv"

# Festival date ranges, verified against real 2026 calendars — see project notes.
# Karva Chauth and Diwali excluded: both fall outside the collection window.
FESTIVAL_RANGES = [
    ("2026-08-24", "2026-08-29", "Onam + Raksha Bandhan"),
    ("2026-09-02", "2026-09-05", "Janmashtami"),
    ("2026-09-14", "2026-09-25", "Ganesh Chaturthi"),
    ("2026-10-11", "2026-10-20", "Navratri + Dussehra"),
]

ROLLING_WINDOW = "3D"  # time-based, not row-count — correct even with scrape
                        # failures/gaps, unlike a fixed row count would be

# Scheduled scrape times are 12am/8am/4pm IST. Each row is assigned to its
# NEAREST anchor time (handling midnight wraparound), rather than a fixed
# hour range — avoids mislabeling, e.g., a late-evening timestamp as
# "afternoon" just because it fell in a wide fixed bucket.
TIME_SLOT_ANCHORS = {"midnight_run": 0, "morning_run": 8, "afternoon_run": 16}


def add_date_features(df):
    df["date"] = df["timestamp"].dt.date
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])
    return df


def add_time_slot(df):
    """Assigns each row to its NEAREST scheduled scrape time (0/8/16 IST),
    using circular distance so hours near midnight (e.g. 23:50) correctly
    map to the midnight run rather than being stranded in a wide 'evening'
    bucket. Robust to a few minutes/hours of drift from the external
    cron trigger."""
    def nearest_slot(hour):
        def circ_dist(h, anchor):
            d = abs(h - anchor)
            return min(d, 24 - d)
        return min(TIME_SLOT_ANCHORS, key=lambda k: circ_dist(hour, TIME_SLOT_ANCHORS[k]))

    df["time_slot"] = df["timestamp"].dt.hour.apply(nearest_slot)
    return df


def add_festival_flag(df):
    df["is_festival"] = False
    df["festival_name"] = None
    for start, end, name in FESTIVAL_RANGES:
        start_date = pd.to_datetime(start).date()
        end_date = pd.to_datetime(end).date()
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        df.loc[mask, "is_festival"] = True
        df.loc[mask, "festival_name"] = name
    return df


def add_price_change_features(df):
    """Requires the df to be sorted by product_id, then timestamp — do that
    before calling this, or the diff()/rolling() will be meaningless."""
    df = df.sort_values(["product_id", "timestamp"]).reset_index(drop=True)

    grouped_sp = df.groupby("product_id")["sp"]
    df["price_change"] = grouped_sp.diff()
    df["price_change_pct"] = grouped_sp.pct_change() * 100

    # Time-based window covering the PRECEDING 3 calendar days, excluding
    # the current observation (closed='left' gives window [t-3D, t), not
    # including t itself) — this way the current price is compared against
    # its own prior history, not partly averaged with itself.
    #
    # Computed via an explicit merge on (product_id, timestamp) rather than
    # positional assignment — this makes correctness independent of row
    # order, instead of relying on groupby's default sort behavior lining
    # up with a prior sort_values() call (which is correct today, but is
    # an easy thing to accidentally break in a future edit).
    temp = df.set_index("timestamp")
    rolling_result = (
        temp.groupby("product_id")["sp"]
        .rolling(ROLLING_WINDOW, closed="left")
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "rolling_price_mean", "std": "rolling_price_std"})
    )
    df = df.merge(rolling_result, on=["product_id", "timestamp"], how="left")
    return df


def add_discount_cross_check(df):
    """Compare Flipkart's displayed discount_percent against what mrp/sp
    actually implies — a large discrepancy is a data quality flag AND a
    signal for the misleading-discount analysis."""
    df["computed_discount_percent"] = None
    has_mrp = df["mrp"].notna() & (df["mrp"] > 0)
    df.loc[has_mrp, "computed_discount_percent"] = (
        (df.loc[has_mrp, "mrp"] - df.loc[has_mrp, "sp"]) / df.loc[has_mrp, "mrp"] * 100
    )
    df["computed_discount_percent"] = pd.to_numeric(df["computed_discount_percent"])
    df["discount_discrepancy"] = df["discount_percent"] - df["computed_discount_percent"]
    return df


def engineer_features(df):
    df = add_date_features(df)
    df = add_time_slot(df)
    df = add_festival_flag(df)
    df = add_price_change_features(df)  # sorts internally
    df = add_discount_cross_check(df)
    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows")

    df = engineer_features(df)

    print(f"\nis_festival counts:\n{df['is_festival'].value_counts()}")
    print(f"\ntime_slot counts:\n{df['time_slot'].value_counts()}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved feature-engineered data to {OUTPUT_CSV}")
