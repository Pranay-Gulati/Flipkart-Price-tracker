"""
Step 6: ML model — predicts next_price_change_pct (the % price change at a
product's NEXT scraped observation), using that next observation's calendar
context plus the product's current price/discount state.

Run this AFTER feature_engineering.py, on flipkart_data_features.csv.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

INPUT_CSV = "flipkart_data_features.csv"

# Features describing the product's OWN current state (known as of "now")
CURRENT_STATE_FEATURES = [
    "discount_percent", "price_change_pct", "rolling_price_mean", "rolling_price_std",
]
STATIC_FEATURES = ["subcategory", "mrp"]
# Calendar features describing the NEXT timestamp — what we're predicting INTO
NEXT_CALENDAR_FEATURES = ["next_day_of_week", "next_is_weekend", "next_is_festival", "next_time_slot"]

TARGET = "next_price_change_pct"


def build_target_and_features(df):
    df = df.sort_values(["product_id", "timestamp"]).reset_index(drop=True)

    # Target: this product's price_change_pct at its NEXT observation
    df[TARGET] = df.groupby("product_id")["price_change_pct"].shift(-1)

    # Calendar features must describe the NEXT timestamp being predicted into,
    # not the current row's own timestamp
    df["next_day_of_week"] = df.groupby("product_id")["day_of_week"].shift(-1)
    df["next_is_weekend"] = df.groupby("product_id")["is_weekend"].shift(-1)
    df["next_is_festival"] = df.groupby("product_id")["is_festival"].shift(-1)
    df["next_time_slot"] = df.groupby("product_id")["time_slot"].shift(-1)

    # mrp is missing specifically for products that never show a strikethrough
    # price — treat that as "no discount structure" (mrp == sp), not a random
    # gap to drop or average over.
    df["mrp"] = df["mrp"].fillna(df["sp"])

    return df


def prepare_model_data(df):
    df = build_target_and_features(df)

    required_cols = [TARGET] + CURRENT_STATE_FEATURES + STATIC_FEATURES + NEXT_CALENDAR_FEATURES
    before = len(df)
    df_model = df.dropna(subset=required_cols).copy()
    dropped = before - len(df_model)
    print(f"Dropped {dropped} rows missing required features/target "
          f"(each product's first observation has no lagged history, "
          f"and each product's last observation has no 'next' target). "
          f"{len(df_model)} rows remain for modeling.")

    # Keep a readable label around for inspection/debugging — get_dummies
    # below consumes the original 'subcategory' column. Named to avoid
    # colliding with the "subcategory_" prefix used to detect dummy columns.
    df_model["subcat_label"] = df_model["subcategory"]

    # One-hot encode categoricals; boolean columns -> int
    df_model["next_is_weekend"] = df_model["next_is_weekend"].astype(int)
    df_model["next_is_festival"] = df_model["next_is_festival"].astype(int)

    cat_cols = ["subcategory", "next_day_of_week", "next_time_slot"]
    df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=False)

    dummy_cols = [c for c in df_model.columns if
                  c.startswith(tuple(f"{cc}_" for cc in cat_cols))]
    feature_cols = CURRENT_STATE_FEATURES + ["mrp", "next_is_weekend", "next_is_festival"] + dummy_cols

    return df_model, feature_cols


def chronological_split(df, timestamp_col="timestamp", train_frac=0.8):
    df_sorted = df.sort_values(timestamp_col)
    cutoff = df_sorted[timestamp_col].quantile(train_frac)
    train = df_sorted[df_sorted[timestamp_col] <= cutoff]
    test = df_sorted[df_sorted[timestamp_col] > cutoff]
    print(f"Chronological split at {cutoff}: {len(train)} train rows, {len(test)} test rows")
    return train, test


def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{name}: RMSE={rmse:.3f}, MAE={mae:.3f}")
    return rmse, mae


def run():
    df = pd.read_csv(INPUT_CSV, parse_dates=["timestamp", "date"])
    df_model, feature_cols = prepare_model_data(df)

    train, test = chronological_split(df_model)
    X_train, y_train = train[feature_cols], train[TARGET]
    X_test, y_test = test[feature_cols], test[TARGET]

    print(f"\nFeatures used ({len(feature_cols)}): {feature_cols}\n")

    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    evaluate("Random Forest", y_test, rf_pred)

    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    evaluate("XGBoost", y_test, xgb_pred)

    # Feature importance from Random Forest
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 10 feature importances (Random Forest):")
    print(importances.head(10))

    return rf, xgb_model, importances


if __name__ == "__main__":
    run()
