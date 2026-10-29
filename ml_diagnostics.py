"""
Diagnostics for the price-change model: target distribution, naive
baseline, model comparison, error investigation, and explicit leakage checks.
Run this AFTER ml_model.py's functions are available (imports from it).
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

from ml_model import (
    INPUT_CSV, TARGET, prepare_model_data, chronological_split,
    CURRENT_STATE_FEATURES, STATIC_FEATURES, NEXT_CALENDAR_FEATURES,
)


def check_1_target_distribution(df_model):
    print("=" * 60)
    print("1. TARGET DISTRIBUTION:", TARGET)
    print("=" * 60)
    print(df_model[TARGET].describe())
    print(f"\nSkew: {df_model[TARGET].skew():.3f}")
    print(f"% of values within ±1 of zero: "
          f"{(df_model[TARGET].abs() <= 1).mean() * 100:.1f}%")
    print()


def check_2_naive_baseline(y_test):
    print("=" * 60)
    print("2. NAIVE BASELINE (predict 0% every time)")
    print("=" * 60)
    baseline_pred = np.zeros(len(y_test))
    rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    mae = mean_absolute_error(y_test, baseline_pred)
    print(f"Baseline: RMSE={rmse:.3f}, MAE={mae:.3f}")
    print()
    return rmse, mae


def check_3_comparison_table(results):
    print("=" * 60)
    print("3. COMPARISON TABLE")
    print("=" * 60)
    table = pd.DataFrame(results, columns=["Model", "RMSE", "MAE"])
    table["RMSE_improvement_vs_baseline_%"] = (
        (table.loc[0, "RMSE"] - table["RMSE"]) / table.loc[0, "RMSE"] * 100
    )
    table["MAE_improvement_vs_baseline_%"] = (
        (table.loc[0, "MAE"] - table["MAE"]) / table.loc[0, "MAE"] * 100
    )
    print(table.to_string(index=False))
    print()
    return table


def check_4_error_investigation(test, y_test, y_pred, feature_cols, model_name, top_n=10):
    print("=" * 60)
    print(f"4. LARGEST ERRORS — {model_name}")
    print("=" * 60)
    errors = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_pred,
        "abs_error": np.abs(y_test.values - y_pred),
    }, index=test.index)
    worst = errors.sort_values("abs_error", ascending=False).head(top_n)
    # bring back identifying info for inspection
    inspect_cols = ["product_name", "subcat_label", "timestamp"] + CURRENT_STATE_FEATURES
    worst_detail = test.loc[worst.index, inspect_cols].join(worst)
    print(worst_detail.to_string())
    print()
    return worst_detail


def check_5_leakage_review(feature_cols):
    print("=" * 60)
    print("5. LEAKAGE REVIEW — where does each feature's information come from?")
    print("=" * 60)
    for f in CURRENT_STATE_FEATURES:
        print(f"  [current state, pre-target]  {f}")
    print(f"  [current state, pre-target]  mrp")
    for f in ["next_is_weekend", "next_is_festival"]:
        print(f"  [calendar fact, legitimately known in advance]  {f}")
    for f in feature_cols:
        if f.startswith("next_day_of_week_") or f.startswith("next_time_slot_"):
            print(f"  [calendar fact, legitimately known in advance]  {f}")
        elif f.startswith("subcategory_"):
            print(f"  [static product info, pre-target]  {f}")
    print("\n  Cross-product boundary check: see leak_count test — must be 0.")
    print("  (rolling_price_mean/std use closed='left', excluding the CURRENT")
    print("   observation itself — conservative, not leaking future info.)")
    print()


def run_diagnostics():
    df = pd.read_csv(INPUT_CSV, parse_dates=["timestamp", "date"])
    df_model, feature_cols = prepare_model_data(df)

    check_1_target_distribution(df_model)

    train, test = chronological_split(df_model)
    X_train, y_train = train[feature_cols], train[TARGET]
    X_test, y_test = test[feature_cols], test[TARGET]

    baseline_rmse, baseline_mae = check_2_naive_baseline(y_test)

    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae = mean_absolute_error(y_test, rf_pred)

    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_mae = mean_absolute_error(y_test, xgb_pred)

    results = [
        ("Baseline (predict 0)", baseline_rmse, baseline_mae),
        ("Random Forest", rf_rmse, rf_mae),
        ("XGBoost", xgb_rmse, xgb_mae),
    ]
    check_3_comparison_table(results)

    check_4_error_investigation(test, y_test, rf_pred, feature_cols, "Random Forest")

    check_5_leakage_review(feature_cols)


if __name__ == "__main__":
    run_diagnostics()
