"""
Filtered variant: models price_change_pct magnitude/direction only for
rows where a real price change actually happened, excluding the ~98% of
rows where nothing changed (next_price_change_pct == 0). The zero-vs-not
question is a separate, easier problem this doesn't try to solve.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

from ml_model import INPUT_CSV, TARGET, prepare_model_data, CURRENT_STATE_FEATURES

ZERO_EPSILON = 1e-9  # floating-point safety margin, not a "small change" threshold


def filter_to_real_changes(df_model):
    before = len(df_model)
    filtered = df_model[df_model[TARGET].abs() > ZERO_EPSILON].copy()
    print(f"Filtered {before} -> {len(filtered)} rows "
          f"({len(filtered)/before*100:.1f}% had a real price change)")
    return filtered


def chronological_split(df, timestamp_col="timestamp", train_frac=0.8):
    df_sorted = df.sort_values(timestamp_col)
    cutoff = df_sorted[timestamp_col].quantile(train_frac)
    train = df_sorted[df_sorted[timestamp_col] <= cutoff]
    test = df_sorted[df_sorted[timestamp_col] > cutoff]
    print(f"Chronological split at {cutoff}: {len(train)} train rows, {len(test)} test rows")
    return train, test


def run_filtered():
    df = pd.read_csv(INPUT_CSV, parse_dates=["timestamp", "date"])
    df_model, feature_cols = prepare_model_data(df)
    df_filtered = filter_to_real_changes(df_model)

    print("\nFiltered target distribution:")
    print(df_filtered[TARGET].describe())

    train, test = chronological_split(df_filtered)
    X_train, y_train = train[feature_cols], train[TARGET]
    X_test, y_test = test[feature_cols], test[TARGET]

    # Appropriate baseline for THIS subset: predict the training mean,
    # not 0 — "predict 0" is no longer meaningful once true-zero rows
    # have been excluded.
    train_mean = y_train.mean()
    baseline_pred = np.full(len(y_test), train_mean)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    print(f"\nBaseline (predict training mean = {train_mean:.3f}): "
          f"RMSE={baseline_rmse:.3f}, MAE={baseline_mae:.3f}")

    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae = mean_absolute_error(y_test, rf_pred)
    print(f"Random Forest: RMSE={rf_rmse:.3f}, MAE={rf_mae:.3f}")

    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    print(f"XGBoost: RMSE={xgb_rmse:.3f}, MAE={xgb_mae:.3f}")

    print(f"\nRF improvement vs baseline — RMSE: "
          f"{(baseline_rmse - rf_rmse) / baseline_rmse * 100:.1f}%, "
          f"MAE: {(baseline_mae - rf_mae) / baseline_mae * 100:.1f}%")
    print(f"XGB improvement vs baseline — RMSE: "
          f"{(baseline_rmse - xgb_rmse) / baseline_rmse * 100:.1f}%, "
          f"MAE: {(baseline_mae - xgb_mae) / baseline_mae * 100:.1f}%")

    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 10 feature importances (Random Forest, filtered model):")
    print(importances.head(10))

    return rf, xgb_model, importances


if __name__ == "__main__":
    run_filtered()
