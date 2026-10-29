"""
Generates comparison plots for the ML section of the presentation:
1. Baseline vs RF vs XGBoost — RMSE/MAE, for BOTH the full model (all rows)
   and the filtered model (real-change rows only)
2. Predicted vs Actual scatter — filtered model, the one showing real signal
3. Feature importance comparison — full model vs filtered model, side by
   side (supports the "festival affects WHETHER, not HOW MUCH" finding)

Run this AFTER ml_model.py and ml_model_filtered.py both work on your data.
Saves PNG files directly — insert these into your slides.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

from ml_model import INPUT_CSV, TARGET, prepare_model_data, chronological_split
from ml_model_filtered import filter_to_real_changes

plt.rcParams["figure.figsize"] = (9, 5)
plt.style.use("seaborn-v0_8-whitegrid")


def train_and_eval(X_train, y_train, X_test, y_test):
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    return rf, rf_pred, xgb_model, xgb_pred


def plot_comparison_bars(results, title, filename):
    """results: dict like {'Baseline': (rmse, mae), 'Random Forest': (...), 'XGBoost': (...)}"""
    labels = list(results.keys())
    rmses = [results[k][0] for k in labels]
    maes = [results[k][1] for k in labels]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, rmses, width, label="RMSE", color="steelblue")
    ax.bar(x + width/2, maes, width, label="MAE", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")


def plot_predicted_vs_actual(y_test, y_pred, title, filename):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6, color="steelblue", edgecolor="white")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="gray", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual next_price_change_pct")
    ax.set_ylabel("Predicted next_price_change_pct")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")


def plot_feature_importance_comparison(imp_full, imp_filtered, filename, top_n=8):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    top_full = imp_full.head(top_n).sort_values()
    axes[0].barh(top_full.index, top_full.values, color="steelblue")
    axes[0].set_title("Full model (all rows, incl. zeros)")

    top_filtered = imp_filtered.head(top_n).sort_values()
    axes[1].barh(top_filtered.index, top_filtered.values, color="coral")
    axes[1].set_title("Filtered model (real changes only)")

    fig.suptitle("Feature importance: full model vs filtered model")
    fig.tight_layout()
    fig.savefig(filename, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")


def run():
    df = pd.read_csv(INPUT_CSV, parse_dates=["timestamp", "date"])
    df_model, feature_cols = prepare_model_data(df)

    # ---- Full model (all rows, including zeros) ----
    train, test = chronological_split(df_model)
    X_train, y_train = train[feature_cols], train[TARGET]
    X_test, y_test = test[feature_cols], test[TARGET]

    baseline_pred_full = np.zeros(len(y_test))
    baseline_rmse_full = np.sqrt(mean_squared_error(y_test, baseline_pred_full))
    baseline_mae_full = mean_absolute_error(y_test, baseline_pred_full)

    rf_full, rf_pred_full, xgb_full, xgb_pred_full = train_and_eval(X_train, y_train, X_test, y_test)
    rf_rmse_full = np.sqrt(mean_squared_error(y_test, rf_pred_full))
    rf_mae_full = mean_absolute_error(y_test, rf_pred_full)
    xgb_rmse_full = np.sqrt(mean_squared_error(y_test, xgb_pred_full))
    xgb_mae_full = mean_absolute_error(y_test, xgb_pred_full)

    plot_comparison_bars(
        {"Baseline (predict 0)": (baseline_rmse_full, baseline_mae_full),
         "Random Forest": (rf_rmse_full, rf_mae_full),
         "XGBoost": (xgb_rmse_full, xgb_mae_full)},
        "Full model: Baseline vs RF vs XGBoost (all rows, incl. zeros)",
        "plot_1_full_model_comparison.png",
    )

    imp_full = pd.Series(rf_full.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # ---- Filtered model (real changes only) ----
    df_filtered = filter_to_real_changes(df_model)
    train_f, test_f = chronological_split(df_filtered)
    Xf_train, yf_train = train_f[feature_cols], train_f[TARGET]
    Xf_test, yf_test = test_f[feature_cols], test_f[TARGET]

    train_mean = yf_train.mean()
    baseline_pred_filt = np.full(len(yf_test), train_mean)
    baseline_rmse_filt = np.sqrt(mean_squared_error(yf_test, baseline_pred_filt))
    baseline_mae_filt = mean_absolute_error(yf_test, baseline_pred_filt)

    rf_filt, rf_pred_filt, xgb_filt, xgb_pred_filt = train_and_eval(Xf_train, yf_train, Xf_test, yf_test)
    rf_rmse_filt = np.sqrt(mean_squared_error(yf_test, rf_pred_filt))
    rf_mae_filt = mean_absolute_error(yf_test, rf_pred_filt)
    xgb_rmse_filt = np.sqrt(mean_squared_error(yf_test, xgb_pred_filt))
    xgb_mae_filt = mean_absolute_error(yf_test, xgb_pred_filt)

    plot_comparison_bars(
        {f"Baseline (mean={train_mean:.2f})": (baseline_rmse_filt, baseline_mae_filt),
         "Random Forest": (rf_rmse_filt, rf_mae_filt),
         "XGBoost": (xgb_rmse_filt, xgb_mae_filt)},
        "Filtered model: Baseline vs RF vs XGBoost (real changes only)",
        "plot_2_filtered_model_comparison.png",
    )

    plot_predicted_vs_actual(
        yf_test, rf_pred_filt,
        "Filtered model — Predicted vs Actual (Random Forest)",
        "plot_3_predicted_vs_actual.png",
    )

    imp_filtered = pd.Series(rf_filt.feature_importances_, index=feature_cols).sort_values(ascending=False)

    plot_feature_importance_comparison(
        imp_full, imp_filtered, "plot_4_feature_importance_comparison.png"
    )

    print("\nAll plots saved. Ready to insert into slides.")


if __name__ == "__main__":
    run()
