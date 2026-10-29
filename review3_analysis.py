"""
Review 3 additions, run AFTER feature_engineering.py:
  1. Does a price change happen? (festival / weekend / time-slot rates)
  2. Classifier: will the NEXT observation show a price change? (AUC / F1)
  3. R-squared for the regression models
  4. Deceptive-discount checks (rule-based)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, r2_score
import xgboost as xgb

from ml_model import prepare_model_data, chronological_split, TARGET

INPUT_CSV = "flipkart_data_features.csv"
CHANGE_THRESHOLD = 0.5  # % — below this a move is treated as no change


def change_rates(df):
    df = df.copy()
    df["changed"] = df["price_change_pct"].abs() > CHANGE_THRESHOLD
    df = df[df["price_change_pct"].notna()]
    print("\n=== 1. Share of observations with a price change ===")
    print(f"Overall: {df['changed'].mean():.2%}")
    for col in ["is_festival", "is_weekend", "time_slot"]:
        print(f"\nBy {col}:")
        print((df.groupby(col)["changed"].mean() * 100).round(2).to_string())
    print("\nBy festival_name (NaN = no festival):")
    print((df.groupby(df["festival_name"].fillna("None"))["changed"].mean() * 100).round(2).to_string())
    print("\nBy subcategory:")
    print((df.groupby("subcategory")["changed"].mean() * 100).round(2).to_string())


def classifier(df):
    df_model, feature_cols = prepare_model_data(df)
    df_model["next_changed"] = (df_model[TARGET].abs() > CHANGE_THRESHOLD).astype(int)
    train, test = chronological_split(df_model)
    X_tr, y_tr = train[feature_cols], train["next_changed"]
    X_te, y_te = test[feature_cols], test["next_changed"]
    print(f"\n=== 2. Change classifier === positives: train {y_tr.mean():.2%}, test {y_te.mean():.2%}")

    pos_w = (1 - y_tr.mean()) / max(y_tr.mean(), 1e-6)
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                                                random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     scale_pos_weight=pos_w, random_state=42),
    }
    importances = None
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        # threshold tuned on train-set probabilities would overfit; use the train base rate quantile
        thr = np.quantile(m.predict_proba(X_tr)[:, 1], 1 - y_tr.mean())
        pred = (p >= thr).astype(int)
        print(f"{name}: AUC={roc_auc_score(y_te, p):.3f}  F1={f1_score(y_te, pred):.3f}  "
              f"precision={precision_score(y_te, pred, zero_division=0):.3f}  "
              f"recall={recall_score(y_te, pred):.3f}")
        if name == "Random Forest":
            importances = pd.Series(m.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("Baseline (always 'no change'): AUC=0.500, F1=0.000")
    print("\nTop 10 classifier features (RF):")
    print(importances.head(10).round(4).to_string())


def r_squared(df):
    df_model, feature_cols = prepare_model_data(df)
    train, test = chronological_split(df_model)
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(train[feature_cols], train[TARGET])
    xg = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    xg.fit(train[feature_cols], train[TARGET])
    print("\n=== 3. R-squared (full data, chronological test set) ===")
    print(f"RF  R2 = {r2_score(test[TARGET], rf.predict(test[feature_cols])):.3f}")
    print(f"XGB R2 = {r2_score(test[TARGET], xg.predict(test[feature_cols])):.3f}")


def deceptive_discounts(df):
    df = df.sort_values(["product_id", "timestamp"]).copy()
    print("\n=== 4. Deceptive-discount checks ===")
    n_disc = df["discount_percent"].notna().sum()

    # Rule A: displayed discount disagrees with (mrp - sp) / mrp
    a = df["discount_discrepancy"].abs() > 2
    print(f"A. Displayed vs computed discount differs by >2pp: {a.sum()} of {n_disc} rows ({a.sum()/n_disc:.2%})")

    # Rule B: MRP raised while selling price did not rise by as much -> discount inflated
    g = df.groupby("product_id")
    df["mrp_chg"] = g["mrp"].pct_change() * 100
    df["sp_chg"] = g["sp"].pct_change() * 100
    b = (df["mrp_chg"] > 1) & (df["sp_chg"] < df["mrp_chg"] / 2)
    print(f"B. MRP raised while SP rose less than half as much: {b.sum()} rows across "
          f"{df.loc[b, 'product_id'].nunique()} products")
    print(df[b].groupby("subcategory").size().sort_values(ascending=False).to_string())

    # Rule C: festival-time discount rise not backed by a real SP drop vs. the product's non-festival median
    base = df[~df["is_festival"]].groupby("product_id").agg(base_sp=("sp", "median"),
                                                            base_disc=("discount_percent", "median"))
    f = df[df["is_festival"]].merge(base, on="product_id", how="left")
    f["sp_vs_base_pct"] = (f["sp"] / f["base_sp"] - 1) * 100
    f["disc_vs_base_pp"] = f["discount_percent"] - f["base_disc"]
    c = (f["disc_vs_base_pp"] >= 5) & (f["sp_vs_base_pct"] > -1)
    fd = f["disc_vs_base_pp"].notna().sum()
    print(f"C. Festival discount >=5pp above baseline but SP not lower than baseline: "
          f"{c.sum()} of {fd} festival rows ({c.sum()/fd:.2%}), "
          f"{f.loc[c, 'product_id'].nunique()} products")
    print(f[c].groupby("subcategory").size().sort_values(ascending=False).to_string())
    print("\nMean festival SP vs baseline by subcategory (%, negative = cheaper in festival):")
    print(f.groupby("subcategory")["sp_vs_base_pct"].mean().round(2).to_string())
    print("\nMean festival discount vs baseline by subcategory (pp):")
    print(f.groupby("subcategory")["disc_vs_base_pp"].mean().round(2).to_string())

    # Products flagged by the most rules
    flags = pd.DataFrame({"A": a.groupby(df["product_id"]).sum(), "B": b.groupby(df["product_id"]).sum()})
    flags["C"] = c.groupby(f["product_id"]).sum()
    flags = flags.fillna(0).astype(int)
    names = df.drop_duplicates("product_id").set_index("product_id")["product_name"].str[:45]
    flags["name"] = names
    print("\nTop 10 products by total flags:")
    print(flags.assign(total=flags[["A", "B", "C"]].sum(axis=1)).sort_values("total", ascending=False)
          .head(10).to_string())


if __name__ == "__main__":
    data = pd.read_csv(INPUT_CSV, parse_dates=["timestamp", "date"])
    change_rates(data)
    classifier(data)
    r_squared(data)
    deceptive_discounts(data)
