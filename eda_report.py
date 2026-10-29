"""
Step 5: EDA — generates one self-contained HTML report covering all 11
analysis sections. Run this AFTER feature_engineering.py, on
flipkart_data_features.csv.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import math
from io import BytesIO

INPUT_CSV = "flipkart_data_features.csv"
OUTPUT_HTML = "eda_report.html"

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def img_tag(fig):
    return f'<img src="data:image/png;base64,{fig_to_base64(fig)}" style="max-width:100%; margin:10px 0;">'


def section(title, question, content_html):
    return f"""
    <div class="section">
        <h2>{title}</h2>
        <p class="question"><em>{question}</em></p>
        {content_html}
    </div>
    """


def section_1_overview(df):
    n_rows = len(df)
    n_products = df["product_id"].nunique()
    n_categories = df["category"].nunique()
    n_subcategories = df["subcategory"].nunique()
    date_min, date_max = df["date"].min(), df["date"].max()
    stats = df[["sp", "mrp", "discount_percent", "rating"]].describe().round(2)

    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    missing_report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_report = missing_report[missing_report["missing_count"] > 0]

    html = f"""
    <ul>
        <li>Rows: {n_rows}</li>
        <li>Products: {n_products} across {n_categories} categories, {n_subcategories} subcategories</li>
        <li>Date range: {date_min} to {date_max}</li>
    </ul>
    {stats.to_html(classes='stats-table')}
    <h3>Missing values</h3>
    {missing_report.to_html(classes='stats-table') if len(missing_report) > 0 else '<p>No missing values.</p>'}
    """
    return section("1. Dataset Overview", "What does this dataset actually cover?", html)


def section_2_price_trend(df):
    subcats = sorted(df["subcategory"].dropna().unique())
    ncols = 3
    nrows = math.ceil(len(subcats) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, sc in enumerate(subcats):
        sub = df[df["subcategory"] == sc]
        trend = sub.groupby("date")["sp"].mean()
        normalized = trend / trend.iloc[0] * 100
        axes[i].plot(normalized.index, normalized.values)
        axes[i].axhline(100, color="gray", linestyle="--", linewidth=0.8)
        axes[i].set_title(sc, fontsize=10)
        axes[i].tick_params(axis="x", rotation=45, labelsize=7)

    for j in range(len(subcats), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Normalized price trend by subcategory (100 = starting price)")
    fig.tight_layout()
    chart = img_tag(fig)

    return section("2. Price Trend",
                    "How does price move over time, WITHIN each subcategory (normalized so different price scales are comparable)?",
                    chart)


def section_3_weekday_weekend(df):
    df_w = df.copy()
    df_w["day_type"] = df_w["is_weekend"].map({True: "Weekend", False: "Weekday"})

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df_w, x="subcategory", y="discount_percent", hue="day_type", ax=ax)
    ax.set_title("Discount % — Weekday vs Weekend, by subcategory")
    ax.set_xlabel("")
    plt.xticks(rotation=25, ha="right")
    chart1 = img_tag(fig)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df_w, x="subcategory", y="price_change_pct", hue="day_type", ax=ax2)
    ax2.set_title("Price change % — Weekday vs Weekend, by subcategory")
    ax2.set_xlabel("")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xticks(rotation=25, ha="right")
    chart2 = img_tag(fig2)

    return section("3. Weekday vs Weekend",
                    "Within each subcategory, does discount or price movement differ weekday vs weekend?",
                    chart1 + chart2)


def section_4_festival(df):
    df_f = df.copy()
    df_f["festival_label"] = df_f["festival_name"].fillna("None")

    pivot_disc = df_f.pivot_table(values="discount_percent", index="subcategory",
                                   columns="festival_label", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_disc, annot=True, fmt=".1f", cmap="Greens", ax=ax)
    ax.set_title("Avg discount %: Subcategory × Festival")
    chart1 = img_tag(fig)

    pivot_pct = df_f.pivot_table(values="price_change_pct", index="subcategory",
                                  columns="festival_label", aggfunc="mean")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_pct, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax2, center=0)
    ax2.set_title("Avg price change %: Subcategory × Festival")
    chart2 = img_tag(fig2)

    return section("4. Festival Behaviour",
                    "Within each subcategory, how does each festival window compare to others?",
                    chart1 + chart2)


def section_5_time_of_day(df):
    order = ["midnight_run", "morning_run", "afternoon_run"]
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="subcategory", y="discount_percent", hue="time_slot",
                hue_order=order, ax=ax)
    ax.set_title("Discount % by time of day, within each subcategory")
    ax.set_xlabel("")
    plt.xticks(rotation=25, ha="right")
    return section("5. Time of Day",
                    "Within each subcategory, does the scrape time (12am/8am/4pm) show any difference?",
                    img_tag(fig))


def section_6_category(df):
    df_cv = df.copy()
    df_cv["cv"] = df_cv["rolling_price_std"] / df_cv["rolling_price_mean"] * 100
    cv_by_subcat = df_cv.groupby("subcategory")["cv"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    cv_by_subcat.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Price volatility by subcategory (coefficient of variation)")
    ax.set_xlabel("CV (%) — std relative to mean price, higher = more volatile proportionally")
    note = "<p><em>Using CV (std ÷ mean × 100) instead of raw std, so expensive products " \
           "don't automatically rank as \"most volatile\" just because their ₹ swings are " \
           "bigger in absolute terms.</em></p>"
    return section("6. Price Volatility", "Which subcategory fluctuates most, PROPORTIONALLY to its own price?",
                    img_tag(fig) + note)


def section_7_product_spotlight(df):
    df_cv = df.copy()
    df_cv["cv"] = df_cv["rolling_price_std"] / df_cv["rolling_price_mean"] * 100
    prod_cv = (
        df_cv.groupby(["product_id", "product_name", "subcategory"])["cv"]
        .mean()
        .dropna()
        .sort_values(ascending=False)
    )
    top5 = prod_cv.head(5).reset_index()
    bottom5 = prod_cv.tail(5).reset_index()
    top5.columns = ["product_id", "product_name", "subcategory", "avg_cv_pct"]
    bottom5.columns = ["product_id", "product_name", "subcategory", "avg_cv_pct"]
    html = "<h3>Most volatile (by CV)</h3>" + top5.round(2).to_html(index=False)
    html += "<h3>Least volatile (by CV)</h3>" + bottom5.round(2).to_html(index=False)
    return section("7. Individual Product Spotlight",
                    "Which SPECIFIC products fluctuate most/least, proportionally?", html)


def section_8_bank_offer(df):
    df_b = df[df["bank_offer_price"].notna()].copy()
    df_b["bank_offer_pct"] = (df_b["sp"] - df_b["bank_offer_price"]) / df_b["sp"] * 100
    df_b["festival_label"] = df_b["is_festival"].map({True: "Festival", False: "Non-festival"})

    pivot = df_b.pivot_table(values="bank_offer_pct", index="subcategory",
                              columns="festival_label", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Purples", ax=ax)
    ax.set_title("Avg bank-offer benefit (% of SP): Subcategory × Festival")
    note = f"<p><em>Based on {len(df_b)} rows where a bank offer was actually shown " \
           f"(excludes {df['bank_offer_price'].isna().sum()} rows with no bank offer, " \
           f"rather than treating missing as ₹0). Shown as % of SP so subcategories at " \
           f"different price levels are comparable.</em></p>"
    return section("8. Bank Offer Analysis",
                    "Is the bank-offer benefit ALSO bigger during festivals, and does it vary by subcategory?",
                    img_tag(fig) + note)


def section_9_discrepancy(df):
    fig, ax = plt.subplots()
    df["discount_discrepancy"].dropna().hist(bins=30, ax=ax, color="coral")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Distribution of discount discrepancy (shown − computed)")
    ax.set_xlabel("Discrepancy (percentage points)")
    chart1 = img_tag(fig)

    subcat_disc = df.groupby("subcategory")["discount_discrepancy"].mean().sort_values()
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    subcat_disc.plot(kind="barh", ax=ax2, color="coral")
    ax2.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax2.set_title("Average discount discrepancy by subcategory (signed)")
    ax2.set_xlabel("Avg discrepancy (shown − computed)")
    chart2 = img_tag(fig2)

    return section("9. Discount Discrepancy — Overall + By Subcategory",
                    "How often, and by how much, does the shown discount disagree with the real one?",
                    chart1 + chart2)


def section_10_cross_check(df):
    df_c = df.copy()
    df_c["festival_label"] = df_c["festival_name"].fillna("None")

    # Normalize price against each subcategory's own overall mean, so a heatmap
    # cell is comparable across subcategories at very different price levels
    subcat_mean_sp = df_c.groupby("subcategory")["sp"].transform("mean")
    df_c["normalized_sp"] = df_c["sp"] / subcat_mean_sp * 100

    pivot_disc = df_c.pivot_table(values="discount_discrepancy", index="subcategory",
                                   columns="festival_label", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_disc, annot=True, fmt=".1f", cmap="RdYlGn_r", ax=ax, center=0)
    ax.set_title("Avg discount discrepancy: Subcategory × Festival")
    chart1 = img_tag(fig)

    pivot_sp = df_c.pivot_table(values="normalized_sp", index="subcategory",
                                 columns="festival_label", aggfunc="mean")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_sp, annot=True, fmt=".1f", cmap="Blues", ax=ax2, center=100)
    ax2.set_title("Normalized price (100 = subcategory's own average): Subcategory × Festival")
    chart2 = img_tag(fig2)

    pivot_pct = df_c.pivot_table(values="discount_percent", index="subcategory",
                                  columns="festival_label", aggfunc="mean")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_pct, annot=True, fmt=".1f", cmap="Greens", ax=ax3)
    ax3.set_title("Avg discount %: Subcategory × Festival")
    chart3 = img_tag(fig3)

    note = "<p><em>These three views are best read together, not the discrepancy " \
           "heatmap alone: a discrepancy value on its own doesn't say whether it " \
           "reflects genuine discounting or an inflated MRP claim — the normalized " \
           "price and discount% patterns alongside it help narrow down which is more " \
           "likely. Any interpretation here is suggestive, not conclusive.</em></p>"

    return section("10. Subcategory × Festival Cross-Check",
                    "Where does discount discrepancy concentrate — and does it worsen during festivals?",
                    chart1 + chart2 + chart3 + note)


def section_11_correlation(df):
    corr_cols = ["sp", "discount_percent", "rating", "rolling_price_std"]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, center=0)
    ax.set_title("Correlation heatmap")
    return section("11. Correlation Heatmap", "Do price, discount, rating, and volatility relate to each other?", img_tag(fig))


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Flipkart Price Fluctuation — EDA Report</title>
<style>
    body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 20px; color: #222; }}
    h1 {{ border-bottom: 3px solid #2874f0; padding-bottom: 10px; }}
    .section {{ margin: 40px 0; padding: 20px 0; border-top: 1px solid #ddd; }}
    h2 {{ color: #2874f0; }}
    .question {{ color: #666; }}
    table {{ border-collapse: collapse; margin: 10px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
    th {{ background: #f5f5f5; }}
</style>
</head>
<body>
<h1>Flipkart Price Fluctuation — EDA Report</h1>
{sections}
</body>
</html>
"""


def build_report(df):
    sections = [
        section_1_overview(df),
        section_2_price_trend(df),
        section_3_weekday_weekend(df),
        section_4_festival(df),
        section_5_time_of_day(df),
        section_6_category(df),
        section_7_product_spotlight(df),
        section_8_bank_offer(df),
        section_9_discrepancy(df),
        section_10_cross_check(df),
        section_11_correlation(df),
    ]
    return HTML_TEMPLATE.format(sections="\n".join(sections))


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV, parse_dates=["timestamp"])
    df["date"] = pd.to_datetime(df["date"])
    print(f"Loaded {len(df)} rows")

    html = build_report(df)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved EDA report to {OUTPUT_HTML}")
