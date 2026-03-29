"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Supply Chain Risk Analytics — Streamlit Web Application                   ║
║   Model  : CatBoost/LightGBM/HistGradientBoosting/RandomForest ensemble     ║
║   Pages  : Prediction | Business Insights | Business Recommendations        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
import shap
from datetime import timedelta

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Risk Analytics",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #E2E8F0;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-label {
        font-size: 13px;
        color: #64748B;
        font-weight: 500;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 32px;
        font-weight: 700;
        margin: 0;
        line-height: 1.1;
    }
    .kpi-red    { color: #DC2626; }
    .kpi-orange { color: #F97316; }
    .kpi-green  { color: #16A34A; }
    .kpi-blue   { color: #2563EB; }
    .kpi-purple { color: #7C3AED; }
    .result-box {
        border-radius: 12px;
        padding: 24px 28px;
        margin-top: 16px;
        border: 1px solid #E2E8F0;
    }
    .result-high   { background: #FEF2F2; border-color: #FECACA; }
    .result-medium { background: #FFFBEB; border-color: #FDE68A; }
    .result-low    { background: #F0FDF4; border-color: #BBF7D0; }
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #1E293B;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #E2E8F0;
    }
    .rec-card {
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        border-left: 5px solid;
    }
    .rec-high   { background: #FEF2F2; border-color: #DC2626; }
    .rec-medium { background: #FFFBEB; border-color: #F97316; }
    .rec-low    { background: #F0FDF4; border-color: #16A34A; }
    .rec-title  { font-weight: 700; font-size: 15px; margin-bottom: 6px; }
    .rec-body   { font-size: 14px; color: #475569; }
    .info-box {
        background: #EFF6FF;
        border-radius: 10px;
        padding: 14px 18px;
        border: 1px solid #BFDBFE;
        margin-bottom: 10px;
        font-size: 14px;
        color: #1E40AF;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    path = os.path.join(BASE_DIR, "best_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_features():
    path = os.path.join(BASE_DIR, "columns.txt")
    with open(path, "r") as f:
        return [line.strip() for line in f]

@st.cache_data
def load_training_data():
    path = os.path.join(BASE_DIR, "cleaned_supply_chain.csv")
    df = pd.read_csv(path)
    df["order_date"]    = pd.to_datetime(df["order_date"],    errors="coerce")
    df["shipping_date"] = pd.to_datetime(df["shipping_date"], errors="coerce")
    return df

@st.cache_data
def load_predictions():
    path = os.path.join(BASE_DIR, "predictions.csv")
    return pd.read_csv(path)

@st.cache_data
def load_feature_importance():
    path = os.path.join(BASE_DIR, "feature_importance.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame({'Feature': ['N/A'], 'Importance': [1.0]})

def kpi_card(label, value, color_class, suffix=""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {color_class}">{value}{suffix}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# SHIPPING MODE → AUTO SCHEDULED DAYS
# ─────────────────────────────────────────────────────────────────────
SHIPPING_DAYS = {
    "Same Day"       : 0,
    "First Class"    : 2,
    "Second Class"   : 3,
    "Standard Class" : 5,
}


# ─────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["order date (DateOrders)", "Shipping Date (DateOrders)"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "order date (DateOrders)" in df.columns:
        df["order_weekday"]    = df["order date (DateOrders)"].dt.weekday
        df["order_month"]      = df["order date (DateOrders)"].dt.month
        df["order_day"]        = df["order date (DateOrders)"].dt.day
        df["order_is_weekend"] = df["order_weekday"].isin([5, 6]).astype(int)

    if "order date (DateOrders)" in df.columns and "Shipping Date (DateOrders)" in df.columns:
        df["actual_workdays"] = (
            df["Shipping Date (DateOrders)"] - df["order date (DateOrders)"]
        ).dt.days

    if "actual_workdays" in df.columns and "Days for shipment (scheduled)" in df.columns:
        df["delta_days"]  = df["actual_workdays"] - df["Days for shipment (scheduled)"]
        df["delta_ratio"] = df["delta_days"] / (df["Days for shipment (scheduled)"] + 1)

    target_col       = "Late_delivery_risk"
    prior_group_cols = ["Order Region", "Shipping Mode", "Customer Segment",
                        "Category Name", "Order Country", "Market"]
    for col in prior_group_cols:
        feat_name = f"{col}_prior_late_rate"
        if col in train_df.columns and col in df.columns:
            prior = train_df.groupby(col)[target_col].mean().rename(feat_name)
            df    = df.merge(prior, on=col, how="left")

    cumulative_cols = ["Customer Id", "Customer Country", "Order Country"]
    for col in cumulative_cols:
        feat_name = f"{col}_cum_late_rate"
        if col in train_df.columns and col in df.columns:
            cum = train_df.groupby(col)[target_col].mean().rename(feat_name)
            df  = df.merge(cum, on=col, how="left")

    if "Order Item Total" in df.columns:
        q75 = train_df["Order Item Total"].quantile(0.75) if "Order Item Total" in train_df.columns else 500
        df["is_high_value"] = (df["Order Item Total"] > q75).astype(int)
    if "Order Item Discount Rate" in df.columns:
        df["has_discount"] = (df["Order Item Discount Rate"] > 0).astype(int)

    return df


def label_encode_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        le      = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚚 Supply Chain")
    st.markdown("### Risk Analytics")
    st.markdown("---")

    page = st.radio(
        "Navigate to",
        ["🎯  Prediction", "📊  Business Insights", "💡  Business Recommendations"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown("- Algorithm: CatBoost / LightGBM")
    st.markdown("- Accuracy: **99.45%**")
    st.markdown("- AUC: **0.9999**")
    st.markdown("- Features: **44**")
    st.markdown("- Orders: **180,519**")
    st.markdown("- Dataset: DataCo Supply Chain")
    st.markdown("---")
    st.caption("Supply Chain Risk Analytics v2.0")


# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════
if page == "🎯  Prediction":

    st.title("🎯 Late Delivery Risk Predictor")
    st.markdown("Fill in the order details. Shipping date and scheduled days are **auto-calculated** from shipping mode.")
    st.markdown("---")

    try:
        model    = load_model()
        features = load_features()
        train_df = load_training_data()
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.stop()

    st.markdown('<div class="section-header">📦 Order Details</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        order_date = st.date_input(
            "Order Date",
            value = pd.to_datetime("2016-03-15"),
            help  = "Date the customer placed the order"
        )
        shipping_date_input = st.date_input(
            "Shipping Date",
            value = pd.to_datetime("2016-03-18"),
            help  = "Date the product is dispatched from warehouse"
        )
        scheduled_days = st.slider(
            "Scheduled Shipping Days", 0, 7, 3,
            help="Number of days scheduled for delivery"
        )
        shipping_mode = st.selectbox(
            "Shipping Mode",
            ["Standard Class", "Second Class", "First Class", "Same Day"],
        )

        # Smart warning
        actual_gap    = (pd.to_datetime(shipping_date_input) - pd.to_datetime(order_date)).days
        expected_days = SHIPPING_DAYS[shipping_mode]
        if actual_gap < 0:
            st.error("Shipping date cannot be before order date.")
        elif shipping_mode == "Same Day" and actual_gap > 0:
            st.warning(f"Same Day shipping selected but gap is {actual_gap} day(s). Expected 0.")
        elif actual_gap != expected_days:
            st.info(f"{shipping_mode} typically takes {expected_days} day(s) but entered gap is {actual_gap} day(s).")

        shipping_date = pd.to_datetime(shipping_date_input)

        order_region = st.selectbox("Order Region",
            sorted(train_df["Order Region"].dropna().unique().tolist()))

    with col2:
        market = st.selectbox("Market",
            sorted(train_df["Market"].dropna().unique().tolist()))
        order_country = st.selectbox("Order Country",
            sorted(train_df["Order Country"].dropna().unique().tolist()))
        order_city = st.selectbox("Order City",
            sorted(train_df["Order City"].dropna().unique().tolist()))
        customer_segment = st.selectbox("Customer Segment",
            ["Consumer", "Corporate", "Home Office"])

    with col3:
        category_name = st.selectbox("Category Name",
            sorted(train_df["Category Name"].dropna().unique().tolist()))
        order_item_total = st.number_input(
            "Order Item Total ($)", 10.0, 10000.0, 250.0, 50.0)
        discount_rate = st.slider(
            "Discount Rate", 0.0, 1.0, 0.0, 0.05,
            help="0 = no discount, 1 = 100% discount")
        customer_id = st.number_input(
            "Customer ID (0 if new customer)", 0, 99999, 0, 1)

    st.markdown("")
    predict_btn = st.button("🔍  Predict Delivery Risk", type="primary", use_container_width=True)

    if predict_btn:

        input_dict = {
            "order date (DateOrders)"       : pd.to_datetime(order_date),
            "Shipping Date (DateOrders)"    : shipping_date,
            "Days for shipment (scheduled)" : scheduled_days,
            "Shipping Mode"                 : shipping_mode,
            "Order Region"                  : order_region,
            "Market"                        : market,
            "Order Country"                 : order_country,
            "Order City"                    : order_city,
            "Customer Segment"              : customer_segment,
            "Category Name"                 : category_name,
            "Order Item Total"              : order_item_total,
            "Order Item Discount Rate"      : discount_rate,
            "Customer Id"                   : customer_id,
            "Customer Country"              : order_country,
            "Order State"                   : "Unknown",
            "Department Name"               : "Unknown",
            "Order Item Product Price"      : order_item_total,
            "Order Item Profit Ratio"       : 0.1,
            "Order Profit Per Order"        : order_item_total * 0.1,
            "Sales"                         : order_item_total,
            "Order Item Quantity"           : 1,
            "Order Item Discount"           : order_item_total * discount_rate,
            "Benefit per order"             : order_item_total * 0.1,
        }

        input_df = pd.DataFrame([input_dict])
        input_df = engineer_features(input_df, train_df)
        input_df = label_encode_input(input_df)
        input_df = input_df.fillna(input_df.median(numeric_only=True))
        input_df = input_df.fillna(0)

        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0

        X = input_df[features]

        probability = model.predict_proba(X)[0][1]
        prediction  = model.predict(X)[0]
        prob_pct    = round(probability * 100, 1)

        if probability >= 0.7:
            tier       = "High Risk"
            result_cls = "result-high"
            icon       = "🔴"
            action     = "Upgrade shipping mode or reroute immediately before dispatch."
            suggestion = "Switch to First Class or Standard Class which have better on-time rates."
        elif probability >= 0.4:
            tier       = "Medium Risk"
            result_cls = "result-medium"
            icon       = "🟡"
            action     = "Monitor closely and notify customer proactively."
            suggestion = "Add 1-2 buffer days or assign to priority fulfillment queue."
        else:
            tier       = "Low Risk"
            result_cls = "result-low"
            icon       = "🟢"
            action     = "No immediate action required. Standard processing."
            suggestion = "Order is on track. Continue with normal fulfillment process."

        st.markdown("---")
        st.markdown("### 📋 Prediction Result")

        r1, r2, r3 = st.columns(3)
        with r1:
            kpi_card("Late Probability", f"{prob_pct}%",
                     "kpi-red" if probability >= 0.7 else
                     "kpi-orange" if probability >= 0.4 else "kpi-green")
        with r2:
            kpi_card("Prediction",
                     "Will Be Late" if prediction == 1 else "On Time",
                     "kpi-red" if prediction == 1 else "kpi-green")
        with r3:
            kpi_card("Risk Tier", tier,
                     "kpi-red" if probability >= 0.7 else
                     "kpi-orange" if probability >= 0.4 else "kpi-green")

        st.markdown(f"""
        <div class="result-box {result_cls}">
            <h3 style="margin:0 0 8px 0">{icon} {tier} — {prob_pct}% probability of late delivery</h3>
            <p style="margin:0 0 6px 0; color:#475569; font-size:15px;">
                <strong>⚡ Immediate Action:</strong> {action}
            </p>
            <p style="margin:0; color:#475569; font-size:15px;">
                <strong>💡 Suggestion:</strong> {suggestion}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        fig = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = prob_pct,
            title = {"text": "Late Delivery Probability (%)", "font": {"size": 16}},
            delta = {"reference": 50, "valueformat": ".1f"},
            gauge = {
                "axis"  : {"range": [0, 100], "tickwidth": 1},
                "bar"   : {"color": "#DC2626" if probability >= 0.7 else
                                    "#F97316" if probability >= 0.4 else "#16A34A"},
                "steps" : [
                    {"range": [0,  40], "color": "#DCFCE7"},
                    {"range": [40, 70], "color": "#FEF3C7"},
                    {"range": [70,100], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line"      : {"color": "#1E293B", "width": 3},
                    "thickness" : 0.75,
                    "value"     : prob_pct
                }
            }
        ))
        fig.update_layout(
            height=280, margin=dict(t=40, b=0, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#1E293B"
        )
        st.plotly_chart(fig, use_container_width=True)

        # SHAP
        st.markdown("---")
        st.markdown("### 🔍 Why this prediction?")
        st.markdown("Top factors influencing this specific order's risk score:")

        try:
            with st.spinner("Computing feature contributions..."):
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X)

                if isinstance(shap_vals, list):
                    sv = shap_vals[1]
                else:
                    sv = shap_vals

                shap_df = pd.DataFrame({
                    "Feature"     : features,
                    "SHAP Value"  : sv[0],
                    "Input Value" : X.iloc[0].values
                })
                shap_df["Abs"] = shap_df["SHAP Value"].abs()
                shap_df = shap_df.sort_values("Abs", ascending=False).head(15)
                shap_df = shap_df.sort_values("SHAP Value")
                shap_df["Direction"] = shap_df["SHAP Value"].apply(
                    lambda v: "→ Late" if v > 0 else "→ On Time")
                shap_df["Color"] = shap_df["SHAP Value"].apply(
                    lambda v: "#DC2626" if v > 0 else "#16A34A")
                shap_df["Label"] = shap_df.apply(
                    lambda r: f"{r['Feature']}  (={round(r['Input Value'], 2)})", axis=1)

                fig_shap = go.Figure()
                fig_shap.add_trace(go.Bar(
                    x=shap_df["SHAP Value"], y=shap_df["Label"],
                    orientation="h", marker_color=shap_df["Color"],
                    text=shap_df["Direction"], textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Impact: %{x:.4f}<extra></extra>"
                ))
                fig_shap.add_vline(x=0, line_color="#1E293B", line_width=1.5)
                fig_shap.update_layout(
                    height=480,
                    title="Feature Contributions — Red pushes Late, Green pushes On Time",
                    xaxis_title="Impact on Prediction (SHAP Value)",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=160, t=50, b=10),
                    yaxis=dict(tickfont=dict(size=11))
                )
                st.plotly_chart(fig_shap, use_container_width=True)

                top_late   = shap_df[shap_df["SHAP Value"] > 0].tail(3)[::-1]
                top_ontime = shap_df[shap_df["SHAP Value"] < 0].head(3)

                col_why1, col_why2 = st.columns(2)
                with col_why1:
                    st.markdown("**🔴 Top reasons pushing toward Late:**")
                    if len(top_late) == 0:
                        st.success("No strong late-pushing factors!")
                    for _, row in top_late.iterrows():
                        st.error(f"**{row['Feature']}** = {round(row['Input Value'], 2)} "
                                 f"(impact: +{row['SHAP Value']:.4f})")

                with col_why2:
                    st.markdown("**🟢 Top reasons pushing toward On Time:**")
                    if len(top_ontime) == 0:
                        st.warning("No strong on-time factors!")
                    for _, row in top_ontime.iterrows():
                        st.success(f"**{row['Feature']}** = {round(row['Input Value'], 2)} "
                                   f"(impact: {row['SHAP Value']:.4f})")

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

        with st.expander("📄 View Input Summary"):
            summary = pd.DataFrame({
                "Field": ["Order Date", "Shipping Date", "Scheduled Days",
                          "Shipping Mode", "Order Region", "Market",
                          "Order Country", "Order City", "Customer Segment",
                          "Category Name", "Order Item Total", "Discount Rate"],
                "Value": [str(order_date), shipping_date.strftime('%Y-%m-%d'),
                          f"{scheduled_days} days", shipping_mode,
                          order_region, market, order_country, order_city,
                          customer_segment, category_name,
                          f"${order_item_total:,.2f}", f"{discount_rate:.0%}"]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════
elif page == "📊  Business Insights":

    st.title("📊 Business Insights Dashboard")
    st.markdown("ML-powered insights from **180,519 real DataCo supply chain orders** (2015–2018).")
    st.markdown("---")

    try:
        pred_df  = load_predictions()
        feat_df  = load_feature_importance()
        train_df = load_training_data()
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.stop()

    # Date filter
    st.markdown('<div class="section-header">📅 Filter by Date Range</div>',
                unsafe_allow_html=True)
    min_date = train_df["order_date"].min()
    max_date = train_df["order_date"].max()

    d1, d2 = st.columns(2)
    with d1:
        start_date = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
    with d2:
        end_date = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date)

    mask     = ((train_df["order_date"] >= pd.to_datetime(start_date)) &
                (train_df["order_date"] <= pd.to_datetime(end_date)))
    filt_df  = train_df[mask].copy()
    st.markdown(f"Showing **{len(filt_df):,}** orders from **{start_date}** to **{end_date}**")
    st.markdown("")

    # KPIs
    st.markdown('<div class="section-header">📈 Key Performance Indicators</div>',
                unsafe_allow_html=True)

    total_orders  = len(filt_df)
    late_rate     = round(filt_df["is_late"].mean() * 100, 1)
    avg_delay     = round(filt_df["delivery_delay"].mean(), 2)
    total_sales   = round(filt_df["Sales"].sum(), 0)
    total_profit  = round(filt_df["Order Profit Per Order"].sum(), 0)
    profit_margin = round(filt_df["profit_margin"].mean() * 100, 1)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1: kpi_card("Total Orders",  f"{total_orders:,}",    "kpi-blue")
    with k2: kpi_card("Late Rate",     f"{late_rate}%",        "kpi-red")
    with k3: kpi_card("Avg Delay",     f"{avg_delay}d",        "kpi-orange")
    with k4: kpi_card("Total Sales",   f"${total_sales:,.0f}", "kpi-green")
    with k5: kpi_card("Total Profit",  f"${total_profit:,.0f}","kpi-purple")
    with k6: kpi_card("Profit Margin", f"{profit_margin}%",    "kpi-blue")

    st.markdown("")

    # Trend chart
    st.markdown('<div class="section-header">📈 Late Delivery Trend Over Time</div>',
                unsafe_allow_html=True)

    trend_df = filt_df.copy()
    trend_df["month"] = trend_df["order_date"].dt.to_period("M").astype(str)
    trend_monthly = (trend_df.groupby("month")
                    .agg(late_rate=("is_late","mean"),
                         total_orders=("is_late","count"),
                         avg_delay=("delivery_delay","mean"))
                    .reset_index())
    trend_monthly["late_rate_pct"] = (trend_monthly["late_rate"] * 100).round(1)

    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(
        go.Scatter(x=trend_monthly["month"], y=trend_monthly["late_rate_pct"],
                   name="Late Rate %", mode="lines+markers",
                   line=dict(color="#DC2626", width=2), marker=dict(size=5)),
        secondary_y=False
    )
    fig_trend.add_trace(
        go.Bar(x=trend_monthly["month"], y=trend_monthly["total_orders"],
               name="Total Orders", marker_color="#BFDBFE", opacity=0.5),
        secondary_y=True
    )
    fig_trend.update_layout(
        height=350, title="Monthly Late Delivery Rate vs Order Volume",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(tickangle=45)
    )
    fig_trend.update_yaxes(title_text="Late Rate (%)", secondary_y=False)
    fig_trend.update_yaxes(title_text="Total Orders",  secondary_y=True)
    st.plotly_chart(fig_trend, use_container_width=True)

    # Charts row 1
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<div class="section-header">🗺️ Late Rate by Region</div>',
                    unsafe_allow_html=True)
        region_df = (filt_df.groupby("Order Region")
                    .agg(late_rate=("is_late","mean"), count=("is_late","count"))
                    .reset_index())
        region_df["late_rate_pct"] = (region_df["late_rate"] * 100).round(1)
        region_df = region_df.sort_values("late_rate_pct", ascending=False)

        fig_r = px.bar(region_df, x="Order Region", y="late_rate_pct",
                       title="Late Delivery Rate by Region (%)",
                       color="late_rate_pct",
                       color_continuous_scale=[[0,"#16A34A"],[0.5,"#F97316"],[1,"#DC2626"]],
                       text="late_rate_pct")
        fig_r.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_r.update_layout(height=380, showlegend=False, coloraxis_showscale=False,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=10, r=10, t=40, b=80),
                            xaxis=dict(tickangle=30))
        st.plotly_chart(fig_r, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">🚚 Late Rate by Shipping Mode</div>',
                    unsafe_allow_html=True)
        ship_df = (filt_df.groupby("Shipping Mode")
                  .agg(late_rate=("is_late","mean"), count=("is_late","count"))
                  .reset_index())
        ship_df["late_rate_pct"] = (ship_df["late_rate"] * 100).round(1)
        ship_df = ship_df.sort_values("late_rate_pct", ascending=False)

        fig_s = px.bar(ship_df, x="Shipping Mode", y="late_rate_pct",
                       title="Late Delivery Rate by Shipping Mode (%)",
                       color="late_rate_pct",
                       color_continuous_scale=[[0,"#16A34A"],[0.5,"#F97316"],[1,"#DC2626"]],
                       text="late_rate_pct")
        fig_s.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_s.update_layout(height=380, showlegend=False, coloraxis_showscale=False,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_s, use_container_width=True)

    # Charts row 2
    c3, c4 = st.columns(2, gap="large")

    with c3:
        st.markdown('<div class="section-header">🌍 Late Rate by Market</div>',
                    unsafe_allow_html=True)
        market_df = (filt_df.groupby("Market")
                    .agg(late_rate=("is_late","mean"), count=("is_late","count"))
                    .reset_index())
        market_df["late_rate_pct"] = (market_df["late_rate"] * 100).round(1)
        market_df = market_df.sort_values("late_rate_pct", ascending=True)

        fig_m = px.bar(market_df, x="late_rate_pct", y="Market", orientation="h",
                       title="Late Delivery Rate by Market (%)",
                       color="late_rate_pct",
                       color_continuous_scale=[[0,"#16A34A"],[0.5,"#F97316"],[1,"#DC2626"]],
                       text="late_rate_pct")
        fig_m.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_m.update_layout(height=380, showlegend=False, coloraxis_showscale=False,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=10, r=60, t=40, b=10))
        st.plotly_chart(fig_m, use_container_width=True)

    with c4:
        st.markdown('<div class="section-header">📦 Revenue at Risk by Category</div>',
                    unsafe_allow_html=True)
        cat_df = (filt_df[filt_df["is_late"] == 1]
                 .groupby("Category Name")
                 .agg(revenue_at_risk=("Sales","sum"), late_orders=("is_late","count"))
                 .reset_index())
        cat_df = cat_df.sort_values("revenue_at_risk", ascending=True).tail(10)

        fig_c = px.bar(cat_df, x="revenue_at_risk", y="Category Name", orientation="h",
                       title="Revenue at Risk by Category (Late Orders Only)",
                       color="revenue_at_risk",
                       color_continuous_scale=[[0,"#FEF3C7"],[0.5,"#F97316"],[1,"#DC2626"]],
                       text="revenue_at_risk")
        fig_c.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_c.update_layout(height=380, showlegend=False, coloraxis_showscale=False,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=10, r=80, t=40, b=10))
        st.plotly_chart(fig_c, use_container_width=True)

    # Feature importance
    st.markdown('<div class="section-header">🏆 Top Factors Driving Late Deliveries</div>',
                unsafe_allow_html=True)

    fi_top = feat_df.head(12).sort_values("Importance").copy()
    fi_top["Score"] = (fi_top["Importance"] / fi_top["Importance"].max() * 100).round(1)

    fig_fi = px.bar(fi_top, x="Score", y="Feature", orientation="h",
                    title="Top 12 Features — Model Importance Score",
                    color="Score",
                    color_continuous_scale=[[0,"#2563EB"],[0.4,"#F97316"],[1,"#DC2626"]],
                    text="Score")
    fig_fi.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_fi.update_layout(height=420, showlegend=False, coloraxis_showscale=False,
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         margin=dict(l=10, r=40, t=40, b=10),
                         yaxis=dict(tickfont=dict(size=11)))
    st.plotly_chart(fig_fi, use_container_width=True)

    # Model performance
    st.markdown('<div class="section-header">🤖 Model Performance Summary</div>',
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1: kpi_card("Model Accuracy", "99.45%", "kpi-green")
    with m2: kpi_card("ROC-AUC Score",  "0.9999", "kpi-green")
    with m3: kpi_card("F1 Score",       "0.99",   "kpi-green")
    with m4: kpi_card("Features Used",  "44",     "kpi-blue")

    st.markdown("")
    st.info("""
    **About the Model:** Trained on 180,519 real DataCo supply chain orders using an ensemble of
    CatBoost, LightGBM, HistGradientBoosting and RandomForest. Best model selected by F1 score.
    Key features include historical prior late rates per region, customer cumulative late history,
    date-based features and delta between actual vs scheduled shipping days.
    """)


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — BUSINESS RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════
elif page == "💡  Business Recommendations":

    st.title("💡 Business Recommendations")
    st.markdown("Data-driven recommendations generated automatically from **180,519 supply chain orders**.")
    st.markdown("---")

    try:
        train_df = load_training_data()
        pred_df  = load_predictions()
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.stop()

    # Compute stats
    late_rate      = round(train_df["is_late"].mean() * 100, 1)
    total_sales    = train_df["Sales"].sum()
    late_revenue   = train_df[train_df["is_late"] == 1]["Sales"].sum()
    late_rev_pct   = round(late_revenue / total_sales * 100, 1)
    avg_delay      = round(train_df["delivery_delay"].mean(), 1)
    worst_region   = train_df.groupby("Order Region")["is_late"].mean().idxmax()
    worst_mode     = train_df.groupby("Shipping Mode")["is_late"].mean().idxmax()
    best_mode      = train_df.groupby("Shipping Mode")["is_late"].mean().idxmin()
    worst_market   = train_df.groupby("Market")["is_late"].mean().idxmax()
    mode_late      = train_df.groupby("Shipping Mode")["is_late"].mean() * 100
    region_late    = train_df.groupby("Order Region")["is_late"].mean() * 100
    seg_late       = train_df.groupby("Customer Segment")["is_late"].mean() * 100
    cat_rev        = (train_df[train_df["is_late"] == 1]
                     .groupby("Category Name")["Sales"].sum()
                     .sort_values(ascending=False))

    worst_mode_rate   = round(mode_late[worst_mode], 1)
    best_mode_rate    = round(mode_late[best_mode], 1)
    worst_region_rate = round(region_late[worst_region], 1)
    seg_worst         = seg_late.idxmax()
    seg_worst_rate    = round(seg_late[seg_worst], 1)
    seg_best          = seg_late.idxmin()
    seg_best_rate     = round(seg_late[seg_best], 1)
    top_risk_cat      = cat_rev.index[0] if len(cat_rev) > 0 else "Unknown"
    top_risk_cat_rev  = round(cat_rev.iloc[0], 0) if len(cat_rev) > 0 else 0
    discount_late     = round(train_df[train_df["Order Item Discount Rate"] > 0]["is_late"].mean() * 100, 1)
    no_discount_late  = round(train_df[train_df["Order Item Discount Rate"] == 0]["is_late"].mean() * 100, 1)

    # Executive Summary
    st.markdown('<div class="section-header">📋 Executive Summary</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-box result-high">
        <h3 style="margin:0 0 12px 0">🔴 Current State — Action Required</h3>
        <p style="margin:0 0 6px 0; font-size:15px; color:#475569;">
            • <strong>{late_rate}%</strong> of all orders are at late delivery risk across 180,519 orders
        </p>
        <p style="margin:0 0 6px 0; font-size:15px; color:#475569;">
            • <strong>${late_revenue:,.0f}</strong> ({late_rev_pct}% of total revenue) is at risk from delayed orders
        </p>
        <p style="margin:0 0 6px 0; font-size:15px; color:#475569;">
            • Average delivery delay of <strong>{avg_delay} days</strong> across all late orders
        </p>
        <p style="margin:0; font-size:15px; color:#475569;">
            • Worst performing region: <strong>{worst_region}</strong> |
              Worst shipping mode: <strong>{worst_mode}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # High Priority
    st.markdown('<div class="section-header">🔴 High Priority Recommendations</div>',
                unsafe_allow_html=True)

    recs_high = [
        {
            "title": f"🚚 Fix {worst_mode} Shipping — {worst_mode_rate}% late rate",
            "body" : (f"{worst_mode} shipping has the highest late delivery rate at {worst_mode_rate}%. "
                      f"Despite being a premium option it consistently underperforms {best_mode} "
                      f"({best_mode_rate}% late rate). Investigate fulfillment bottlenecks, improve "
                      f"SLA compliance, or reprice this tier to reflect actual performance.")
        },
        {
            "title": f"🌍 Urgent: Address {worst_region} Region — {worst_region_rate}% late rate",
            "body" : (f"{worst_region} shows the highest late delivery rate at {worst_region_rate}%. "
                      f"Immediate actions: partner with local last-mile carriers, build regional buffer "
                      f"stock, and add 1-2 day buffer to scheduled shipping times for this region.")
        },
        {
            "title": f"💰 Protect Revenue in {top_risk_cat} — ${top_risk_cat_rev:,.0f} at risk",
            "body" : (f"The {top_risk_cat} category has the highest revenue exposure from late deliveries. "
                      f"High-value items here should be routed through fast-track lanes with mandatory "
                      f"pre-dispatch quality checks and real-time tracking.")
        },
    ]

    for rec in recs_high:
        st.markdown(f"""
        <div class="rec-card rec-high">
            <div class="rec-title">{rec['title']}</div>
            <div class="rec-body">{rec['body']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Medium Priority
    st.markdown('<div class="section-header">🟡 Medium Priority Recommendations</div>',
                unsafe_allow_html=True)

    recs_medium = [
        {
            "title": f"👥 Customer Segment Strategy — {seg_worst} at {seg_worst_rate}% late rate",
            "body" : (f"The {seg_worst} segment has a {seg_worst_rate}% late rate vs {seg_best_rate}% "
                      f"for {seg_best}. Introduce segment-specific SLA agreements and set up automated "
                      f"proactive notifications when {seg_worst} orders enter high-risk zones.")
        },
        {
            "title": f"🏷️ Discount Impact — Discounted orders are {round(discount_late - no_discount_late, 1)}% more likely to be late",
            "body" : (f"Orders with discounts show a {discount_late}% late rate vs {no_discount_late}% "
                      f"for full-price orders. Cap discount campaigns during peak logistics stress periods "
                      f"and avoid stacking discounts with premium shipping modes.")
        },
        {
            "title": "📅 Seasonal Capacity Planning",
            "body" : ("Late delivery rates show monthly variation across 2015-2018. Pre-position inventory "
                      "and increase warehouse staffing during historically high-risk months. Use the ML "
                      "model to forecast upcoming high-risk periods and adjust capacity proactively.")
        },
    ]

    for rec in recs_medium:
        st.markdown(f"""
        <div class="rec-card rec-medium">
            <div class="rec-title">{rec['title']}</div>
            <div class="rec-body">{rec['body']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Strategic
    st.markdown('<div class="section-header">🟢 Strategic Recommendations</div>',
                unsafe_allow_html=True)

    recs_low = [
        {
            "title": "🤖 Deploy ML Model for Real-Time Risk Flagging",
            "body" : ("Integrate the prediction API into the order management system so every new order "
                      "is automatically scored at placement. Orders above 70% risk threshold should "
                      "trigger automatic escalation workflows without human intervention.")
        },
        {
            "title": "📊 Build Carrier Performance Scorecards",
            "body" : ("Use historical delivery data to score each carrier by region. Route orders "
                      "dynamically to the best-performing carrier for each origin-destination pair "
                      "rather than using fixed carrier assignments.")
        },
        {
            "title": "🔄 Implement Continuous Model Retraining",
            "body" : ("Retrain the ML model quarterly with fresh data to capture evolving patterns. "
                      "Set up automated drift detection to alert when accuracy drops below 95%. "
                      "This keeps predictions reliable as supply chain conditions change.")
        },
    ]

    for rec in recs_low:
        st.markdown(f"""
        <div class="rec-card rec-low">
            <div class="rec-title">{rec['title']}</div>
            <div class="rec-body">{rec['body']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Potential savings
    st.markdown('<div class="section-header">💰 Potential Business Impact</div>',
                unsafe_allow_html=True)

    potential_10 = round(late_revenue * 0.10, 0)
    potential_25 = round(late_revenue * 0.25, 0)
    potential_50 = round(late_revenue * 0.50, 0)

    p1, p2, p3 = st.columns(3)
    with p1: kpi_card("10% Late Rate Reduction", f"${potential_10:,.0f}", "kpi-green", " saved")
    with p2: kpi_card("25% Late Rate Reduction", f"${potential_25:,.0f}", "kpi-orange", " saved")
    with p3: kpi_card("50% Late Rate Reduction", f"${potential_50:,.0f}", "kpi-red",    " saved")

    st.markdown("")
    st.success(f"""
    **💡 Key Takeaway:** Reducing the late delivery rate by just 25% through the above recommendations
    could recover **${potential_25:,.0f}** in at-risk revenue. The ML model provides the foundation
    for proactive intervention — turning reactive logistics into a predictive operation.
    """)
