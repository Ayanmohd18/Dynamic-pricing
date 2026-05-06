import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# --- Configuration & Styling ---
st.set_page_config(page_title="Dynamic Pricing Dashboard", layout="wide")

# Custom Sleek UI / UX CSS (Uber Aesthetic)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
        letter-spacing: -0.01em;
    }
    
    /* Metrics / Cards */
    [data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        padding: 24px;
        border-radius: 0px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
    }
    [data-testid="stMetricLabel"] {
        font-weight: 500;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.6;
    }
    [data-testid="stMetricValue"] {
        font-weight: 600;
        font-size: 2.2rem;
    }

    /* Buttons */
    .stButton>button {
        background-color: var(--text-color);
        color: var(--background-color);
        border: none;
        border-radius: 0px;
        font-weight: 600;
        padding: 12px 24px;
        transition: opacity 0.2s ease, transform 0.1s ease;
    }
    .stButton>button:hover, .stButton>button:focus {
        opacity: 0.8;
        background-color: var(--text-color);
        color: var(--background-color);
        border: none;
    }
    .stButton>button:active {
        transform: scale(0.98);
    }

    /* Inputs */
    .stTextInput>div>div>input, .stSelectbox>div>div>div {
        border-radius: 0px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* Table */
    [data-testid="stTable"] {
        border-radius: 0px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        font-weight: 600 !important;
    }
    hr {
        border-bottom-color: rgba(128, 128, 128, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Data & Model Loading ---
@st.cache_resource
def load_assets():
    # Robust path resolution relative to the script location
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, 'models', 'xgboost_pricing_v1.pkl')
    data_path = os.path.join(project_root, 'data', 'processed', 'features.parquet')
    
    if not os.path.exists(model_path):
        # Fallback for different structures if needed
        st.error(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    df = pd.read_parquet(data_path)
    
    # Precompute category stats for Confidence Intervals and Dropdowns
    cat_stats = df.groupby('category').agg({'optimal_price': ['std', 'mean']}).reset_index()
    cat_stats.columns = ['category', 'std', 'mean']
    categories = sorted(df['category'].unique())
    
    return model, df, cat_stats, categories

model, df, cat_stats, categories = load_assets()

# --- Sidebar Controls ---
st.sidebar.header("🛠️ Model Controls")
sku_id = st.sidebar.text_input("SKU ID", value="olist_prod_999")
selected_cat = st.sidebar.selectbox("Product Category", options=categories)
demand_score = st.sidebar.slider("Demand Score (7d)", 0, 1000, 150)
inventory_ratio = st.sidebar.slider("Inventory Ratio", 0.0, 1.0, 0.4)
comp_delta = st.sidebar.slider("Competitor Delta", -0.5, 0.5, 0.05)
is_flash_sale = st.sidebar.toggle("⚡ Is Flash Sale", value=False)

st.sidebar.header("🌧️ Real-Time Logistics (Surge)")
weather_condition = st.sidebar.selectbox("Weather Condition", options=["Clear", "Rain", "Heavy Rain/Snow", "Extreme"])
fleet_availability = st.sidebar.slider("Delivery Fleet Availability", 0.0, 1.0, 0.8)

# --- Helper Functions ---
def get_prediction(inputs):
    # Map inputs to feature vector (must match training feature order)
    feature_cols = [
        'price', 'freight_value', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'is_weekend', 'is_month_end', 'is_holiday', 'days_since_last_order',
        'demand_score_7d', 'demand_score_30d', 'demand_velocity',
        'inventory_ratio', 'price_percentile_in_category',
        'competitor_delta', 'review_elasticity'
    ]
    
    # Derived/Simulated features for the dashboard
    input_data = {
        'price': 100.0, # Placeholder base
        'freight_value': 15.0,
        'hour_sin': 0.5, 'hour_cos': 0.8,
        'day_sin': 0.7, 'day_cos': 0.7,
        'is_weekend': 0, 'is_month_end': 0, 'is_holiday': 0,
        'days_since_last_order': 2.0,
        'demand_score_7d': demand_score,
        'demand_score_30d': demand_score * 3,
        'demand_velocity': 0.1 if demand_score > 200 else -0.05,
        'inventory_ratio': inventory_ratio,
        'price_percentile_in_category': 0.5,
        'competitor_delta': comp_delta,
        'review_elasticity': 1.0
    }
    
    X_input = pd.DataFrame([input_data])[feature_cols]
    pred = model.predict(X_input)[0]
    return pred, X_input

# --- Main Dashboard Logic ---
ml_price, X_input = get_prediction(sku_id)

# 1. Multiplier Logic (Simulated for breakdown)
inv_mult = 1.10 if inventory_ratio < 0.2 else 1.0
dem_mult = 1.05 if demand_score > 500 else 1.0
flash_mult = 1.25 if is_flash_sale else 1.0

# Surge Pricing Logic (Uber/Lyft style)
weather_mult_map = {"Clear": 1.0, "Rain": 1.05, "Heavy Rain/Snow": 1.15, "Extreme": 1.30}
weather_mult = weather_mult_map[weather_condition]
fleet_mult = 1.0 + ((1.0 - fleet_availability) * 0.4) # Up to 40% surge if 0 fleet

final_price = ml_price * inv_mult * dem_mult * flash_mult * weather_mult * fleet_mult

# CI and Guardrails
cat_std = cat_stats[cat_stats['category'] == selected_cat]['std'].values[0]
p_floor = final_price * 0.85
p_ceiling = final_price * 1.5

# --- Main Panel ---
st.title("🎯 Dynamic Pricing Engine v1.0")

# 1. PRICE RECOMMENDATION
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Recommended Price", f"₹{final_price:.2f}", delta=f"{((final_price/100)-1)*100:.1f}% vs Avg")
with c2:
    st.metric("Confidence Interval (±1σ)", f"₹{final_price-cat_std:.1f} - ₹{final_price+cat_std:.1f}")
with c3:
    st.metric("Price Guardrails", f"F: ₹{p_floor:.1f} | C: ₹{p_ceiling:.1f}")

# 2. FEATURE IMPACT (SHAP)
st.divider()
st.subheader("🧬 Real-Time Feature Impact (SHAP)")

# Functional SHAP for stability
predict_fn = lambda x: model.predict(pd.DataFrame(x, columns=X_input.columns))
# For dashboard performance, we use a pre-calculated or very small masker
masker = shap.maskers.Independent(df[X_input.columns].sample(50), max_samples=50)
explainer = shap.Explainer(predict_fn, masker)
shap_values = explainer(X_input)

shap_df = pd.DataFrame({
    'Feature': X_input.columns,
    'Impact': shap_values.values[0]
}).sort_values('Impact', ascending=True)

fig_shap = px.bar(shap_df, x='Impact', y='Feature', orientation='h', 
             title="How inputs are shifting the price",
             color='Impact', color_continuous_scale='gray')
st.plotly_chart(fig_shap, use_container_width=True, theme="streamlit")

# 3. PRICE HISTORY
st.divider()
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 30-Day Category Trend")
    cat_df = df[df['category'] == selected_cat].tail(30)
    fig_hist = px.line(cat_df, x='order_purchase_timestamp', y='optimal_price',
                  title=f"Historical Price: {selected_cat}",
                  line_shape='spline', render_mode='svg')
    fig_hist.update_traces(line_color='gray', line_width=2)
    st.plotly_chart(fig_hist, use_container_width=True, theme="streamlit")

with col_right:
    # 4. ADJUSTMENT BREAKDOWN
    st.subheader("📋 Surge & Adjustment Breakdown")
    breakdown_data = {
        "Step": ["Base ML Price", "Inventory Mult.", "Demand Mult.", "Flash Sale", "Weather Surge", "Fleet Surge", "Final Price"],
        "Value": [f"₹{ml_price:.2f}", f"x{inv_mult}", f"x{dem_mult}", f"x{flash_mult}", f"x{weather_mult}", f"x{fleet_mult:.2f}", f"₹{final_price:.2f}"],
        "Impact": ["-", 
                   f"{'+' if inv_mult > 1 else ''}{int((inv_mult-1)*100)}%", 
                   f"{'+' if dem_mult > 1 else ''}{int((dem_mult-1)*100)}%", 
                   f"{'+' if flash_mult > 1 else ''}{int((flash_mult-1)*100)}%",
                   f"{'+' if weather_mult > 1 else ''}{int((weather_mult-1)*100)}%",
                   f"{'+' if fleet_mult > 1 else ''}{int((fleet_mult-1)*100)}%",
                   "Total Recommendation"]
    }
    st.table(pd.DataFrame(breakdown_data))

# --- Export ---
st.divider()
report_df = pd.DataFrame([{
    "timestamp": datetime.now(),
    "sku_id": sku_id,
    "category": selected_cat,
    "rec_price": final_price,
    "demand": demand_score,
    "inventory": inventory_ratio
}])

csv = report_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Pricing Report (CSV)",
    data=csv,
    file_name=f"price_report_{sku_id}.csv",
    mime="text/csv",
)
