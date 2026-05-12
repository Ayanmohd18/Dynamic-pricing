import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

# --- MASTER ENGINE SELECTOR ---
st.sidebar.title("🚀 Engine Selector")
selected_engine = st.sidebar.selectbox(
    "Active Implementation", 
    ["Olist E-commerce (Existing)", "Online Retail Production (New)"]
)

if selected_engine == "Online Retail Production (New)":
    # --- RENDER NEW PRODUCTION ENGINE ---
    # We dynamically execute the new engine code to avoid set_page_config conflicts
    # and keep files separate.
    new_engine_path = os.path.join(os.getcwd(), "dynamic_pricing", "dashboard", "app.py")
    if os.path.exists(new_engine_path):
        with open(new_engine_path, encoding="utf-8") as f:
            code = f.read()
            # Clean up code for dynamic execution within existing app
            code = code.replace("st.set_page_config", "# st.set_page_config")
            exec(code)
    else:
        st.error(f"New engine not found at {new_engine_path}")

else:
    # --- ORIGINAL OLIST IMPLEMENTATION (PRESERVED) ---
    st.title("🚀 AI-Powered Dynamic Pricing Engine")
    st.markdown("### Phase 2: Real-Time Inference & Competitor Monitoring (Olist)")

    # Sidebar for SKU and Category Context
    st.sidebar.header("Product Configuration")
    sku_id = st.sidebar.text_input("SKU ID", value="prod_12345")
    base_price = st.sidebar.number_input("Current Base Price (₹)", value=150.0)
    cost = st.sidebar.number_input("Unit Cost (₹)", value=100.0)
    msrp = st.sidebar.number_input("MSRP (₹)", value=250.0)

    # Main Dashboard Columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Real-Time Market Signals")
        
        st.markdown("#### Demand & Inventory")
        demand_7d = st.slider("7-Day Order Volume", 0, 500, 45)
        inventory = st.slider("Inventory Ratio (Stock/Demand)", 0.0, 1.0, 0.4)
        
        st.markdown("#### Competitor Landscape")
        comp_avg = st.number_input("Competitor Avg Price (₹)", value=145.0)
        comp_delta = (base_price - comp_avg) / comp_avg
        
        st.markdown("#### Temporal Signals")
        is_weekend = st.checkbox("Weekend Peak")
        is_holiday = st.checkbox("Festive Holiday")

    with col2:
        st.subheader("🎯 Recommended Pricing")
        
        payload = {
            "sku_id": sku_id,
            "price": base_price,
            "freight_value": 20.0,
            "hour_sin": 0.5, "hour_cos": 0.8,
            "day_sin": 0.7, "day_cos": 0.7,
            "is_weekend": 1 if is_weekend else 0,
            "is_month_end": 0,
            "is_holiday": 1 if is_holiday else 0,
            "days_since_last_order": 1.2,
            "demand_score_7d": float(demand_7d),
            "demand_score_30d": float(demand_7d * 4),
            "demand_velocity": 0.05 if demand_7d > 20 else -0.05,
            "inventory_ratio": inventory,
            "price_percentile_in_category": 0.6,
            "competitor_delta": comp_delta,
            "review_elasticity": 1.0,
            "competitor_avg": comp_avg,
            "cost": cost,
            "msrp": msrp
        }

        try:
            response = requests.post("http://localhost:8000/api/v1/price", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Final Price", f"₹{data['final_price']:.2f}", delta=f"{data['final_price'] - base_price:.2f}")
                m2.metric("ML Recommendation", f"₹{data['ml_price']:.2f}")
                m3.metric("Competitor Avg", f"₹{comp_avg:.2f}")
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = data['final_price'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Optimal Price Position"},
                    delta = {'reference': base_price},
                    gauge = {
                        'axis': {'range': [cost, msrp*1.2]},
                        'steps': [
                            {'range': [cost, cost*1.05], 'color': "red"},
                            {'range': [cost*1.05, msrp], 'color': "lightgreen"},
                            {'range': [msrp, msrp*1.2], 'color': "orange"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': data['final_price']
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                if data['applied_guardrail']:
                    st.warning("⚠️ Warning: Price was adjusted by safety guardrails (Margin/MSRP checks).")
            else:
                st.error("Olist API connection failed. Start the legacy server or switch to Online Retail.")
                
        except Exception as e:
            st.info("💡 Start the FastAPI server (src/api/main.py) to see Olist price updates.")

    st.divider()
    st.markdown("#### Logic Trace (Olist Phase 2)")
    st.code("""
    1. Prediction: XGBoost computes ML_Base
    2. Blending: 0.7 * ML_Base + 0.3 * Competitor_Avg
    3. Guardrail: Clamp within [Cost * 1.05, MSRP * 1.50]
    """)
