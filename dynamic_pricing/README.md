# Production-Grade Dynamic Pricing Engine

This directory contains a complete, end-to-end implementation of a Dynamic Pricing Engine trained on two real-world UCI retail datasets (Online Retail I & II).

## Features
- **Ensemble ML**: XGBoost, LightGBM, CatBoost, and Random Forest stacked with Ridge regression.
- **Complex Feature Engineering**: Temporal, demand velocity, price elasticity (MR=MC), inventory proxies, and competitor simulation.
- **Real-Time API**: FastAPI with Redis caching for <10ms inference.
- **CEP Streaming**: Flash sale detection using spike multipliers and baseline demand rates.
- **Dashboard**: Advanced Streamlit UI with Live Recommender, Product Analytics, and Simulator.

## Quick Start
1. **Train the Models**:
   ```bash
   python run_pipeline.py
   ```
2. **Start the API**:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
3. **Launch the Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```
   *Note: You can also switch to this engine from the main dashboard at `d:/DP/dashboard/app.py`.*

## Project Structure
- `src/`: Core logic (ingest, clean, features, train, etc.)
- `api/`: FastAPI implementation
- `streaming/`: Flash sale detection logic
- `dashboard/`: Streamlit interface
- `data/`: Processed and feature-engineered data
- `models/`: Serialized artifacts and metrics
