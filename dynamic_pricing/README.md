# ⚙️ dynamic_pricing — Engine Package

This directory contains the **core implementation** of the Dynamic Pricing Engine: the full ML pipeline, REST API, streaming layer, and dashboard. See the [root README](../README.md) for system-level documentation.

---

## 📂 Package Layout

```
dynamic_pricing/
├── config.py              ← Central config: paths, thresholds, hyperparameters
├── run_pipeline.py        ← Single-command end-to-end runner
├── requirements.txt       ← All Python dependencies
├── Dockerfile
├── docker-compose.yml     ← Redis + API + Dashboard stack
│
├── src/                   ← ML pipeline (run in order)
│   ├── ingest.py          ← Load & merge UCI Online Retail I & II
│   ├── clean.py           ← Filter invalid rows, standardise schema
│   ├── features.py        ← Feature engineering (~50 features)
│   ├── pricing_target.py  ← Target variable construction
│   ├── train.py           ← Train XGB / LGBM / CatBoost / RF + Ridge stack
│   ├── evaluate.py        ← Evaluation metrics (MAE, R², MAPE)
│   ├── predict.py         ← Inference wrapper (used by API)
│   └── simulator.py       ← Scenario simulator
│
├── api/
│   ├── main.py            ← FastAPI: /price/recommend, /price/bulk, /health
│   └── schemas.py         ← Pydantic v2 request/response models
│
├── streaming/
│   ├── flash_sale_detector.py  ← 5-min sliding-window CEP spike detector
│   └── producer.py             ← Kafka order-event producer
│
└── dashboard/
    └── app.py             ← Streamlit UI (recommender, analytics, simulator)
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (ingest → features → train → evaluate)

```bash
python run_pipeline.py
```

> Artifacts saved to `models/saved/` and `data/processed/`.

### 3. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: `http://localhost:8000/docs`

### 4. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

UI: `http://localhost:8501`

### 5. Full stack via Docker

```bash
docker-compose up --build
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## ⚙️ Key Config Values (`config.py`)

| Key | Value | Purpose |
|---|---|---|
| `DEMAND_WINDOWS` | `[7, 14, 30]` | Rolling demand windows |
| `FLASH_SALE_MULTIPLIER_MIN` | `1.5` | CEP spike trigger |
| `COMPETITOR_BLEND_RATIO` | `0.30` | Competitor weight in price blend |
| `CACHE_TTL_SECONDS` | `60` | Redis TTL |
| `API_PORT` | `8000` | FastAPI port |

> Full parameter list in [`config.py`](config.py).
