# 🏷️ Dynamic Pricing Engine

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-FF6600)](https://xgboost.readthedocs.io)
[![Redis](https://img.shields.io/badge/Cache-Redis-DC382D?logo=redis&logoColor=white)](https://redis.io)
[![Kafka](https://img.shields.io/badge/Streaming-Kafka-231F20?logo=apachekafka&logoColor=white)](https://kafka.apache.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose)

A **production-grade, end-to-end dynamic pricing system** trained on real-world UCI retail datasets. It combines ensemble machine learning, real-time streaming, a Redis-cached REST API, and an interactive Streamlit dashboard to deliver sub-second, data-driven price recommendations.

---

## 📑 Table of Contents

- [Industry Problem](#-industry-problem)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Datasets](#-datasets)
- [Quick Start](#-quick-start)
- [ML Pipeline](#-ml-pipeline)
- [API Reference](#-api-reference)
- [Streaming & CEP](#-streaming--cep)
- [Dashboard](#-dashboard)
- [Docker Deployment](#-docker-deployment)
- [Configuration](#-configuration)

---

## 🏭 Industry Problem

E-commerce platforms suffer significant revenue leakage from **static pricing**. They cannot react in real time to:

- Demand spikes and flash sale events
- Inventory depletion and stockout risk
- Competitor price movements
- Temporal patterns (weekday vs. weekend, seasonal trends)

This engine solves all of the above by continuously learning from historical transaction data and serving optimized prices through a low-latency API.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Dynamic Pricing Engine                       │
├──────────────┬─────────────────────┬──────────────┬──────────────────┤
│  Data Layer  │    ML Pipeline      │  Serving     │   Monitoring     │
│              │                     │              │                  │
│ UCI Retail I │ Phase 1: Ingest     │ FastAPI      │ Streamlit        │
│ UCI Retail II│ Phase 2: Clean      │   └─ Redis   │   Dashboard      │
│              │ Phase 3: Features   │      Cache   │                  │
│ .parquet     │ Phase 4: Target Eng.│ /price/      │ Live Recommender │
│ (processed)  │ Phase 5: Train      │  recommend   │ Product Analytics│
│              │   ├─ XGBoost        │ /price/bulk  │ Price Simulator  │
│              │   ├─ LightGBM       │ /health      │ Flash Sale Alerts│
│              │   ├─ CatBoost       │              │                  │
│              │   ├─ RandomForest   │              │                  │
│              │   └─ Ridge Stack    │   Kafka      │                  │
│              │ Phase 6: Evaluate   │   Producer   │                  │
│              │                     │   + CEP      │                  │
└──────────────┴─────────────────────┴──────────────┴──────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Stacking Ensemble** | XGBoost, LightGBM, CatBoost, and Random Forest blended via Ridge meta-learner with learned weights |
| **Price Elasticity** | MR=MC optimization-inspired elasticity features computed over 30-day rolling windows |
| **Redis Caching** | API responses cached with configurable TTL for <10ms repeated-query latency |
| **CEP Flash Sale Detection** | Complex Event Processing detects demand spikes (≥1.5× baseline) within a 5-minute sliding window and emits Kafka alerts |
| **Kafka Streaming** | Order events published to Kafka; flash sale price multipliers (up to 1.4×) broadcast to downstream consumers |
| **Interactive Dashboard** | Streamlit UI with live pricing recommender, product analytics, and price-change simulator |
| **Docker-Ready** | Full `docker-compose.yml` orchestrating API, Redis, and Dashboard services |
| **Time-Series Safe Splits** | Training / validation / test sets split strictly by date to prevent leakage |

---

## 📁 Project Structure

```
Dynamic-pricing/
│
├── README.md                    ← This file
├── .gitignore
│
├── dashboard/
│   └── app.py                   ← Root-level Streamlit entry point
│
└── dynamic_pricing/             ← Core engine package
    ├── config.py                ← All paths, thresholds, and hyperparameters
    ├── run_pipeline.py          ← End-to-end pipeline runner
    ├── requirements.txt
    ├── Dockerfile
    ├── docker-compose.yml
    │
    ├── src/                     ← ML pipeline stages
    │   ├── __init__.py
    │   ├── ingest.py            ← Phase 1: Load xlsx datasets
    │   ├── clean.py             ← Phase 2: Filter, standardise, deduplicate
    │   ├── features.py          ← Phase 3: ~50 features (temporal, demand, elasticity)
    │   ├── pricing_target.py    ← Phase 4: Target variable engineering
    │   ├── train.py             ← Phase 5: Ensemble training & stacking
    │   ├── evaluate.py          ← Phase 6: MAE, R², MAPE metrics
    │   ├── predict.py           ← Inference wrapper used by the API
    │   └── simulator.py         ← Price-change scenario simulator
    │
    ├── api/
    │   ├── main.py              ← FastAPI app (single + bulk pricing endpoints)
    │   └── schemas.py           ← Pydantic request / response models
    │
    ├── streaming/
    │   ├── flash_sale_detector.py  ← CEP spike detector (5-min window)
    │   └── producer.py             ← Kafka producer for order events
    │
    ├── dashboard/
    │   └── app.py               ← Streamlit UI
    │
    ├── notebooks/               ← Exploratory notebooks
    ├── tests/                   ← Pytest test suite
    ├── data/                    ← Processed parquet files (gitignored)
    └── models/                  ← Serialised model artifacts (gitignored)
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Language** | Python 3.11+ |
| **ML / Modelling** | XGBoost, LightGBM, CatBoost, scikit-learn (RandomForest, Ridge), joblib |
| **Data Processing** | Pandas, NumPy, PyArrow, openpyxl |
| **API** | FastAPI, Uvicorn, Pydantic v2 |
| **Caching** | Redis 7 (async via `redis.asyncio`) |
| **Streaming** | Apache Kafka, kafka-python |
| **Dashboard** | Streamlit, Plotly, streamlit-autorefresh |
| **Containerisation** | Docker, Docker Compose |
| **Testing** | Pytest, httpx |

---

## 📦 Datasets

The engine is trained on two publicly available UCI retail datasets:

| Dataset | Period | Records (approx.) |
|---|---|---|
| **Online Retail I** (`Online Retail.xlsx`) | 2010–2011 | ~540K transactions |
| **Online Retail II** (`online_retail_II.xlsx`) | 2009–2011 | ~1M transactions |

> **Source:** [UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/ml/datasets/online+retail)
> 
> Raw data files are **not** included in the repository due to size. Download and place them in `online+retail+ii/`.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Redis (for API caching — optional, degrades gracefully)
- Kafka (for streaming — optional)
- Raw dataset files in `online+retail+ii/`

### 1. Clone & Install

```bash
git clone https://github.com/Ayanmohd18/Dynamic-pricing.git
cd Dynamic-pricing/dynamic_pricing

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Full ML Pipeline

This ingests data, engineers features, trains all models, and saves artifacts:

```bash
python run_pipeline.py
```

### 3. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: `http://localhost:8000/docs`

### 4. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Navigate to: `http://localhost:8501`

---

## 🤖 ML Pipeline

The pipeline runs in sequential phases, all configurable via `config.py`:

```
Phase 1 → Ingest      Load both Excel sheets, combine into one DataFrame
Phase 2 → Clean       Remove returns, bad prices, duplicates; standardise columns
Phase 3 → Features    Engineer ~50 features:
                         • Temporal  : hour, dow, month, is_weekend, quarter
                         • Demand    : rolling 7/14/30-day sums & lags
                         • Elasticity: ΔPrice / ΔDemand over 30-day window
                         • Inventory : cumulative quantity proxies
                         • Competitor: simulated price blending (30/70 split)
Phase 4 → Target      Derive optimal target price using demand-weighted logic
Phase 5 → Train       Train 4 base models + Ridge stacking meta-learner
                         • Time-series split: train < Sep 2011 | val Sep–Nov | test Nov+
                         • MinMaxScaler applied after split (no leakage)
Phase 6 → Evaluate    Report MAE, R², MAPE on held-out test set
```

### Ensemble Blend Weights

The Ridge meta-learner learns optimal weights from validation-set OOF predictions. Typical results:

| Model | Role |
|---|---|
| XGBoost | Base learner (highest weight typically) |
| LightGBM | Base learner |
| CatBoost | Base learner |
| Random Forest | Base learner |
| Ridge | Meta-learner (stacking) |

---

## 🔌 API Reference

Base URL: `http://localhost:8000`

### `GET /health`

```json
{
  "status": "healthy",
  "uptime_seconds": 142.5,
  "redis_connected": true
}
```

### `POST /price/recommend`

**Request:**
```json
{
  "stock_code": "85123A",
  "current_price": 2.55,
  "quantity": 10,
  "country": "United Kingdom",
  "timestamp": "2024-11-15T14:30:00",
  "adjustment_factors": {}
}
```

**Response:**
```json
{
  "recommended_price": 2.89,
  "price_change_pct": 13.3,
  "confidence": 0.87,
  "latency_ms": 4.2,
  "cached": false
}
```

### `POST /price/bulk`

Accepts an array of `PriceRequest` objects; processes them in parallel via `asyncio.gather`.

---

## 📡 Streaming & CEP

The `FlashSaleDetector` implements a **sliding 5-minute window** per product:

1. Each order event is ingested with `ingest_event(event)`
2. Events older than the window are purged from a `deque`
3. Current window demand is compared to a historical **median baseline**
4. If `current_rate / baseline ≥ 1.5` → **spike detected**

| Spike Ratio | Severity | Price Multiplier |
|---|---|---|
| 1.5× – 2.0× | LOW | 1.04× – 1.07× |
| 2.0× – 3.0× | MEDIUM | 1.07× – 1.11× |
| 3.0× – 5.0× | HIGH | 1.11× – 1.16× |
| ≥ 5.0× | CRITICAL | up to **1.40×** |

Alerts are published to the `flash_sale_alerts` Kafka topic.

---

## 📊 Dashboard

The Streamlit dashboard (`dashboard/app.py`) provides:

- **Live Recommender** — Enter a stock code and get real-time price recommendations from the API
- **Product Analytics** — Demand trends, price history, and feature importance charts
- **Price Simulator** — Adjust quantity / date sliders and preview price impact
- **Flash Sale Monitor** — Live feed of active CEP alerts from the streaming layer

---

## 🐳 Docker Deployment

Spin up the full stack (Redis + API + Dashboard) with one command:

```bash
cd dynamic_pricing
docker-compose up --build
```

| Service | Port |
|---|---|
| Redis | 6379 |
| FastAPI | 8000 |
| Streamlit | 8501 |

---

## ⚙️ Configuration

All tuneable parameters live in `dynamic_pricing/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `DEMAND_WINDOWS` | `[7, 14, 30]` | Rolling demand window sizes (days) |
| `LAG_DAYS` | `[1, 7, 14, 30]` | Lag feature offsets |
| `PRICE_ELASTICITY_WINDOW` | `30` | Days for elasticity calculation |
| `FLASH_SALE_MULTIPLIER_MIN` | `1.5` | Spike detection threshold |
| `COMPETITOR_BLEND_RATIO` | `0.30` | Weight of competitor price in blend |
| `CACHE_TTL_SECONDS` | `60` | Redis cache expiry |
| `KAFKA_BROKER` | `localhost:9092` | Kafka broker address |

---

## 📄 License

This project is for educational and portfolio purposes. Dataset attribution: Daqing Chen, UCI Machine Learning Repository.
