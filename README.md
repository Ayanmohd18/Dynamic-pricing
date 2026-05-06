# 🎯 State-of-the-Art Dynamic Pricing Engine

An elite, AI-powered dynamic pricing engine for e-commerce platforms. Inspired by the real-time surge pricing architectures of Uber, Lyft, and Amazon, this system processes millions of data points to deliver sub-second, highly optimized pricing adjustments based on market demand, logistics, competitor actions, and inventory levels.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20TensorFlow-orange)
![Reinforcement Learning](https://img.shields.io/badge/RL-PyTorch%20DQN-red)
![Streaming](https://img.shields.io/badge/Streaming-Apache%20Flink%20%7C%20Kafka-black)
![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

## 🏢 Industry Problem
E-commerce platforms face intense competition where static pricing causes severe revenue leaks. The inability to react to weather disruptions, delivery fleet shortages, competitor undercutting, or flash sale spikes results in simultaneous stockouts (selling too cheap during high demand) and dead stock (overpricing during low demand).

This pricing engine solves this by dynamically altering prices in real-time, targeting a **10-20% revenue uplift** and a **15% reduction in stockouts**.

---

## 🚀 Key Features

*   **Uber-Style Surge Logistics:** Dynamically scales prices up to 40% based on real-time physics—specifically **Weather Conditions** (Rain/Snow multipliers) and **Delivery Fleet Availability** to prevent logistics bottlenecks.
*   **Ensemble ML Predictor:** Blends predictions from an advanced **XGBoost** regressor (70% weight) and a **TensorFlow Deep Learning Baseline** (30% weight) to calculate the ultimate base price.
*   **Reinforcement Learning Agent:** Features a state-of-the-art **Deep Q-Network (DQN)** built in PyTorch that learns optimal pricing policies via trial-and-error in a simulated marketplace to maximize long-term revenue.
*   **Real-Time Stream Processing (Apache Flink):** Ingests clickstreams and order events using Flink and Kafka. Implements **Complex Event Processing (CEP)** to detect 3x volume spikes and trigger emergency "Flash Sale" multipliers.
*   **Competitor Blending & Guardrails:** Integrates competitor REST APIs, blending their average pricing with ML predictions, firmly guarded by strict Cost-Floor and MSRP-Ceiling constraints.
*   **Generative AI Explainability:** An LLM module acts as a virtual Pricing Analyst, turning raw mathematical outputs into human-readable rationales (e.g., *"Price increased by 15% to balance exceptionally high demand against critically low inventory"*).
*   **Minimalist Dashboard:** A sleek, dark-mode Streamlit UI meticulously designed with Uber-aesthetics, featuring real-time SHAP feature impact visualization.

---

## 🧠 System Architecture

1.  **`src/models/pricing_engine.py`**: The core logic engine that blends XGBoost, TensorFlow, and Competitor data while enforcing margin guardrails.
2.  **`src/models/tensorflow_baseline.py`**: Keras/TF Dense Neural Network serving as the baseline price regressor.
3.  **`src/models/rl_pricing_agent.py`**: PyTorch DQN Agent and simulated `MarketplaceEnv` learning optimal multi-step pricing policies.
4.  **`src/api/llm_explainability.py`**: LLM Prompt-engineering module translating price math into business logic.
5.  **`src/streaming/flink_pipeline.py`**: PyFlink streaming pipeline managing 5-minute sliding windows and Flash Sale detection.
6.  **`dashboard/streamlit_app.py`**: Real-time analytical dashboard with surge mechanics.

---

## ⚙️ Installation & Setup

### Prerequisites
*   Python 3.11+
*   Java 11 (For Apache Flink & Kafka)
*   AWS EC2 (For production deployment)

### 1. Clone & Install
```bash
git clone https://github.com/Ayanmohd18/Dynamic-pricing.git
cd Dynamic-pricing
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard
```bash
streamlit run dashboard/streamlit_app.py
```
*Navigate to `http://localhost:8501` to view the UI.*

### 3. Run the Streaming Pipeline (Flink)
```bash
python src/streaming/flink_pipeline.py --local
```

### 4. Train Models
```bash
# Train Deep Learning Baseline
python src/models/tensorflow_baseline.py

# Train Reinforcement Learning Agent
python src/models/rl_pricing_agent.py
```

---

## 🛠 Tech Stack
*   **Data Science:** Pandas, NumPy, Scikit-Learn
*   **Machine Learning:** XGBoost, TensorFlow/Keras, PyTorch
*   **Streaming & MLOps:** Apache Flink, Kafka, MLflow
*   **Backend & Frontend:** FastAPI, Streamlit, Plotly
*   **Deployment:** AWS EC2, S3

---

*Designed for high-frequency pricing adjustments and optimized revenue capture.*
