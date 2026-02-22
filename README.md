# 📈 සිංහ AI: Sri Lankan Rupee (LKR) Forex Forecaster

An end-to-end Machine Learning system designed to predict LKR exchange rates using **Stochastic Gradient Boosting (LightGBM)**. This project integrates ground-truth data from the **Central Bank of Sri Lanka (CBSL)** with global macroeconomic indicators to provide multi-day recursive forecasts.

---

## 🚀 Key Features (Bonus Marks Integration)

* **Full-Stack Integration:** A containerized system using **FastAPI** (Backend) and **Streamlit** (Frontend).
* **Interactive Macro Stress Testing:** A "Shock Factor" slider allows users to simulate sudden daily changes in the **Global USD Index (DXY)**.
* **Recursive Stochastic Engine:** Feeds predictions back into the model to simulate 30-day trends with **95% Confidence Intervals**.
* **Model Explainability (XAI):** Built-in **SHAP analysis** modules to interpret the influence of Lags, Moving Averages, and Macro drivers.
* **Price-Scaled Volatility:** Specialized logic to handle JPY/LKR scaling separately from USD/LKR, ensuring realistic simulations across all unit values.

---

## 🧠 Technical Innovations

To ensure "Production-Grade" financial realism, the system implements:
* **$\sigma \sqrt{t}$ Scaling:** Uses the Square Root of Time rule to scale stochastic noise, preventing "volatility explosion" in long-term recursive forecasts.
* **L1 Regularization (Lasso):** Applied during training to "zero out" noise, ensuring the model respects the current LKR recovery stability.
* **Dynamic Lag Shifting:** Unlike static models, the backend dynamically updates $Lag_1, Lag_2, \dots, Lag_7$ during the recursive loop to maintain temporal integrity.

---

## 📂 Project Structure & Pipeline

### **1. Research & Development (Notebooks)**
* **`1_data_collection.ipynb`**: Fetches global DXY data and merges with the local CBSL dataset.
* **`2_preprocessing.ipynb`**: Handles feature engineering (Lags, MA, Volatility, and Crisis structural breaks).
* **`3_training_evaluation.ipynb`**: Implements the Time-Series split, LightGBM training, and performance metrics.
* **`4_explainability.ipynb`**: Utilizes SHAP analysis to align model logic with economic domain knowledge.

### **2. Production System**
* **`backend/`**: FastAPI server and the `predict.py` engine.
* **`frontend/`**: Streamlit interactive dashboard.
* **`models/`**: Serialized `.joblib` files (Features model binary + metadata).
* **`src/`**: Core logic for model training and recursive forecasting.

---

## 🛠️ Execution Guide

**Prerequisite:** Docker Desktop installed.

1.  **Start the System:**
    ```bash
    docker-compose up --build
    ```
2.  **Access the Dashboard:**
    * **Frontend:** [http://localhost:8501](http://localhost:8501)
    * **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📖 Evaluation Guide

1.  **Managed Float Baseline:** Notice that for USD/LKR, the model predicts a stable recovery trend. This reflects the current LKR stability observed in 2024-2025 data.
2.  **Stress Testing:** Adjust the **"Global USD Shock"** slider to 3.0%. Observe how the AI trend line pivots upwards, illustrating the Rupee's sensitivity to global dollar surges.
3.  **Explainability:** Scroll to the bottom of the dashboard to view the **Feature Importance** breakdown, proving the model prioritizes recent momentum and macro trends.