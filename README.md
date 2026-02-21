# 📈 Sri Lankan Rupee (LKR) AI Forex Forecaster

An end-to-end Machine Learning system designed to predict LKR exchange rates using **Stochastic Gradient Boosting (LightGBM)**. This project integrates ground-truth data from the **Central Bank of Sri Lanka (CBSL)** with global macroeconomic indicators to provide multi-day recursive forecasts.

---

## 🚀 Key Features (Bonus Marks Integration)

* **Full-Stack Integration:** A containerized system using **FastAPI** (Backend) and **Streamlit** (Frontend).
* **Interactive User Inputs:** Users can manually override the starting exchange rate to simulate "What-If" scenarios.
* **Macro Stress Testing:** A "Shock Factor" slider allows users to simulate sudden changes in the Global US Dollar Index (DXY).
* **Model Explainability:** Built-in explanation modules describe the influence of Lags, Moving Averages, and Macro drivers.
* **Stochastic Simulations:** Uses Gaussian noise and recursive loops to provide a 95% confidence interval for future trends.

---

## 📂 Project Structure & Pipeline

The project is organized into a sequential pipeline for clarity and reproducibility:

## 📂 Dataset Description
The model is powered by a multi-source dataset obtained via:
1. **Exchange Rate Data:** Historical daily "Indicative Rates" sourced from the **Central Bank of Sri Lanka (CBSL)** Open Data portal (2010–2025).
2. **Global Macro Indicators:** The **USD Index (DXY)** fetched via the `yfinance` API to provide global context to the LKR's performance.
3. **Derived Features:** Engineered lags, moving averages, and crisis-period indicators developed during the preprocessing phase.

### **1. Jupyter Notebooks (Research & Development)**
* **`1_data_collection.ipynb`**: Fetches global DXY data via `yfinance` and merges it with the local CBSL dataset.
* **`2_preprocessing.ipynb`**: Handles feature engineering (Lags, Moving Averages, Volatility, and Crisis structural breaks).
* **`3_training_evaluation.ipynb`**: Implements the Time-Series split, LightGBM training, and performance metrics (MAE/RMSE).
* **`4_explainability.ipynb`**: Utilizes SHAP analysis to interpret model logic and align it with economic domain knowledge.

### **2. Production System**
* **`backend/`**: Contains `api.py` (FastAPI) and the AI prediction engine.
* **`frontend/`**: Contains `app.py` (Streamlit) for the interactive dashboard.
* **`models/`**: Stores serialized `.joblib` files for each local currency model.
* **`data/`**: Organized into `/raw` and `/processed` directories.
* **`src/`**: Core logic for training and recursive prediction.

---

## 🛠️ Execution Guide (Docker Desktop Required)

The most reliable way to run the system is using Docker Compose, which configures the environment and dependencies automatically.

1.  **Build and Start the Containers:**
    Open a terminal in the project root and run:
    ```bash
    docker-compose up --build
    ```

2.  **Access the Dashboard:**
    * **Frontend UI:** [http://localhost:8501](http://localhost:8501)
    * **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

3.  **Stopping the System:**
    ```bash
    docker-compose down
    ```

---

## 🧠 Model Methodology

* **Architecture:** Local Model approach (One independent LightGBM model per currency pair).
* **Objective Function:** Mean Absolute Error (MAE) - chosen for its robustness against extreme price spikes during economic crises.
* **Forecasting Engine:** Recursive Multi-step forecasting where each predicted value ($t+1$) is fed back as an input feature for the next step ($t+2$).
* **Explainability:** Feature Importance analysis proves the model prioritizes recent momentum ($Lags$) and global macro trends ($USD Index$).

---

## 📖 User Guide for Evaluators

1.  **Select Currency:** Choose from USD, EUR, GBP, JPY, etc., from the sidebar.
2.  **Manual Entry:** Notice the "Manual Entry" field. You can type in a specific LKR rate to see how the AI reacts to that starting point.
3.  **Apply Shock:** Adjust the "Global USD Shock" slider to see the immediate bearish/bullish impact on the LKR forecast.
4.  **Download:** Use the "Download Forecast CSV" button at the bottom to export the simulated data.