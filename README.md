# HVAC Energy Consumption Forecasting System

## 1. Project Overview
This project focuses on analyzing and forecasting the **HVAC_Level** (the intensity or operational load of an HVAC system) based on environmental and operational sensor data. By understanding the relationship between various temporal and sensor features, the system aims to optimize building energy management.

The project implements and benchmarks three time-series forecasting models: **OLS (Ordinary Least Squares)**, **Holt-Winters**, and **ARIMA**, ultimately selecting the OLS model for future predictions due to its stability.

## 2. Dataset Statistics
The dataset consists of time-series data collected between **December 2020 and March 2022**.

* **Total Observations:** 444 data points.
* **Total Features:** 36 variables.
* **Target Variable:** `HVAC_Level`.
* **Key Predictors:** Based on correlation analysis, the top 10 features selected for modeling include `'A'`, `'CC'`, `'H'`, `'T'`, `'Z'`, `'E'`, `'F'`, `'BB'`, and the back-shifted target `'HVAC_Level_t-1'`.

## 3. Methodology
The data science pipeline involves rigorous preprocessing and feature engineering:
1.  **Data Treatment:** Included Backshifting (1 day lag), Exponential Moving Average (EMA), and Outlier removal.
2.  **Decomposition:** Analysis of Trend, Seasonality, and Residuals to understand signal components.
3.  **Model Selection:**
    * **OLS Regression:** Uses correlated features and lagged variables.
    * **Holt-Winters:** Handles seasonality and trends via exponential smoothing.
    * **ARIMA (1,1,1):** Auto-Regressive Integrated Moving Average for univariate time series.

## 4. Model Evaluation (RMSE)
Models were evaluated using **Root Mean Squared Error (RMSE)** across two different back-testing periods to ensure consistency.

| Model Type | Backtest 1 RMSE | Backtest 2 RMSE | Performance Verdict |
| :--- | :--- | :--- | :--- |
| **OLS Regression** | **0.7610** | **0.7885** | [cite_start]**Best & Most Stable** [cite: 274] |
| Holt-Winters | 1.2922 | 0.8472 | Good on BT2, poor on BT1 |
| ARIMA (1,1,1) | 1.1292 | 0.6636 | High variance between tests |

**Conclusion:** The OLS model was selected as the final production model because it maintained low and stable error rates across both testing periods.

## 5. Installation & Usage

Follow the steps below to set up the environment and run the forecasting pipeline.

### Prerequisites
* Python 3.10+
* Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `scikit-learn`.

### 1. Setup Environment
```bash
# Clone the repository
git clone [https://github.com/your-username/hvac-prediction.git](https://github.com/your-username/hvac-prediction.git)
cd hvac-prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
```

### 2. Run the Pipeline

1. **Data Preprocessing:** Cleans raw data, handles missing values, and creates dummy variables/scaled features.
```bash
python data_preprocess.py
```

2. **Analysis & Visualization:** Generates statistical reports, correlation heatmaps, and time-series decomposition plots.
```bash
python data_evaluate.py
```

3. **Train Models:** Train and evaluate the three candidate models.
```bash
python ols_model.py
python arima_model.py
python holt_winters_model.py
```

4. **Predict:** Generate future forecasts (next 6 days) using the best-performing model (OLS).
```bash
python hvac_level_predictions.py
```
