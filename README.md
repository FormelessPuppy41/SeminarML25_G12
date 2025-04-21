
# Solar Energy Forecast Combination Project

## Overview

This project focuses on **combining forecasts for solar energy production** for **50Hertz**, a transmission system operator (TSO) in Germany.  
We generate synthetic forecasts, combine them using various machine learning methods, and analyze the results to improve prediction accuracy.

> **This work replicates and extends the study by Nikodinoska et al. (2022):**  
> *Solar and wind power generation forecasts using elastic net in time-varying forecast combinations*, Applied Energy, 306, 117983.  
> [DOI: 10.1016/j.apenergy.2021.117983](https://doi.org/10.1016/j.apenergy.2021.117983)

---

## Key Components

### 1. Initial Forecast Generation
- **Folder:** `initial_forecasting`
- **Script:** `run_forecasts_real_errors.py`
- **Description:** 
  - Generates **six synthetic forecasts** to simulate real-world conditions.
  - These forecasts serve as inputs for the combined forecasting methods.

### 2. Forecast Combination Methods
- **Folder:** `methods`
- **Scripts:** 
  - Combination models include `lasso.py`, `ridge.py`, `elastic_net.py`, `xgboost.py`, `simple_average.py`, `adaptive_elastic_net.py`, and `transformer_model.py`.
- **Controller Scripts:**
  - `forecast_controller.py`
  - `forecast_result_controller.py`
- **Description:**
  - Implements multiple ensemble and machine learning models for combining forecasts.

### 3. Helpers
- **Folder:** `helpers`
- **Scripts:** 
  - `forecast_result_processor.py`, `forecast_runner.py`, `forecast_writer.py`
- **Description:**
  - Assists with running forecasts, processing results, and writing outputs.

### 4. Data Handling
- **Folder:** `data`
- **Input Files:**
  - `error_model_combined_forecasts2.csv` – for HR data (solar production error model)
  - `data_ander_groepje.csv` – for price data
- **Output Files:**
  - Located in the `output_files` directory
  - Results are stored separately for **HR** (high-resolution solar forecast) and **Price** (energy price forecasts).

### 5. Main Script
- **File:** `main.py`
- **Functions:**
  1. **Data Generation** – Create synthetic forecast data.
  2. **Model Estimation** – Train and estimate combination models.
  3. **Visualization** – Visualize model performance and comparison.

### 6. Configuration
- **File:** `configuration.py`
- **Purpose:**
  - Adjust model settings, forecast parameters, and processing configurations.

---

## How to Run the Project

1. **Set up the environment**  
   Make sure you have a Python environment (recommend using `.venv`) and install the necessary dependencies.

2. **Configure settings**  
   Modify `configuration.py` if you want to change model parameters or data paths.

3. **Run the main file**  
   Use the `main.py` script to generate data, estimate models, and visualize results:
   ```bash
   python main.py
   ```

---

## Authors

**Group G12** – Seminar on Machine Learning 2024/25

---


## Project Structure

```
SEMINARML25_G12/
├── combined_forecast/
│   ├── __pycache__/
│   ├── helpers/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── forecast_result_processor.py
│   │   ├── forecast_runner.py
│   │   └── forecast_writer.py
│   ├── initial_forecasting/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── forecast.py
│   │   ├── forecasters.py
│   │   ├── forecasts_real_errors.py
│   │   ├── PCA_real_errors.png
│   │   ├── run_forecast.py
│   │   ├── run_forecasts_real_errors.py
│   │   └── run_forecasts.py
│   ├── methods/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── adaptive_elastic_net.py
│   │   ├── elastic_net.py
│   │   ├── lasso.py
│   │   ├── ridge.py
│   │   ├── simple_average.py
│   │   ├── transformer_model.py
│   │   ├── utils.py
│   │   ├── xgboost.py
│   │   ├── forecast_controller.py
│   │   └── forecast_result_controller.py
├── data/
│   ├── __pycache__/
│   ├── data_files/
│   │   └── 50hertz_files/
│   ├── input_files/
│   │   ├── __init__.py
│   │   ├── combined_forecasts.csv
│   │   ├── data_ander_groepje.csv
│   │   ├── error_model_combined_forecasts.csv
│   │   ├── error_model_combined_forecasts2.csv
│   │   ├── params.csv
│   │   └── SolarDataCombined.csv
│   ├── model_results/
│   │   ├── __init__.py
│   │   ├── adaptive_elastic_net_forecast.csv
│   │   ├── elastic_net_forecast.csv
│   │   ├── lasso_forecast.csv
│   │   ├── ridge_forecast.csv
│   │   ├── simple_average_forecast.csv
│   │   ├── tune_elastic_net_forecast.csv
│   │   └── xgboost_forecast.csv
│   ├── output_files/
│   │   ├── __init__.py
│   │   ├── hr/
│   │   │   ├── adaptive_elastic_net_forecast.csv
│   │   │   ├── elastic_net_forecast.csv
│   │   │   ├── lasso_forecast.csv
│   │   │   ├── ridge_forecast.csv
│   │   │   ├── tune_elastic_net_forecast.csv
│   │   │   └── xgboost_forecast.csv
│   │   └── price/
│   │   │   ├── adaptive_elastic_net_forecast.csv
│   │   │   ├── elastic_net_forecast.csv
│   │   │   ├── lasso_forecast.csv
│   │   │   ├── ridge_forecast.csv
│   │   │   ├── tune_elastic_net_forecast.csv
│   │   │   └── xgboost_forecast.csv
│   ├── data_loader.py
│   └── data_processing.py
├── findings/
├── utils/
│   └── __init__.py
├── configuration.py
├── main.py
└── README.md

```

---