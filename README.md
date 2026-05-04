# 🏎️ Formula 1: Tire Degradation & Pit-Stop Strategy Optimizer

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![FastF1](https://img.shields.io/badge/Data-FastF1-yellow.svg)](https://theoehrly.github.io/Fast-F1/)

## 🏁 Project Overview
In modern Formula 1, a race is rarely won purely on the track; it is won on the pit wall. This project is an end-to-end Machine Learning pipeline and Interactive Dashboard that transitions from **Predictive Analytics** (guessing lap times) to **Prescriptive Analytics** (dictating optimal mathematical decisions).

Using real-world, high-frequency telemetry data from the Bahrain Grand Prix, this tool trains an XGBoost algorithm to learn the complex, non-linear degradation curves of different Pirelli tire compounds. It then wraps that AI brain in a simulation engine to calculate the exact optimal pit-stop strategy to minimize total race time.


![Strategy Dashboard Placeholder](Screenshot.pdf) 

## 🧠 The Architecture
This project mimics a production-level data science workflow:

1. **Data Ingestion (`data_ingestion.py`):** Utilizes the `fastf1` API to pull official timing data, caching the large telemetry files locally to prevent rate-limiting.
2. **Feature Engineering (`feature_engineering.py`):** Cleans noisy track data (filtering out Safety Cars and In/Out laps). Translates the physics into mathematics by one-hot encoding tire compounds and utilizing `LapNumber` as a proxy for fuel-load weight reduction.
3. **Machine Learning (`model_training.py`):** Trains an **XGBoost Regressor** to model the competing forces of fuel burn (car gets faster) and tire degradation (car gets slower).
4. **Operations Research (`app.py`):** A dynamic **Streamlit** dashboard that simulates race conditions, calculates pit-stop time penalties, and visually highlights the crossover point between 1-stop and 2-stop strategies.

## 🛠️ Tech Stack
* **Data Processing:** `pandas`, `numpy`, `fastf1`
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Frontend & Visualization:** `streamlit`, `plotly`

## 📊 The Machine Learning Engine
The core of the prediction relies on understanding the relationship:
`Predicted Lap Time = Base Pace - (Fuel Burn Advantage) + (Tire Degradation Penalty)`

The XGBoost model successfully reverse-engineered this relationship from raw track data, achieving a **Mean Absolute Error (MAE) of ~0.60 seconds** on unseen test data. The feature importance output confirmed that `TyreLife` and `LapNumber` were the driving factors in the decision tree splits.

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR-USERNAME/F1-Strategy-Optimizer.git](https://github.com/YOUR-USERNAME/F1-Strategy-Optimizer.git)
cd F1-Strategy-Optimizer
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Data Pipeline (Optional - if you want to retrain the model)**
```bash
python data_ingestion.py
python feature_engineering.py
python model_training.py
```

**4. Launch the Strategy Dashboard**
```bash
streamlit run app.py
```

## 📂 Project Structure
```text
F1-Strategy-Optimizer/
├── data_ingestion.py        # Connects to FastF1 API
├── feature_engineering.py   # Prepares data for machine learning
├── model_training.py        # Trains the XGBoost AI 
├── app.py                   # The interactive Streamlit dashboard
├── f1_tire_model.json       # The saved AI weights
├── requirements.txt         # Package dependencies
└── README.md                # Project documentation
```

## 🔮 Future Improvements
* **Weather Integration:** Incorporating track temperature (`TrackTemp`) from the FastF1 weather API to dynamically adjust wear coefficients.
* **Traffic Simulation:** Adding a "dirty air" penalty to the lap times if a simulated driver exits the pit lane behind slower cars.
