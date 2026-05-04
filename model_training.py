import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("Starting Model Training...")

# --- STEP 1: Load the prepped data ---
df = pd.read_csv('engineered_f1_data.csv')

# Ensure X and y are ready (re-declaring just in case)
features = ['LapNumber', 'TyreLife', 'Compound_SOFT', 'Compound_MEDIUM', 'Compound_HARD']
X = df[features]
y = df['LapTime_Sec']

# --- STEP 2: The Train/Test Split ---
# test_size=0.2 means 20% of the data is hidden for the Final Exam
# random_state=42 is just a seed so we get the exact same random split every time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Studying {len(X_train)} laps... Hiding {len(X_test)} laps for the test.")

# --- STEP 3: Create the AI Brain ---
# XGBoost is an algorithm that builds hundreds of decision trees
model = xgb.XGBRegressor(
    n_estimators=100,      # Build 100 decision trees
    learning_rate=0.1,     # How fast it learns (too fast = sloppy, too slow = takes forever)
    max_depth=5,           # How complex each tree is allowed to get
    random_state=42
)

# --- STEP 4: Train the AI ---
print("Training the XGBoost model (This might take a few seconds)...")
model.fit(X_train, y_train)

# --- STEP 5: The Final Exam ---
print("Taking the Final Exam...")
predictions = model.predict(X_test)

# --- STEP 6: Grading the Exam ---
# Mean Absolute Error (MAE) tells us how many seconds off the AI was, on average.
error = mean_absolute_error(y_test, predictions)

print(f"\nModel Graded! On average, the AI guesses the lap time within: {error:.3f} seconds.")
model.save_model('f1_tire_model.json')
print("\nModel successfully saved as 'f1_tire_model.json'!")

# --- STEP 2: The Professional Model Summary ---
# 1. RMSE (Root Mean Squared Error): Punishes the AI heavily for being REALLY wrong on a single lap
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# 2. R-squared (R2): The "Accuracy" equivalent for regression. 
# It scores from 0 to 1.0. A score of 0.85 means the AI understands 85% of what causes lap times to change.
r2 = r2_score(y_test, predictions)

print("\n=== AI MODEL REPORT CARD ===")
print(f"Mean Absolute Error (MAE):     {error:.3f} seconds (Average mistake)")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f} seconds (Mistake including outliers)")
print(f"R-squared Score (R2):          {r2:.3f} (How well it learned the physics)")
print("============================")

# --- STEP 3: Feature Importance (Visual Summary) ---
# This prints a chart showing exactly WHICH clues the AI found most useful
xgb.plot_importance(model, importance_type='weight', title='What forces affect lap time the most?')
plt.show()