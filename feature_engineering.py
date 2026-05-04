import pandas as pd

print("Starting Feature Engineering...")

# Load previous data
training_data = pd.read_csv('clean_f1_data.csv')

# --- STEP 1: Filter out the Rain ---
# We only want dry tires (Soft, Medium, Hard) for this model.
# If it rained, the lap times would ruin our dry-weather math.
valid_compounds = ['SOFT', 'MEDIUM', 'HARD']
df = training_data[training_data['Compound'].isin(valid_compounds)].copy()

# --- STEP 2: One-Hot Encoding ---
# This translates the words into 1s and 0s for the AI
df = pd.get_dummies(df, columns=['Compound'], dtype=int)

# --- STEP 3: Define X (The Clues) and y (The Answer) ---
# We use LapNumber as a proxy for fuel load (higher lap = less fuel)
# We use TyreLife to track how degraded the rubber is
features = ['LapNumber', 'TyreLife', 'Compound_SOFT', 'Compound_MEDIUM', 'Compound_HARD']

# If a compound column is missing (e.g., nobody used Hards), add it as 0s so the AI doesn't break
for col in ['Compound_SOFT', 'Compound_MEDIUM', 'Compound_HARD']:
    if col not in df.columns:
        df[col] = 0

X = df[features]
y = df['LapTime_Sec']

print("\nSuccess! Here are your features (X) ready for the AI:")
print(X.head())
print("\nAnd here is the target answer (y) we want to predict:")
print(y.head())

df.to_csv('engineered_f1_data.csv', index=False)
print("\nEngineered data saved to 'engineered_f1_data.csv'!")