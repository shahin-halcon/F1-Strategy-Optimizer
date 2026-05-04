import fastf1
import pandas as pd
import os

# --- STEP 1: Set up Caching ---
# We create a folder on your computer to save the data.
if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')
    
fastf1.Cache.enable_cache('f1_cache')

# --- STEP 2: Load the Race Data ---
print("Downloading 2024 Bahrain Grand Prix data...")
# 2024 is the year. 'Bahrain' is the track. 'R' stands for Race.
session = fastf1.get_session(2024, 'Bahrain', 'R')
session.load()

# --- STEP 3: Clean the Data ---
# Extract all the laps completed by every driver
all_laps = session.laps

# Remove the "junk" laps (Safety cars, pit stops, out-laps)
print("Cleaning the data...")
clean_laps = all_laps.pick_quicklaps().copy()

# --- STEP 4: Prepare the numbers for Machine Learning ---
# AI models need pure numbers, not time formats like "1:35.200"
clean_laps['LapTime_Sec'] = clean_laps['LapTime'].dt.total_seconds()

# We only want to keep the columns that actually matter for tire wear
columns_to_keep = ['Driver', 'LapNumber', 'Compound', 'TyreLife', 'LapTime_Sec']
training_data = clean_laps[columns_to_keep]

# Drop any rows where a sensor might have failed and left a blank value (NaN)
training_data = training_data.dropna()

print("\nHere is a sneak peek at your clean data:")
print(training_data.head())

training_data.to_csv('clean_f1_data.csv', index=False)
print("Data saved to clean_f1_data.csv!")