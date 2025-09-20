import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import pickle
import os

# --- Step 1: Data Collection and Loading ---
# Make sure the 'garments_worker_productivity.csv' file is in the same directory as this script.
print("Loading dataset...")
data = pd.read_csv('garments_worker_productivity.csv')
print("Dataset loaded successfully.")

# --- Step 2: Data Preprocessing and Feature Engineering ---
print("Starting data preprocessing...")

# Handle date and drop unnecessary columns
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data['quarter'] = data['quarter'].str.lower()
data['quarter'] = data['quarter'].replace({'quarter1': 'quarter_1', 'quarter2': 'quarter_2', 'quarter3': 'quarter_3', 'quarter4': 'quarter_4', 'quarter5': 'quarter_5'})
data['month'] = data['date'].dt.month
data.drop(['date'], axis=1, inplace=True)

# Drop the 'wip' column due to a high number of null values
if 'wip' in data.columns:
    data.drop(['wip'], axis=1, inplace=True)

# Clean and unify 'department' column values
data['department'] = data['department'].replace({'finishing ':'finishing'})

# Handle categorical variables using Label Encoding
le_quarter = LabelEncoder()
le_department = LabelEncoder()
le_day = LabelEncoder()

# Fit and transform categorical columns
data['quarter'] = le_quarter.fit_transform(data['quarter'])
data['department'] = le_department.fit_transform(data['department'])
data['day'] = le_day.fit_transform(data['day'])

# Convert all columns to numeric, coercing errors to NaN
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with any remaining NaN values after conversion
data.dropna(inplace=True)
print("Data preprocessing complete.")

# --- Step 3: Model Building, Training, and Saving ---
# Define features (X) and target (y)
X = data.drop('actual_productivity', axis=1)
y = data['actual_productivity']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
print("Training the XGBoost model...")
model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# Create the Flask/ directory if it doesn't exist
output_dir = 'Flask'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the trained model and encoders to the Flask folder
print(f"Saving model and encoders to '{output_dir}/'...")
with open(os.path.join(output_dir, 'gwp.pkl'), 'wb') as file:
    pickle.dump(model, file)

with open(os.path.join(output_dir, 'gwp_quarter_encoder.pkl'), 'wb') as file:
    pickle.dump(le_quarter, file)

with open(os.path.join(output_dir, 'gwp_department_encoder.pkl'), 'wb') as file:
    pickle.dump(le_department, file)

with open(os.path.join(output_dir, 'gwp_day_encoder.pkl'), 'wb') as file:
    pickle.dump(le_day, file)

print("Model and encoders saved successfully!")
