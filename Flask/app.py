from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import os

# Flask app with correct template folder
app = Flask(__name__, template_folder="template")

# --- Load model and encoders safely ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = pickle.load(open(os.path.join(BASE_DIR, 'gwp.pkl'), 'rb'))
    quarter_encoder = pickle.load(open(os.path.join(BASE_DIR, 'gwp_quarter_encoder.pkl'), 'rb'))
    department_encoder = pickle.load(open(os.path.join(BASE_DIR, 'gwp_department_encoder.pkl'), 'rb'))
    day_encoder = pickle.load(open(os.path.join(BASE_DIR, 'gwp_day_encoder.pkl'), 'rb'))
except FileNotFoundError as e:
    print("Error: Model or encoder files not found.")
    print("Details:", e)
    exit()


# --- Routes ---
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # --- Get values from form ---
            quarter_input = request.form.get('quarter').lower().strip()
            department_input = request.form.get('department').lower().strip()
            day_input = request.form.get('day').lower().strip()
            team = int(request.form.get('team'))
            targeted_productivity = float(request.form.get('targeted_productivity'))
            smv = float(request.form.get('smv'))
            incentive = float(request.form.get('incentive'))
            over_time = float(request.form.get('over_time'))
            idle_time = float(request.form.get('idle_time'))
            idle_men = float(request.form.get('idle_men'))
            no_of_style_change = int(request.form.get('no_of_style_change'))
            no_of_workers = float(request.form.get('no_of_workers'))
            month = int(request.form.get('month'))

            # --- Encode categorical variables ---
            encoded_quarter = quarter_encoder.transform([quarter_input])[0]
            encoded_department = department_encoder.transform([department_input])[0]
            encoded_day = day_encoder.transform([day_input])[0]

            # --- Prepare dataframe for model ---
            input_data = pd.DataFrame([[
                encoded_quarter, encoded_department, encoded_day,
                team, targeted_productivity, smv, incentive, over_time,
                idle_time, idle_men, no_of_style_change, no_of_workers, month
            ]], columns=[
                'quarter', 'department', 'day', 'team',
                'targeted_productivity', 'smv', 'incentive',
                'over_time', 'idle_time', 'idle_men',
                'no_of_style_change', 'no_of_workers', 'month'
            ])

            # --- Prediction ---
            prediction = model.predict(input_data)[0]

            if prediction < 0.5:
                prediction_text = "Low Productive."
            elif 0.5 <= prediction < 0.8:
                prediction_text = "Medium Productive."
            else:
                prediction_text = "Highly Productive."

            return render_template('submit.html', prediction_text=prediction_text)

        except Exception as e:
            error_message = f"Error processing input: {e}"
            return render_template('predict.html', error=error_message)

    return render_template('predict.html')


# --- Run the app ---
if __name__ == '__main__':
    app.run(debug=True)
