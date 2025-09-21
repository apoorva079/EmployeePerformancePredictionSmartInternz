# Employee Performance Prediction - SmartInternz


## 📌 Project Overview

**Employee Performance Prediction** is a Flask-based web application that predicts the productivity level of employees in a garments manufacturing setup using a machine learning model.  

The app takes multiple input parameters such as **department, day, team size, targeted productivity, and other operational metrics**, then predicts whether an employee is:  
- **Low Productive**  
- **Medium Productive**  
- **Highly Productive**  

This project was developed as part of the **SmartInternz Internship Program**.

---

## 🚀 Features

- Predict employee performance based on real input data.  
- Uses a trained ML model for accurate predictions.  
- User-friendly interface with responsive forms.  
- Built with Flask, HTML, CSS, and Python.  
- Handles errors gracefully and provides feedback.  

---

## 🛠️ Technologies Used

- **Python** - Core scripting and ML integration  
- **Flask** - Web application framework  
- **Pandas & NumPy** - Data manipulation and preprocessing  
- **Scikit-learn** - Machine Learning model  
- **HTML, CSS** - Frontend interface  
- **Pickle** - Model and encoder serialization  

---

## 📸 Screenshots  

**Home Page**  
![Home Page](https://github.com/apoorva079/EmployeePerformancePredictionSmartInternz/blob/master/Screenshots/Screenshot%202025-07-20%20175105.png)

**Prediction Form Page**  
![Prediction Form](https://github.com/apoorva079/EmployeePerformancePredictionSmartInternz/blob/master/Screenshots/Screenshot%202025-07-20%20174833.png)

**Prediction Result Page**  
![Prediction Result](https://github.com/apoorva079/EmployeePerformancePredictionSmartInternz/blob/master/Screenshots/Screenshot%202025-07-20%20174634.png)

**About Page**
![About Page](https://github.com/apoorva079/EmployeePerformancePredictionSmartInternz/blob/master/Screenshots/Screenshot%202025-07-20%20175134.png)

---

##⚙️ Installation and Setup

1. Clone the repository

git clone https://github.com/apoorva079/EmployeePerformancePredictionSmartInternz.git


2. Navigate to the project folder

cd EmployeePerformancePredictionSmartInternz/Flask


3. Create a virtual environment

python -m venv venv
source venv/bin/activate      # For Linux/Mac
venv\Scripts\activate         # For Windows


4. Install dependencies

pip install -r requirements.txt


5. Run the Flask app

python app.py


6. Open your browser and go to

http://127.0.0.1:5000

---

##📂 Project Structure

EmployeePerformancePredictionSmartInternz/
│
└── Flask/
    │
    ├── template/
    │   ├── about.html
    │   ├── home.html
    │   ├── predict.html
    │   └── submit.html
    │
    ├── gwp_day_encoder.pkl
    ├── gwp_department_encoder.pkl
    ├── gwp_quarter_encoder.pkl
    ├── gwp.pkl
    ├── garments_worker_productivity.csv
    ├── app.py
    └── train_and_save_model.py

---

📜 License

This project is licensed under the MIT License.

👩‍💻 Developed by: Apoorva Verma
📌 Internship: SmartInternz


