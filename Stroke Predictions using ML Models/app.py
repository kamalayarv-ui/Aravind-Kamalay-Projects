from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Feature names expected by the model
feature_names = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'hypertension_heart_disease', 'age_bmi_ratio', 'gender_Female', 'gender_Male',
    'ever_married_No', 'ever_married_Yes', 'work_type_Govt_job', 'work_type_Never_worked',
    'work_type_Private', 'work_type_Self-employed', 'work_type_children',
    'smoking_status_Unknown', 'smoking_status_formerly smoked', 'smoking_status_never smoked',
    'smoking_status_smokes', 'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Raw numeric inputs
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])

        # Categorical inputs
        gender = request.form['gender']  # Male or Female
        ever_married = request.form['ever_married']  # Yes or No
        work_type = request.form['work_type']  # Private, Self-employed, etc.
        smoking_status = request.form['smoking_status']  # never smoked, etc.

        # Engineered features
        hypertension_heart_disease = hypertension * heart_disease
        age_bmi_ratio = age / bmi if bmi != 0 else 0

        # One-hot encodings (initialize all 0s)
        input_dict = dict.fromkeys(feature_names, 0)

        # Fill in raw + engineered features
        input_dict['age'] = age
        input_dict['hypertension'] = hypertension
        input_dict['heart_disease'] = heart_disease
        input_dict['avg_glucose_level'] = avg_glucose_level
        input_dict['bmi'] = bmi
        input_dict['hypertension_heart_disease'] = hypertension_heart_disease
        input_dict['age_bmi_ratio'] = age_bmi_ratio

        # One-hot encode categorical fields
        input_dict[f'gender_{gender}'] = 1
        input_dict[f'ever_married_{ever_married}'] = 1
        input_dict[f'work_type_{work_type}'] = 1
        input_dict[f'smoking_status_{smoking_status}'] = 1

        # BMI category
        if bmi < 18.5:
            pass  # Underweight not in model
        elif 18.5 <= bmi < 25:
            input_dict['bmi_category_Normal'] = 1
        elif 25 <= bmi < 30:
            input_dict['bmi_category_Overweight'] = 1
        else:
            input_dict['bmi_category_Obese'] = 1

        # Convert to DataFrame in expected column order
        X_input = pd.DataFrame([input_dict])[feature_names]

        prediction = model.predict(X_input)[0]
        prediction_text = "Stroke" if prediction == 1 else "No Stroke"

        return render_template('next.html', prediction=prediction_text)

    except Exception as e:
        return f"Error occurred: {e}", 400

if __name__ == '__main__':
    app.run(debug=True)
