from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

def clean(df):
    df.columns = df.columns.str.strip().str.lower()
    return df

def safe_lookup(df, disease, col):
    try:
        return df[df["disease"] == disease][col].values[0]
    except:
        return "Not available" 

model = joblib.load("models/model.joblib")

description = clean(pd.read_csv("data/description.csv"))
diets = clean(pd.read_csv("data/diets.csv"))
meds = clean(pd.read_csv("data/medications.csv"))
workout = clean(pd.read_csv("data/workout_df.csv"))
precautions = clean(pd.read_csv("data/precautions_df.csv"))
severity = clean(pd.read_csv("data/Symptom-severity.csv"))
train_df = clean(pd.read_csv("data/Training.csv"))

SYMPTOMS = train_df.columns[:-1].tolist()

def calculate_severity(symptoms):
    score = 0
    for s in symptoms:
        row = severity[severity["symptom"] == s]
        if not row.empty:
            score += int(row["weight"].values[0])

    if score < 5:
        return "Mild"
    elif score < 10:
        return "Moderate"
    else:
        return "Severe"
    
def predict_full(symptoms_list):
    
    input_data = [1 if s in symptoms_list else 0 for s in SYMPTOMS]
    disease = model.predict([input_data])[0]
    probs = model.predict_proba([input_data])[0]
    confidence = round(max(probs) * 100, 2)
    result = {
        "Disease": disease,
        "Confidence": confidence,
        "Severity": calculate_severity(symptoms_list),
        "Description": safe_lookup(description, disease, "description"),
        "Diet" :safe_lookup(diets, disease, "diet"),
        "Medicines":safe_lookup(meds,disease,"medication"),
        "Workout":safe_lookup(workout,disease,"workout"),
        "Precautions":precautions[precautions["disease"] == disease].iloc[:,1:].values.tolist()
    }

    return result

@app.route("/")
def home():
    return render_template("index.html", symptoms=list(SYMPTOMS))

@app.route("/predict", methods=["POST"])
def predict():
    selected = request.form.getlist("symptoms")
    result = predict_full(selected)
    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True,port=8000)
