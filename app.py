# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load("model/xgb_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        data = {
            "Gender": request.form["gender"],
            "Married": request.form["married"],
            "Dependents": int(request.form["dependents"]),
            "Education": request.form["education"],
            "Self_Employed": request.form["self_employed"],
            "ApplicantIncome": float(request.form["applicant_income"]),
            "CoapplicantIncome": float(request.form["coapplicant_income"]),
            "LoanAmount": float(request.form["loan_amount"]),
            "Loan_Amount_Term": int(request.form["loan_term"]),
            "Credit_History": float(request.form["credit_history"]),
            "Property_Area": request.form["property_area"]
        }

        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]  # Probability of not defaulting

        return render_template("result.html", result=prediction, proba=round(proba * 100, 2))

if __name__ == "__main__":
    app.run(debug=True)
