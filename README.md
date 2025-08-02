# Loan Default Prediction using Gradient Boosting

A mini machine learning project that predicts whether a loan applicant is likely to default based on demographic and financial input features. Built using **Flask** for the backend and **Gradient Boosting Classifier** as the ML algorithm.

---

##  Project Features

- Clean and modern HTML/CSS UI
- Trained Gradient Boosting model using `sklearn`
- User inputs details via web form
- Model predicts **Loan Status** (Approved/Defaulted)
- Lightweight and easy to deploy

---

##  Why Gradient Boosting?

Gradient Boosting is ideal for this project because:

- It handles mixed-type data (categorical + numerical) well.
- It performs well with imbalanced datasets.
- It avoids overfitting using built-in regularization.
- It provides high accuracy for binary classification tasks.

---

##  Project Structure
```
loan_default_prediction/
│
├── static/
│ └── style.css # Modern styled CSS
│
├── templates/
│ ├── index.html # Input form page
│ └── result.html # Output prediction page
│
├── loan_data.csv # Dataset with loan applicant info
├── model.pkl # Trained Gradient Boosting model
├── app.py # Flask application
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

---

##  Sample Dataset (loan_data.csv)
```
| Gender | Married | Education | Self_Employed | ApplicantIncome | LoanAmount | Credit_History | Property_Area | Loan_Status |
|--------|---------|-----------|----------------|------------------|-------------|----------------|----------------|--------------|
| Male   | Yes     | Graduate  | No             | 4583             | 128         | 1              | Urban          | Y            |
| Female | No      | Not Graduate | Yes          | 3000             | 100         | 0              | Rural          | N            |
| Male   | Yes     | Graduate  | No             | 6000             | 200         | 1              | Semiurban      | Y            |
```
---

##  How to Run the Project

 1. Clone this repository
```
git clone https://github.com/your-username/loan-default-flask.git
cd loan-default-flask
```
2. install dependencies
```
pip install -r requirements.txt
```
3. Train the Model 
```
python train_model.py     
```
4. Run the Flask App
```
python app.py
```
Visit http://127.0.0.1:5000 in your browser.

# Model Info
Algorithm: GradientBoostingClassifier

Library: scikit-learn

Accuracy: ~85% on test dataset (depending on split)
---
# Tech Stack
  Machine Learning: scikit-learn

  Backend: Flask

  Frontend: HTML5, CSS3

  Data: CSV File (simulated loan applications)

# Screenshots
![alt text](<Screenshot 2025-08-02 224711.png>)
![alt text](<Screenshot 2025-08-02 224746.png>)
![alt text](<Screenshot 2025-08-02 224807.png>)