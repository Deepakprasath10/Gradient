
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report

df = pd.read_csv("loan_data.csv")

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

cat_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
num_cols = ["Dependents", "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
], remainder='passthrough')


model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")

clf = Pipeline([
    ("pre", preprocessor),
    ("model", model)
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(clf, "model/xgb_model.pkl")
