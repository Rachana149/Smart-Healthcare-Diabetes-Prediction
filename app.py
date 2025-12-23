import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Smart Healthcare Diabetes Prediction",
    layout="wide"
)

st.title("🏥 Smart Healthcare Diabetes Prediction System")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

st.subheader("🧾 Enter Patient Details")


age = st.number_input("Age", 0, 120, 30)
bmi = st.number_input("BMI", 0.0, 50.0, 22.0)
hba1c = st.number_input("HbA1c Level", 0.0, 15.0, 5.0)
glucose = st.number_input("Blood Glucose Level", 0, 300, 100)

hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
smoking = st.selectbox(
    "Smoking History",
    ["never", "former", "current", "not current"]
)


input_df = pd.DataFrame(0, index=[0], columns=columns)


input_df["age"] = age
input_df["bmi"] = bmi
input_df["HbA1c_level"] = hba1c
input_df["blood_glucose_level"] = glucose
input_df["hypertension"] = hypertension
input_df["heart_disease"] = heart_disease


if gender != "Female":
    input_df[f"gender_{gender}"] = 1

if smoking != "never":
    input_df[f"smoking_history_{smoking}"] = 1


input_scaled = scaler.transform(input_df)


if st.button("🔍 Predict Disease"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Diabetes ({(1 - probability)*100:.2f}%)")


