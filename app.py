import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("student_model.pkl")

st.title("Student Performance Predictor")

# Input fields
gender = st.selectbox("Gender", [0, 1])  # 0 = female, 1 = male
race_ethnicity = st.selectbox("Race/Ethnicity", [0, 1, 2, 3, 4])
parent_edu = st.selectbox("Parental Education Level", [0, 1, 2, 3, 4, 5])
lunch = st.selectbox("Lunch", [0, 1])
test_prep = st.selectbox("Test Preparation", [0, 1])

# Predict button
if st.button("Predict"):
    input_data = np.array([[gender, race_ethnicity, parent_edu, lunch, test_prep]])
    prediction = model.predict(input_data)
    result = "PASS ✅" if prediction[0] == 1 else "FAIL ❌"
    st.success(f"The student is predicted to: **{result}**")
