import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("student_pass_model.pkl")

st.set_page_config(page_title="Student Pass Predictor", layout="centered")
st.title("🎓 Student Pass/Fail Predictor")
st.write("Enter the student's details below to check if they are likely to pass.")

# User Inputs
gender = st.selectbox("Gender", ["female", "male"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
education = st.selectbox("Parental Level of Education", [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
prep = st.selectbox("Test Preparation Course", ["none", "completed"])

# Encoding manually based on training LabelEncoder
encode = lambda val, options: options.index(val)

gender_encoded = encode(gender, ["female", "male"])
race_encoded = encode(race, ["group A", "group B", "group C", "group D", "group E"])
education_encoded = encode(education, [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch_encoded = encode(lunch, ["free/reduced", "standard"])
prep_encoded = encode(prep, ["none", "completed"])

# Prediction
features = np.array([[gender_encoded, race_encoded, education_encoded, lunch_encoded, prep_encoded]])

if st.button("Predict"):
    result = model.predict(features)[0]
    if result == 1:
        st.success("✅ The student is likely to PASS.")
    else:
        st.error("❌ The student is likely to FAIL.")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Scikit-learn")
