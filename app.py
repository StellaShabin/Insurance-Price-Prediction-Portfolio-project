import streamlit as st
import numpy as np
import joblib

# Load your pre-trained model
model = joblib.load('premium_price_model.pkl')

st.header("Insurance Price Prediction", divider="gray")

# Define input fields
age = st.number_input("Age", min_value=0)
diabetes = st.selectbox("Diabetes", [0, 1])
bp_problems = st.selectbox("Blood Pressure Problems", [0, 1])
transplants = st.selectbox("Any Transplants", [0, 1])
chronic_diseases = st.selectbox("Any Chronic Diseases", [0, 1])
height = st.number_input("Height in cm", min_value=0.0)
weight = st.number_input("Weight in kg", min_value=0.0)
allergies = st.selectbox("Known Allergies", [0, 1])
cancer_history = st.selectbox("History Of Cancer In Family", [0, 1])
num_surgeries = st.number_input("Number Of Major Surgeries", min_value=0)

# Calculate derived features
if height > 0:
    bmi = weight / ((height / 100) ** 2)  # Calculate BMI
    
else:
    st.error("Please enter a valid height greater than 0 to calculate BMI.")
    bmi = 0 

if bmi > 0:    
    bmi_squared = bmi ** 2
    bmi_age = bmi * age
    bmi_num_surgeries = bmi * num_surgeries
    age_squared = age ** 2
    age_num_surgeries = age * num_surgeries
    num_surgeries_squared = num_surgeries ** 2

    # Combine inputs to match model’s expected feature order
    input_data = np.array([
    diabetes, bp_problems, transplants, chronic_diseases, height, weight,
    allergies, cancer_history, bmi_squared, bmi_age, bmi_num_surgeries,
    age_squared, age_num_surgeries, num_surgeries_squared
    ]).reshape(1, -1)  # Reshape to match model's input format

    # Check that input_data has the correct number of features
    if input_data.shape[1] == 14:
        # Make prediction
        prediction = model.predict(input_data)
        if st.button("Predict Price", type="primary"):
            st.write(f"Predicted Premium Price: ${prediction[0]:.2f}")
    else:
        st.write("Error: Feature count mismatch. Please check input processing.")
else:
    st.write("BMI could not be calculated due to invalid height/Weight.")
