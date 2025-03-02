import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained classifier and scaler using joblib
classifier = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the prediction function
def predict_survival(d):
    sample_data = pd.DataFrame([d])
    scaled_data = scaler.transform(sample_data)
    pred = classifier.predict(scaled_data)[0]
    prob = classifier.predict_proba(scaled_data)[0][pred]
    return pred, prob

# Streamlit UI components
st.title("dibatic or not Prediction")

# Input fields for each parameter
Pregnancies = st.number_input("Pregnancies", min_value=0.0, max_value=100.0, value=1.0)
BloodPressure = st.number_input("BloodPressure",min_value=50.0, max_value=198.0, value=50.0)
SkinThickness = st.number_input("SkinThickness", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
Insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=1.0)
BMI = st.number_input("BMI", min_value=0.0, max_value=100.0, value=1.0)
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=500.0, value=7.25, step=0.1)
Age = st.number_input("Age",min_value=0.0, max_value=100.0, value=50.0, step=0.1)
                        

# Create the input dictionary for prediction
input_data = {
    'Pregnancies': Pregnancies,
    'BloodPressure': BloodPressure,
    'SkinThickness': SkinThickness,
    'Insulin': Insulin,
    'BMI': BMI,
    'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
    'Age':Age
}

# When the user clicks the "Predict" button
if st.button("Predict"):
    with st.spinner('Making prediction...'):
        pred, prob = predict_survival(input_data)

        if pred == 1:
            # Dibatic
            st.success(f"Prediction: Dibatic with probability {prob:.2f}")
        else:
            # Non Dibatic
            st.error(f"Prediction: Does Not Suffer With Dibaties with probability {prob:.2f}")
