import streamlit as st
import pickle
import numpy as np
import pandas as pd


st.sidebar.title("Heart Disease Prediction App")
html_temp = """
<div style="background-color:blue;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit ML Cloud Heart Disease project </h2>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)
st.markdown("""
This application predicts the likelihood of heart disease based on user inputs. 
Please adjust the sliders or selections to match the individual's profile.
""")


# ---- Load Model ----
model_path = "bestRF_model.pkl"
with open(model_path, "rb") as f:
    heart_model = pickle.load(f)


# Collect User Inputs
age_mapping = {
    'Age 18 to 24': 0,
    'Age 25 to 29': 1,
    'Age 30 to 34': 2,
    'Age 35 to 39': 3,
    'Age 40 to 44': 4,
    'Age 45 to 49': 5,
    'Age 50 to 54': 6,
    'Age 55 to 59': 7,
    'Age 60 to 64': 8,
    'Age 65 to 69': 9,
    'Age 70 to 74': 10,
    'Age 75 to 79': 11,
    'Age 80 or older': 12
}

HadAngina = st.selectbox("Have you had angina?", [0, 1])
selected_age_label = st.selectbox(
    "Select your age range:",
    list(age_mapping.keys())  # ["Age 18 to 24", "Age 25 to 29", ...]
)
age = age_mapping[selected_age_label]
ChestScan = st.selectbox("Abnormal chest scan? (0=No, 1=Yes)", [0, 1])
gen_health = st.selectbox("General Health (0=Excellent → 4=Poor)", [0, 1, 2, 3, 4])
DifficultyWalking = st.selectbox("Difficulty Walking? (0=No, 1=Yes)", [0, 1])
HadStroke = st.selectbox("Have you had a stroke?", [0, 1])
BMI = st.slider("BMI (Body Mass Index)", 12, 98, step = 2)

# Create dataframe from model input
input_data = pd.DataFrame({
    "HadAngina": [HadAngina],
    "age": [age],
    "ChestScan": [ChestScan],
    "gen_health": [gen_health],
    "DifficultyWalking": [DifficultyWalking],
    "HadStroke": [HadStroke],
    "BMI": [BMI]
})

st.write("### Your Input Summary")
st.write(input_data)

 
if st.button("Predict Heart Disease"):
    prob_heart_disease = heart_model.predict_proba(input_data)[0][1]
    st.success(f"Predicted probability of Heart Disease: {prob_heart_disease:.2f}")
    if prob_heart_disease >= 0.5:
        st.error("**Model indicates a HIGH likelihood of heart disease.**")
    else:
        st.success("**Model indicates a LOW likelihood of heart disease.**")

# Write file to directory with saved model
