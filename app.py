import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("nabeeh_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load symptom list from the training CSV
df = pd.read_csv("Training.csv")  # Make sure it's in the same folder or update the path
symptom_list = df.columns[:-1].tolist()  # All columns except the last one (prognosis)

# Streamlit App UI
st.set_page_config(page_title="Nabeeh - Disease Prediction", page_icon="🧠", layout="centered")
st.title("🧠 Nabeeh - Intelligent Diagnosis Assistant")
st.markdown("Select the symptoms you're experiencing, and Nabeeh will predict the most likely disease.")

# User input
selected_symptoms = st.multiselect("Select symptoms:", options=symptom_list)

# Predict button
if st.button("🔍 Predict Disease"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        input_data = np.zeros(len(symptom_list))
        for symptom in selected_symptoms:
            index = symptom_list.index(symptom)
            input_data[index] = 1
        input_data = input_data.reshape(1, -1) 
        prediction = model.predict(input_data)[0]
        st.success(f"✅ Predicted Disease: **{prediction}**")
