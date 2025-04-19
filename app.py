import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load trained model
with open("model/nabeeh_model.pkl", "rb") as f:
    model = pickle.load(f)


# Load symptom list from training data
df = pd.read_csv("data/Training.csv")
symptom_list = df.columns[:-1].tolist()  # All columns except 'prognosis'

# Streamlit UI setup
st.set_page_config(page_title="Nabeeh - Disease Prediction", page_icon="🧠", layout="centered")
st.title("🧠 Nabeeh - Intelligent Diagnosis Assistant")
st.markdown("Select the symptoms you're experiencing, and Nabeeh will predict the most likely disease.")

# User symptom selection
selected_symptoms = st.multiselect("Select symptoms:", options=symptom_list)

# Predict disease
if st.button("🔍 Predict Disease"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        # Create binary input array
        input_data = np.zeros(len(symptom_list))
        for symptom in selected_symptoms:
            index = symptom_list.index(symptom)
            input_data[index] = 1
        input_data = input_data.reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]
        st.success(f"✅ Predicted Disease: **{prediction}**")
