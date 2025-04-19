import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df = pd.read_csv("Training.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
model = RandomForestClassifier()
model.fit(X, y)

# Load symptom list
df = pd.read_csv("data/Training.csv")
symptom_list = df.columns[:-1].tolist()  # All columns except 'prognosis'

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

        prediction = model.predict([input_data])[0]
        st.success(f"✅ Predicted Disease: **{prediction}**")
