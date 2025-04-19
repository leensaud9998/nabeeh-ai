import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load model
with open("nabeeh_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load training data to get symptom list and for fitting scaler and PCA
df = pd.read_csv("Training.csv")
symptom_list = df.columns[:-1].tolist()
X = df.drop(columns='prognosis')

# Setup scaler and PCA exactly as in training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=65)
X_pca = pca.fit_transform(X_scaled)

# Streamlit UI
st.set_page_config(page_title="Nabeeh - Disease Prediction", page_icon="🧠", layout="centered")
st.title("🧠 Nabeeh - Intelligent Diagnosis Assistant")
st.markdown("Select the symptoms you're experiencing, and Nabeeh will predict the most likely disease.")

# User input
selected_symptoms = st.multiselect("Select symptoms:", options=symptom_list)

# Predict
if st.button("🔍 Predict Disease"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        # Convert selected symptoms to input vector
        input_data = np.zeros(len(symptom_list))
        for symptom in selected_symptoms:
            index = symptom_list.index(symptom)
            input_data[index] = 1

        # Apply same preprocessing: scale + PCA
        input_scaled = scaler.transform([input_data])
        input_pca = pca.transform(input_scaled)

        # Predict
        prediction = model.predict(input_pca)[0]
        st.success(f"✅ Predicted Disease: **{prediction}**")
