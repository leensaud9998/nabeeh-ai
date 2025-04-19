import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load trained model
with open("nabeeh_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load training data (for fitting scaler and PCA)
df = pd.read_csv("Training.csv")
symptom_list = df.columns[:-1].tolist()
X = df.drop(columns='prognosis')

# Handle missing values if any
X.fillna(0, inplace=True)

# Setup scaler and PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=65)
X_pca = pca.fit_transform(X_scaled)

# Streamlit UI
st.set_page_config(page_title="Nabeeh - Chronic Disease Prediction", page_icon="🧠", layout="centered")
st.title("Nabeeh 🧠 - How much is knowing early worth to you?")
st.markdown("Select the symptoms you're experiencing, and Nabeeh will predict the most likely disease.")

# User Input
selected_symptoms = st.multiselect("Select symptoms:", options=symptom_list)

# Predict Button
# Predict Button
if st.button("🔍 Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom! ⚠️")
    else:
        input_data = np.zeros(len(symptom_list))
        for symptom in selected_symptoms:
            index = symptom_list.index(symptom)
            input_data[index] = 1

        # Apply same preprocessing
        input_scaled = scaler.transform([input_data])
        input_pca = pca.transform(input_scaled)

        # Get prediction probabilities
        proba = model.predict_proba(input_pca)[0]
        top_indices = np.argsort(proba)[::-1][:10]  # Top 3

        st.success("Top 10 Predicted Diseases:")
        for idx in top_indices:
            disease = model.classes_[idx]
            probability = proba[idx] * 100
            st.markdown(f"- **{disease}**: {probability:.2f}%")
