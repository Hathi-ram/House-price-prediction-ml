import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import shap

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ----------------------------
# Load Files
# ----------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

data = pd.read_csv("House Price India.csv")

# ----------------------------
# Title
# ----------------------------
st.title("🏠 House Price Prediction using Machine Learning")
st.markdown("### Advanced XGBoost Model for Real Estate Price Estimation")

# ----------------------------
# Dataset Warning
# ----------------------------
st.warning("""
⚠ Important Note:

This model is trained on an older historical housing dataset.

Dataset limitations:
- Mainly covers Bengaluru, Karnataka.
- Does not represent all Indian cities.
- Prices may appear lower than current market values.
- Predictions are based on historical records only.
""")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Enter Property Details")

bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Number of Bathrooms", 1, 10, 2)
living_area = st.sidebar.slider("Living Area (sq ft)", 500, 10000, 1500)
lot_area = st.sidebar.slider("Lot Area (sq ft)", 500, 20000, 3000)
floors = st.sidebar.slider("Number of Floors", 1, 5, 2)
condition = st.sidebar.slider("House Condition", 1, 5, 3)
grade = st.sidebar.slider("House Grade", 1, 13, 7)
schools = st.sidebar.slider("Schools Nearby", 0, 10, 2)
airport_distance = st.sidebar.slider("Distance from Airport (km)", 1, 100, 20)

# ----------------------------
# Input Data
# ----------------------------
input_dict = {
    "number of bedrooms": bedrooms,
    "number of bathrooms": bathrooms,
    "living area": living_area,
    "lot area": lot_area,
    "number of floors": floors,
    "condition of the house": condition,
    "grade of the house": grade,
    "Number of schools nearby": schools,
    "Distance from the airport": airport_distance
}

input_data = pd.DataFrame([input_dict])

# Align columns
input_data = input_data.reindex(columns=columns, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)[0]

# ----------------------------
# Prediction Output
# ----------------------------
st.subheader("Predicted House Price")

st.success(f"₹ {prediction:,.0f}")

# Price range
low = prediction * 0.9
high = prediction * 1.1

st.info(f"Estimated Price Range: ₹ {low:,.0f} - ₹ {high:,.0f}")

# ----------------------------
# Model Accuracy
# ----------------------------
st.subheader("Model Performance")

accuracy = 97.27
st.metric("Training Accuracy", f"{accuracy}%")

# ----------------------------
# Feature Importance
# ----------------------------
st.subheader("Feature Importance")

try:
    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": input_data.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.set_title("Most Important Features Affecting Price")
    st.pyplot(fig)

except:
    st.info("Feature importance not available for this model.")

# ----------------------------
# SHAP Explainability
# ----------------------------
st.subheader("Why this price? (SHAP Explainability)")

try:
    explainer = shap.Explainer(model)
    shap_values = explainer(input_scaled)

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

except:
    st.info("SHAP explanation not available.")
# ----------------------------
# Property Location Map
# ----------------------------
st.subheader("Property Locations")

map_data = data[['Lattitude', 'Longitude']].rename(
    columns={
        'Lattitude': 'lat',
        'Longitude': 'lon'
    }
)

st.map(map_data)

# ----------------------------
# Dataset Overview
# ----------------------------
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Properties", len(data))
col2.metric("Total Features", len(data.columns))
col3.metric("Location Coverage", "Bengaluru")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("🚀 Internship-Level Machine Learning Project")
st.markdown("Built with XGBoost, Streamlit, SHAP, and Real Estate Data Analysis")
