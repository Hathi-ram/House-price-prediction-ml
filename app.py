import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ----------------------------
# Load Model and Data
# ----------------------------
model = pickle.load(open("xgb_model.pkl", "rb"))
data = pd.read_csv("House Price India.csv")

# ----------------------------
# Title
# ----------------------------
st.title("🏠 House Price Prediction using Machine Learning")
st.markdown("### Predict house prices with advanced XGBoost model")

# ----------------------------
# Important Dataset Note
# ----------------------------
st.warning("""
⚠ **Important Note**

This project uses an older historical housing dataset.

Dataset limitations:
- Mainly covers **Bengaluru, Karnataka**
- Does not represent all Indian cities
- Prices may appear lower than today’s market rates
- Predictions are based on historical data only
""")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Enter House Details")

bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)
living_area = st.sidebar.slider("Living Area (sq ft)", 500, 10000, 1500)
lot_area = st.sidebar.slider("Lot Area (sq ft)", 500, 20000, 3000)
floors = st.sidebar.slider("Floors", 1, 5, 2)
condition = st.sidebar.slider("Condition", 1, 5, 3)
grade = st.sidebar.slider("Grade", 1, 13, 7)
schools = st.sidebar.slider("Schools Nearby", 0, 10, 2)
airport_distance = st.sidebar.slider("Distance from Airport (km)", 1, 100, 20)

# ----------------------------
# Prediction
# ----------------------------
input_data = pd.DataFrame({
    'number of bedrooms': [bedrooms],
    'number of bathrooms': [bathrooms],
    'living area': [living_area],
    'lot area': [lot_area],
    'number of floors': [floors],
    'condition of the house': [condition],
    'grade of the house': [grade],
    'Number of schools nearby': [schools],
    'Distance from the airport': [airport_distance]
})

prediction = model.predict(input_data)[0]

# ----------------------------
# Display Prediction
# ----------------------------
st.subheader("Predicted House Price")

st.success(f"₹ {prediction:,.0f}")

# Price Range
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
# Feature Importance (Best Visualization)
# ----------------------------
st.subheader("Feature Importance")

feature_names = input_data.columns
importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance_df["Feature"], importance_df["Importance"])
ax.set_title("Most Important Features Affecting House Price")
st.pyplot(fig)

# ----------------------------
# Location Map
# ----------------------------
st.subheader("Property Location Map")

st.map(data[['Lattitude', 'Longitude']])

# ----------------------------
# Dataset Overview
# ----------------------------
st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric("Total Houses", len(data))
col2.metric("Total Features", len(data.columns))
col3.metric("City Coverage", "Bengaluru")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("Built for Machine Learning Project")
st.markdown("Developed using XGBoost, Streamlit, and Real Estate Data Analysis")

import shap

st.subheader("Why this price? (SHAP Explainability)")

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(
        shap_values[0],
        show=False
    )

    st.pyplot(fig)

except Exception as e:
    st.error("SHAP explanation could not be generated.")
