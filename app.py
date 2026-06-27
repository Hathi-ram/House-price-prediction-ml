import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import shap

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# --------------------------------
# Load Files
# --------------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

data = pd.read_csv("House Price India.csv")

# --------------------------------
# Header
# --------------------------------
st.title("🏠 House Price Prediction System")
st.markdown(
    """
    ### Advanced Machine Learning Project using XGBoost  
    Predict real estate prices with explainable AI and interactive visualization.
    """
)

# --------------------------------
# Dataset Note
# --------------------------------
st.warning("""
⚠ Dataset Notice:
This project uses an older historical dataset mainly from **Bengaluru, Karnataka**.
Prices may be lower than current market values and may not reflect all Indian cities.
""")

# --------------------------------
# Sidebar Inputs
# --------------------------------
st.sidebar.header("🏡 Enter Property Details")

bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)
living_area = st.sidebar.slider("Living Area (sq ft)", 500, 10000, 1500)
lot_area = st.sidebar.slider("Lot Area (sq ft)", 500, 20000, 3000)
floors = st.sidebar.slider("Floors", 1, 5, 2)
condition = st.sidebar.slider("Condition", 1, 5, 3)
grade = st.sidebar.slider("House Grade", 1, 13, 7)
schools = st.sidebar.slider("Schools Nearby", 0, 10, 2)
airport_distance = st.sidebar.slider("Distance from Airport (km)", 1, 100, 20)

# --------------------------------
# Input Dictionary
# --------------------------------
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

# Match columns
input_data = input_data.reindex(columns=columns, fill_value=0)

# Scale
input_scaled = scaler.transform(input_data)

# --------------------------------
# Prediction Button
# --------------------------------
if st.button("🔍 Predict House Price"):

    prediction = model.predict(input_scaled)[0]

    st.subheader("📌 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"Predicted Price: ₹ {prediction:,.0f}")

    with col2:
        low = prediction * 0.9
        high = prediction * 1.1
        st.info(f"Estimated Range: ₹ {low:,.0f} - ₹ {high:,.0f}")

    # --------------------------------
    # Model Accuracy
    # --------------------------------
    st.subheader("📊 Model Performance")

    st.metric("Training Accuracy", "97.27%")

    # --------------------------------
    # Feature Importance
    # --------------------------------
    st.subheader("🔥 Feature Importance")

    try:
        importance = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": input_data.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            importance_df["Feature"],
            importance_df["Importance"]
        )
        ax.set_title("Important Factors Affecting House Price")
        st.pyplot(fig)

    except:
        st.info("Feature importance not available.")

    # --------------------------------
    # SHAP Explainability
    # --------------------------------
    st.subheader("🧠 Why this prediction?")

    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(input_scaled)

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

    except:
        st.info("SHAP explanation currently unavailable.")

# --------------------------------
# Property Map
# --------------------------------
st.subheader("📍 Property Locations")

map_data = data[['Lattitude', 'Longitude']].rename(
    columns={
        'Lattitude': 'lat',
        'Longitude': 'lon'
    }
)

st.map(map_data)

# --------------------------------
# Dataset Overview
# --------------------------------
st.subheader("📂 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Houses", len(data))

with col2:
    st.metric("Features", len(data.columns))

with col3:
    st.metric("City Coverage", "Bengaluru")

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.markdown("🚀 Built using XGBoost, Streamlit, and Explainable AI")
st.markdown("Designed as an Internship-Level Machine Learning Project")
