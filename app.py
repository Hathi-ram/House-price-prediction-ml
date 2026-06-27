import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import shap
import urllib.parse

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="House Price Prediction System",
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
# Title
# --------------------------------
st.title("🏠 House Price Prediction System")

st.markdown("""
### Advanced Machine Learning Project using XGBoost  
Predict property prices using intelligent machine learning models, feature analysis, and location support.
""")

# --------------------------------
# Project Note
# --------------------------------
st.info("""
ℹ **Project Note**

This application predicts house prices using a historical housing dataset and machine learning techniques.

### Key Highlights:
✔ Built using **XGBoost Regressor**  
✔ Achieved **97.27% training accuracy**  
✔ Uses property features like bedrooms, bathrooms, area, quality, and nearby facilities  
✔ Includes **Feature Importance Analysis**  
✔ Includes **SHAP Explainability**  
✔ Includes **Real Location Navigation Support**

**Important:**  
Since the model is trained on historical records, predicted prices may differ from current market values.
""")

# --------------------------------
# Sidebar Inputs
# --------------------------------
st.sidebar.header("🏡 Property Details")

# Location Details
st.sidebar.markdown("### Location Details")

city = st.sidebar.selectbox(
    "Select City",
    ["Hyderabad", "Mumbai", "Chennai", "Bangalore", "Delhi"]
)

area_name = st.sidebar.text_input(
    "Enter Area / Locality",
    "Gachibowli"
)

# Basic Information
st.sidebar.markdown("### Basic Information")

bedrooms = st.sidebar.selectbox("Bedrooms", [1,2,3,4,5,6,7,8,9,10])
bathrooms = st.sidebar.selectbox("Bathrooms", [1,2,3,4,5,6,7,8])
floors = st.sidebar.selectbox("Floors", [1,2,3,4,5])

# Property Size
st.sidebar.markdown("### Property Size")

living_area = st.sidebar.number_input(
    "Living Area (sq ft)",
    min_value=500,
    max_value=10000,
    value=1500,
    step=100
)

lot_area = st.sidebar.number_input(
    "Lot Area (sq ft)",
    min_value=500,
    max_value=50000,
    value=3000,
    step=100
)

# Quality
st.sidebar.markdown("### Property Quality")

condition = st.sidebar.slider("House Condition (1-5)", 1, 5, 3)
grade = st.sidebar.slider("House Grade (1-13)", 1, 13, 7)

# Nearby Facilities
st.sidebar.markdown("### Nearby Facilities")

schools = st.sidebar.slider("Schools Nearby", 0, 10, 2)

airport_distance = st.sidebar.number_input(
    "Distance from Airport (km)",
    min_value=1,
    max_value=100,
    value=20
)

st.sidebar.markdown("---")
st.sidebar.info("""
Enter all details and click **Predict House Price**
to estimate property value.
""")

# --------------------------------
# Prepare Input
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

input_data = input_data.reindex(columns=columns, fill_value=0)
input_scaled = scaler.transform(input_data)

# --------------------------------
# Predict Button
# --------------------------------
if st.button("🔍 Predict House Price"):

    prediction = model.predict(input_scaled)[0]

    # Prediction Result
    st.subheader("📌 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"Predicted Price: ₹ {prediction:,.0f}")

    with col2:
        low = prediction * 0.9
        high = prediction * 1.1
        st.info(f"Estimated Range: ₹ {low:,.0f} - ₹ {high:,.0f}")

    # Location Display
    st.subheader("📍 Selected Location")

    st.write(f"**City:** {city}")
    st.write(f"**Area / Locality:** {area_name}")

    # Google Maps Link (real user location)
    full_location = f"{area_name}, {city}"
    encoded_location = urllib.parse.quote(full_location)

    google_maps_url = (
        f"https://www.google.com/maps/search/?api=1&query={encoded_location}"
    )

    st.markdown(
        f"[🗺 Open Selected Location in Google Maps]({google_maps_url})"
    )

    # Model Performance
    st.subheader("📊 Model Performance")
    st.metric("Training Accuracy", "97.27%")

    # Feature Importance
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
        st.info("Feature importance unavailable.")

    # SHAP Explainability
    st.subheader("🧠 Prediction Explainability")

    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(input_scaled)

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

    except:
        st.warning("SHAP explainability currently unavailable.")

# --------------------------------
# Dataset Overview
# --------------------------------
st.subheader("📂 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Properties", len(data))

with col2:
    st.metric("Total Features", len(data.columns))

with col3:
    st.metric("Model Used", "XGBoost")

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.markdown("🚀 Built using XGBoost, Streamlit, SHAP, and Google Maps")
st.markdown("Designed as an Advanced Internship-Level Machine Learning Project")
