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
# Title
# --------------------------------
st.title("🏠 House Price Prediction System")
st.markdown("""
### Advanced Machine Learning Project using XGBoost  
Predict real estate prices with explainable AI and smart property analysis.
""")

# --------------------------------
# Project Note
# --------------------------------
st.info("""
ℹ **Project Note**

This house price prediction system is built using a historical housing dataset.  
As the dataset is based on older records, predicted prices may differ from current market values.  
The model is intended for educational and analytical purposes and may not fully represent all real estate markets or cities.
""")

# --------------------------------
# Sidebar Inputs
# --------------------------------
st.sidebar.header("🏡 Enter Property Details")

# Basic Info
st.sidebar.markdown("### Basic Information")

bedrooms = st.sidebar.selectbox(
    "Number of Bedrooms",
    [1,2,3,4,5,6,7,8,9,10]
)

bathrooms = st.sidebar.selectbox(
    "Number of Bathrooms",
    [1,2,3,4,5,6,7,8]
)

floors = st.sidebar.selectbox(
    "Number of Floors",
    [1,2,3,4,5]
)

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

# Property Quality
st.sidebar.markdown("### Property Quality")

condition = st.sidebar.slider(
    "Condition Rating (1 = Poor, 5 = Excellent)",
    1, 5, 3
)

grade = st.sidebar.slider(
    "Construction Grade (1 = Low, 13 = Premium)",
    1, 13, 7
)

# Nearby Facilities
st.sidebar.markdown("### Nearby Facilities")

schools = st.sidebar.slider(
    "Schools Nearby",
    0, 10, 2
)

airport_distance = st.sidebar.number_input(
    "Distance from Airport (km)",
    min_value=1,
    max_value=100,
    value=20
)

st.sidebar.markdown("---")
st.sidebar.info("""
Fill the property details and click **Predict House Price**  
to estimate the property value.
""")

# --------------------------------
# Input Data
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

# Match training columns
input_data = input_data.reindex(columns=columns, fill_value=0)

# Scale data
input_scaled = scaler.transform(input_data)

# --------------------------------
# Prediction Button
# --------------------------------
if st.button("🔍 Predict House Price"):

    prediction = model.predict(input_scaled)[0]

    # --------------------------------
    # Prediction Result
    # --------------------------------
    st.subheader("📌 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"Predicted Price: ₹ {prediction:,.0f}")

    with col2:
        low = prediction * 0.9
        high = prediction * 1.1
        st.info(f"Estimated Range: ₹ {low:,.0f} - ₹ {high:,.0f}")

    # --------------------------------
    # Model Performance
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
    # Similar Property Location
    # --------------------------------
    st.subheader("📍 Similar Property Location")

    try:
        data["difference"] = (
            abs(data["number of bedrooms"] - bedrooms) +
            abs(data["number of bathrooms"] - bathrooms) +
            abs(data["living area"] - living_area) +
            abs(data["lot area"] - lot_area)
        )

        nearest_house = data.sort_values("difference").head(1)

        map_data = nearest_house[['Lattitude', 'Longitude']].rename(
            columns={
                "Lattitude": "lat",
                "Longitude": "lon"
            }
        )

        st.map(map_data)

        st.success("Showing the nearest similar property location.")

    except:
        st.info("Location data not available.")

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
    st.metric("Model Type", "XGBoost")

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.markdown("🚀 Built using XGBoost, Streamlit, and Explainable AI")
st.markdown("Designed as an Internship-Level Machine Learning Project")
