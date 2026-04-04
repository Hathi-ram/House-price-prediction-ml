import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# =========================
# LOAD FILES
# =========================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Load dataset (for graphs)
data = pd.read_csv("House Price India.csv")

st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title(" Advanced House Price Prediction System")

# =========================
# INPUT SECTION
# =========================
st.sidebar.header("Enter House Details")

living_area = st.sidebar.slider("Living Area", 500, 10000, 1500)
lot_area = st.sidebar.slider("Lot Area", 500, 20000, 3000)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)
floors = st.sidebar.selectbox("Floors", [1, 2, 3])

condition = st.sidebar.selectbox("Condition", [1, 2, 3, 4, 5])
grade = st.sidebar.selectbox("Grade", [1, 2, 3, 4, 5])

schools = st.sidebar.slider("Schools Nearby", 0, 10, 2)
distance_airport = st.sidebar.slider("Distance from Airport", 1, 50, 10)

lat = st.sidebar.slider("Latitude", -90.0, 90.0, 12.97)
lon = st.sidebar.slider("Longitude", -180.0, 180.0, 77.59)

# =========================
# CREATE INPUT DATA
# =========================
input_dict = {col: 0 for col in columns}

# Fill values
if 'living area' in input_dict:
    input_dict['living area'] = living_area

if 'lot area' in input_dict:
    input_dict['lot area'] = lot_area

if 'number of bedrooms' in input_dict:
    input_dict['number of bedrooms'] = bedrooms

if 'number of bathrooms' in input_dict:
    input_dict['number of bathrooms'] = bathrooms

if 'number of floors' in input_dict:
    input_dict['number of floors'] = floors

if 'condition of the house' in input_dict:
    input_dict['condition of the house'] = condition

if 'grade of the house' in input_dict:
    input_dict['grade of the house'] = grade

if 'Number of schools nearby' in input_dict:
    input_dict['Number of schools nearby'] = schools

if 'Distance from the airport' in input_dict:
    input_dict['Distance from the airport'] = distance_airport

if 'Lattitude' in input_dict:
    input_dict['Lattitude'] = lat

if 'Longitude' in input_dict:
    input_dict['Longitude'] = lon

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# =========================
# SCALE INPUT
# =========================
input_scaled = scaler.transform(input_df)

# =========================
# PREDICTION
# =========================
st.subheader(" Prediction")

if st.button("Predict Price"):
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")

# =========================
# MAP
# =========================
st.subheader(" Location Map")

map_data = pd.DataFrame({
    'lat': [lat],
    'lon': [lon]
})
st.map(map_data)

# =========================
# GRAPHS
# =========================
st.subheader(" Data Visualizations")

if st.checkbox("Show Price Distribution"):
    fig, ax = plt.subplots()
    sns.histplot(data['Price'], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

if st.checkbox("Show Living Area vs Price"):
    fig, ax = plt.subplots()
    ax.scatter(data['living area'], data['Price'])
    ax.set_xlabel("Living Area")
    ax.set_ylabel("Price")
    st.pyplot(fig)

if st.checkbox("Show Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(data.corr(numeric_only=True), ax=ax)
    st.pyplot(fig)

# =========================
# SHAP EXPLANATION
# =========================
st.subheader(" Model Explainability (SHAP)")

try:
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    if st.button("Explain Prediction"):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.plots.waterfall(shap_values[0])
        st.pyplot(bbox_inches='tight')

except:
    st.warning("SHAP not supported for this model")

# =========================
# END
# =========================
st.write("🚀 Built for Internship-Level ML Project")
