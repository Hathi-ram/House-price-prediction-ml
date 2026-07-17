####  House Price Prediction using Machine Learning

## Live Demo
 [https://your-app-link.streamlit.app](https://house-price-prediction-ml-app.streamlit.app/)

## Experience the live House Price Prediction System built with Machine Learning and Streamlit.

This interactive web application allows users to:

Enter property details such as bedrooms, bathrooms, living area, lot area, and house quality
Get real-time predicted house prices
View estimated price ranges
Analyze feature importance affecting house prices
Understand predictions using SHAP explainability
Explore similar property reference locations on Google Maps
Access direct property location view and navigation support

## Overview

This project predicts house prices using advanced machine learning techniques.
It leverages XGBoost Regressor for high prediction accuracy and integrates Explainable AI (SHAP) for transparent predictions.

The project demonstrates a complete machine learning pipeline including:

Data preprocessing

Feature engineering

Exploratory Data Analysis (EDA)

Model training

Model deployment using Streamlit

## Features
Real-time house price prediction
Interactive web application using Streamlit
Professional property input interface
Feature importance visualization
SHAP model explainability
Similar property location mapping
Google Maps integration for property viewing and navigation
Dataset insights and model performance display.

## Technologies Used

Python

Pandas

NumPy

Scikit-learn

XGBoost

Streamlit

Matplotlib

SHAP

## Model Details
Model Used: XGBoost Regressor

Model Accuracy: 97.27%

Train-Test Split: 80% Training, 20% Testing

Prediction Type: Regression

Scaling: StandardScaler

## Input Features

The model uses the following property features:

Number of Bedrooms

Number of Bathrooms

Number of Floors

Living Area

Lot Area

Condition of House

Grade of House

Number of Schools Nearby

Distance from Airport

## Installation & Run Locally

Clone the repository:

git clone https://github.com/Hathi-ram/house-price-prediction-ml.git

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

## Screenshots

Add screenshots of:

Home Page

Prediction Result

Feature Importance

SHAP Explainability

Similar Property Location

## Important Note

This model is trained on historical housing data.
Predicted prices may differ from current market values.

The location shown in the application is based on the nearest similar property from the dataset and is provided for reference purposes only.

## Author

Vislavath Hathiram
B.Tech –IIITDM Kancheepuram

GitHub: Hathi-ram GitHub
LinkedIn: Vislavath Hathiram LinkedIn
