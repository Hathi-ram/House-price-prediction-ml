##  House Price Prediction using Machine Learning

## Live Demo
 [https://your-app-link.streamlit.app](https://house-price-prediction-ml-app.streamlit.app/)

An interactive machine learning web application that predicts house prices based on property characteristics using an **XGBoost Regression model**.

---

## Problem Statement

House prices depend on several factors such as property size, number of bedrooms and bathrooms, construction quality, nearby facilities, and other property characteristics.

The objective of this project is to build a machine learning system that can:

* Predict house prices from property details
* Identify the factors that influence house prices
* Explain individual model predictions
* Provide an easy-to-use interface for real-time prediction

---

##  Solution

I developed an end-to-end machine learning solution using **XGBoost**.

The project includes:

1. Data preprocessing
2. Exploratory Data Analysis
3. Feature preparation
4. XGBoost model training
5. Model evaluation
6. Feature importance analysis
7. SHAP-based explainability
8. Streamlit web application deployment
9. Similar-property location visualization
10. Google Maps integration

---
## Methodology

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/d3c626a1-d8d0-415a-93a3-f684cf41af0c" />


## Model Performance

**Model:** XGBoost Regressor
**Training Accuracy (R²):** **97.27%**
**Train-Test Split:** 80% Training / 20% Testing
**Problem Type:** Regression

The model learns relationships between property features and historical house prices to generate price predictions.

---

##  Important Features

The application allows users to enter:

* Number of bedrooms
* Number of bathrooms
* Number of floors
* Living area
* Lot area
* House condition
* Construction grade
* Number of schools nearby
* Distance from airport

---

## Explainable AI

A major part of this project is understanding **why the model produces a particular prediction**.

I incorporated:

* Feature Importance
* SHAP Explainability

These techniques help identify which property characteristics have the greatest influence on the predicted price and make the machine learning model more transparent.

---

##  Location Feature

The application also provides a **similar-property reference location** using latitude and longitude available in the dataset.

Users can:

* View the reference property on the map
* Open the location in Google Maps
* Request directions when a valid route is available

The location represents a similar property from the available dataset and is **not a recommendation of a property for purchase**.

---

##  Web Application

The model is deployed using **Streamlit**, allowing users to interact with the trained model without running Python code.

Users can enter property information and receive:

**Input → ML Model → Predicted Price → Explanation → Location Reference**

---

## Important Dataset Limitation

This project uses an **older historical housing dataset**.

Therefore:

* Predicted prices may be lower or different from current market prices.
* The model should not be treated as a real-time property valuation system.
* The available location information depends on the original dataset.
* The model may not accurately represent every city or current real-estate market.

This limitation was identified during project development and is clearly communicated in the deployed application.

---

## 🛠 Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* SHAP
* Streamlit

---

##  Project Structure

```text
house-price-prediction-ml/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── columns.pkl
├── House Price India.csv
├── requirements.txt
└── README.md
```

---

##  Run Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

##  Key Learning Outcomes

Through this project, I gained practical experience in:

* Data preprocessing
* Exploratory Data Analysis
* Regression modeling
* XGBoost
* Model evaluation
* Feature importance
* SHAP Explainable AI
* Streamlit development
* Model deployment
* Handling real-world dataset limitations
* Integrating machine learning with location-based visualization

---

##  Author

**Vislavath Hathiram**
B.Tech – Smart Manufacturing Engineering
IIITDM Kancheepuram

GitHub: https://github.com/Hathi-ram
LinkedIn: https://linkedin.com/in/vislavathhathiram
