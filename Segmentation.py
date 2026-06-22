import streamlit as st
import pandas as pd
import numpy as np
import joblib

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Segmentation App")
st.write("Enter customer details to predict the segment.")

Age = st.number_input("Age", min_value=18, max_value=100, value=35)
Income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
Total_Spending = st.number_input("Total Spending (sum of purchases)", min_value=0, max_value=5000, value=1000)
NumWebPurchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
NumStorePurchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10)
NumWebVisitsMonth = st.number_input("Number of Web Visits per Month", min_value=0, max_value=50, value=3)
Recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

input_data = pd.DataFrame({
    "Age": [Age],
    "Income": [Income],
    "Total_Spending": [Total_Spending],
    "NumWebPurchases": [NumWebPurchases],
    "NumStorePurchases": [NumStorePurchases],
    "NumWebVisitsMonth": [NumWebVisitsMonth],
    "Recency": [Recency]
})

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    st.success(f"Predicted Segment: Cluster {cluster}")