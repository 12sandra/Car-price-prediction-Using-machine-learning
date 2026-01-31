# 🚗 Car Price Prediction Web App

A **Machine Learning–powered web application** that predicts the **resale price of a car** based on user inputs such as present price, kilometers driven, age, fuel type, seller type, and transmission.  
The app is built using **Python, Flask, and Scikit-learn** with a clean **dark-themed UI**.

---

## 🔍 Project Overview

Predicting car resale prices is a real-world regression problem.  
This project demonstrates an **end-to-end ML workflow**, including:

- Data preprocessing & feature engineering
- Model training and evaluation
- Hyperparameter tuning
- Model serialization using Pickle
- Deployment using Flask
- User-friendly web interface

---

## 🧠 Machine Learning Model

- **Algorithm Used:** Random Forest Regressor (Hyperparameter Tuned)
- **Target Variable:** `Selling_Price`
- **Evaluation Metric:** R² Score ≈ 0.95
- **Feature Scaling:** StandardScaler

### 📊 Model Input Features (in order)

['Present_Price',
'Kms_Driven',
'Owner',
'age',
'Fuel_Type_Diesel',
'Fuel_Type_Petrol',
'Seller_Type_Individual',
'Transmission_Manual']

👩‍💻 Author

Sandra A S
B.Tech Computer Science Graduate
Aspiring Data Scientist 
