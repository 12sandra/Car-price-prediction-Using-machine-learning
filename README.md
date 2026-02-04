#  Car Price Prediction Web App



A **Machine Learning–powered web application** that predicts the **resale price of a car** based on user inputs such as present price, kilometers driven, age, fuel type, seller type, and transmission.  
The app is built using **Python, Flask, and Scikit-learn** with a clean **dark-themed UI**.

---
## 🎥 Project Demo




https://github.com/user-attachments/assets/1713e6f3-f914-4880-bcdb-71e7f740cdbb

Python anywhere website Link : https://mlpredictionsandra.pythonanywhere.com/

---

##  Project Overview

Predicting car resale prices is a real-world regression problem.  
This project demonstrates an **end-to-end ML workflow**, including:

- Data preprocessing & feature engineering
- Model training and evaluation
- Hyperparameter tuning
- Model serialization using Pickle
- Deployment using Flask
- User-friendly web interface

---

##  Technologies Used

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Matplotlib, Seaborn**
- **Flask**
- **HTML / CSS**
- **VS Code**

---

## Machine Learning Models Used

- Linear Regression  
- Random Forest Regressor (Base Model)  
- Random Forest Regressor (Hyperparameter Tuned)

---


##  Machine Learning Model

- **Algorithm Used:** Random Forest Regressor (Hyperparameter Tuned)
- **Target Variable:** `Selling_Price`
- **Evaluation Metric:** R² Score ≈ 0.95
- **Feature Scaling:** StandardScaler

- ###Result

- <img width="379" height="153" alt="result_screenshot" src="https://github.com/user-attachments/assets/05aaaddc-a822-4942-99fa-3a858775f5d2" />


### 📊 Model Input Features (in order)

['Present_Price',
'Kms_Driven',
'Owner',
'age',
'Fuel_Type_Diesel',
'Fuel_Type_Petrol',
'Seller_Type_Individual',
'Transmission_Manual']

##  How to Run the Project Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/12sandra/Car-price-prediction-Using-machine-learning.git
cd Car-price-prediction-Using-machine-learning

2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the Flask app
python app.py

4️⃣ Open in browser
http://127.0.0.1:5000/

👩‍💻 Author

Sandra A S


**B.Tech Computer Science Graduate**


**Aspiring Data Scientist**
