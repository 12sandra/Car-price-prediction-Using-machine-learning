from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# 1️⃣ Load Model and Scaler ONCE (global scope)
# try:
#     model = pickle.load(open('car_price_model.pkl', 'rb'))
#     scaler = pickle.load(open('scaler.pkl', 'rb'))
# except FileNotFoundError:
#     print("Error: model or scaler file not found.")
# Load model and scaler
try:
    model = pickle.load(open('car_price_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except Exception as e:
    print("Error loading model or scaler:", e)

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':

        # 2️⃣ Extract numerical features
        present_price = float(request.form['Present_Price'])
        kms_driven = int(request.form['Kms_Driven'])
        owner = int(request.form['Owner'])
        age = int(request.form['age'])

        # 3️⃣ Encode categorical features
        fuel_type = request.form['Fuel_Type']
        fuel_diesel = 1 if fuel_type == 'Diesel' else 0
        fuel_petrol = 1 if fuel_type == 'Petrol' else 0

        seller_type = request.form['Seller_Type']
        seller_individual = 1 if seller_type == 'Individual' else 0

        transmission = request.form['Transmission']
        transmission_manual = 1 if transmission == 'Manual' else 0

        # 4️⃣ Arrange features in EXACT training order
        final_features = np.array([[ 
            present_price,
            kms_driven,
            owner,
            age,
            fuel_diesel,
            fuel_petrol,
            seller_individual,
            transmission_manual
        ]])

        # 5️⃣ SCALE features (this was missing earlier 🚨)
        final_features = scaler.transform(final_features)

        # 6️⃣ Predict
        prediction = model.predict(final_features)[0]
        output = round(prediction, 2)

        # 7️⃣ Optional sanity check
        if output < 0:
            return render_template(
                'index.html',
                prediction_text="Sorry, this car has no resale value."
            )

        return render_template(
            'index.html',
            prediction_text=f"Estimated Selling Price: ₹ {output} Lakhs"
        )

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)



