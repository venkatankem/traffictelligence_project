import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scale = pickle.load(open('scale.pkl', 'rb'))

@app.route('/')  # Route for home page
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    # Reading the inputs given by the user
    input_feature = [float(x) for x in request.form.values()]
    features_values = [np.array(input_feature)]

    # Define feature names (must match model training)
    names = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year',
             'month', 'day', 'hours', 'minutes', 'seconds']

    # Preprocess inputs
    data = pd.DataFrame(features_values, columns=names)
    data_scaled = scale.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=names)

    # Make prediction
    prediction = model.predict(data_scaled)
    text = "Estimated Traffic Volume is: "

    return render_template('index.html', prediction_text=text + str(prediction[0]))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=True, use_reloader=False)
