from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Model coefficients extracted from your trained model
MODEL_COEFFICIENTS = [257.9491632, -30.65990398, 330.97801469, 438.77713593, -24280.92636906, 144.64658202]
INTERCEPT = 11734.679265420064

@app.route('/')
def home():
    with open('index.html', 'r', encoding='utf-8') as file:
        return file.read()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features in the correct order: age, sex, bmi, children, smoker, region
        features = [
            float(data['age']),
            float(data['sex']),
            float(data['bmi']),
            float(data['children']),
            float(data['smoker']),
            float(data['region'])
        ]
        
        # Make prediction using linear regression formula: y = intercept + sum(coef * feature)
        prediction = INTERCEPT
        for i, feature in enumerate(features):
            prediction += MODEL_COEFFICIENTS[i] * feature
        
        # Ensure prediction is not negative
        prediction = max(0, prediction)
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"Model coefficients: {MODEL_COEFFICIENTS}")
    print(f"Model intercept: {INTERCEPT}")
    print("Server running at http://localhost:5000")
    app.run(debug=True, port=5000)