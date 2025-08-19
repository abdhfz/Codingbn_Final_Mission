# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Load preprocessor and model
preprocessor = joblib.load("model/titanic_preprocessor.joblib")
nn_model = tf.keras.models.load_model("model/titanic_nn.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or request.form
    try:
        row = {
            'pclass': int(data.get('pclass', 3)),
            'sex': data.get('sex', 'male'),
            'age': float(data.get('age')) if data.get('age') else np.nan,
            'sibsp': int(data.get('sibsp', 0)),
            'parch': int(data.get('parch', 0)),
            'fare': float(data.get('fare', 0.0)),
            'embarked': data.get('embarked', None)
        }
    except ValueError:
        return jsonify({"error": "Invalid input format"}), 400

    df = pd.DataFrame([row])
    X_proc = preprocessor.transform(df)
    proba = float(nn_model.predict(X_proc, verbose=0)[0][0])
    pred = int(proba >= 0.5)
    
    return jsonify({"survived": bool(pred), "survival_probability": proba})

if __name__ == "__main__":
    app.run(debug=True)
