from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
import os
import requests

app = Flask(__name__)

# MongoDB setup
mongo_client = MongoClient(os.getenv('MONGO_URI'))
db = mongo_client[os.getenv('MONGO_DB')]
models_collection = db['models']

HEALTH_TRACKER_URI = os.getenv('HEALTH_TRACKER_URI')
API_TOKEN = os.getenv('API_TOKEN', 'my_secret_token')

def get_model(user_id):
    model_data = models_collection.find_one({'user_id': user_id})
    if model_data:
        model = pickle.loads(model_data['model'])
        scaler = pickle.loads(model_data['scaler'])
    else:
        model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
        scaler = StandardScaler()
    return model, scaler

def save_model(user_id, model, scaler):
    model_data = {
        'model': pickle.dumps(model),
        'scaler': pickle.dumps(scaler)
    }
    models_collection.update_one({'user_id': user_id}, {'$set': model_data}, upsert=True)

def fetch_user_data(user_id):
    headers = {'api-token': API_TOKEN}
    response = requests.get(f'{HEALTH_TRACKER_URI}/user_activities/{user_id}', headers=headers)
    if response.status_code == 200:
        return response.json()['activities']
    else:
        return []
    
def preprocess_data(data, scaler=None, fit_scaler=False):
    df = pd.DataFrame(data)
    df.dropna(inplace=True)  # Remove rows with missing values
    X = df[['Udaljenost', 'Vrijeme', 'Ukupni uspon']]
    if fit_scaler:
        scaler = scaler.fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled, df['Tezina'].values, scaler

@app.route('/train', methods=['POST'])
def train():
    app.logger.debug(f"Training model")
    user_id = request.json['user_id']
    data = request.json['data']
    
    model, scaler = get_model(user_id)
    X, y, scaler = preprocess_data(data, scaler, fit_scaler=True)

    if not hasattr(model, 'classes_'):  # First time training
        model.partial_fit(X, y, classes=[0, 1, 2])
    else:
        model.partial_fit(X, y)
    save_model(user_id, model, scaler)
    
    return jsonify({'status': 'success', 'message': 'Model trained successfully'})

@app.route('/predict', methods=['POST'])
def predict():
    user_id = request.json['user_id']
    input_data = request.json['input_data']
    
    # Log input data for debugging
    app.logger.debug(f"Input data: {input_data}")

    model, scaler = get_model(user_id)
    if not hasattr(model, 'classes_'):
        app.logger.debug(f"Training initial model for user {user_id}")
        user_data = fetch_user_data(user_id)
        if not user_data:
            return jsonify({'error': 'No user data available to train model'}), 400
        
        X, y, scaler = preprocess_data(user_data, scaler, fit_scaler=True)
        model.partial_fit(X, y, classes=[0, 1, 2])
        save_model(user_id, model, scaler)
    
    df = pd.DataFrame([input_data])
    
    # Log DataFrame columns for debugging
    app.logger.debug(f"DataFrame columns: {df.columns.tolist()}")

    # Ensure that DataFrame has the expected columns
    if not all(col in df.columns for col in ['Udaljenost', 'Vrijeme', 'Ukupni uspon']):
        return jsonify({'error': 'Invalid input data format'}), 400
    
    df.dropna(inplace=True)  # Remove rows with missing values for prediction
    if df.empty:
        return jsonify({'error': 'Input data contains missing values'}), 400

    X_scaled = scaler.transform(df[['Udaljenost', 'Vrijeme', 'Ukupni uspon']])
    
    prediction = model.predict(X_scaled)[0]
    
    # Convert prediction to a standard Python int
    prediction = int(prediction)
    app.logger.debug(f"Prediction: {prediction}")
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
