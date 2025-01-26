from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import requests
import mlflow
import mlflow.sklearn
import pickle
import time
from mlflow import register_model
import tempfile
from mlflow.data.pandas_dataset import PandasDataset

app = Flask(__name__)

# -----------------------------------------------------------------------------
# Environment / Configuration
# -----------------------------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow_server:5000')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Default experiment if needed (we'll override with per-user experiments)
DEFAULT_EXPERIMENT = "global_experiment"

# Name of the global model in the MLflow Model Registry
GLOBAL_MODEL_NAME = "global_sgd_model"

HEALTH_TRACKER_URI = os.getenv('HEALTH_TRACKER_URI', 'http://health_tracker:5001')
API_TOKEN = os.getenv('API_TOKEN', 'my_secret_token')

# -----------------------------------------------------------------------------
# Data Fetching & Preprocessing
# -----------------------------------------------------------------------------

def fetch_user_data(user_id):
    """
    Fetch training data for a given user from your 'health_tracker' service.
    Returns a list of activity records (dicts) or an empty list if none found.
    """
    headers = {'api-token': API_TOKEN}
    response = requests.get(f'{HEALTH_TRACKER_URI}/user_activities/{user_id}', headers=headers)
    if response.status_code == 200:
        return response.json()['activities']
    return []

def preprocess_data(data, scaler=None, fit_scaler=False):
    """
    Given a list of dicts (each containing at least 'Udaljenost', 'Vrijeme', 'Ukupni uspon', 'Tezina'),
    returns X_scaled, y, and the updated scaler.
    """
    df = pd.DataFrame(data)
    df.dropna(inplace=True)  # Remove rows with missing values
    X = df[['Udaljenost', 'Vrijeme', 'Ukupni uspon']]
    y = df['Tezina'].values

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        scaler = scaler.fit(X)  # or scaler.partial_fit(...) if truly incremental
    X_scaled = scaler.transform(X)
    return X_scaled, y, scaler

# -----------------------------------------------------------------------------
# MLflow Registry Helper Functions
# -----------------------------------------------------------------------------

import mlflow.pyfunc
import mlflow.exceptions
from mlflow.tracking import MlflowClient

client = MlflowClient()

class ModelWrapper:
    """
    Simple wrapper so we can store both the sklearn model and scaler in one artifact.
    """
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)[0]

def load_model_from_registry(model_name, stage="Production"):
    """
    Load a model from the MLflow Model Registry at the given stage.
    Returns (model, scaler) or (None, None) if not found.
    """
    model_uri = f"models:/{model_name}/{stage}"
    try:
        loaded_obj = mlflow.sklearn.load_model(model_uri)
        if hasattr(loaded_obj, 'model') and hasattr(loaded_obj, 'scaler'):
            return loaded_obj.model, loaded_obj.scaler
        else:
            # If the object doesn't have these attributes,
            # it might just be a raw sklearn model with no separate scaler
            return loaded_obj, None
    except mlflow.exceptions.MlflowException:
        # Model or stage not found
        return None, None

def register_new_model(run_id, model_name):
    """
    Registers (or creates) a model in the MLflow Model Registry
    from the artifacts of the specified run_id. 
    Returns the newly created version.
    """
    register_response = register_model(
        model_uri=f"runs:/{run_id}/model",
        name=model_name
    )
    return register_response.version

def transition_model_version_to_stage(model_name, version, new_stage="Production", archive_existing=True):
    """
    Transition a given model version to a new stage (e.g., 'Production', 'Staging').
    """
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=new_stage,
        archive_existing_versions=archive_existing
    )

def evaluate_model(model, X_val, y_val):
    """
    Simple evaluation function returning accuracy.
    Feel free to expand for more robust metrics.
    """
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    precision = precision_score(y_val, preds, average='macro', zero_division=0)
    recall = recall_score(y_val, preds, average='macro', zero_division=0)
    f1 = f1_score(y_val, preds, average='macro', zero_division=0)
    return accuracy, precision, recall, f1

# -----------------------------------------------------------------------------
# "Train" Endpoints
# -----------------------------------------------------------------------------

@app.route('/train', methods=['POST'])
def train():
    """
    Train a user-specific model on new data.
    Steps:
      1. Load existing user model from registry or fallback to global model if user model doesn't exist.
      2. Preprocess input data and partial_fit.
      3. Evaluate vs. current Production version, promote if improved.
    """
    app.logger.debug("Training user-specific model")

    user_id = request.json['user_id']
    data = request.json.get('data', [])
    raw_data = pd.DataFrame(data)
    dataset = mlflow.data.from_pandas(
        raw_data, name="new training data", targets="Tezina"
    )
    # Switch to an experiment for this user
    experiment_name = f"user_{user_id}"
    mlflow.set_experiment(experiment_name)

    # 1. Load existing user model from Production
    user_model_name = f"user_{user_id}_sgd_model"
    existing_model, existing_scaler = load_model_from_registry(user_model_name, stage="Production")

    if existing_model is None:
        # If user model doesn't exist, try to load the global Production model as a starting point
        global_model, global_scaler = load_model_from_registry(GLOBAL_MODEL_NAME, stage="Production")
        if global_model is not None:
            model = global_model
            scaler = global_scaler
            app.logger.debug(f"Initialized user model from global Production model.")
        else:
            # If no global model exists yet, just create a fresh one
            model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
            scaler = StandardScaler()
            app.logger.debug(f"No global model found; created new SGD model.")
    else:
        model, scaler = existing_model, existing_scaler

    # 2. Preprocess new data
    X, y, scaler = preprocess_data(data, scaler, fit_scaler=True)

    # 3. Partial fit
    # Classes might be [0,1,2] or something else depending on your domain
    if not hasattr(model, 'classes_'):
        model.partial_fit(X, y, classes=np.unique(y))
    else:
        model.partial_fit(X, y)

    # 4. Evaluate new model on some validation data
    # For demonstration, let's just do a quick check on the same data (not recommended in production).
    # In practice, fetch a hold-out set or do cross-validation.
    new_accuracy, new_precision, new_recall, new_f1 = evaluate_model(model, X, y)
    app.logger.debug(f"New model's accuracy on current training batch: {new_accuracy}")

    # 5. Compare to existing Production model’s performance
    current_prod_accuracy = 0.0
    if existing_model is not None:
        current_prod_accuracy, urrent_prod_precision, urrent_prod_recall, urrent_prod_f1 = evaluate_model(existing_model, X, y)
        app.logger.debug(f"Current production model's accuracy on same data: {current_prod_accuracy}")

    # Start an MLflow run to log metrics, artifacts
    with mlflow.start_run() as run:
        mlflow.log_metric("accuracy", new_accuracy)
        mlflow.log_metric("precision", new_precision)
        mlflow.log_metric("recall", new_recall)
        mlflow.log_metric("f1", new_f1)

        if isinstance(model, SGDClassifier):
            mlflow.log_param("sgd_alpha", model.alpha)
            mlflow.log_param("sgd_loss", model.loss)
            mlflow.log_param("sgd_penalty", model.penalty)
            mlflow.log_param("sgd_max_iter", model.max_iter)
            mlflow.log_param("sgd_tol", model.tol)

        run_id = run.info.run_id
        mlflow.log_param("run_id", run_id)

        df_batch = pd.DataFrame(data)
        with tempfile.NamedTemporaryFile(prefix="train_batch_", suffix=".csv", delete=False) as tmp_file:
            df_batch.to_csv(tmp_file.name, index=False)
            mlflow.log_artifact(tmp_file.name, artifact_path="data")
        
        mlflow.log_input(dataset, context="training")

        # Log the new model artifact
        wrapper = ModelWrapper(model, scaler)
        mlflow.sklearn.log_model(wrapper, artifact_path="model")

        # 6. If new model is better, register & promote
        if new_accuracy > current_prod_accuracy:
            app.logger.debug("New model is better; registering and promoting to Production.")
            new_version = register_new_model(run.info.run_id, user_model_name)
            transition_model_version_to_stage(user_model_name, new_version, "Production", archive_existing=True)
        else:
            app.logger.debug("New model did NOT improve. Not promoting to Production.")

    return jsonify({'status': 'success', 'message': 'User model trained', 'new_accuracy': new_accuracy})


@app.route('/train_global', methods=['POST'])
def train_global():
    """
    Train the global model on all users' data or a subset of aggregated data.
    Steps:
      1. Fetch data from all users (or partial).
      2. Load current global Production model if exists, else new.
      3. partial_fit on aggregated data.
      4. Evaluate and possibly promote if improved.
    """
    app.logger.debug("Training global model")

    # Switch to a global experiment
    mlflow.set_experiment(DEFAULT_EXPERIMENT)

    # 1. Gather data from all users (or a strategy you define)
    # Example: Suppose we have an endpoint that returns data from all users
    headers = {'api-token': API_TOKEN}
    response = requests.get(f'{HEALTH_TRACKER_URI}/all_activities', headers=headers)  
    if response.status_code == 200:
        all_data = response.json().get('activities', [])
    else:
        return jsonify({'error': 'Could not fetch global data'}), 400

    # 2. Load current global Production model
    global_prod_model, global_prod_scaler = load_model_from_registry(GLOBAL_MODEL_NAME, stage="Production")
    if global_prod_model is None:
        global_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
        global_scaler = StandardScaler()
        app.logger.debug("No global production model found; created a fresh one.")
    else:
        global_model, global_scaler = global_prod_model, global_prod_scaler

    # 3. Preprocess & partial_fit
    X, y, global_scaler = preprocess_data(all_data, global_scaler, fit_scaler=True)
    if not hasattr(global_model, 'classes_'):
        global_model.partial_fit(X, y, classes=np.unique(y))
    else:
        global_model.partial_fit(X, y)

    # Evaluate new global model on the same data or a hold-out set
    new_accuracy = evaluate_model(global_model, X, y)

    # Evaluate existing Production model if it exists
    current_prod_accuracy = 0.0
    if global_prod_model is not None:
        current_prod_accuracy = evaluate_model(global_prod_model, X, y) 

    # 4. Log to MLflow, compare, promote if better
    with mlflow.start_run() as run:
        mlflow.log_metric("train_accuracy", new_accuracy)
        mlflow.log_metric("old_model_accuracy", current_prod_accuracy)

        wrapper = ModelWrapper(global_model, global_scaler)
        mlflow.sklearn.log_model(wrapper, artifact_path="model")

        if new_accuracy > current_prod_accuracy:
            app.logger.debug("Global model improved. Registering and promoting to Production.")
            new_version = register_new_model(run.info.run_id, GLOBAL_MODEL_NAME)
            transition_model_version_to_stage(GLOBAL_MODEL_NAME, new_version, "Production", archive_existing=True)
        else:
            app.logger.debug("Global model did NOT improve. Not promoting.")

    return jsonify({'status': 'success', 'message': 'Global model trained', 'new_accuracy': new_accuracy})


# -----------------------------------------------------------------------------
# Prediction Endpoint
# -----------------------------------------------------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict for a given user using their Production model. If user model doesn't exist,
    fallback to global Production model. If global also doesn't exist, error out.
    """
    user_id = request.json['user_id']
    input_data = request.json['input_data']

    app.logger.debug(f"Predict request for user: {user_id}, input_data={input_data}")

    # Load user model first
    user_model_name = f"user_{user_id}_sgd_model"
    model, scaler = load_model_from_registry(user_model_name, stage="Production")

    if model is None:
        # Fall back to global Production if user model doesn't exist
        app.logger.debug("User model not found; trying global model fallback.")
        global_model, global_scaler = load_model_from_registry(GLOBAL_MODEL_NAME, stage="Production")
        if global_model is None:
            return jsonify({'error': 'No user-specific or global model available'}), 400
        model, scaler = global_model, global_scaler

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # Ensure required columns exist
    required_cols = ['Udaljenost', 'Vrijeme', 'Ukupni uspon']
    if not all(col in df.columns for col in required_cols):
        return jsonify({'error': 'Invalid input data format'}), 400

    df.dropna(inplace=True)
    if df.empty:
        return jsonify({'error': 'Input data contains missing values'}), 400

    # Scale
    if scaler is None:
        # If for some reason there's no scaler in the model, create a new one
        scaler = StandardScaler().fit(df[required_cols])
    X_scaled = scaler.transform(df[required_cols])

    # Predict
    prediction = model.predict(X_scaled)[0]
    prediction = int(prediction)  # Convert to plain Python int

    app.logger.debug(f"Prediction for user {user_id}: {prediction}")
    return jsonify({'prediction': prediction})

# -----------------------------------------------------------------------------
# Global model initialization
# -----------------------------------------------------------------------------

@app.route('/init_global_model', methods=['POST'])
def init_global_model():
    """
    One-time endpoint to load a pretrained global model into MLflow registry.
    Expects JSON with optional 'model_file' key indicating the pickle file path on disk.
    Example request body:
        {
          "model_file": "/global_models/global_model.pkl"
        }
    If no 'model_file' is provided, a default path is used.
    """

    app.logger.debug("Initializing global model in MLflow registry...")

    # 1. Determine model file path (use default if none provided)
    mlflow.set_experiment(DEFAULT_EXPERIMENT)
    model_file = request.json.get('model_file', '/global_models/global_model.pkl')

    # 2. Load the pretrained model from the file
    try:
        with open(model_file, 'rb') as f:
            pretrained_model = pickle.load(f)
    except Exception as e:
        app.logger.error(f"Could not load model from {model_file}: {e}")
        return jsonify({'error': f"Failed to load model file {model_file}"}), 400

    # 3. Start an MLflow run to log the model artifact
    with mlflow.start_run(run_name="init_global_model") as run:
        mlflow.log_param("source", model_file)
        # For scikit-learn model: mlflow.sklearn.log_model(...)
        # If you have a scaler or pipeline, wrap them as needed. 
        mlflow.sklearn.log_model(pretrained_model, artifact_path="model")

        # 4. Register the model in the Model Registry under GLOBAL_MODEL_NAME
        client = MlflowClient()
        result = client.create_model_version(
            name=GLOBAL_MODEL_NAME,  # references your global model name
            source=f"runs:/{run.info.run_id}/model",
            run_id=run.info.run_id
        )

        # 5. Wait until the model version is ready
        for _ in range(10):
            model_version_details = client.get_model_version(
                name=GLOBAL_MODEL_NAME,
                version=result.version
            )
            if model_version_details.status == "READY":
                break
            time.sleep(1)

        # 6. (Optional) Promote to Production stage
        client.transition_model_version_stage(
            name=GLOBAL_MODEL_NAME,
            version=result.version,
            stage="Production",
            archive_existing_versions=True
        )

    return jsonify({
        'status': 'success',
        'message': f"Pretrained global model registered as version {result.version}",
        'model_file': model_file
    })

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
