# rai_service/main.py
import os
import tempfile
import shutil
import threading
import mlflow
import pandas as pd

from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, render_template_string
from mlflow.tracking import MlflowClient

# Responsible AI Toolbox
from responsibleai import RAIInsights
from raiwidgets import ResponsibleAIDashboard

app = Flask(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Where we'll store the RAIInsights results
RAI_INSIGHTS_DIR = "rai_insights_dir"

# We'll store a global reference to the RAI dashboard so we can run it in a thread
global_dashboard = None


# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def fetch_dataset_from_artifact(run_id: str, artifact_path: str) -> pd.DataFrame:
    """
    Downloads the CSV from MLflow artifacts (e.g., 'data/train_batch_123.csv')
    and returns it as a DataFrame.
    """
    client = MlflowClient()
    with tempfile.TemporaryDirectory() as local_dir:
        local_path = client.download_artifacts(run_id, artifact_path, local_dir)
        df = pd.read_csv(local_path)
    return df

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

def create_rai_insights(model, df: pd.DataFrame, label_col: str = "Tezina"):
    """
    Creates a RAIInsights object with Error Analysis, Interpretability, Data Balance.
    We'll treat 'df' as test data for demonstration. 
    """
    if label_col not in df.columns:
        raise ValueError(f"Data must contain a '{label_col}' column.")

    #y = df[label_col]
    #X = df.drop(columns=[label_col])

    # We'll assume it's a classification task
    task_type = "classification"  # or "regression"
    rai_insights = RAIInsights(
        model=model,
        train=df,  # In a real scenario, pass your actual training data
        test=df,   # Re-using X for demonstration
        target_column=label_col,
        task_type=task_type
    )
    # Add components
    rai_insights.error_analysis.add()
    rai_insights.explainer.add()
    rai_insights.data_balance.add(cols_of_interest=['Udaljenost', 'Vrijeme', 'Ukupni uspon', 'Tezina'])
    # Compute them
    rai_insights.compute()
    return rai_insights

# ------------------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------------------

@app.route("/")
def index():
    return (
        "<h3>Welcome to the RAI Service</h3>"
        "<p>Use /analyze_artifact?run_id=XYZ&batch_id=ABC&model_name=... to run analysis</p>"
        "<p>Then visit /dashboard to see the interactive results</p>"
    )

@app.route("/analyze_artifact", methods=["GET"])
def analyze_artifact():
    """
    Example usage:
    GET /analyze_artifact?run_id=some_run_id&batch_id=123&model_name=user_sgd_model&stage=Production

    We build artifact_path as "data/train_batch_{batch_id}.csv".
    """
    run_id = request.args.get("run_id")
    batch_id = request.args.get("batch_id")
    model_name = request.args.get("model_name", "global_sgd_model")
    stage = request.args.get("stage", "Production")

    if not run_id or not batch_id:
        return jsonify({"error": "Missing required query params: run_id, batch_id"}), 400

    artifact_path = f"data/train_batch_{batch_id}.csv"

    # 1. Fetch dataset from artifact
    try:
        df = fetch_dataset_from_artifact(run_id, artifact_path)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch dataset artifact: {str(e)}"}), 400

    # 2. Load model from registry
    try:
        model, scalar = load_model_from_registry(model_name, stage)
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 400

    # 3. Build RAIInsights
    try:
        rai_insights = create_rai_insights(model, df, label_col="Tezina")
    except Exception as e:
        return jsonify({"error": f"RAIInsights creation error: {str(e)}"}), 400

    # 4. Save RAIInsights
    if os.path.exists(RAI_INSIGHTS_DIR):
        shutil.rmtree(RAI_INSIGHTS_DIR)
    os.makedirs(RAI_INSIGHTS_DIR, exist_ok=True)
    rai_insights.save(RAI_INSIGHTS_DIR)

    # 5. Launch the RAI dashboard in a background thread
    def run_dashboard(rai_insights):
        # This will block if run in the main thread, so we use a background thread
        ResponsibleAIDashboard(rai_insights, public_ip="0.0.0.0", port=5010)

    try:
        thread = threading.Thread(target=run_dashboard, args=(rai_insights,), daemon=True)
        thread.start()
    except Exception as e:
        return jsonify({"error": f"Failed to launch RAI dashboard: {str(e)}"}), 500


    return jsonify({
        "status": "Analysis generated from artifact data",
        "run_id": run_id,
        "batch_id": batch_id,
        "model_name": model_name,
        "stage": stage,
        "dashboard_url": url_for("dashboard", _external=True)
    })


@app.route("/dashboard")
def dashboard():
    """
    A simple page that embeds the RAI Dashboard in an <iframe> from port 5010.
    If you prefer a direct redirect, you can link directly to http://localhost:5010
    """
    html_content = """
    <html>
    <head>
        <title>RAI Dashboard</title>
    </head>
    <body>
        <iframe src="http://localhost:5010" width="100%" height="800px"></iframe>
    </body>
    </html>
    """
    return render_template_string(html_content)


if __name__ == "__main__":
    # In Docker, you typically rely on CMD in Dockerfile, but for local dev:
    app.run(host="0.0.0.0", port=5004, debug=True)
