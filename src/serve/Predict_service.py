from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import requests
import pymongo
import joblib
from sklearn.metrics import mean_squared_error
from flask_cors import CORS, cross_origin
import onnxruntime as ort
from mlflow.tracking import MlflowClient
import mlflow
import dagshub
import datetime
from apscheduler.schedulers.background import BackgroundScheduler

# Configure MLflow
dagshub_token = '22495012faf69bd7449136c47feddea65bd1ff8c'
dagshub.auth.add_app_token(dagshub_token)
dagshub.init(repo_name="IIS_pro", repo_owner="CesarMitja", mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/CesarMitja/IIS_pro.mlflow')
mlflow.set_experiment('Real_Estate_Prediction_Service')

app = Flask(__name__)
CORS(app)

CONNECTION_STRING = "mongodb+srv://cesi:Hondacbr125.@ptscluster.gkdlocr.mongodb.net/?retryWrites=true&appName=PTScluster"
client = pymongo.MongoClient(CONNECTION_STRING)

db = client.IIS
rent_predictions_collection = db.iis2_rent
price_predictions_collection = db.iis2_price
actual_collection = db.iis2_a

mlflow_client = MlflowClient()

# Load Rent Prediction Model and Artifacts
rent_model_name = "Rent_Prediction_Model"
rent_latest_version = mlflow_client.get_latest_versions(rent_model_name)[0]

def load_model(run_id, artifact_path):
    local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{artifact_path}")
    ort_session = ort.InferenceSession(local_model_path)
    return ort_session

def load_artifact(run_id, artifact_path):
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{artifact_path}")
    return joblib.load(local_path)

rent_model = load_model(rent_latest_version.run_id, "quantized_onnx_model/quantized_model_rent.onnx")
rent_preprocessor = load_artifact(rent_latest_version.run_id, "preprocessor/preprocessor.pkl")
rent_label_encoder = load_artifact(rent_latest_version.run_id, "label_encoder/label_encoder.pkl")

rent_features = ['Price', 'Bedrooms', 'Bathrooms', 'Living Area', 'Type']
rent_input_features = ['Price', 'Bedrooms', 'Bathrooms', 'Living Area', 'Type']

# Load Price Prediction Model and Artifacts
price_model_name = "Price_Prediction_Model"
price_latest_version = mlflow_client.get_latest_versions(price_model_name)[0]

price_model = load_model(price_latest_version.run_id, "quantized_onnx_model/quantized_model_price.onnx")
price_preprocessor = load_artifact(price_latest_version.run_id, "preprocessor/preprocessor.pkl")
price_label_encoder = load_artifact(price_latest_version.run_id, "label_encoder/label_encoder.pkl")

price_features = ['Bedrooms', 'Bathrooms', 'Living Area', 'Lot Area', 'Type']
price_input_features = ['Bedrooms', 'Bathrooms', 'Living Area', 'Lot Area', 'Type']

def preprocess_input(input_data, preprocessor, features):
    df = pd.DataFrame(input_data, columns=features)
    return preprocessor.transform(df)

def make_prediction(input_data, model, preprocessor, label_encoder, features):
    input_data = preprocess_input(input_data, preprocessor, features)
    input_tensor = np.array(input_data, dtype=np.float32)
    ort_inputs = {model.get_inputs()[0].name: input_tensor}
    ort_outs = model.run(None, ort_inputs)
    predictions = label_encoder.inverse_transform(ort_outs[0].argmax(axis=1))
    return predictions

@app.route('/predict_rent', methods=['POST'])
@cross_origin(origin='*')
def predict_rent():
    try:
        input_json = request.json
        input_data = [input_json[feature] for feature in rent_input_features]
        input_data = np.array([input_data])
        predictions = make_prediction(input_data, rent_model, rent_preprocessor, rent_label_encoder, rent_features)
        results = {"predictions": predictions.tolist()}
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_price', methods=['POST'])
@cross_origin(origin='*')
def predict_price():
    try:
        input_json = request.json
        input_data = [input_json[feature] for feature in price_input_features]
        input_data = np.array([input_data])
        predictions = make_prediction(input_data, price_model, price_preprocessor, price_label_encoder, price_features)
        results = {"predictions": predictions.tolist()}
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)})

def load_last_24_rows(csv_file_path, features):
    df = pd.read_csv(csv_file_path)
    df = df[features]
    if len(df) >= 24:
        df = df.tail(24)
    else:
        raise ValueError("Not enough data. Need at least 24 rows.")
    return df

def predict_and_store_rent():
    try:
        input_data = load_last_24_rows(csv_file_path, rent_features)
        predictions = make_prediction(input_data, rent_model, rent_preprocessor, rent_label_encoder, rent_features)
        doc = {
            "timestamp": datetime.datetime.utcnow(),
            "predictions": predictions.tolist()}
        rent_predictions_collection.insert_one(doc)
        return True
    except Exception as e:
        return False

def predict_and_store_price():
    try:
        input_data = load_last_24_rows(csv_file_path, price_features)
        predictions = make_prediction(input_data, price_model, price_preprocessor, price_label_encoder, price_features)
        doc = {
            "timestamp": datetime.datetime.utcnow(),
            "predictions": predictions.tolist()}
        price_predictions_collection.insert_one(doc)
        return True
    except Exception as e:
        return False

def calculate_metrics():
    yesterday = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    start_of_day = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = yesterday.replace(hour=23, minute=59, second=59, microsecond=999)

    rent_predictions_records = list(rent_predictions_collection.find({
        "timestamp": {"$gte": start_of_day, "$lt": end_of_day}
    }))
    price_predictions_records = list(price_predictions_collection.find({
        "timestamp": {"$gte": start_of_day, "$lt": end_of_day}
    }))
    actuals_records = list(actual_collection.find({
        "timestamp": {"$gte": start_of_day, "$lt": end_of_day}
    }))

    actuals = {record['timestamp'].replace(minute=0, second=0, microsecond=0): record['actual']
               for record in actuals_records}

    for record in rent_predictions_records:
        base_time = record['timestamp']
        if isinstance(base_time, int):  # Convert from UNIX timestamp if necessary
            base_time = datetime.datetime.utcfromtimestamp(base_time / 1000)
        base_time = base_time.replace(minute=0, second=0, microsecond=0)
        
        prediction_values = [pred for pred in record['predictions']]
        mse_values = []
        
        with mlflow.start_run():
            for i, prediction in enumerate(prediction_values):
                pred_time = base_time + datetime.timedelta(hours=i)
                if pred_time in actuals:
                    mse = mean_squared_error([prediction], [actuals[pred_time]])
                    mse_values.append(mse)
                    mlflow.log_metric(f'rent_prediction_{base_time.hour:02d}_{i}_mse', mse)
            
            if mse_values:
                avg_mse = np.mean(mse_values)
                mlflow.log_metric(f'rent_prediction_{base_time.hour:02d}_avg_mse', avg_mse)
                print(f"Average MSE for rent predictions starting at {base_time.hour:02d}:00: {avg_mse}")
            else:
                print(f"No matching actual data for rent predictions starting at {base_time.hour:02d}:00.")

    for record in price_predictions_records:
        base_time = record['timestamp']
        if isinstance(base_time, int):  # Convert from UNIX timestamp if necessary
            base_time = datetime.datetime.utcfromtimestamp(base_time / 1000)
        base_time = base_time.replace(minute=0, second=0, microsecond=0)
        
        prediction_values = [pred for pred in record['predictions']]
        mse_values = []
        
        with mlflow.start_run():
            for i, prediction in enumerate(prediction_values):
                pred_time = base_time + datetime.timedelta(hours=i)
                if pred_time in actuals:
                    mse = mean_squared_error([prediction], [actuals[pred_time]])
                    mse_values.append(mse)
                    mlflow.log_metric(f'price_prediction_{base_time.hour:02d}_{i}_mse', mse)
            
            if mse_values:
                avg_mse = np.mean(mse_values)
                mlflow.log_metric(f'price_prediction_{base_time.hour:02d}_avg_mse', avg_mse)
                print(f"Average MSE for price predictions starting at {base_time.hour:02d}:00: {avg_mse}")
            else:
                print(f"No matching actual data for price predictions starting at {base_time.hour:02d}:00.")

def save_data():
    API_URL = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"
    TARGET_STATION_NAME = "GOSPOSVETSKA C. - TURNERJEVA UL."
    response = requests.get(API_URL)
    response.raise_for_status()
    stations = response.json()
    for station in stations:
        if station['name'] == TARGET_STATION_NAME:
            df1 = pd.DataFrame([station])
    df2 = df1['available_bike_stands'].values
    df2 = df2[0]
    doc = {
        "timestamp": datetime.datetime.utcnow(),
        "actual": df2.tolist()}
    actual_collection.insert_one(doc)

scheduler = BackgroundScheduler()
scheduler.add_job(func=calculate_metrics, trigger='cron', hour=0)
scheduler.add_job(func=predict_and_store_rent, trigger='cron', minute=0)
scheduler.add_job(func=predict_and_store_price, trigger='cron', minute=0)
scheduler.add_job(func=save_data, trigger='cron', minute=0)
scheduler.start()

if __name__ == '__main__':
    app.run(debug=True)
