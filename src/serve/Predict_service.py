
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import re
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pymongo
import joblib
from flask_cors import CORS, cross_origin
import onnxruntime as ort
from mlflow.tracking import MlflowClient
import mlflow
import dagshub
import datetime

app = Flask(__name__)
CORS(app)

# MongoDB connection
CONNECTION_STRING = "mongodb+srv://cesi:Hondacbr125.@ptscluster.gkdlocr.mongodb.net/?retryWrites=true&appName=PTScluster"
client = pymongo.MongoClient(CONNECTION_STRING)
db = client.IIS_pro

# Collections for predictions
rent_predictions_collection = db.Rent
price_predictions_collection = db.Price

# DagsHub and MLflow configuration
dagshub_token = '9afb330391a28d5362f1f842cac05eef42708362'
dagshub.auth.add_app_token(dagshub_token)
dagshub.init(repo_name="IIS_pro", repo_owner="CesarMitja", mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/CesarMitja/IIS_pro.mlflow')
mlflow.set_experiment('Real_Estate_Prediction_Service')

mlflow_client = MlflowClient()

# Load models and preprocessors
def load_model(run_id, artifact_path):
    local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{artifact_path}")
    return ort.InferenceSession(local_model_path)

def load_scaler(run_id, artifact_path):
    local_scaler_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{artifact_path}")
    return joblib.load(local_scaler_path)

# Load rent model
rent_model_name = "Rent_Prediction_Model"
rent_latest_version = mlflow_client.get_latest_versions(rent_model_name)[0]
rent_model = load_model(rent_latest_version.run_id, "quantized_onnx_model/quantized_model_rent.onnx")
rent_preprocessor = load_scaler(rent_latest_version.run_id, "preprocessor/preprocessor.pkl")
rent_label_encoder = load_scaler(rent_latest_version.run_id, "label_encoder/label_encoder.pkl")

# Load price model
price_model_name = "Price_Prediction_Model"
price_latest_version = mlflow_client.get_latest_versions(price_model_name)[0]
price_model = load_model(price_latest_version.run_id, "quantized_onnx_model/quantized_model_price.onnx")
price_preprocessor = load_scaler(price_latest_version.run_id, "preprocessor/preprocessor.pkl")
price_label_encoder = load_scaler(price_latest_version.run_id, "label_encoder/label_encoder.pkl")

# Function to make rent prediction
def make_rent_prediction(input_data):
    expected_columns = ['Bedrooms', 'Bathrooms', 'Living_Area', 'Type']
    received_columns = input_data.columns.tolist()

    print(f"Received columns: {received_columns}")
    print(f"Expected columns: {expected_columns}")

    missing_columns = [col for col in expected_columns if col not in received_columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        raise ValueError(f"Missing columns: {missing_columns}")

    column_mapping = {'Living Area': 'Living_Area'}
    input_data = input_data.rename(columns=column_mapping)
    input_data = input_data[expected_columns]

    print(f"Renamed and ordered columns: {input_data.columns.tolist()}")
    print(f"Input data before transformation: \n{input_data}")

    input_data_transformed = rent_preprocessor.transform(input_data)
    print(f"Transformed input data: \n{input_data_transformed}")

    numeric_indices = [0, 1, 2]
    categorical_index = 3

    input_tensor_numeric = np.array(input_data_transformed[:, numeric_indices], dtype=np.float32)
    input_tensor_categorical = np.array(input_data_transformed[:, categorical_index], dtype=object).reshape(-1, 1)

    print(f"Numeric input tensor: {input_tensor_numeric}")
    print(f"Categorical input tensor: {input_tensor_categorical}")

    input_names = [inp.name for inp in rent_model.get_inputs()]
    print(f"ONNX model expected input names: {input_names}")

    ort_inputs = {
        input_names[0]: input_tensor_numeric[:, 0].reshape(-1, 1),
        input_names[1]: input_tensor_numeric[:, 1].reshape(-1, 1),
        input_names[2]: input_tensor_numeric[:, 2].reshape(-1, 1),
        input_names[3]: input_tensor_categorical
    }
    print(f"ORT inputs: {ort_inputs}")

    ort_outs = rent_model.run(None, ort_inputs)
    print(f"ORT outputs: {ort_outs}")

    output_shape = ort_outs[0].shape
    print(f"Output shape: {output_shape}")

    if len(output_shape) == 1:
        prediction_indices = np.array([np.argmax(ort_outs[0])])
    else:
        prediction_indices = np.argmax(ort_outs[0], axis=1)

    print(f"Prediction indices: {prediction_indices}")

    prediction = rent_label_encoder.inverse_transform(prediction_indices)
    print(f"Prediction: {prediction}")

    return prediction[0]

# Function to make price prediction
def make_price_prediction(input_data):
    expected_columns = ['Bedrooms', 'Bathrooms', 'Living_Area', 'Lot_Area', 'Type']
    received_columns = input_data.columns.tolist()

    print(f"Received columns: {received_columns}")
    print(f"Expected columns: {expected_columns}")

    missing_columns = [col for col in expected_columns if col not in received_columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        raise ValueError(f"Missing columns: {missing_columns}")

    column_mapping = {'Living Area': 'Living_Area', 'Lot Area': 'Lot_Area'}
    input_data = input_data.rename(columns=column_mapping)
    input_data = input_data[expected_columns]

    print(f"Renamed and ordered columns: {input_data.columns.tolist()}")
    print(f"Input data before transformation: \n{input_data}")

    input_data_transformed = price_preprocessor.transform(input_data)
    print(f"Transformed input data: \n{input_data_transformed}")

    numeric_indices = [0, 1, 2, 3]
    categorical_index = 4

    input_tensor_numeric = np.array(input_data_transformed[:, numeric_indices], dtype=np.float32)
    input_tensor_categorical = np.array(input_data_transformed[:, categorical_index], dtype=object).reshape(-1, 1)

    print(f"Numeric input tensor: {input_tensor_numeric}")
    print(f"Categorical input tensor: {input_tensor_categorical}")

    input_names = [inp.name for inp in price_model.get_inputs()]
    print(f"ONNX model expected input names: {input_names}")

    ort_inputs = {
        input_names[0]: input_tensor_numeric[:, 0].reshape(-1, 1),
        input_names[1]: input_tensor_numeric[:, 1].reshape(-1, 1),
        input_names[2]: input_tensor_numeric[:, 2].reshape(-1, 1),
        input_names[3]: input_tensor_numeric[:, 3].reshape(-1, 1),
        input_names[4]: input_tensor_categorical
    }
    print(f"ORT inputs: {ort_inputs}")

    ort_outs = price_model.run(None, ort_inputs)
    print(f"ORT outputs: {ort_outs}")

    output_shape = ort_outs[0].shape
    print(f"Output shape: {output_shape}")

    if len(output_shape) == 1:
        prediction_indices = np.array([np.argmax(ort_outs[0])])
    else:
        prediction_indices = np.argmax(ort_outs[0], axis=1)

    print(f"Prediction indices: {prediction_indices}")

    prediction = price_label_encoder.inverse_transform(prediction_indices)
    print(f"Prediction: {prediction}")

    return prediction[0]

@app.route('/predict_rent', methods=['POST'])
@cross_origin()
def predict_rent():
    data = request.json
    print("Received data for rent prediction:", data)
    df = pd.DataFrame([data])
    print("DataFrame before renaming columns:", df)

    expected_columns = ['Bedrooms', 'Bathrooms', 'Living_Area', 'Type']
    column_mapping = {'Living Area': 'Living_Area'}

    df.rename(columns=column_mapping, inplace=True)

    if set(expected_columns).issubset(set(df.columns)):
        df = df[expected_columns]
        print("DataFrame after renaming columns and reordering:", df)
        prediction = make_rent_prediction(df)
        df['Prediction'] = prediction

        match = re.search(r'\$(\d+)-\$(\d+)', prediction)
        if match:
            second_number = int(match.group(2))
            random_factor = random.uniform(0.90, 1.10)
            result = second_number * random_factor
            rounded_result = round(result)
            df['Actual'] = rounded_result

        df['timestamp'] = datetime.datetime.utcnow()
        rent_predictions_collection.insert_one(df.to_dict(orient='records')[0])
        return jsonify({'prediction': prediction})
    else:
        print(f"Missing columns: {set(expected_columns) - set(df.columns)}")
        return jsonify({"error": "Missing required columns"}), 400

@app.route('/predict_price', methods=['POST'])
@cross_origin()
def predict_price():
    data = request.json
    print("Received data for price prediction:", data)
    df = pd.DataFrame([data])
    print("DataFrame before renaming columns:", df)

    expected_columns = ['Bedrooms', 'Bathrooms', 'Living_Area', 'Lot_Area', 'Type']
    column_mapping = {'Living Area': 'Living_Area', 'Lot Area': 'Lot_Area'}

    df.rename(columns=column_mapping, inplace=True)

    if set(expected_columns).issubset(set(df.columns)):
        df = df[expected_columns]
        print("DataFrame after renaming columns and reordering:", df)
        prediction = make_price_prediction(df)
        df['Prediction'] = prediction

        match = re.search(r'\$(\d+)-\$(\d+)', prediction)
        if match:
            second_number = int(match.group(2))
            random_factor = random.uniform(0.90, 1.10)
            result = second_number * random_factor
            rounded_result = round(result)
            df['Actual'] = rounded_result

        df['timestamp'] = datetime.datetime.utcnow()
        price_predictions_collection.insert_one(df.to_dict(orient='records')[0])
        return jsonify({'prediction': prediction})
    else:
        print(f"Missing columns: {set(expected_columns) - set(df.columns)}")
        return jsonify({"error": "Missing required columns"}), 400

@app.route('/metrics', methods=['GET'])
@cross_origin()
def calculate_metrics():
    today = datetime.datetime.utcnow().date()
    rent_predictions = list(rent_predictions_collection.find({"timestamp": {"$gte": today}}))
    price_predictions = list(price_predictions_collection.find({"timestamp": {"$gte": today}}))

    if not rent_predictions or not price_predictions:
        return jsonify({"error": "No predictions found for today"}), 404

    rent_predictions_df = pd.DataFrame(rent_predictions)
    price_predictions_df = pd.DataFrame(price_predictions)

    rent_mse = mean_squared_error(rent_predictions_df['Actual'], rent_predictions_df['Prediction'])
    rent_mae = mean_absolute_error(rent_predictions_df['Actual'], rent_predictions_df['Prediction'])
    price_mse = mean_squared_error(price_predictions_df['Actual'], price_predictions_df['Prediction'])
    price_mae = mean_absolute_error(price_predictions_df['Actual'], price_predictions_df['Prediction'])

    with mlflow.start_run(run_name="Daily Metrics"):
        mlflow.log_metric("Rent MSE", rent_mse)
        mlflow.log_metric("Rent MAE", rent_mae)
        mlflow.log_metric("Price MSE", price_mse)
        mlflow.log_metric("Price MAE", price_mae)

    return jsonify({
        'rent_mse': rent_mse,
        'rent_mae': rent_mae,
        'price_mse': price_mse,
        'price_mae': price_mae
    })

if __name__ == '__main__':
    app.run(debug=True)
