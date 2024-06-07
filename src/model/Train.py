import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import dagshub
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from onnxruntime.quantization import quantize_dynamic, QuantType

# Konfiguracija MLflow
dagshub_token = '22495012faf69bd7449136c47feddea65bd1ff8c'
dagshub.auth.add_app_token(dagshub_token)
dagshub.init(repo_name="IIS_pro", repo_owner="CesarMitja", mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/CesarMitja/IIS_pro.mlflow')
mlflow.set_experiment('Real_Estate_Price_Prediction')

client = MlflowClient()
try:
    latest_version = client.get_latest_versions("Price_Prediction_Model")[0]
    best_test_accuracy = client.get_metric_history(latest_version.run_id, "Test Accuracy")[-1].value
except (IndexError, ValueError, mlflow.exceptions.MlflowException):
    best_test_accuracy = 0

# Function to handle missing data
def handle_missing_data(df):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].median())

    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])
    return df

# Load the data
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
listings_df_path = os.path.join(project_root, 'data', 'raw', 'listings.csv')

# Load the data with error handling
try:
    listings_df = pd.read_csv(listings_df_path)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Handle missing values
listings_df = handle_missing_data(listings_df)

# Convert 'Price' to a numeric type, coercing any errors
listings_df['Price'] = pd.to_numeric(listings_df['Price'], errors='coerce')

# Drop any rows that now have NaN in 'Price' after this operation
listings_df = listings_df.dropna(subset=['Price'])

# Categorize the 'Price' into bins of $10,000
try:
    price_bins = np.arange(listings_df['Price'].min(), listings_df['Price'].max() + 10000, 10000)
    price_labels = [f"${int(b)}-${int(b) + 10000}" for b in price_bins[:-1]]
    listings_df['Price Category'] = pd.cut(listings_df['Price'], bins=price_bins, labels=price_labels, right=False)
except Exception as e:
    print(f"Error categorizing prices: {e}")
    exit()
listings_df.rename(columns={'Living Area': 'Living_Area','Lot Area': 'Lot_Area'}, inplace=True)
# Prepare the data for modeling
X = listings_df.drop(['Price Category', 'Price', 'Address', 'Zipcode', 'City', 'State'], axis=1, errors='ignore')
y = listings_df['Price Category']

print(X.head())
# Label Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Select categorical and numeric features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Create transformers for numeric and categorical features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine into a single preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define a list of models to evaluate
models = {
    'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42),
    'Random Forest Classifier': RandomForestClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Classifier': SVC(probability=True)
}


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define input names for ONNX based on feature names
initial_type = [(name, FloatTensorType([None, 1])) for name in numeric_features] + \
               [(name, StringTensorType([None, 1])) for name in categorical_features]

# Evaluate each model and check for overfitting
results = {}
best_model = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Create the modeling pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Predict and evaluate the model on the training data
        y_train_pred = pipeline.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Predict and evaluate the model on the test data
        y_pred = pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)
        
        # Log parameters, metrics, and model to MLflow
        mlflow.log_param("Model", name)
        mlflow.log_metric("Train Accuracy", train_accuracy)
        mlflow.log_metric("Test Accuracy", test_accuracy)
        mlflow.log_metric("Test MSE", test_mse)
        
        # Store results
        results[name] = {'Train Accuracy': train_accuracy, 'Test Accuracy': test_accuracy, 'Test MSE': test_mse}
        
        print(f"{name} - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}, Test MSE: {test_mse:.4f}")
        
        # Update the best model if this model is better
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_model = pipeline
            # Save preprocessor and label encoder
            preprocessor_path = os.path.join(project_root, 'models', 'preprocessor.pkl')
            label_encoder_path = os.path.join(project_root, 'models', 'label_encoder.pkl')
            joblib.dump(preprocessor, preprocessor_path)
            joblib.dump(label_encoder, label_encoder_path)
            mlflow.log_artifact(preprocessor_path, "preprocessor")
            mlflow.log_artifact(label_encoder_path, "label_encoder")

            # Save the model as ONNX
            onnx_model = skl2onnx.convert_sklearn(best_model, initial_types=initial_type)
            onnx_model_path = os.path.join(project_root, 'models', 'best_model_price.onnx')
            with open(onnx_model_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            # Quantize the model
            quantized_model_path = os.path.join(project_root, 'models', 'quantized_model_price.onnx')
            quantize_dynamic(onnx_model_path, quantized_model_path, weight_type=QuantType.QUInt8)

            # Log the ONNX models
            mlflow.log_artifact(onnx_model_path, "onnx_model")
            mlflow.log_artifact(quantized_model_path, "quantized_onnx_model")
            mlflow.sklearn.log_model(pipeline, "model", registered_model_name="Price_Prediction_Model")

# Display all results
print("All model performances:", results)

# Save the best model
if best_model is not None:
    best_model_path = os.path.join(project_root, 'models', 'best_model_price.pkl')
    joblib.dump(best_model, best_model_path)
    mlflow.log_artifact(best_model_path, "best_model")
    print(f"Best model saved to {best_model_path}")
else:
    print("No best model found.")
