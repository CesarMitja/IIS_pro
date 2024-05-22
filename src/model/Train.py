import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

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

# Prepare the data for modeling
X = listings_df.drop(['Price Category', 'Address'], axis=1, errors='ignore')  # Drop Address if present
y = listings_df['Price Category']

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
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
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
    'K-Nearest Neighbors': KNeighborsClassifier(3),
    'Extra Trees Classifier': ExtraTreesClassifier(random_state=42),
    'Random Forest Classifier': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(kernel='linear'),
    'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42)
}

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Evaluate each model and check for overfitting
results = {}
best_model = None
best_accuracy = 0

for name, model in models.items():
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
    
    # Store results
    results[name] = {'Train Accuracy': train_accuracy, 'Test Accuracy': test_accuracy}
    
    print(f"{name} - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Update the best model if this model is better
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model = pipeline

# Display all results
print("All model performances:", results)

# Save the best model
if best_model is not None:
    best_model_path = os.path.join(project_root, 'models', 'best_model.pkl')
    joblib.dump(best_model, best_model_path)
    print(f"Best model saved to {best_model_path}")
else:
    print("No best model found.")