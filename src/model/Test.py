import pandas as pd
import joblib
import os

# Load the best model
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

#Address,City,State,Zipcode,Price,Bedrooms,Bathrooms,Living Area,Lot Area,Type
#15510 Sierra Valle Dr,Houston,TX,77083,290000.0,4.0,2.0,1971.0,7200.468,SINGLE_FAMILY

# Path to the saved model
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
best_model_path = os.path.join(project_root, 'models', 'best_model.pkl')
model = load_model(best_model_path)

if model is not None:
    # Example of new data (adjust according to your features)
    new_data = {
        'Bedrooms': [4],
        'Bathrooms': [2],
        'Living Area': [1970],
        'Lot Area': [7200.3],
        'Type': ['SINGLE_FAMILY']
    }
    new_df = pd.DataFrame(new_data)
    
    # Predict using the loaded model
    prediction = model.predict(new_df)
    
    # Show prediction
    print("Predictions:", prediction)
else:
    print("Model could not be loaded.")
