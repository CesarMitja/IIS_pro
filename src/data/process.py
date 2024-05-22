import pandas as pd
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) 
project_root1 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) 
pro_data_path = os.path.join(project_root, 'data', 'processed','listings_processed.csv')


# Load the data from a CSV file
raw_data_path = os.path.join(project_root1,'train.csv')
data = pd.read_csv(raw_data_path)

# Define a mapping from the building type to a more readable format
bldg_type_map = {
    '1Fam': 'SINGLE_FAMILY',
    '2fmCon': 'MULTI_FAMILY',
    'Duplex': 'MULTI_FAMILY',
    'Twnhs': 'TOWNHOUSE',
    'TwnhsE': 'TOWNHOUSE'
}

# Transform the data
transformed_data = pd.DataFrame()
transformed_data['Address'] = data['ID'].astype(str) + " Example St"
transformed_data['City'] = "Houston"
transformed_data['State'] = "TX"
transformed_data['Zipcode'] = 77000
transformed_data['Price'] = data['SalePrice']
transformed_data['Bedrooms'] = data['BedroomAbvGr']
transformed_data['Bathrooms'] = data['FullBath'] + data['HalfBath'] * 0.5
transformed_data['Living Area'] = data['GrLivArea']
transformed_data['Lot Area'] = data['LotArea'] / 43560.0  # Convert square feet to acres
transformed_data['Type'] = data['BldgType'].map(bldg_type_map)

output_file_path = pro_data_path
transformed_data.to_csv(output_file_path, index=False)

