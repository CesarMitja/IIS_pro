import pandas as pd
import great_expectations as ge
import json
import os
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, NumTargetDriftTab

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
listings_df_path1 = os.path.join(project_root, 'data', 'raw', 'listings.csv')
listings_df_path2 = os.path.join(project_root, 'data', 'raw', 'listings_rent.csv')
listings_df_path1_ref = os.path.join(project_root, 'data', 'raw', 'listings_ref.csv')
listings_df_path2_ref = os.path.join(project_root, 'data', 'raw', 'listings_rent_ref.csv')
report_dir = os.path.join(project_root, 'reports')

# Branje podatkov
data1 = pd.read_csv(listings_df_path1)
data2 = pd.read_csv(listings_df_path2)
data1_ref = pd.read_csv(listings_df_path1_ref)
data2_ref = pd.read_csv(listings_df_path2_ref)

# Ustvarjanje Great Expectations dataframe
ge_data1 = ge.from_pandas(data1)
ge_data2 = ge.from_pandas(data2)

# Definiranje pričakovanj za prvi dataframe
ge_data1.expect_column_values_to_not_be_null('Address')
ge_data1.expect_column_values_to_not_be_null('City')
ge_data1.expect_column_values_to_not_be_null('State')
ge_data1.expect_column_values_to_not_be_null('Zipcode')
ge_data1.expect_column_values_to_not_be_null('Price')
ge_data1.expect_column_values_to_not_be_null('Bedrooms')
ge_data1.expect_column_values_to_not_be_null('Bathrooms')
ge_data1.expect_column_values_to_not_be_null('Living Area')
ge_data1.expect_column_values_to_not_be_null('Type')

ge_data1.expect_column_values_to_be_between('Price', min_value=0)
ge_data1.expect_column_values_to_be_between('Bedrooms', min_value=0)
ge_data1.expect_column_values_to_be_between('Bathrooms', min_value=0)
ge_data1.expect_column_values_to_be_between('Living Area', min_value=0)

ge_data1.expect_column_values_to_be_of_type('Address', 'str')
ge_data1.expect_column_values_to_be_of_type('City', 'str')
ge_data1.expect_column_values_to_be_of_type('State', 'str')
ge_data1.expect_column_values_to_be_of_type('Zipcode', 'int64')
ge_data1.expect_column_values_to_be_of_type('Price', 'float64')
ge_data1.expect_column_values_to_be_of_type('Bedrooms', 'float64')
ge_data1.expect_column_values_to_be_of_type('Bathrooms', 'float64')
ge_data1.expect_column_values_to_be_of_type('Living Area', 'float64')
ge_data1.expect_column_values_to_be_of_type('Type', 'str')

# Definiranje pričakovanj za drugi dataframe
ge_data2.expect_column_values_to_not_be_null('Price')
ge_data2.expect_column_values_to_not_be_null('Bedrooms')
ge_data2.expect_column_values_to_not_be_null('Bathrooms')
ge_data2.expect_column_values_to_not_be_null('Living Area')
ge_data2.expect_column_values_to_not_be_null('Type')

ge_data2.expect_column_values_to_be_between('Price', min_value=0)
ge_data2.expect_column_values_to_be_between('Bedrooms', min_value=0)
ge_data2.expect_column_values_to_be_between('Bathrooms', min_value=0)
ge_data2.expect_column_values_to_be_between('Living Area', min_value=0)

ge_data2.expect_column_values_to_be_of_type('Price', 'float64')
ge_data2.expect_column_values_to_be_of_type('Bedrooms', 'float64')
ge_data2.expect_column_values_to_be_of_type('Bathrooms', 'float64')
ge_data2.expect_column_values_to_be_of_type('Living Area', 'float64')
ge_data2.expect_column_values_to_be_of_type('Type', 'str')

# Validacija podatkov
result1 = ge_data1.validate()
result2 = ge_data2.validate()

# Pretvorba rezultatov v JSON serializirano obliko
result1_json = result1.to_json_dict()
result2_json = result2.to_json_dict()

# Shranjevanje rezultatov v datoteko
os.makedirs(report_dir, exist_ok=True)
with open(os.path.join(report_dir, 'validation_results_price.json'), 'w') as f:
    json.dump(result1_json, f, indent=4)

with open(os.path.join(report_dir, 'validation_results_rent.json'), 'w') as f:
    json.dump(result2_json, f, indent=4)

print("Rezultati validacije so bili shranjeni v datoteko.")

# Evidently - Ustvarjanje poročil
# Preverimo prisotnost potrebnih stolpcev in jih odstranimo, če so prazni
# required_columns = ['Lot Area', 'Zipcode']
# for column in required_columns:
#     if column not in data1.columns or data1[column].isnull().all():
#         data1.drop(columns=[column], inplace=True)
#     if column not in data1_ref.columns or data1_ref[column].isnull().all():
#         data1_ref.drop(columns=[column], inplace=True)
#     if column not in data2.columns or data2[column].isnull().all():
#         data2.drop(columns=[column], inplace=True)
#     if column not in data2_ref.columns or data2_ref[column].isnull().all():
#         data2_ref.drop(columns=[column], inplace=True)

# Ustvarjanje Evidently dashboard za listings
dashboard1 = Dashboard(tabs=[DataDriftTab(), NumTargetDriftTab()])
dashboard1.calculate(data1_ref, data1)
dashboard1.save(os.path.join(report_dir, 'evidently_report_listings.html'))

# Ustvarjanje Evidently dashboard za listings_rent
dashboard2 = Dashboard(tabs=[DataDriftTab(), NumTargetDriftTab()])
dashboard2.calculate(data2_ref, data2)
dashboard2.save(os.path.join(report_dir, 'evidently_report_listings_rent.html'))

print("Evidently poročila so bila shranjena v datoteke.")