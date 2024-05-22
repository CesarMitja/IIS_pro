import pandas as pd
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) 
input_file_path = os.path.join(project_root, 'data', 'raw','listings.csv')

 

# Read the data into a pandas DataFrame
df = pd.read_csv(input_file_path)

# Drop the last two columns
df_filtered = df.iloc[:, :-2]

# Define the file path for the output CSV file
output_file_path = os.path.join(project_root, 'data', 'raw','listings.csv')

# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv(output_file_path, index=False)

# Output the path to the new file
print(output_file_path)