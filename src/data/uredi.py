import pandas as pd
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) 
input_file_path = os.path.join(project_root, 'data', 'raw','listings.csv')

 

df = pd.read_csv(input_file_path)

df_filtered = df.iloc[:, :-2]

output_file_path = os.path.join(project_root, 'data', 'raw','listings.csv')

df_filtered.to_csv(output_file_path, index=False)

print(output_file_path)