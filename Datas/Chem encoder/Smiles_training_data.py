import pandas as pd
import json

# Load the CSV file
df = pd.read_csv(r'C:\Users\George\Desktop\ISEF-2023\Datas\Chem encoder\SMILES_Big_Data_Set.csv')

# Drop the 2nd, 3rd, 4th, and 5th columns
df = df.drop(columns=['pIC50','mol','num_atoms','logP'])

# Convert the DataFrame column to a list
result_list = df['SMILES'].tolist()

# Save the list as a JSON file
with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Chem encoder\SMILES_list.json', 'w') as json_file:
    json.dump(result_list, json_file, indent=2)

print('JSON file created successfully!')