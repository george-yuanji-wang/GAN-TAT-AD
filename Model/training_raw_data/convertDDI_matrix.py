import pandas as pd
import numpy as np
import json

# Load drug list and DDI data
with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\back up\drug_list_final.json', 'r') as f:
    drug_list = json.load(f)

with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\DDI\backup data\DDI copy.json', 'r') as f:
    ddi_data = json.load(f)

# Create a DataFrame with drugs as columns and rows
adj_matrix = pd.DataFrame(index=drug_list, columns=drug_list)

# Fill the DataFrame with 1 if drugs interact (directed), 0 otherwise
for drug, interactions in ddi_data.items():
    adj_matrix.loc[drug, interactions] = 1

# Fill NaN with 0 (no interaction)
adj_matrix = adj_matrix.fillna(0)

# Convert DataFrame to a numpy array
adj_matrix_array = adj_matrix.to_numpy()

# Save the adjacency matrix as a JSON file
with open('directed_adjacency_matrix.json', 'w') as f:
    json.dump(adj_matrix_array.tolist(), f)

# Save the adjacency matrix as a TXT file for better visualization
np.savetxt('directed_adjacency_matrix.txt', adj_matrix_array, fmt='%d')

print('Directed adjacency matrix created successfully!')