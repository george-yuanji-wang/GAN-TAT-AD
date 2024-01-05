import pandas as pd
import json
import numpy as np
import csv

with open(r'C:\Users\George\Desktop\ISEF-2023\temp_matrix const\merged_protein_file2_with string.json', 'r') as f:
    protein_list = json.load(f)

#num_proteins = len(protein_list)
#adjacency_matrix = np.zeros((num_proteins, num_proteins))
adj_matrix = pd.DataFrame(index=protein_list, columns=protein_list)

with open(r'C:\Users\George\Desktop\ISEF-2023\temp_matrix const\PPI copy.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    next(reader)
    for row in reader:

        protein1 = row[0]
        protein2 = row[1]
        score = float(row[2])

        if protein1 in protein_list and protein2 in protein_list:

            adj_matrix.loc[protein1, protein2] = score

a_matrix = adj_matrix.fillna(0)
# Convert DataFrame to a numpy array
j_matrix = a_matrix.to_numpy()
np.savetxt('33PPI_matrix.txt', j_matrix, fmt='%.3f')

print("PPI")