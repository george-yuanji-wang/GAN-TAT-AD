import pandas as pd
import json
import numpy as np
import csv

# Load signal transduction network data
with open(r'C:\Users\George\Desktop\ISEF-2023\temp_matrix const\cell_communication_edge_node_relation_cleaned copy.json', 'r') as f:
    network_data = json.load(f)

# Load compound list
with open(r'C:\Users\George\Desktop\ISEF-2023\temp_matrix const\compound_list_final.json', 'r') as f:
    compound_list = json.load(f)

# Load drug-drug interaction data
with open(r'C:\Users\George\Desktop\ISEF-2023\temp_matrix const\DDI copy.json', 'r') as f:
    ddi_data = json.load(f)

# Load drug list
with open(r'C:\Users\George\Desktop\ISEF-2023\temp_matrix const\drug_list_final.json', 'r') as f:
    drug_list = json.load(f)

# Load drug-target interaction data
with open(r'C:\Users\George\Desktop\ISEF-2023\temp_matrix const\DTI_json copy.json', 'r') as f:
    dti_data = json.load(f)

# Load protein list
with open(r'C:\Users\George\Desktop\ISEF-2023\temp_matrix const\merged_protein_file2_with string.json', 'r') as f:
    protein_list = json.load(f)

# Load reaction map data
with open(r'C:\Users\George\Desktop\ISEF-2023\temp_matrix const\Reaction_Map_Uniprot_Pubchem_formatted copy.json', 'r') as f:
    reaction_map_data = json.load(f)

# Load reactions list
with open(r'C:\Users\George\Desktop\ISEF-2023\temp_matrix const\reactions_list_final.json', 'r') as f:
    reactions_list = json.load(f)
a = []
for i in reactions_list:
    a.append(i['Reaction'])
reactions_list = a



STPs = {}
for p, i in network_data.items():
    if p not in STPs.keys():
        STPs[p] = list((i['interactions']).keys())
    else:
        STPs[p].append((i['interactions']).keys())

STP_adj_matrix = pd.DataFrame(index=protein_list, columns=protein_list)
for protein, protein2 in STPs.items():
    for i in protein2:
        STP_adj_matrix.loc[protein, i] = 1

# Fill NaN with 0 (no interaction)
adj_matrix = STP_adj_matrix.fillna(0)

# Convert DataFrame to a numpy array
STP_adj_matrix = adj_matrix.to_numpy()
np.savetxt('STP_adj_matrix.txt', STP_adj_matrix, fmt='%d')

print("done STP")

adj_matrix = pd.DataFrame(index=drug_list, columns=drug_list)

for drug, interactions in ddi_data.items():
    adj_matrix.loc[drug, interactions] = 1

# Fill NaN with 0 (no interaction)
adj_matrix = adj_matrix.fillna(0)

# Convert DataFrame to a numpy array
adj_matrix_array = adj_matrix.to_numpy()
# Save the adjacency matrix as a TXT file for better visualization
np.savetxt('DDI_adj_matrix.txt', adj_matrix_array, fmt='%d')

print("done DDI")

DTIs = pd.DataFrame(index=protein_list, columns=drug_list)
for drug, protein in dti_data.items():
    for i in protein:
        DTIs.loc[i, drug] = 1

adj_matrix = DTIs.fillna(0)
# Convert DataFrame to a numpy array
adj_matrix_array = adj_matrix.to_numpy()
# Save the adjacency matrix as a TXT file for better visualization
np.savetxt('DTI_adj_matrix.txt', adj_matrix_array, fmt='%d')

print("done DTI")

#compound to reaction -> metabolic
CRs = pd.DataFrame(index=compound_list, columns=reactions_list)
for reaction, list in reaction_map_data.items():
    for compound in list[0]:
        CRs.loc[compound, reaction] = 1

adj_matrix = CRs.fillna(0)
# Convert DataFrame to a numpy array
adj_matrix_array = adj_matrix.to_numpy()
# Save the adjacency matrix as a TXT file for better visualization
np.savetxt('CR_matrix.txt', adj_matrix_array, fmt='%d')

print("done CR")

#protein to reaction -> metabolic
ERs = pd.DataFrame(index=protein_list, columns=reactions_list)
for reaction, list in reaction_map_data.items():
    for protein in list[1]:
        ERs.loc[protein, reaction] = 1

adj_matrix = ERs.fillna(0)
# Convert DataFrame to a numpy array
adj_matrix_array = adj_matrix.to_numpy()
# Save the adjacency matrix as a TXT file for better visualization
np.savetxt('PR_matrix.txt', adj_matrix_array, fmt='%d')

print("done PR")

#PPI
adj_matrix = pd.DataFrame(index=protein_list, columns=protein_list)

with open(r'C:\Users\George\Desktop\ISEF-2023\temp_matrix const\PPI copy.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    next(reader)
    for row in reader:

        protein1 = row[0]
        protein2 = row[1]
        score = float(row[2])

        if protein1 in protein_list and protein2 in protein_list:

            adj_matrix.loc[protein1, protein2] = score*1000

a_matrix = adj_matrix.fillna(0)
# Convert DataFrame to a numpy array
j_matrix = a_matrix.to_numpy()
np.savetxt('PPI_matrix.txt', j_matrix, fmt='%d')

print("PPI")
