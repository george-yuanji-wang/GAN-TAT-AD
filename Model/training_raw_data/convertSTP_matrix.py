import pandas as pd
import json

# Load protein list and signal transduction network data
with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\back up\merged_protein_file.json', 'r') as f:
    protein_list = json.load(f)

with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Signal pathway\back up data\cell_communication_edge_node_relation_cleaned copy.json', 'r') as f:
    network_data = json.load(f)

adj_matrix = pd.DataFrame(index=protein_list, columns=protein_list)

# Create a list of interaction scores
interaction_scores = []

# Fill the interaction_scores list with interaction information
for protein, protein_data in network_data.items():
    for interacted_protein in protein_data['interactions'].keys():
        interaction_scores.append([protein, interacted_protein, 1])

# Create a DataFrame from the interaction_scores list
interaction_df = pd.DataFrame(interaction_scores, columns=['protein', 'interacted_protein', 'interaction_score'])

# Use pd.concat to join all columns at once
adj_matrix = pd.concat([
    interaction_df.pivot(index='protein', columns='interacted_protein', values='interaction_score').fillna(0),
    adj_matrix
])

# Save the directed adjacency matrix as a JSON file
with open('directed_adjacency_matrix_signal_transduction.json', 'w') as f:
    json.dump(adj_matrix.to_dict(), f)

print('Directed adjacency matrix for signal transduction network created successfully!')