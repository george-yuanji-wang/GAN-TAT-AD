import networkx as nx
from node2vec import Node2Vec
from igraph import Graph
import igraph as ig
import numpy as np
import json
import pandas as pd
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import DataLoader


node_list_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\back up\current_protein_Signal+meta+targets.json'
with open(node_list_path, 'r') as file:
    node_list = json.load(file)
graph = r'C:\Users\George\Desktop\ISEF-2023\Network construction\PPI_homo_graph_features_loaded.graphml'
# Create an igraph object
PPI_graph = ig.Graph.Load(graph, format='graphml')

feature_keys = [
    "Indegree", "Outdegree", "Closeness", "Pagerank", "Cluster_coefficients",
    "Nearest_Neighbor_Degree", "Similarity", "Subunit", "Transmembrane",
    "Catalytic_activity", "Interaction", "Tissue_Specificity", "Disease",
    "Sequence_conflict", "Modified_residue", "Function", "Binding_site",
    "Natural_variant", "Alternative_products", "Subcellular_location",
    "Active_site", "Disulfide_bond", "Mutagenesis", "PTM", "STP_involvement"
]

features = torch.tensor([
    PPI_graph.vs[key] for key in feature_keys
], dtype=torch.float).t()

edge_indices = torch.tensor(PPI_graph.get_edgelist(), dtype=torch.long).t()

# Assuming you have a label attribute in your graph
labels = torch.tensor(PPI_graph.vs["label"], dtype=torch.float).view(-1, 1)

# Create a PyTorch Geometric Data object
data = Data(x=features, edge_index=edge_indices, y=labels)
print(data)
# Define the GraphSAGE model
in_channels = data.num_features
hidden_channels = 16
num_layers = 3  # You can adjust this based on your needs
out_channels = 1 # Set to None if you don't want a final linear layer

model = GraphSAGE(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    num_layers=num_layers,
    out_channels=out_channels,
)

# Define the DataLoader
loader = DataLoader([data], batch_size=1, shuffle=True)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10  # Adjust based on your needs

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.binary_cross_entropy_with_logits(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}')

# Save node embeddings as JSON
model.eval()
with torch.no_grad():
    embeddings = model(data.x, data.edge_index).numpy().tolist()

with open(r'C:\Users\George\Desktop\ISEF-2023\Model\test\GraphSage\node_embeddings.json', 'w') as f:
    json.dump(embeddings, f)