import numpy as np
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DDI_matrix_path = r'C:\Users\George\Desktop\ISEF-2023\Model\matrix\DDI_adj_matrix.txt'
PPI_matrix_path = r'C:\Users\George\Desktop\ISEF-2023\Model\matrix\_PPSS_matrix.txt'
protein_list_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\back up\merged_protein_without_PPI.json'

with open(protein_list_path, "r") as json_file:
    protein_list = json.load(json_file)
    pl = len(protein_list)

PP_matrix = np.loadtxt(PPI_matrix_path)

print("done loading data")

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_sigmoid=True):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU() if not use_sigmoid else nn.Sigmoid()

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = self.linear(x)
        x = self.activation(x)
        return x

class GraphConvolutionalAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(GraphConvolutionalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            GraphConvolution(input_dim, 4096),
            GraphConvolution(4096, 2048),
            GraphConvolution(2048, 1024),
            GraphConvolution(1024, 512),
            GraphConvolution(512, 256),
            GraphConvolution(256, 128),
            GraphConvolution(128, 64)
        )
        self.decoder = nn.Sequential(
            GraphConvolution(64, 128),
            GraphConvolution(128, 256),
            GraphConvolution(256, 512),
            GraphConvolution(512, 1024),
            GraphConvolution(1024, 2048),
            GraphConvolution(2048, 4096),
            GraphConvolution(4096, input_dim, use_sigmoid=True)
        )
    def forward(self, x, adj):
        encoded = x
        for layer in self.encoder:
            encoded = layer(encoded, adj)

        decoded = encoded
        for layer in self.decoder:
            decoded = layer(decoded, adj)

        return decoded

# Example usage
adjacency_matrix = PP_matrix  # Example adjacency matrix

# Convert adjacency matrix to PyTorch tensor
adj_tensor = torch.Tensor(adjacency_matrix)

# Create the GCN autoencoder model
input_dim = PP_matrix.shape[0]  # Dimension based on the number of nodes in the graph
model = GraphConvolutionalAutoencoder(input_dim)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
threshold = 0.01  # Set the difference threshold

print("Done initialization")

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(adj_tensor, adj_tensor)
    loss = criterion(output, adj_tensor)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Compute the difference between decoded output and original adjacency matrix
    diff = torch.abs(output - adj_tensor).mean().item()

    # Print loss and difference
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Difference: {diff:.4f}")

    # Check if the difference is below the threshold
    if diff < threshold:
        print("Training stopped as the difference threshold is reached.")
        break

# Get the embedded version
embedded_adjacency = adj_tensor
for layer in model.encoder:
    embedded_adjacency = layer(embedded_adjacency, adj_tensor)

print("Embedded adjacency matrix:", embedded_adjacency)