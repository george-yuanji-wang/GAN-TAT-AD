import numpy as np
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Move a tensor to the GPU

random_matrix = np.random.choice([0, 1], size=(200, 200), p=[0.5, 0.5])
np.fill_diagonal(random_matrix, 0)

#other_PPI = r'C:\Users\George\Desktop\PIN_Data.csv'
#df = pd.read_csv(other_PPI, skiprows=0, index_col=0)
#ppionline = df.to_numpy()
#print(ppionline.shape)


# Load your graph from the adjacency matrix
path = r'C:\Users\George\Desktop\ISEF-2023\Model\matrix\_PPSS.txt'
matrix = np.loadtxt(path)
#matrix = ppionline
#PPI_graph = Graph.Adjacency(matrix[:, :])

# Assuming you have a matrix 'matrix' as your input data
data_tensor = torch.tensor(matrix, dtype=torch.float32)

# Define the model
class RowAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(RowAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 3000),
            nn.ReLU(),
            nn.Linear(3000, 1500),
            nn.ReLU(),
            nn.Linear(1500, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(100, 150),
            nn.ReLU(),
            nn.Linear(150, 250),
            nn.ReLU(),
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.Linear(500, 1500),
            nn.ReLU(),
            nn.Linear(1500, 3000),
            nn.ReLU(),
            nn.Linear(3000, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define dimensions
input_dim = 7393  # Adjust based on your data
embedding_dim = 100  # Adjust as needed

embedmatrix = np.array([])

# Define hyperparameters
batch_size = 10  # Adjust as needed
num_epochs = 10  # Adjust as needed

# Create a DataLoader for batching the data
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate the autoencoder model
model = RowAutoencoder(input_dim, embedding_dim)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    total_loss = 0.0
    t = 0
    for batch in dataloader:
        # Forward pass
        inputs = batch[0]
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, inputs)

        # Accumulate the loss
        total_loss += loss.item()
        print(f"batch{t}: {loss.item()}")
        t += 1

        # Backpropagation
        loss.backward()

    # Average the gradients
    for param in model.parameters():
        param.grad /= t

    # Parameter update
    optimizer.step()

    # Print the average loss for the epoch
    average_loss = total_loss / t
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss}")


# After training, you can access the embeddings of the entire matrix
embeddings = model.encoder(data_tensor)