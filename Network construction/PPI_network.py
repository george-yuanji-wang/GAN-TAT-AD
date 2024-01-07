
import numpy as np
import json
from igraph import Graph
import igraph as ig
import torch
path = r'C:\Users\George\Desktop\ISEF-2023\Model\matrix\PPI_matrix.txt'
matrix = np.loadtxt(path)
protein_list_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\back up\current_protein_Signal+meta+targets.json'
with open(protein_list_path, "r") as json_file:
    protein_list = json.load(json_file)
    pl = len(protein_list)

PPI_ = Graph.Adjacency(matrix[:, :])
edge_index = torch.tensor(PPI_.get_edgelist(), dtype=torch.long).t().contiguous()

count = 0
for i in matrix:
    for j in i:
        if j == 1:
            count += 1

print(count)

