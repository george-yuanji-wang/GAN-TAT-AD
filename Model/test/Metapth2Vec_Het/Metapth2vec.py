import networkx as nx
from node2vec import Node2Vec
from igraph import Graph
import igraph as ig
import numpy as np
import json
import pandas as pd
from gensim.models import Word2Vec
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, log_loss, matthews_corrcoef, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_recall_fscore_support as prf_support
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from torch_geometric.nn import MetaPath2Vec
from typing import Dict, Tuple
import torch
import tqdm

graphss = r'C:\Users\George\Desktop\ISEF-2023\Network construction\Het_graph_final.graphml'
HetG = ig.Graph.Load(graphss, format='graphml')
graph = HetG


def create_edge_index_dict(graph: ig.Graph) -> Dict[Tuple[str, str, str], torch.Tensor]:
    edge_index_dict = {}

    for edge in graph.es:
        src_node_type = graph.vs[edge.source]["type"]
        rel_type = edge["type"]
        dst_node_type = graph.vs[edge.target]["type"]

        key = (src_node_type, rel_type, dst_node_type)

        if key not in edge_index_dict:
            edge_index_dict[key] = []

        edge_index_dict[key].append((edge.source, edge.target))

    # Convert the lists of edge indices to torch.Tensor
    for key in edge_index_dict:
        edge_index_dict[key] = torch.tensor(edge_index_dict[key], dtype=torch.long).t()

    return edge_index_dict


# Assuming your graph is named HetG
edge_index_dict = create_edge_index_dict(HetG)
print(edge_index_dict)

device = "cpu"

metapath = [
    ('Protein', 'Protein-Protein-Physical', 'Protein')
]


model = MetaPath2Vec(edge_index_dict,
                     embedding_dim=128,
                     metapath=metapath,
                     walk_length=5,
                     context_size=3,
                     walks_per_node=3,
                     num_negative_samples=1,
                     sparse=True
                    ).to(device)

# loader = model.loader(batch_size=128, shuffle=True, num_workers=3)

#try:
#  for idx, (pos_rw, neg_rw) in enumerate(loader):
#    if idx == 10: break
#    print(idx, pos_rw.shape, neg_rw.shape)
#except IndexError:

optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

def train(epoch, log_steps=500, eval_steps=1000):
    model.train()

    total_loss = 0
    for i, (pos_rw, neg_rw) in enumerate(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_steps == 0:
            print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                   f'Loss: {total_loss / log_steps:.4f}'))
            total_loss = 0

        if (i + 1) % eval_steps == 0:
            acc = test()
            print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                   f'Acc: {acc:.4f}'))

@torch.no_grad()
def test(train_ratio=0.1):
    model.eval()

    z = model('author', batch=data.y_index_dict['author'])
    y = data.y_dict['author']

    perm = torch.randperm(z.size(0))
    train_perm = perm[:int(z.size(0) * train_ratio)]
    test_perm = perm[int(z.size(0) * train_ratio):]

    return model.test(z[train_perm], y[train_perm], z[test_perm],
                      y[test_perm], max_iter=150)

for epoch in range(1, 2):
    train(epoch)
    acc = test()
    print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')
'''















