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
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE
import neptune
import wandb
import os
from bunch_of_KNN_ import run_KNNss_100

csv_file_path = r'C:\Users\George\Desktop\ISEF-2023\Model\test\GAE_PPI\embed_8.csv'

run_KNNss_100(csv_file_path)
