
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
from torch_geometric.nn.models.autoencoder import ARGA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import (v_measure_score, homogeneity_score, completeness_score)
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import neptune
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc


def plot_roc_curve(fpr, tpr, model_name, auc_score):
    plt.plot(fpr, tpr, label=f'{model_name} (auc = {auc_score:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc=4)
    plt.show()

# Function to compute AUC-ROC for binary classification
def compute_roc_auc(model, X_test, y_test):
    # Compute probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    # Compute AUC-ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score

node_list_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\back up\current_protein_Signal+meta+targets.json'
with open(node_list_path, 'r') as file:
    node_list = json.load(file)
graph = r'C:\Users\George\Desktop\ISEF-2023\Network construction\PPI_homo_graph_features_loaded.graphml'
# Create an igraph object
PPI_graph = ig.Graph.Load(graph, format='graphml')

feature_keys = [
    "Indegree", "Outdegree", "Closeness", "Betweenness", "Pagerank", "Cluster_coefficients",
    "Nearest_Neighbor_Degree", "Similarity", "Subunit", "Transmembrane",
    "Catalytic_activity", "Interaction", "Tissue_Specificity", "Disease",
    "Sequence_conflict", "Modified_residue", "Function", "Binding_site",
    "Natural_variant", "Alternative_products", "Subcellular_location",
    "Active_site", "Disulfide_bond", "Mutagenesis", "PTM", "STP_involvement"
]

features = torch.tensor([
    PPI_graph.vs[key] for key in feature_keys
], dtype=torch.float).t()

# Assuming you have a label attribute in your graph
labels = torch.tensor(PPI_graph.vs["label"], dtype=torch.float)

df = pd.DataFrame(features.numpy())
df['label'] = labels.numpy()

from sklearn.model_selection import train_test_split

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Initialize models
lr = LogisticRegression()
rf = RandomForestClassifier()
svc = SVC()
knn = KNeighborsClassifier()

model = rf

model.fit(X_train_smote, y_train_smote)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X)
fpr, tpr, auc_score = compute_roc_auc(model, X_test, y_test)
name = 'Random Forest'
print(f"{name} AUC-ROC: {auc_score:.2f}")
plot_roc_curve(fpr, tpr, name, auc_score)
print(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"{name} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")


prob = y_prob[:, 1]
names = np.array(PPI_graph.vs["name"])

print(prob.shape, names.shape)

output = pd.DataFrame({
    "Name": names,
    "Probability": prob
})

output = output.sort_values(by="Probability", ascending=False)
csv_file_path = 'prediction_base_nograph_all.csv'
output.to_csv(csv_file_path, index=False)

my_list = ['P18507', 'P23416', 'P21918', 'P31513', 'P28472', 'P35367', 'O00591', 'P19838', 'P19634', 'P08908', 'P47869', 'Q00653', 'P18505', 'P31645', 'Q12809', 'P02768', 'P98066', 'P35348', 'P10635', 'P05067', 'Q12879', 'P08173', 'O15399', 'P14867', 'P20813', 'Q8N1C3', 'P47870', 'P08913', 'P06276', 'P21728', 'P14416', 'P35368', 'P24462', 'P46098', 'P31644', 'P20309', 'P11712', 'O14764', 'P08172', 'P20815', 'O60391', 'Q13224', 'P50406', 'P08588', 'P28566', 'Q14957', 'P22303', 'P35462', 'Q96FL8', 'Q9UNQ0', 'P25100', 'P28222', 'Q9UN88', 'P08684', 'P33261', 'P02763', 'P11229', 'P07550', 'P28223', 'O75311', 'Q9HB55', 'P21917', 'P48167', 'Q9H015', 'P22310', 'P28221', 'Q16445', 'P28335', 'P13945', 'P23415', 'P78334', 'P36544', 'P05177', 'P34903', 'P30939', 'Q99928', 'Q05586', 'P08183', 'P48169', 'Q8TCU5', 'P11509', 'P05181', 'P18089']

output = output[~output['Name'].isin(my_list)]
print(output.all)

csv_file_path = 'prediction_base_nograph_unknown.csv'
output.to_csv(csv_file_path, index=False)
# Train and evaluate models
'''
models = {'Logistic Regression': lr, 'Random Forest': rf, 'SVC': svc, 'KNN': knn}
for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    fpr, tpr, auc_score = compute_roc_auc(model, X_test, y_test)
    print(f"{name} AUC-ROC: {auc_score:.2f}")
    plot_roc_curve(fpr, tpr, name, auc_score)
    print(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"{name} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")
'''