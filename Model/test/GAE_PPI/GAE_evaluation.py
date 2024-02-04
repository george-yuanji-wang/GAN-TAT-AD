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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from torch_geometric.nn import GAE
import neptune
import wandb
import os
from bunch_of_KNN_ import run_KNNss_100
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt

csv_file_path = r'C:\Users\George\Desktop\ISEF-2023\Model\test\GAE_PPI\embed_38.csv'

df = pd.read_csv(csv_file_path)
name = np.array(df.iloc[:, 0].values)
features = np.array(df.iloc[:, 1:].values)

target = ['P18507', 'P23416', 'P21918', 'P31513', 'P28472', 'P35367', 'O00591', 'P19838', 'P19634', 'P08908', 'P47869', 'Q00653', 'P18505', 'P31645', 'Q12809', 'P02768', 'P98066', 'P35348', 'P10635', 'P05067', 'Q12879', 'P08173', 'O15399', 'P14867', 'P20813', 'Q8N1C3', 'P47870', 'P08913', 'P06276', 'P21728', 'P14416', 'P35368', 'P24462', 'P46098', 'P31644', 'P20309', 'P11712', 'O14764', 'P08172', 'P20815', 'O60391', 'Q13224', 'P50406', 'P08588', 'P28566', 'Q14957', 'P22303', 'P35462', 'Q96FL8', 'Q9UNQ0', 'P25100', 'P28222', 'Q9UN88', 'P08684', 'P33261', 'P02763', 'P11229', 'P07550', 'P28223', 'O75311', 'Q9HB55', 'P21917', 'P48167', 'Q9H015', 'P22310', 'P28221', 'Q16445', 'P28335', 'P13945', 'P23415', 'P78334', 'P36544', 'P05177', 'P34903', 'P30939', 'Q99928', 'Q05586', 'P08183', 'P48169', 'Q8TCU5', 'P11509', 'P05181', 'P18089']

labels = np.array([0 if x not in target else 1 for x in name])

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_resampled, y_train_resampled)

# Predict probabilities for the test set
y_pred_prob = rf_classifier.predict_proba(X_test)[:, 1]

# Calculate the AUC-ROC score
roc_auc = roc_auc_score(y_test, y_pred_prob)
print("AUC-ROC Score:", roc_auc)

# Generate the ROC curve plot
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Generate the confusion matrix
y_pred = rf_classifier.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Display classification report
print(classification_report(y_test, y_pred))