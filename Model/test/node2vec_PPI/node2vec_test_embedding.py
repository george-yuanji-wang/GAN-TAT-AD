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
from bunch_of_KNN_ import run_KNNss_100
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

node_list_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\back up\current_protein_Signal+meta+targets.json'
with open(node_list_path, 'r') as file:
    node_list = json.load(file)

# Load your adjacency matrix
path = r'C:\Users\George\Desktop\ISEF-2023\Model\matrix\_PPSS.txt'
matrix = np.loadtxt(path)
print(matrix.sum())
graph = r'C:\Users\George\Desktop\ISEF-2023\Network construction\PPI_homo_graph_features_loaded.graphml'
# Create an igraph object
PPI_graph = ig.Graph.Load(graph, format='graphml')

# Set node names for igraph graph
PPI_graph.vs['name'] = node_list

print(PPI_graph.summary())

PPI_graphx = Graph.to_networkx(PPI_graph)

# Specify the parameters for node2vec
dimensions = 128  # Dimensionality of node embeddings
num_walks = 100
walk_length = 50

p=1
q=1

# Create a node2vec model
node2vec = Node2Vec(PPI_graphx, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=10, p=p,q=q)

# Fit the model to generate embeddings
model = node2vec.fit(window=10, min_count=1)

embedding_data = []
for node in model.wv.key_to_index:
    vector = model.wv.get_vector(node)
    node_name = PPI_graphx.nodes[int(node)]['name']  # Retrieve node name from the graph
    embedding_data.append([node_name] + vector.tolist())

column_names = ['node_name'] + [f'dim_{i}' for i in range(model.vector_size)]
embedding_df = pd.DataFrame(embedding_data, columns=column_names)


print(embedding_df.shape)  # Output: (7393, 129)
print(embedding_df.head())  # Output: Display the first few rows of the DataFrame

embedding_df.to_csv(r'C:\Users\George\Desktop\ISEF-2023\Model\test\node2vec_embedding_100walks.csv', index=False)

#run_KNNss_100('node2vec_embedding_100walks.csv')

colors = [
    '#ffc0cb', '#bada55'
]

# Extract the embedding vectors
embedding_vectors = embedding_df.iloc[:, 1:-1].values
labels = embedding_df.iloc[:, -1].values

# Apply t-SNE to reduce the dimensionality to 2
tsne = TSNE(n_components=2, random_state=0)
embeddings_2d = tsne.fit_transform(embedding_vectors)

colors = [
    '#ffc0cb', '#bada55'
]

# Create a scatter plot of the embeddings
plt.figure(figsize=(8, 8))
for i in range(len(labels)):
        if labels[i] == 0:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=20, color=colors[0])
        elif labels[i] == 1:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=20, color=colors[1])

plt.title('t-SNE Plot of Node Embeddings')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig(r'C:\Users\George\Desktop\ISEF-2023\Model\test\node2vec_PPI\embedded_photo.png', dpi=300)  # Save the plot as a PNG file with 300 dpi resolution
plt.show()

