import json
import torch
import random
import itertools
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import random
import igraph as ig

seed=42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Load Wiki-CS Dataset
def load_PPI():

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
    
    features_list = []

    # Iterate over all nodes and extract the features
    for vertex in PPI_graph.vs:
        # Extract and concatenate the features for this node
        node_features = [vertex[attribute] for attribute in feature_keys]
        features_list.append(node_features)

    # Convert the features list to a PyTorch FloatTensor
    features = torch.FloatTensor(np.array(features_list))

    # Extract labels (assuming 'label' is the attribute name for labels)
    labels = torch.LongTensor(np.array([vertex['label'] for vertex in PPI_graph.vs]))

    # Create edge list
    edge_list = [(edge.source, edge.target) for edge in PPI_graph.es]

    # Create adjacency matrix
    src, dst = tuple(zip(*edge_list))
    adj = np.unique(np.array([src, dst]).T, axis=0)
    adj = sp.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(adj.max()+1, adj.max()+1), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = torch.FloatTensor(np.array(adj.todense()))

    idx_targets = [x for x in labels if x == 1]
    idx_ntargets = [x for x in labels if x == 0]

    idx_targets_train, idx_targets_test = train_test_split(idx_targets, test_size=0.3, random_state=1)
    idx_ntargets_train, idx_ntargets_test = train_test_split(idx_ntargets, test_size=0.3, random_state=1)

    idx_targets_train, idx_targets_val = train_test_split(idx_targets_train, test_size=0.1429, random_state=1)
    idx_ntargets_train, idx_ntargets_val = train_test_split(idx_ntargets_train, test_size=0.1429, random_state=1)
    
    idx_train = np.concatenate((np.array(idx_ntargets_train), np.array(idx_targets_train)))
    idx_test = np.concatenate((np.array(idx_ntargets_test), np.array(idx_targets_test)))
    idx_val = np.concatenate((np.array(idx_ntargets_val), np.array(idx_targets_val)))

    idx_train = torch.tensor(idx_train)
    idx_test = torch.tensor(idx_test)
    idx_val = torch.tensor(idx_val)

    return idx_train, idx_val, idx_test, adj, features, labels


def refine_label_order(labels):
    max_label = labels.max()
    j = 0

    for i in range(labels.max(), 0, -1):
        if sum(labels==i) >= 101 and i > j:
            while sum(labels==j) >= 101 and i > j:
                j = j+1
            if i > j:
                head_ind = labels == j
                tail_ind = labels == i
                labels[head_ind] = i
                labels[tail_ind] = j
                j = j + 1
            else:
                break
        elif i <= j:
            break

    return labels


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_cora(num_per_class=20, num_im_class=3, im_ratio=0.5):

    print('Loading cora dataset...')

    idx_features_labels = np.genfromtxt("../data/cora/cora.content", dtype=np.dtype(str))
    edges_unordered = np.genfromtxt("../data/cora/cora.cites", dtype=np.int32)

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1]
    
    classes_dict = {'Neural_Networks': 0, 'Reinforcement_Learning': 1, 'Probabilistic_Methods': 2, 'Case_Based': 3, 'Theory': 4, 'Rule_Learning': 5, 'Genetic_Algorithms': 6}
    labels = np.array(list(map(classes_dict.get, labels)))

    idx_dict = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_dict.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = torch.FloatTensor(np.array(normalize(features).todense()))
    labels = torch.LongTensor(labels)
    adj = torch.FloatTensor(np.array(adj.todense()))


    num_per_class_list = []
    for i in range(labels.max().item() + 1):
        if i > labels.max().item() - num_im_class:
            num_per_class_list.append(int(num_per_class * im_ratio))
        else:
            num_per_class_list.append(num_per_class)

    num_classes = len(set(labels.tolist()))
    train_idx = []
    val_idx = []
    test_idx = []

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i, len(c_idx)))
        random.shuffle(c_idx)

        train_idx = train_idx + c_idx[:num_per_class_list[i]]
        val_idx = val_idx + c_idx[num_per_class_list[i]:num_per_class_list[i]+25]
        test_idx = test_idx + c_idx[num_per_class_list[i]+25:num_per_class_list[i]+80]

    random.shuffle(train_idx)

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, adj, features, labels


# Load BlogCatalog Dataset
def load_BlogCatalog():
    mat = loadmat('../data/BlogCatalog/blogcatalog.mat')
    embed = np.loadtxt('../data/BlogCatalog/blogcatalog.embeddings_64')

    feature = np.zeros((embed.shape[0],embed.shape[1]-1))
    feature[embed[:,0].astype(int),:] = embed[:,1:]
    features = normalize(feature)

    adj = mat['network']
    label = mat['group']

    labels = np.array(label.todense().argmax(axis=1)).squeeze()
    labels[labels>16] = labels[labels>16]-1
    labels = refine_label_order(labels)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    adj = torch.FloatTensor(np.array(adj.todense()))


    num_classes = len(set(labels.tolist()))
    train_idx = []
    val_idx = []
    test_idx = []

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)

        c_num = len(c_idx)
        if c_num < 4:
            if c_num < 3:
                print("too small class type")
            batch_train = 1
            batch_val = 1
            batch_test = 1
        else:
            batch_train = int(c_num/4)
            batch_val = int(c_num/4)
            batch_test = int(c_num/2)

        train_idx = train_idx + c_idx[:batch_train]
        val_idx = val_idx + c_idx[batch_train:batch_train+batch_val]
        test_idx = test_idx + c_idx[batch_train+batch_val:batch_train+batch_val+batch_test]

    random.shuffle(train_idx)

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, adj, features, labels


# Load Wiki-CS Dataset
def load_wiki_cs():
    raw = json.load(open('../data/wiki-cs/data.json'))
    features = torch.FloatTensor(np.array(raw['features']))
    labels = torch.LongTensor(np.array(raw['labels']))

    edge_list = list(itertools.chain(*[[(i, nb) for nb in nbs] for i, nbs in enumerate(raw['links'])]))
    src, dst = tuple(zip(*edge_list))
    adj = np.unique(np.array([src, dst]).T, axis=0)
    adj = sp.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(adj.max()+1, adj.max()+1), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = torch.FloatTensor(np.array(adj.todense()))


    num_classes = len(set(labels.tolist()))
    train_idx = []
    val_idx = []
    test_idx = []

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)

        c_num = len(c_idx)
        if c_num < 4:
            if c_num < 3:
                print("too small class type")
            batch_train = 1
            batch_val = 1
            batch_test = 1
        else:
            batch_train = int(c_num/4)
            batch_val = int(c_num/4)
            batch_test = int(c_num/2)

        train_idx = train_idx + c_idx[:batch_train]
        val_idx = val_idx + c_idx[batch_train:batch_train+batch_val]
        test_idx = test_idx + c_idx[batch_train+batch_val:batch_train+batch_val+batch_test]

    random.shuffle(train_idx)

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, adj, features, labels