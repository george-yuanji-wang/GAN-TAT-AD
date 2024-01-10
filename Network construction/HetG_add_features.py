import igraph as ig
import json
import numpy as np
import pandas as pd
import csv
import math

HetG = ig.Graph.Load(r'C:\Users\George\Desktop\ISEF-2023\Network construction\Het_graph_initialize.graphml', format='graphml')
print(HetG.summary())
df_proteins = pd.read_csv(r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\protein_features\protein_features_beforefilter.csv')

print(df_proteins.head())

i=0
pagerank = HetG.pagerank()
cluster_coeffs = HetG.transitivity_local_undirected(mode="zero")
nnd = HetG.neighborhood_size(order=1, mode="all")
betweenness = HetG.betweenness(directed=True)

for index, row in df_proteins.iterrows():
    protein_id = row["Protein"]
    
    # Find the index of the node with the corresponding protein_id
    node_index = HetG.vs.find(name=protein_id).index

    # Calculate Indegree for protein nodes
    protein_indegree = HetG.degree(node_index, mode="in")
    protein_outdegree = HetG.degree(node_index, mode="out")
    protein_closeness = HetG.closeness(vertices=node_index)
    if math.isnan(protein_closeness):
        protein_closeness = 0
    protein_pagerank = pagerank[node_index]
    protein_cluster_coefficients = cluster_coeffs[node_index]
    protein_nnd = nnd[node_index]
    protein_betweenness = betweenness[node_index]

    HetG.vs[node_index]["Indegree"] = protein_indegree
    HetG.vs[node_index]["Outdegree"] = protein_outdegree
    HetG.vs[node_index]["Betweenness"] = protein_betweenness
    HetG.vs[node_index]["Closeness"] = protein_closeness
    HetG.vs[node_index]["Pagerank"] = protein_pagerank
    HetG.vs[node_index]["Cluster_coefficients"] = protein_cluster_coefficients
    HetG.vs[node_index]["Nearest_Neighbor_Degree"] = protein_nnd
    
    # Update attributes for the node
    HetG.vs[node_index]["Similarity"] = row["SIMILARITY"]
    HetG.vs[node_index]["Subunit"] = row["SUBUNIT"]
    HetG.vs[node_index]["Transmembrane"] = row["Transmembrane"]
    HetG.vs[node_index]["Catalytic_activity"] = row["CATALYTIC ACTIVITY"]
    HetG.vs[node_index]["Interaction"] = row["INTERACTION"]
    HetG.vs[node_index]["Tissue_Specificity"] = row["TISSUE SPECIFICITY"]
    HetG.vs[node_index]["Disease"] = row["DISEASE"]
    HetG.vs[node_index]["Sequence_conflict"] = row["Sequence conflict"]
    HetG.vs[node_index]["Modified_residue"] = row["Modified residue"]
    HetG.vs[node_index]["Function"] = row["FUNCTION"]
    HetG.vs[node_index]["Binding_site"] = row["Binding site"]
    HetG.vs[node_index]["Natural_variant"] = row["Natural variant"]
    HetG.vs[node_index]["Alternative_products"] = row["ALTERNATIVE PRODUCTS"]
    HetG.vs[node_index]["Subcellular_location"] = row["SUBCELLULAR LOCATION"]
    HetG.vs[node_index]["Active_site"] = row["Active site"]
    HetG.vs[node_index]["Disulfide_bond"] = row["Disulfide bond"]
    HetG.vs[node_index]["Mutagenesis"] = row["Mutagenesis"]
    HetG.vs[node_index]["PTM"] = row["PTM"]
    HetG.vs[node_index]["STP_involvement"] = row["STP involvement"]
    
    # Update other attributes similarly

    # Set the label as an attribute
    HetG.vs[node_index]["label"] = row["label"]

    i+=1
    print(i)


# Assuming you want to check the status of the first few nodes
selected_nodes = HetG.vs[2534:2539]  # Change the slice as needed

# Print information for each selected node
for node in selected_nodes:
    print(f"\nNode ID: {node.index}")
    print(f"Protein ID: {node['name']}")
    print(f"Indegree: {node['Indegree']}")
    print(f"Outdegree: {node['Outdegree']}")
    #print(f"Betweenness: {node['Betweenness']}")
    print(f"Closeness: {node['Closeness']}")
    print(f"Pagerank: {node['Pagerank']}")
    print(f"Cluster Coefficients: {node['Cluster_coefficients']}")
    print(f"Nearest Neighbor Degree: {node['Nearest_Neighbor_Degree']}")
    print(f"Similarity: {node['Similarity']}")
    print(f"Subunit: {node['Subunit']}")
    # Print other attributes similarly

    print("------------")

graph_file = r"C:\Users\George\Desktop\ISEF-2023\Network construction\Het_graph_final.graphml.graphml"
HetG.save(graph_file, format="graphml")
print("saved")