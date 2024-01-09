import igraph as ig
import json
import numpy as np
import pandas as pd
import csv

PPI = ig.Graph.Load(r'C:\Users\George\Desktop\ISEF-2023\Network construction\PPI_homo_graph_initialize.graphml', format='graphml')
print(PPI.summary())
df_proteins = pd.read_csv(r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\protein_features\protein_features_beforefilter.csv')

print(df_proteins.head())

i=0
pagerank = PPI.pagerank()
cluster_coeffs = PPI.transitivity_local_undirected(mode="zero")
nnd = PPI.neighborhood_size(order=1, mode="all")
for index, row in df_proteins.iterrows():
    protein_id = row["Protein"]
    
    # Find the index of the node with the corresponding protein_id
    node_index = PPI.vs.find(name=protein_id).index

    # Calculate Indegree for protein nodes
    protein_indegree = PPI.degree(node_index, mode="in")
    protein_outdegree = PPI.degree(node_index, mode="out")
    #protein_betweenness = PPI.betweenness(vertices=node_index)
    protein_closeness = PPI.closeness(vertices=node_index)
    protein_pagerank = pagerank[node_index]
    protein_cluster_coefficients = cluster_coeffs[node_index]
    protein_nnd = nnd[node_index]

    PPI.vs[node_index]["Indegree"] = protein_indegree
    PPI.vs[node_index]["Outdegree"] = protein_outdegree
    #PPI.vs[node_index]["Betweenness"] = protein_betweenness
    PPI.vs[node_index]["Closeness"] = protein_closeness
    PPI.vs[node_index]["Pagerank"] = protein_pagerank
    PPI.vs[node_index]["Cluster_coefficients"] = protein_cluster_coefficients
    PPI.vs[node_index]["Nearest_Neighbor_Degree"] = protein_nnd
    
    # Update attributes for the node
    PPI.vs[node_index]["Similarity"] = row["SIMILARITY"]
    PPI.vs[node_index]["Subunit"] = row["SUBUNIT"]
    PPI.vs[node_index]["Transmembrane"] = row["Transmembrane"]
    PPI.vs[node_index]["Catalytic_activity"] = row["CATALYTIC ACTIVITY"]
    PPI.vs[node_index]["Interaction"] = row["INTERACTION"]
    PPI.vs[node_index]["Tissue_Specificity"] = row["TISSUE SPECIFICITY"]
    PPI.vs[node_index]["Disease"] = row["DISEASE"]
    PPI.vs[node_index]["Sequence_conflict"] = row["Sequence conflict"]
    PPI.vs[node_index]["Modified_residue"] = row["Modified residue"]
    PPI.vs[node_index]["Function"] = row["FUNCTION"]
    PPI.vs[node_index]["Binding_site"] = row["Binding site"]
    PPI.vs[node_index]["Natural_variant"] = row["Natural variant"]
    PPI.vs[node_index]["Alternative_products"] = row["ALTERNATIVE PRODUCTS"]
    PPI.vs[node_index]["Subcellular_location"] = row["SUBCELLULAR LOCATION"]
    PPI.vs[node_index]["Active_site"] = row["Active site"]
    PPI.vs[node_index]["Disulfide_bond"] = row["Disulfide bond"]
    PPI.vs[node_index]["Mutagenesis"] = row["Mutagenesis"]
    PPI.vs[node_index]["PTM"] = row["PTM"]
    PPI.vs[node_index]["STP_involvement"] = row["STP involvement"]
    
    # Update other attributes similarly

    # Set the label as an attribute
    PPI.vs[node_index]["label"] = row["label"]

    i+=1
    print(i)


# Assuming you want to check the status of the first few nodes
selected_nodes = PPI.vs[2534:2539]  # Change the slice as needed

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

graph_file = r"C:\Users\George\Desktop\ISEF-2023\Network construction\PPI_homo_graph_features_loaded.graphml"
PPI.save(graph_file, format="graphml")
print("saved")

