import igraph as ig
import json
import numpy as np
import pandas as pd
import csv

# Load dictionary data
dictionary_file_1 = "C:/Users/George/Desktop/ISEF-2023/Datas/DDI/backup data/DDI copy.json"
dictionary_file_2 = "C:/Users/George/Desktop/ISEF-2023/Datas/DTI - SNAP/back up/DTI_json copy.json"
dictionary_file_3 = "C:/Users/George/Desktop/ISEF-2023/Datas/labels/alzheimer_disease/label_dictionary.json"
dictionary_file_4 = "C:/Users/George/Desktop/ISEF-2023/Datas/Metabolic/Back up datas/Reaction_Map_Uniprot_Pubchem_formatted copy.json"
dictionary_file_5 = r'C:\Users\George\Desktop\ISEF-2023\Datas\Signal pathway\back up data\cell_communication_edge_node_relation_cleaned copy.json'
csv_file_5 = r'C:\Users\George\Desktop\ISEF-2023\Datas\PPI\back up\PPI.csv'

adjacency_matrix_file = "C:/Users/George/Desktop/ISEF-2023/Model/matrix/_PPSS.txt"

node_list_file_1 = "C:/Users/George/Desktop/ISEF-2023/Datas/Node list/back up/current_protein_Signal+meta+targets.json"
node_list_file_2 = "C:/Users/George/Desktop/ISEF-2023/Datas/Node list/back up/drug_list_final.json"
node_list_file_3 = "C:/Users/George/Desktop/ISEF-2023/Datas/Node list/back up/reactions_list_final.json"
node_list_file_4 = "C:/Users/George/Desktop/ISEF-2023/Datas/Node list/back up/compound_list_final.json"

with open(dictionary_file_1, 'r') as f:
    DDI = json.load(f)
with open(dictionary_file_2, 'r') as f:
    DTI = json.load(f)
with open(dictionary_file_3, 'r') as f:
    Labels = json.load(f)
with open(dictionary_file_4, 'r') as f:
    Metabolic = json.load(f)
with open(dictionary_file_5, 'r') as f:
    STP = json.load(f)

# Load adjacency matrix
adjacency_matrix = np.loadtxt(adjacency_matrix_file)

# Load node list data
with open(node_list_file_1, 'r') as f:
    Proteins = json.load(f)
with open(node_list_file_2, 'r') as f:
    Drugs = json.load(f)
with open(node_list_file_3, 'r') as f:
    Reactions = json.load(f)
with open(node_list_file_4, 'r') as f:
    Compounds = json.load(f)

# Create an empty graph
HetG = ig.Graph()

# Add Drugs as nodes
for drug in Drugs:
    HetG.add_vertex(name=drug, type="Drug")
for protein in Proteins:
    HetG.add_vertex(name=protein, type="Protein")
for reaction in Reactions:
    reaction_ = reaction["Reaction"]
    HetG.add_vertex(name=reaction_, type="Reaction")
for compound in Compounds:
    HetG.add_vertex(name=compound, type="Compound")

n = HetG.vcount()
print(n)

i=0
# Read the CSV file and extract the protein pairs
with open(csv_file_5, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        protein1 = row[0]
        protein2 = row[1]
        if protein1 in Proteins and protein2 in Proteins:
            HetG.add_edge(protein1, protein2, type="Protein-Protein-Physical")
            i+=1
            print(i)

i=0
for protein1, info in STP.items():
    proteins = list(info['interactions'].keys())
    for protein2 in proteins:
        HetG.add_edge(protein1, protein2, type="Protein-Protein-STP")
        i+=1
        print(i)


i=0
for reaction, participant in Metabolic.items():
    compound = participant[0]
    protein = participant[1]
    for t in compound:
        HetG.add_edge(t, reaction, type="Metabolite-Reaction")
        i+=1
    for h in protein:
        HetG.add_edge(h, reaction, type="Enzyme-Reaction")
        i+=1

    print(i)

i = 0
for drug1, drugs in DDI.items():
    for drug2 in drugs:
        HetG.add_edge(drug1, drug2, type="Drug-Drug")
        i+=1
        print(i)

i=0
for drug, proteins in DTI.items():
    for protein in proteins:
        HetG.add_edge(drug, protein, type="Drug-Target")
        i+=1
        print(i)

# Print the summary of the graph
print(HetG.summary())
graph_file = r"C:\Users\George\Desktop\ISEF-2023\Network construction\het_graph.graphml"
HetG.save(graph_file, format="graphml")