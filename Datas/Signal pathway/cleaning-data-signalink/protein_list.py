import csv
import json

# CSV file path
json_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Signal pathway\cleaning-data-signalink\cell_communication_edge_node_relation_cleaned.json'

with open(json_file_path, 'r') as json_file:
    data_dict = dict(json.load(json_file))

protein_list = []

for i, j in data_dict.items():
    protein_list.append(i)
    for a in j["interactions"].keys():
        protein_list.append(a)

protein_list = list(set(protein_list))

print(len(protein_list))

with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Signal pathway\protein_list.json', 'w') as jsonfile:
    json.dump(protein_list, jsonfile)