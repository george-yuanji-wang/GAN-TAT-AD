import csv
import json

# CSV file path
json_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Signal pathway\cell_communication_edge_node_relation.json'


with open(json_file_path, 'r') as json_file:
    data_dict = dict(json.load(json_file))

protein_list = list(data_dict.keys())

ref = {}
b=0
for key in protein_list:

    path = (data_dict[key])['pathways']
    new_path = []
    
    for i in path:
        if '|' not in i:
            new_path.append(i)
        else:
            a = i.split('|')
            for t in a:
                new_path.append(t)
    new_path = list(set(new_path))
    data_dict[key]['pathways'] = new_path

    int_d = (data_dict[key])['interactions']

    for ke in int_d.keys():
        interact = int_d[ke][0]

        id = int((interact.split(":")[1]).split("(")[0])
        name = interact.split("(")[1][:-1]

        if id not in ref.keys():
            ref[id] = name
            
        int_d[ke] = id

with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Signal pathway\cell_communication_edge_node_relation_cleaned.json', 'w') as jsonfile:
    json.dump(data_dict, jsonfile, indent=4)

with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Signal pathway\interaction_ref.json', 'w') as jsonfile:
    json.dump(ref, jsonfile, indent=4)

with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Signal pathway\protein_list.json', 'w') as jsonfile:
    json.dump(protein_list, jsonfile)



