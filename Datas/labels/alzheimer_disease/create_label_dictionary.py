import numpy as np
import json
path = r'C:\Users\George\Desktop\ISEF-2023\Datas\labels\alzheimer_disease\alzheimer_disease.txt'
protein_l = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\back up\current_protein_Signal+meta+targets.json'


targets = []


with open(protein_l, 'r') as file:
    protein_list = json.load(file)


with open(path, 'r') as file:
    contents = file.readlines()
    for i in contents:
        targets.append(i[:-1])

targets.pop()
targets.append('P18089')


proteins = {}
for i in protein_list:
    if i in targets:
        proteins[i] = 1
    else:
        proteins[i] = 0

with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\labels\alzheimer_disease\label_dictionary.json', 'w') as file:
    json.dump(proteins, file)