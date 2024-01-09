import requests
from bs4 import BeautifulSoup
import urllib.request
import json

# URL of the UniProt API for the protein P28332 in JSON format
file_path = r"C:\Users\George\Desktop\ISEF-2023\Datas\Node list\protein_features\temp_uniprot_blast.json"
protein_list_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\back up\current_protein_Signal+meta+targets.json'
with open(protein_list_path, "r") as json_file:
    protein_list = json.load(json_file)
    pl = len(protein_list)
with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\protein_features\protein_features_uniprot_blast.json', 'r') as json_file:
    protein_features = json.load(json_file)

j=0
for protein in protein_list[7000:]:
    j+=1
    protein_feature = {}
    url = "https://rest.uniprot.org/uniprotkb/" + protein + ".json"
    urllib.request.urlretrieve(url, file_path)
    with open(file_path, "r") as file:
        data = json.load(file)
    if "extraAttributes" in data:
        data = data["extraAttributes"]
    else:
        data = {'a':{"Binding site":0}, 'b':{'Active site': 0}}
    for i in list(data.keys())[:-1]:
        x = list(data[i].items())
        for feature in x:
            if feature[0] not in protein_feature.keys():
                protein_feature[feature[0]] = feature[1]
    
    protein_features[protein] = protein_feature
    print(protein)

    if j % 500 == 0:
        with open(r"C:\Users\George\Desktop\ISEF-2023\Datas\Node list\protein_features\protein_features_uniprot_blast.json", "w") as file:
            json.dump(protein_features, file)

with open(r"C:\Users\George\Desktop\ISEF-2023\Datas\Node list\protein_features\protein_features_uniprot_blast.json", "w") as file:
            json.dump(protein_features, file)

print("Dictionary dumped into JSON file successfully.")