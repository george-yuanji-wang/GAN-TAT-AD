import csv
import json

# Input and output file paths
csv_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\DTI - SNAP\DTI.csv'
json_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\DTI - SNAP\DTI_json.json'

# Read CSV and organize data into a dictionary
interaction_dict = {}
with open(csv_file_path, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
 
        drug = row['#Drug']
        gene = row['Gene']

        # Add drug-gene interaction to the dictionary
        if drug not in interaction_dict:
            interaction_dict[drug] = [gene]
        else:
            interaction_dict[drug].append(gene)

# Write the dictionary to a JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(interaction_dict, json_file, indent=2)

print(f"JSON file has been created: {json_file_path}")
