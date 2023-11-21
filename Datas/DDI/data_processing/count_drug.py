import csv
from collections import Counter
import json

# Input and output file paths
csv_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\DDI\data_processing\DDI.csv'
json_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\DDI\data_processing\Drug_list.json'

unique_drugs = set()

with open(csv_file_path, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        unique_drugs.add(row['#Drug1'])
        unique_drugs.add(row[' Drug2'])

# Convert the set to a list
unique_drugs_list = list(unique_drugs)

# Write the list to a JSON file
with open(json_file_path, mode='w') as json_file:
    json.dump(unique_drugs_list, json_file, indent=2)

# Print or use the list as needed
print(f"Unique Drugs List: {unique_drugs_list}")
print(f"JSON file saved at: {json_file_path}")