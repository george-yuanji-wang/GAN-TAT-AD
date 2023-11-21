import csv
import json

# Input CSV file path
csv_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\DDI\DDI.csv'

# Output JSON file path
json_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\DDI\DDI.json'

# Read the CSV file and create a dictionary for drug interactions
drug_interactions = {}

with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    
    for row in csv_reader:
        drug1 = row['#Drug1']
        drug2 = row[' Drug2']

        # Initialize empty lists for drug1 and drug2 if not present
        drug_interactions.setdefault(drug1, []).append(drug2)
        drug_interactions.setdefault(drug2, []).append(drug1)

# Write the drug interactions dictionary to a JSON file
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(drug_interactions, json_file, indent=2)

print(f'Conversion completed. JSON file saved at: {json_file_path}')