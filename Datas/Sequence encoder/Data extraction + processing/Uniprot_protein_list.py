import csv
import json

# Input file path
csv_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Sequence encoder\Data extraction + processing\uniprotkb_AND_reviewed_true_AND_model_o_2023_11_20.csv'
json_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Sequence encoder\Data extraction + processing\uniprot_protein_list.json'  # Provide a path for the output JSON file

# Read the first column of the CSV and store unique values in a set
unique_values = set()

with open(csv_file_path, mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        if row:  # Check if the row is not empty
            unique_values.add(row[0])

# Convert the set to a list (optional)
unique_values_list = list(unique_values)

# Write the set to a JSON file
with open(json_file_path, mode='w') as json_file:
    json.dump(list(unique_values), json_file, indent=2)

# Print or use the set as needed
print(f"Unique Values Set: {unique_values}")
print(f"JSON file saved at: {json_file_path}")
