import csv
import json

def process_csv(input_file):
    proteins = set()

    with open(input_file, 'r') as input_csv:
        reader = csv.DictReader(input_csv)

        for row in reader:
            protein1 = row['protein1']
            protein2 = row['protein2']

            proteins.add(protein1)
            proteins.add(protein2)

    return list(proteins)

# Provide the path to your input file
input_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\PPI\data_cleaning\PPI.csv'

# Call the function to process the CSV and get the list of proteins
protein_list = process_csv(input_file_path)

print(len(protein_list))
# Store the protein list as JSON
output_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\PPI\data_cleaning\protein_list.json'
with open(output_file_path, 'w') as output_file:
    json.dump(protein_list, output_file)

print("Processing completed successfully.")