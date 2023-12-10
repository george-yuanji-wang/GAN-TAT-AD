import json
import csv

# Read the JSON file
with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Sequence encoder\Data extraction + processing\sequence_homosapien_genome.json') as json_file:
    data = json.load(json_file)

i = 0
# Prepare the CSV data
csv_data = []
for item in data:

    header = item['header']
    protein_id = header.split('|')[1]
    protein_name = header.split('|')[2].split(' ')[0]
    if 'GN=' in header:
        gene_name = header.split('GN=')[1].split(' ')[0]
    pe = header.split('PE=')[1].split(' ')[0]
    sv = header.split('SV=')[1]
    sequence = item['sequence']
    csv_data.append([protein_id, protein_name, gene_name, pe, sv, sequence])

# Write the CSV data to a file
with open('HUMAN_GENOME_PROTEIN_AMINO_ACID_SEQUENCE.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['protein_id', 'protein_name', 'gene_name', 'PE', 'SV', 'sequence'])
    writer.writerows(csv_data)