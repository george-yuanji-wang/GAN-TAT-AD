import csv

import csv

def map_ensp_to_uniprot(mapping_file, input_file, output_file):
    ensp_to_uniprot = {}  # Dictionary to store ENSP to UniProt mapping

    # Read the mapping file and store the ENSP to UniProt mapping
    with open(mapping_file, 'r') as map_file:
        reader = csv.reader(map_file, delimiter='\t')
        next(reader)  # Skip header row if present

        for row in reader:
            ensp_id = row[0]
            uniprot_id = row[1]
            ensp_to_uniprot[ensp_id] = uniprot_id

    # Perform the mapping and write the result to the output file
    with open(input_file, 'r') as input_csv, open(output_file, 'w') as output_csv:
        reader = csv.DictReader(input_csv)
        fieldnames = reader.fieldnames

        writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            ensp1 = row['protein1']
            ensp2 = row['protein2']

            # Map ENSP IDs to UniProt IDs
            uniprot1 = ensp_to_uniprot.get(ensp1)
            uniprot2 = ensp_to_uniprot.get(ensp2)

            # Skip the row if mapping is not possible for either protein
            if uniprot1 is None or uniprot2 is None:
                continue

            # Update the row with UniProt IDs
            row['protein1'] = uniprot1
            row['protein2'] = uniprot2

            writer.writerow(row)

    print("Mapping completed successfully.")

# Provide the path to your mapping file
mapping_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\PPI\data_cleaning\idmapping_2023_12_10.tsv'

# Provide the path to your input file
input_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\PPI\data_cleaning\9606.protein.physical.links.full.v12.0.csv'

# Provide the path for the output file
output_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\PPI\data_cleaning\PPI_raw.csv'

# Call the function to perform the mapping
map_ensp_to_uniprot(mapping_file_path, input_file_path, output_file_path)