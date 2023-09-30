import csv

# Replace these with your actual file paths.
idmapping_leftover_file = r"C:\Users\George\Desktop\ISEF-2023\Datas\ISEF datasets\Homo Sapien PPIN\idmapping_leftover.csv"
idmapping_file = r"C:\Users\George\Desktop\ISEF-2023\Datas\ISEF datasets\Homo Sapien PPIN\idmapping_.csv"
output_file = r"C:\Users\George\Desktop\ISEF-2023\Datas\ISEF datasets\Homo Sapien PPIN\updated_idmapping.csv"

# Read the idmapping CSV file into a list of dictionaries.
idmapping_data = []
with open(idmapping_file, 'r') as idmapping_csv:
    idmapping_reader = csv.DictReader(idmapping_csv)
    for row in idmapping_reader:
        idmapping_data.append(row)

# Read the idmapping_leftover CSV file into a list of dictionaries.
idmapping_leftover_data = []
with open(idmapping_leftover_file, 'r') as idmapping_leftover_csv:
    idmapping_leftover_reader = csv.DictReader(idmapping_leftover_csv)
    for row in idmapping_leftover_reader:
        row['converted_alias'] = '9606.' + row['converted_alias']  # Add "9606." to the 'converted_alias'
        idmapping_leftover_data.append(row)

# Iterate through rows in the idmapping_leftover data.
for leftover_row in idmapping_leftover_data:
    uniprot_format_leftover = leftover_row['initial_alias']
    ensp_format_leftover = leftover_row['converted_alias']

    # Find the corresponding row in the idmapping data where "From" column matches ensp_format_leftover.
    matching_row = None
    for idmapping_row in idmapping_data:
        if idmapping_row['From'] == ensp_format_leftover:
            matching_row = idmapping_row
            break

    # If there is a match, update the corresponding "Entry" value in the idmapping data.
    if matching_row:
        original_uniprot = matching_row['Entry']
        matching_row['Entry'] = uniprot_format_leftover
        print(f"Replacement: {original_uniprot} -> {uniprot_format_leftover}")

# Save the updated idmapping data to a new CSV file.
header = idmapping_data[0].keys()
with open(output_file, 'w', newline='') as output_csv:
    writer = csv.DictWriter(output_csv, fieldnames=header)
    writer.writeheader()
    writer.writerows(idmapping_data)

print(f"Updated idmapping data saved to '{output_file}'")