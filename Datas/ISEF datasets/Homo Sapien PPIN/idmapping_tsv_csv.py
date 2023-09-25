import csv

input_tsv_file = r"C:\Users\George\Desktop\ISEF datasets\Homo Sapien PPIN\idmapping_2023_09_24.tsv"
output_csv_file = r"C:\Users\George\Desktop\ISEF datasets\Homo Sapien PPIN\idmapping_2023_09_24.csv"

# Initialize a list to store rows of data.
data = []

# Read the TSV file and split columns based on multiple spaces.
with open(input_tsv_file, 'r') as tsv_file:
    for line in tsv_file:
        # Split columns using regular expression to handle multiple spaces.
        columns = line.strip().split('\t')
        
        # Append the split columns as a row to the data list.
        data.append(columns)

# Write the data to a CSV file.
with open(output_csv_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # Write the data to the CSV file.
    for row in data:
        writer.writerow(row)

print(f"TSV file '{input_tsv_file}' has been successfully converted to '{output_csv_file}'")