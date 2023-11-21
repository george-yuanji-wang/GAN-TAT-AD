import csv

# Input TSV file path
tsv_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\DDI\ChCh-Miner_durgbank-chem-chem.tsv'

# Output CSV file path
csv_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\DDI\DDI.csv'

# Read the TSV file and write to a CSV file
with open(tsv_file_path, 'r', newline='', encoding='utf-8') as tsv_file, \
        open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:

    # Create a TSV reader and CSV writer
    tsv_reader = csv.reader(tsv_file, delimiter='\t')
    csv_writer = csv.writer(csv_file)

    # Write each row from TSV to CSV
    for row in tsv_reader:
        csv_writer.writerow(row)

print(f'Conversion completed. CSV file saved at: {csv_file_path}')