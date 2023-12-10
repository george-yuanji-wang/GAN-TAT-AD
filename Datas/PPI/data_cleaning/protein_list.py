import csv

csv_file = 'Datas/PPI/PP-Pathways_ppi.csv'
output_file = 'Datas/PPI/PP-Pathways_ppi_list.txt'

with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

with open(output_file, 'w') as file:
    for row in data:
        file.write(row[0] + '\n')
        file.write(row[1] + '\n')