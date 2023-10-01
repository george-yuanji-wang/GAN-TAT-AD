import csv

# Replace these with your actual file paths.
output_csv_file = r"C:\Users\George\Desktop\ISEF-2023\Datas\ISEF datasets\Pancreate cancer chemical database\list.csv"
input_csv_file = r"C:\Users\George\Desktop\ISEF-2023\Datas\ISEF datasets\Pancreate cancer chemical database\Extracted_pancreatic_cancer_related_chemicals.csv"

with open(input_csv_file, 'r', newline='', encoding='utf-8') as input_file, \
     open(output_csv_file, 'w', newline='', encoding='utf-8') as output_file:
    
    # Create CSV readers and writers.
    csv_reader = csv.reader(input_file)
    csv_writer = csv.writer(output_file)

    # Process the header (first row).
    header = next(csv_reader)
    new_header = [header[0]]  # Keep only the second and fourth columns.
    csv_writer.writerow(new_header)

    # Process each row in the input file.
    for row in csv_reader:

        new_row = ["".join([*row[0]][1:])]
        csv_writer.writerow(new_row)

print(f"Processing complete. Result saved to '{output_csv_file}'")