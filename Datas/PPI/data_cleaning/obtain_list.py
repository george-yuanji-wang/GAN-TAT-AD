import csv

def extract_columns_to_txt(csv_file, output_file):
    elements = set()  # Set to store unique elements

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row if present

        for row in reader:
            # Extract elements from the first and second columns
            element1 = row[0]
            element2 = row[1]

            # Add elements to the set
            elements.add(element1)
            elements.add(element2)

    with open(output_file, 'w') as txt_file:
        for element in elements:
            txt_file.write(f"{element}\n")

    print("Extraction completed successfully.")

# Provide the path to your CSV file
csv_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\PPI\data_cleaning\PPI.csv'

# Provide the path for the output text file
output_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\PPI\data_cleaning\plist.txt'

# Call the function to extract columns to a single text file
extract_columns_to_txt(csv_file_path, output_file_path)