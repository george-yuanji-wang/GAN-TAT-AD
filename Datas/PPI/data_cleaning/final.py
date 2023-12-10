import csv

def process_csv(input_file, output_file):
    with open(input_file, 'r') as input_csv, open(output_file, 'w', newline='') as output_csv:
        reader = csv.reader(input_csv)
        writer = csv.writer(output_csv)

        header = next(reader)
        writer.writerow(header[:2] + [header[9]])
        i = 0 
        for row in reader:
            i+=1
            if i % 2 == 0:
                new_row = row[:2] + [float(row[9]) / 1000]
                writer.writerow(new_row)

    print("Processing completed successfully.")

# Provide the path to your input file
input_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\PPI\data_cleaning\PPI_raw.csv'

# Provide the path for the output file
output_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\PPI\data_cleaning\PPI.csv'

# Call the function to process the CSV
process_csv(input_file_path, output_file_path)