import csv

input_csv_file = r'C:\Users\George\Desktop\ISEF datasets\Pancreate cancer database\prognostic_pancreatic.csv'
output_csv_file = r'C:\Users\George\Desktop\ISEF datasets\Pancreate cancer database\prognostic_list_protein.csv'

# Initialize a list to store the values from the first column.
first_column_values = []

# Read the CSV file and extract the first column values.
with open(input_csv_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        if row:  # Ensure the row is not empty

            first_column_values.append([row[3].strip("[]'"), ''.join([*(row[4].strip("{}'':"))[-7:]]), row[5]])

# Save the first column values as a new CSV file.
with open(output_csv_file, 'w', newline='') as output_file:
    csv_writer = csv.writer(output_file)
    for value in first_column_values:
        csv_writer.writerow(value)

print(f"Values from the first column saved to '{output_csv_file}'")