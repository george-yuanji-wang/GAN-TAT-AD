import json
import csv

# Replace 'input.json' with the path to your JSON file.
input_json_file = r'C:\Users\George\Desktop\ISEF datasets\Pancreate cancer database\prognostic_pancreatic.json'

# Replace 'output.csv' with the desired path for your CSV output file.
output_csv_file = r'C:\Users\George\Desktop\ISEF datasets\Pancreate cancer database\prognostic_pancreatic.csv'

# Read the JSON file.
with open(input_json_file, 'r') as json_file:
    data = json.load(json_file)

# Extract header (column names) from the first element of the JSON data.
header = data[0].keys()

# Write the data to a CSV file.
with open(output_csv_file, 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=header)
    
    # Write the header row.
    writer.writeheader()
    
    # Write the JSON data to the CSV file.
    for row in data:
        writer.writerow(row)

print(f"JSON data has been successfully converted to '{output_csv_file}'")