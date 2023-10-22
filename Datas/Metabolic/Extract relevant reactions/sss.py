
import json
import csv

# Dictionary to store unique values of 'a' and corresponding lists of different 'b'
a_dict = []

# Read the CSV file
with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Metabolic\Extract relevant reactions\reactions.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    # Iterate through rows in the CSV
    for row in csv_reader:
        a_value = row['Reaction']
        b_value = row['Entry']

        # Check if 'a' value is already in the dictionary
        found = False
        for entry in a_dict:
            if entry['Reaction'] == a_value:
                found = True
                # Check if 'b' value is different, if yes, add it to the list
                if b_value not in entry['Module']:
                    entry['Module'].append(b_value)
                break

        if not found:
            # If 'a' value is not in the dictionary, add it with an initial list containing 'b' value
            a_dict.append({'Reaction': a_value, 'Module': [b_value]})

# Write the results to a JSON file
output_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Metabolic\Extract reactions information\reactions_list.csv'
with open(output_file_path, 'w') as json_file:
    json.dump(a_dict, json_file, indent=2)

print(f"Results written to {output_file_path}")