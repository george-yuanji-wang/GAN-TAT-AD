import json
import csv
import re

# Load the JSON file
with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Metabolic\Extract relevant reactions\modules_extracted_list.json', 'r') as json_file:
    data = json.load(json_file)

# Initialize a list to store the results
result_list = []

# Iterate through dictionaries in the JSON data
for entry_dict in data:
    entry_value = entry_dict.get('Entry')
    reactions_list = entry_dict.get('Reactions', [])

    # Iterate through reactions in the list
    for reaction in reactions_list:
        # Find all occurrences of "R" followed by 5 digits using regular expression
        r_matches = re.findall(r'R\d{5}', reaction[0])

        # Append the results to the list
        result_list.extend([(r, entry_value) for r in r_matches])

# Write the results to a CSV file
with open('reactions.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write the header
    csv_writer.writerow(['Reaction', 'Entry'])
    
    # Write the data
    csv_writer.writerows(result_list)

    print(1)