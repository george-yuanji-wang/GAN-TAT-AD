import json
import re

# Load the JSON file
with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Metabolic\Extract reactions information\Reaction_Map.json', 'r') as json_file:
    data = json.load(json_file)

# Extract all entries in the format "x.x.x.x"
x_entries = set()

# Define a regular expression pattern
pattern = re.compile(r'\b\d+\.\d+\.\d+\.\d+\b')

# Iterate through the data
for reaction, values in data.items():
    for sublist in values:
        for entry in sublist:
            match = re.search(pattern, entry)
            if match:
                x_entries.add(match.group())

# Convert the set to a list before saving it to a new JSON file
x_entries_list = list(x_entries)

# Save the extracted "x.x.x.x" entries as a new JSON file
with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Metabolic\Create_ID_convertor\enzymes_list.json', 'w') as output_file:
    json.dump(x_entries_list, output_file)

# Print or use the extracted "x.x.x.x" entries
print(len(x_entries_list))