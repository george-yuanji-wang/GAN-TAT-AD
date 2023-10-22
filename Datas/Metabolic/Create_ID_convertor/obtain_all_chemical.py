import json

# Load the JSON file
with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Metabolic\Extract reactions information\Reaction_Map.json', 'r') as json_file:
    data = json.load(json_file)

# Extract all "C...." entries
c_entries = set()
for reaction, values in data.items():
    for sublist in values:
        c_entries.update(entry for entry in sublist if entry.startswith("C"))

# Convert the set to a list before saving it to a new JSON file
c_entries_list = list(c_entries)

# Save the extracted "C...." entries as a new JSON file
with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Metabolic\Create_ID_convertor\chemical_list.json', 'w') as output_file:
    json.dump(c_entries_list, output_file)

# Print or use the extracted "C...." entries
print(len(c_entries_list))
