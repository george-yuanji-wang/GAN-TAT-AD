import json

# Input file paths
json_file_path1 = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\data processing\DTI_protein_list copy.json'
json_file_path2 = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\data processing\enzyme_list copy.json'
json_file_path3 = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\data processing\signal_protein_list copy.json'
json_file_path4 = r'C:\Users\George\Desktop\ISEF-2023\Datas\PPI\data_cleaning\protein_list.json'
merged_json_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\data processing\merged_protein_without_PPIandSTP.json'  # Provide a path for the merged JSON file

# Read data from the first JSON file
with open(json_file_path1, 'r') as file1:
    data1 = json.load(file1)

# Read data from the second JSON file
with open(json_file_path2, 'r') as file2:
    data2 = json.load(file2)

with open(json_file_path3, 'r') as file3:
    data3 = json.load(file3)

with open(json_file_path4, 'r') as file4:
    data4 = json.load(file4)

# Merge the lists and remove duplicates
merged_data = list(set(data1 + data2))

# Write the merged data to a new JSON file
with open(merged_json_file_path, 'w') as merged_file:
    json.dump(merged_data, merged_file, indent=2)

print(len(merged_data))
# Print or use the merged data as needed
print(f"Merged Data List: {merged_data}")
print(f"Merged JSON file saved at: {merged_json_file_path}")
