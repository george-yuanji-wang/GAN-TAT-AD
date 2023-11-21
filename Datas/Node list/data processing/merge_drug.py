import json

# Input file paths
json_file_path1 = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\data processing\Drug_list copy.json'
json_file_path2 = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\data processing\DTI_drug_list copy.json'
merged_json_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Node list\data processing\merged_json_file.json'  # Provide a path for the merged JSON file

# Read data from the first JSON file
with open(json_file_path1, 'r') as file1:
    data1 = json.load(file1)

# Read data from the second JSON file
with open(json_file_path2, 'r') as file2:
    data2 = json.load(file2)

# Merge the lists and remove duplicates
merged_data = list(set(data1 + data2))

# Write the merged data to a new JSON file
with open(merged_json_file_path, 'w') as merged_file:
    json.dump(merged_data, merged_file, indent=2)

# Print or use the merged data as needed
print(f"Merged Data List: {merged_data}")
print(f"Merged JSON file saved at: {merged_json_file_path}")
