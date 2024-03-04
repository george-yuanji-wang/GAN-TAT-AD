import pandas as pd
import json

# Read the JSON file
with open(r'C:\Users\Srisharan\OneDrive\Desktop\ISEF\ISEF-2023\Outcome_data\gsea_gene.json') as f:
    data = json.load(f)

# Initialize lists to store data
names = []
descriptions = []
all_predicted = []
known = []
predicted_unknown = []

# Iterate through the JSON data and extract values
for gene, details in data.items():
    names.append(gene)
    descriptions.append(details.get('Protein', ''))
    all_predicted.append(details.get('all_predicts', 0))
    known.append(details.get('known', 0))
    predicted_unknown.append(details.get('predicted_unknown', 0))

# Create DataFrame
df = pd.DataFrame({
    'NAME': names,
    'DESCRIPTION': "na",
    'All PREDICTED': all_predicted,
    'KNOWN': known,
    'PREDICTED_UNKNOWN': predicted_unknown
})

# Display DataFrame
print("DataFrame:")
print(df)

df.to_csv('GSEA_expression_data.txt', sep='\t', index=False)
