import pandas as pd

# Replace these with your actual file paths.
ppin_file = r"C:\Users\George\Desktop\ISEF-2023\Datas\ISEF datasets\Homo Sapien PPIN\PPIN edges\9606.protein.links.detailed.v12.0.csv"
idmapping_file = r"C:\Users\George\Desktop\ISEF-2023\Datas\ISEF datasets\Homo Sapien PPIN\updated_idmapping.csv"
output_file = r"C:\Users\George\Desktop\ISEF-2023\Datas\ISEF datasets\Homo Sapien PPIN\PPIN_UniProt.csv"

# Read the idmapping CSV file into a DataFrame considering only the 'From' and 'Entry' columns.
idmapping_df = pd.read_csv(idmapping_file, usecols=['From', 'Entry'])

# Read the PPIN CSV file into a DataFrame.
ppin_df = pd.read_csv(ppin_file)

# Merge the PPIN DataFrame with the idmapping DataFrame to replace ENSP with UniProt.
ppin_df = ppin_df.merge(idmapping_df, left_on='protein1', right_on='From', how='inner')
ppin_df.drop(columns=['protein1', 'From'], inplace=True)
ppin_df.rename(columns={'Entry': 'protein1'}, inplace=True)

ppin_df = ppin_df.merge(idmapping_df, left_on='protein2', right_on='From', how='inner')
ppin_df.drop(columns=['protein2', 'From'], inplace=True)
ppin_df.rename(columns={'Entry': 'protein2'}, inplace=True)

# Write the result to a new CSV file.
ppin_df.to_csv(output_file, index=False)

print(f"CSV file '{output_file}' with UniProt format has been created.")