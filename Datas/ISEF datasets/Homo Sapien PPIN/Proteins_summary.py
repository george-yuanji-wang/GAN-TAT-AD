import pandas as pd
import csv

# Load the original dataset
csv_file = r"C:\Users\George\Desktop\ISEF datasets\Homo Sapien PPIN\9606.protein.links.detailed.v12.0.csv"
data = pd.read_csv(csv_file)  # Replace 'your_ppin_dataset.csv' with the actual file path

# Get unique proteins from both 'protein1' and 'protein2' columns
unique_proteins = pd.concat([data['protein1'], data['protein2']]).unique()

# Create a DataFrame with the list of unique proteins
protein_list_df = pd.DataFrame({'Protein': unique_proteins})

# Save the list of proteins to a new CSV file
protein_list_df.to_csv('protein_list.csv', index=False)  # Change 'protein_list.csv' to your desired output file name