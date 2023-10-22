import pandas as pd
import csv

df = pd.read_csv(r'C:\Users\George\Desktop\ISEF-2023\Datas\Signal pathway\cell_signal_communication.csv')

# Drop specified columns"
columns_to_drop = ["source_name", "source_speciesID","source_species","source_topology","target_speciesID","target_species","target_topology","target_pathways","layer","effect","references","source", "confidence_score","tissues","score_from_the_source", "target_name", "directness"]
df = df.drop(columns=columns_to_drop)

# Save the modified DataFrame to a new CSV file
df.to_csv('modified_file.csv', index=False)

