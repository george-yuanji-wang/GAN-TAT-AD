import pandas as pd

# Load the dataset from the CSV file
data = pd.read_csv(r'C:\Users\George\Desktop\ISEF-2023\Datas\PPI\data_cleaning\9606.protein.physical.links.full.v12.0.csv')

# Filter rows based on the combined_score threshold
data = data[data['combined_score'] > 825]
data = data[data['textmining'] > 200]

# Reset the index of the resulting DataFrame
data = data.reset_index(drop=True)

# Save the updated dataset as a new CSV file
data.to_csv(r'C:\Users\George\Desktop\ISEF-2023\Datas\PPI\data_cleaning\9606.protein.physical.links.full.v12.0.csv', index=False)