import pandas as pd

# Load the dataset from the CSV file
data = pd.read_csv(r'C:\Users\George\Desktop\ISEF-2023\Datas\PPI\data_cleaning\9606.protein.physical.links.full.v12.0.csv')

# Drop rows with 0 in experiments or textmining columns
data = data[(data['experiments'] != 0) & (data['textmining'] != 0)]

# Reset the index of the resulting DataFrame
data = data.reset_index(drop=True)

# Save the cleaned dataset as a new CSV file
data.to_csv(r'C:\Users\George\Desktop\ISEF-2023\Datas\PPI\data_cleaning\new9606.protein.physical.links.full.v12.0.csv', index=False)