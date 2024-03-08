import json

# Load the data from the JSON file
with open('Outcome_data\DoGSiteScorer\DogSiteScorer_data.json', 'r') as file:
    data = json.load(file)

# Initialize dictionary to store protein IDs and count of subpockets with VAL > 0.8
output_data = {}

# Iterate through each protein
for protein_id, properties in data.items():
    count_above_threshold = 0
    for key, value in properties.items():
        if key.startswith('P_'):
            # Check if the value for key "VAL" is greater than 0.8
            if float(value.get("VAL", 0)) > 0.6:
                count_above_threshold += 1
    # Store protein ID and count in the dictionary
    output_data[protein_id] = count_above_threshold

# Save the output as a JSON file
with open('Outcome_data\DoGSiteScorer\Pocket_Count.json', 'w') as file:
    json.dump(output_data, file, indent=4)

import json
import matplotlib.pyplot as plt

# Load data from the first JSON file containing counts of protein pockets above 0.8
with open('Outcome_data\DoGSiteScorer\Pocket_Count.json', 'r') as file:
    pockets_data = json.load(file)

# Load data from the second JSON file containing probability values
with open('Outcome_data\DoGSiteScorer\MaxDrugScore.json', 'r') as file:
    probability_data = json.load(file)

# Extract protein IDs and counts of protein pockets above 0.8
protein_ids = list(pockets_data.keys())
pocket_counts = list(pockets_data.values())

# Extract probability values for the corresponding protein IDs
probabilities = [float(probability_data.get(protein_id, {}).get("probability", 0)) for protein_id in protein_ids]

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(pocket_counts, probabilities, alpha=0.5)

# Set plot labels and title
plt.title('Number of Protein Pockets vs. Probability')
plt.xlabel('Number of Protein Pockets Above 0.6')
plt.ylabel('Probability')
plt.grid(True)
plt.tight_layout()
plt.show()