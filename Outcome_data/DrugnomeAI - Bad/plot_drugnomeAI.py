import json
import matplotlib.pyplot as plt


json_file_path = r'C:\Users\George\Desktop\ISEF-2023\Outcome_data\DrugnomeAI\Drugnome_Score.json'

# Load the JSON file into a dictionary
with open(json_file_path, 'r') as json_file:
    data_dict = json.load(json_file)
druggability_scores = []
probabilities = []
for key, value in data_dict.items():

    if value['probability'] > 0.1:
        druggability_scores.append(value['drugnome_score'])
        probabilities.append(value['probability'])

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(probabilities, druggability_scores)

# Set plot title and labels
plt.title('Probability vs. Druggability Score')
plt.xlabel('Probability')
plt.ylabel('Druggability Score')

# Show plot
plt.grid(True)
plt.show()