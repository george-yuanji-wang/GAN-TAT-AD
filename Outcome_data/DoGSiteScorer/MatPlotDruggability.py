import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
import time
import subprocess
import csv

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load the final output from the JSON file
with open('output.json', 'r') as file:
    final_output = json.load(file)

# Initialize lists to store data
predicted_probabilities = []
druggability_scores = []

# Extract data from final output
for data in final_output.values():
    predicted_probabilities.append(float(data['probability']))
    druggability_scores.append(float(data['max_VAL']))

# Perform linear regression
slope, intercept, _, _, _ = linregress(predicted_probabilities, druggability_scores)

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(predicted_probabilities, druggability_scores, alpha=0.5)

# Plot the regression line
plt.plot(predicted_probabilities, intercept + slope * np.array(predicted_probabilities), color='red')

# Set plot labels and title
plt.title('Predicted Probability vs. Druggability Score')
plt.xlabel('Predicted Probability')
plt.ylabel('Druggability Score')

# Set axis limits
plt.xlim(0.875, 1)
plt.ylim(0.5, 1)

# Show plot
plt.grid(True)
plt.tight_layout()
plt.show()




