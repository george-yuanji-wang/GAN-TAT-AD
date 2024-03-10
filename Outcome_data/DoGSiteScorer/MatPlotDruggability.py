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
with open('Outcome_data\DoGSiteScorer\MaxDrugScore.json', 'r') as file:
    final_output = json.load(file)

# Initialize lists to store data
predicted_probabilities = []
druggability_scores = []
i = 0

protein_threshold = 2000


# Extract data from final output
druggability_scores = {'Max_Score': [], 'Average_Score': [], 'Weighted_Average_Score': [], 
                       'Median_Score': [], 'Trimmed_Mean_Score': [], 'Upper_Quartile_Score': [], 
                       'Composite_Score': [], 'Harmonic_Mean_Top_N': [], 'Score_Variance': [],
                'score_std_dev': [],
                'score_range': [],
                'score_skewness': [],
                'score_kurtosis': [],
                'cumulative_score_above_threshold': []
            }

i = 0
for data in final_output:
    if i < protein_threshold:
        predicted_probabilities.append(float(data['Probability']))
        for score_type in druggability_scores.keys():
            druggability_scores[score_type].append(float(data['Score'][score_type]))
        i += 1

# Now, loop through each score type to perform linear regression and create plots
for score_type, scores in druggability_scores.items():
    slope, intercept, _, _, _ = linregress(predicted_probabilities, scores)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(predicted_probabilities, scores, alpha=0.5)
    
    # Plot the regression line
    plt.plot(predicted_probabilities, intercept + slope * np.array(predicted_probabilities), color='red')
    
    # Set plot labels and title
    plt.title(f'Predicted Probability vs. {score_type}')
    plt.xlabel('Predicted Probability')
    plt.ylabel(score_type)
    
    # Set axis limits
    plt.xlim(0.1, 1)
    plt.ylim(0, 1)
    
    # Show plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

slope, intercept, _, _, _ = linregress(predicted_probabilities, druggability_scores['Upper_Quartile_Score'])
plt.figure(figsize=(10, 6))
plt.scatter(predicted_probabilities, druggability_scores['Upper_Quartile_Score'], alpha=0.5)
    
    # Plot the regression line
plt.plot(predicted_probabilities, intercept + slope * np.array(predicted_probabilities), color='red')
    
    # Set plot labels and title
plt.title(f'Predicted Probability vs. Upper_Quartile_Score')
plt.xlabel('Predicted Probability')
plt.ylabel('Upper_Quartile_Score')
    
    # Set axis limits
plt.xlim(0.1, 1)
plt.ylim(0.2, 0.8)
    
    # Show plot
plt.grid(True)
plt.tight_layout()
plt.show()




