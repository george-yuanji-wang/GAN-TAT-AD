import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset from the CSV file
data = pd.read_csv(r'C:\Users\George\Desktop\ISEF-2023\Datas\PPI\data_cleaning\9606.protein.physical.links.full.v12.0.csv')

# Specify the columns for analysis
score_columns = ['experiments', 'textmining', 'combined_score']

# Create a figure with subplots for each score
num_scores = len(score_columns)
num_cols = 2
num_rows = (num_scores + 1) // num_cols
fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 6 * num_rows))
fig.subplots_adjust(hspace=0.4)

# Iterate over each score column and create a histogram plot
for i, column in enumerate(score_columns):
    row = i // num_cols
    col = i % num_cols
    scores = data[column]
    axs[row, col].hist(scores, bins=20, edgecolor='black')
    axs[row, col].set_xlabel(column)
    axs[row, col].set_ylabel('Frequency')
    axs[row, col].set_title(f'Distribution of {column}')

    # Calculate statistical summary
    mean = scores.mean()
    median = scores.median()
    std = scores.std()
    min_val = scores.min()
    max_val = scores.max()

    # Display statistical summary
    summary_text = f"Statistical Summary:\nMean: {mean}\nMedian: {median}\nStandard Deviation: {std}\nMinimum Value: {min_val}\nMaximum Value: {max_val}"
    axs[row, col].text(0.05, 0.95, summary_text, transform=axs[row, col].transAxes, verticalalignment='top')

# Hide empty subplots if the number of scores is not a multiple of 2
if num_scores % 2 != 0:
    axs[num_rows-1, num_cols-1].axis('off')

plt.tight_layout()
plt.show()