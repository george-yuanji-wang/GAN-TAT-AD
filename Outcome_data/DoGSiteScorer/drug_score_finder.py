import json
import csv
from scipy import stats
import numpy as np
from scipy.stats import variation, skew, kurtosis

# Function to read data from CSV file
def read_csv(file_path):
    data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

# Function to read data from JSON file
def read_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

# Function to find the maximum VAL for each protein
def find_max_val(data):
    max_vals = {}
    for key, value in data.items():
        max_val = max(value.items(), key=lambda x: float(x[1]))
        max_vals[key] = float(max_val[1])
    return max_vals

# Function to write data to JSON file
def write_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Main function
def main():
    # Read probabilities from the CSV file
    trim_percentage = 10 
    top_n = 3 
    probabilities = read_csv('Outcome_data\All_predicted_proteins.csv')
    count = 0
    # Read VAL values from the second JSON file
    val_data = read_json(r'C:\Users\George\Desktop\ISEF-2023\Outcome_data\DoGSiteScorer\fixed_DogSiteScorer_data_withALLPredicted.json')

    # Extracting VAL values for each protein
    max_vals = {}
    for key, pockets in val_data.items():
        scores = [float(attribute['VAL']) for pocket, attribute in pockets.items()]
        if scores:
        
            # Basic metrics
            average_score = np.mean(scores)
            median_score = np.median(scores)
            upper_quartile_score = np.percentile(scores, 75)
            
            # Trimmed mean
            trimmed_scores = stats.trim_mean(scores, trim_percentage / 100)
            
            # Harmonic mean of top N scores
            top_n_scores = sorted(scores, reverse=True)[:top_n]
            harmonic_mean_top_n = stats.hmean(top_n_scores) if top_n_scores else 0

            # Finding best pocket as before
            best_pocket = ""
            max_score = 0
            for pocket, attribute in pockets.items():
                if float(attribute['VAL']) > max_score:
                    max_score = float(attribute['VAL'])
                    best_pocket = pocket
                    
            # Assuming equal weights for weighted average, which simplifies to just the average
            weighted_average_score = average_score  # Adjust this line if you have specific weights

            # Composite score - Example: simple average of median and upper quartile
            composite_score = (median_score + upper_quartile_score) / 2

            score_variance = np.var(scores)
            score_std_dev = np.std(scores)
            score_range = max(scores) - min(scores)
            score_skewness = skew(scores)
            score_kurtosis = kurtosis(scores)
            cumulative_score_above_threshold = sum(score for score in scores if score > 0.8)  # Define threshold
            
            # Update max_vals with all metrics for each protein
            max_vals[key] = {
                'Best_Pocket': best_pocket,
                'Max_Score': max_score,
                'Average_Score': average_score,
                'Weighted_Average_Score': weighted_average_score,
                'Median_Score': median_score,
                'Trimmed_Mean_Score': trimmed_scores,
                'Upper_Quartile_Score': upper_quartile_score,
                'Composite_Score': composite_score,
                'Harmonic_Mean_Top_N': harmonic_mean_top_n,
                'Score_Variance': score_variance,
                'score_std_dev': score_std_dev,
                'score_range': score_range,
                'score_skewness': score_skewness,
                'score_kurtosis': score_kurtosis,
                'cumulative_score_above_threshold': cumulative_score_above_threshold
            }
    print(count)
    # Combine protein IDs, probabilities, and max VALs
    result = []
    key = max_vals.keys()
    for i in probabilities:
        if i[0] != 'Name':
            probability = i[1]
            protein = i[0]
            if protein in key:
                max_val = max_vals[protein]
                result.append({"Protein ID": protein, "Probability": probability, "Score": max_val})
            

    # Write result to a new JSON file
    write_json(result, r'C:\Users\George\Desktop\ISEF-2023\Outcome_data\DoGSiteScorer\MaxDrugScore.json')

    print("Done")

if __name__ == "__main__":
    main()
