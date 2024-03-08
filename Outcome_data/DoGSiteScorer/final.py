import json
import csv

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
    probabilities = read_csv('Outcome_data\All_predicted_proteins.csv')

    # Read VAL values from the second JSON file
    val_data = read_json('Outcome_data\DoGSiteScorer\DogSiteScorer_data_withALLPredicted.json')

    # Extracting VAL values for each protein
    max_vals = {}
    for key, value in val_data.items():
        max_vals[key] = float(value['VAL'])

    # Combine protein IDs, probabilities, and max VALs
    result = []
    for row in probabilities:
        protein_id, probability = row
        max_val = max_vals.get(protein_id, 0)
        result.append({"Protein ID": protein_id, "Probability": probability, "Max_Val": max_val})

    # Write result to a new JSON file
    write_json(result, 'result.json')

if __name__ == "__main__":
    main()
