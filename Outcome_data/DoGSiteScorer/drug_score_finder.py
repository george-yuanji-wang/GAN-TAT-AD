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
    count = 0
    # Read VAL values from the second JSON file
    val_data = read_json(r'C:\Users\George\Desktop\ISEF-2023\Outcome_data\DoGSiteScorer\fixed_DogSiteScorer_data_withALLPredicted.json')

    # Extracting VAL values for each protein
    max_vals = {}
    for key, pockets in val_data.items():
        best_pocket = ""
        max_score = 0
        for pocket, attribute in pockets.items():
            if float(attribute['VAL']) > max_score:
                max_score = float(attribute['VAL'])
                best_pocket = pocket


        count += 1
        max_vals[key] = {'druggability_score':max_score, 'Best_Pocket': best_pocket}
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
