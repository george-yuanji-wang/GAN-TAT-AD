import requests
import json
import time
import subprocess
import csv

# Read the protein names from the text file
with open(r'Outcome_data\All_predicted_proteins.csv', 'r') as file:
    next(file)  # Skip the header line
    protein_names = [line.split(',')[0].strip() for line in file][:1109]  # Limit to the first 5 names

datasss = {}
# Loop through each of the first 5 protein names
for uniprot_name in protein_names:
    consecutive_202 = 0
    if uniprot_name == 'P03989':
        continue  # Skip the iteration if uniprot_name is '86'
    # Define the JSON payload
    print(uniprot_name)
    payload = {
        "dogsite": {
            "pdbCode": f'{uniprot_name}',
            "analysisDetail": "1",
            "bindingSitePredictionGranularity": "1",
            "ligand": "",
            "chain": ""
        }
    }

    # Define headers
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    # Define the URL
    url = "https://proteins.plus/api/dogsite_rest"

    # Send POST request
    response = requests.post(url, json=payload, headers=headers)

    status_code = None

    # Keep looping until status code is 200
    while status_code != 200:
        # Send POST request
        response = requests.post(url, json=payload, headers=headers)
        # Get status code from the response
        status_code = response.status_code
        # Check if status code is not 200
        if status_code != 200:
            if status_code == 202 or status_code == 400:
                consecutive_202 += 1
                if consecutive_202 >= 50:
                    print(f"Encountered 50 consecutive 202 status codes for {uniprot_name}. Moving on to the next one.")
                    break  # Break out of the loop to move to the next UniProt ID
            print("Waiting for status code 200. Current status code:", status_code)
            # Wait for 5 seconds before checking again
            time.sleep(5)
        else:
            consecutive_202 = 0  # Reset the counter if status code is 200

    if consecutive_202 >= 50:
        continue 

    # Once the status code is 200, parse and print the response
    response_json = response.json()

    new_url = response_json["location"]

    job_id = new_url.split("/")[-1]

    # Make the GET request
    status_code = None
    while status_code != 200:
          # Send POST request
          response_two = requests.get(new_url)         # Get status code from the response
          status_code = response_two.status_code
          # Check if status code is not 200
          if status_code != 200:
              print("Waiting for status code 200. Current status code:", status_code)
              # Wait for 5 seconds before checking again
              time.sleep(5)

    result_table = response_two.json()["result_table"]

    data = requests.get(result_table).text


    rows = data.strip().split('\n')
    headers = rows[0].split()

    data = {}
    for row in rows[1:]:
        values = row.split()
        row_dict = {}
        i = 0
        for header, value in zip(headers, values):
          if i > 3:
            row_dict[header] = value

          i += 1

        data[values[0]] = row_dict


    datasss[uniprot_name] = data
    
    # Save the data to a file after each iteration
    with open("DogSiteScorer_data_withALLPredicted.json", "a") as json_file:
        json.dump({uniprot_name: data}, json_file, indent=4)
        json_file.write("\n")


# Read the JSON file
with open('Outcome_data\DoGSiteScorer\DogSiteScorer_data_withALLPredicted.json', 'r') as file:
    json_data = json.load(file)

# Read the CSV file
csv_data = {}
with open('Outcome_data\All_predicted_proteins.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        protein_id, probability = row
        csv_data[protein_id] = probability

# Create a dictionary to store the final output
final_output = {}

# Iterate through the JSON data and store the corresponding probability from the CSV
for protein_id, values in json_data.items():
    max_val = float('-inf')
    for item_id, item_data in values.items():
        if 'VAL' in item_data:
            val = float(item_data['VAL'])
            if val > max_val:
                max_val = val
    # Check if the protein ID exists in the CSV data
    if protein_id in csv_data:
        probability = csv_data[protein_id]
        final_output[protein_id] = {"probability": probability, "max_VAL": max_val}

    # Save the final output to a file after each iteration
    with open('output.json', 'a') as outfile:
        json.dump({protein_id: final_output[protein_id]}, outfile, indent=4)
        outfile.write("\n")
