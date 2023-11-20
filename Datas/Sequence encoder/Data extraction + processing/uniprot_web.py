import requests
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def extract_first_column(csv_file_path):
    column_data = []

    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row:  # Check if the row is not empty
                column_data.append(row[0])

    return column_data

# Replace 'your_file.csv' with the actual path to your CSV file
csv_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Sequence encoder\uniprotkb_AND_reviewed_true_AND_model_o_2023_11_20.csv'

# Extract the first column and turn it into a list
gene_list = extract_first_column(csv_file_path)[1:]


def download_gene_sequence(gene):
    base_url = 'https://www.uniprot.org/uniprot/'
    url = f'{base_url}{gene}.fasta'
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error downloading sequence for {gene}: {e}"

def process_genes_in_batches(gene_list, batch_size=100):
    with ThreadPoolExecutor() as executor:
        gene_batches = [gene_list[i:i + batch_size] for i in range(0, len(gene_list), batch_size)]
        results = []

        for batch in tqdm(gene_batches, desc="Processing Genes", unit="batch"):
            batch_results = list(executor.map(download_gene_sequence, batch))
            results.extend(batch_results)

    return results

# Batch size for processing
batch_size = 10

# Download gene sequences in batches
results = process_genes_in_batches(gene_list, batch_size)

# Output filename for the sequences
output_filename = r'C:\Users\George\Desktop\ISEF-2023\Datas\Sequence encoder\gene_sequences.fasta'

# Filter out errors and save valid sequences to a file
valid_sequences = [result for result in results if not result.startswith("Error")]
with open(output_filename, 'w') as output_file:
    output_file.write('\n'.join(valid_sequences))

print(f"Gene sequences downloaded and saved to {output_filename}")

