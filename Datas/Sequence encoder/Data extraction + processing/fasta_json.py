import re
import json

def fasta_to_json(fasta_file_path):
    sequences = []
    current_sequence = None

    with open(fasta_file_path, 'r') as fasta_file:
        for line in fasta_file:
            if line.startswith('>'):
                # Header line
                if current_sequence:
                    sequences.append(current_sequence)
                current_sequence = {'header': line[1:].strip(), 'sequence': ''}
            else:
                # Sequence line
                current_sequence['sequence'] += line.strip()

    # Add the last sequence
    if current_sequence:
        sequences.append(current_sequence)

    return sequences

# Replace 'your_fasta_file.fasta' with the actual file path
fasta_file_path = r'C:\Users\George\Desktop\ISEF-2023\Datas\Sequence encoder\gene_sequences.fasta'

# Convert FASTA to JSON
json_data = fasta_to_json(fasta_file_path)

# Save to JSON file
with open(r'C:\Users\George\Desktop\ISEF-2023\Datas\Sequence encoder\sequence_homosapien_genome.json', 'w') as json_file:
    json.dump(json_data, json_file, indent=2)

print("Data saved to output.json")