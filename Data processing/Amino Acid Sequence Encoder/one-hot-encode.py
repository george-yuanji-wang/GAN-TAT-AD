import json
import numpy as np


path = r"/Users/michelle/jupyter/ISEF/Datas/Sequence encoder/back up data/sequence_homosapien_genome_copy.json"

with open(path, "r") as file:
    data = json.load(file)
    
# Assuming data is a list of dictionaries with 'header' and 'sequences' keys
sequences_list = [entry['sequence'] for entry in data]

concatenated_sequences = ''.join(sequences_list)

unique_characters = list(set([*concatenated_sequences]))

unique_characters.sort()

print(unique_characters)


def encode_amino_acid_sequence(sequence):
    all_amino_acids = "ACDEFGHIKLMNPQRSTUVWY"
    amino_acid_to_int = {acid: i for i, acid in enumerate(all_amino_acids)}
    integer_sequence = np.array([amino_acid_to_int[acid] for acid in sequence])

    # One-hot encode the sequence using NumPy
    one_hot_encoded_sequence = np.eye(len(all_amino_acids))[integer_sequence]

    return one_hot_encoded_sequence

encoded_sequences = []

for seq in sequences_list:
    e = encode_amino_acid_sequence(seq)
    encoded_sequences.append(e)

# Convert the list to a NumPy array
encoded_sequences = np.array(encoded_sequences, dtype=object)

print("Memory size of encoded_sequences array: {} bytes".format(encoded_sequences.nbytes))

np.save(r'/Users/michelle/jupyter/ISEF/Data processing/Amino Acid Sequence Encoder/encoded_sequences.py', encoded_sequences)


