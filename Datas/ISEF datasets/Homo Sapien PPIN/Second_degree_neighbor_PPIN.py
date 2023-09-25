from tqdm import tqdm

# Path to the interacted proteins CSV file
interacted_proteins_file = r"C:\Users\George\Desktop\ISEF datasets\Homo Sapien PPIN\PPIN_UniProt.csv"
disease_related_proteins_file = r"C:\Users\George\Desktop\ISEF datasets\Pancreate cancer database\prognostic_protein_UniProt.csv"
unmatched_proteins_file = r"C:\Users\George\Desktop\ISEF datasets\Homo Sapien PPIN\Left_unmatched_proteins.csv"

# Read the disease-related proteins CSV file and store them in a list
disease_related_proteins = []

with open(disease_related_proteins_file, 'r') as disease_file:
    for line in disease_file:
        protein = (line.split(','))[0]
        disease_related_proteins.append(protein)

# Create a dictionary to store interacted proteins and their neighbors
interacted_proteins_dict = {}

# Read the interacted proteins CSV file and build the dictionary
with open(interacted_proteins_file, 'r') as interacted_file:
    for line in interacted_file:
        *_, protein1, protein2 = line.strip().split(',')
        
        if protein1 not in interacted_proteins_dict:
            interacted_proteins_dict[protein1] = set()
        if protein2 not in interacted_proteins_dict:
            interacted_proteins_dict[protein2] = set()
        
        interacted_proteins_dict[protein1].add(protein2)
        interacted_proteins_dict[protein2].add(protein1)

# Create a dictionary to store neighbors for disease-related proteins
all_neighbors = list(disease_related_proteins.copy())
not_found_proteins = []

# Function to find neighbors for a given protein
def find_second_neighbors(protein):
    global all_neighbors
    global not_found_proteins

    if protein in interacted_proteins_dict.keys():
        temp = list(interacted_proteins_dict[protein])

        '''for proteins in temp:
            
            temp = temp + list(interacted_proteins_dict[proteins])
            temp = set(temp)
            temp = list(temp)'''
        
        all_neighbors += temp
        all_neighbors = set(all_neighbors)
        all_neighbors = list(all_neighbors)

    else:

        not_found_proteins.append(protein)


# Iterate through disease-related proteins and find their second-degree neighbors
for protein in tqdm(disease_related_proteins[1:], desc="Expanding Neighbors"):
    
    find_second_neighbors(protein)

# Write the results to a CSV file
with open('first_degree_neighbor_proteins.csv', 'w') as output_file:
    output_file.write("Expanded pancreatic proteins\n")
    for neighbor in set(all_neighbors):
        output_file.write(f"{neighbor}\n")

# Write the left proteins to a CSV file
with open('fLeft_unmatched_proteins.csv', 'w') as output_file:
    output_file.write("Left over proteins\n")
    for left in set(not_found_proteins):
        output_file.write(f"{left}\n")

print("Neighbors expansion complete.")
