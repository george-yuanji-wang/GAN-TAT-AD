import pandas as pd
import json

drugnome = r'C:\Users\George\Desktop\ISEF-2023\Outcome_data\DrugnomeAI\DrugnomeAI-generic_model-percentile_scores.csv'
gene_protein = r'C:\Users\George\Desktop\ISEF-2023\Datas\HUMAN_GENOME_PROTEIN_AMINO_ACID_SEQUENCE.csv'
probability = r'C:\Users\George\Desktop\ISEF-2023\Outcome_data\All_predicted_proteins.csv'

drugnome_df = pd.read_csv(drugnome)
gene_protein_df = pd.read_csv(gene_protein)
probability_df = pd.read_csv(probability)

# Display the first few rows of the dataframes to verify

gene_protein_dict = pd.Series(gene_protein_df.iloc[:, 1].values,index=gene_protein_df.iloc[:, 0]).to_dict()
protein_prob_dict = pd.Series(probability_df.iloc[:, 1].values,index=probability_df.iloc[:, 0]).to_dict()
drugnome_dict = pd.Series(drugnome_df.iloc[:, 2].values, index=drugnome_df.iloc[:, 0]).to_dict()

final_dict = {}
for protein, probability in probability_df.values:
    gene = None
    # Find the gene corresponding to the protein
    for key, value in gene_protein_dict.items():
        
        if key == protein:
            gene = value[:-6]
  
            break
    # If the gene is found and it has a drugnome score, add it to the final dictionary
    if gene and gene in drugnome_dict:
        final_dict[protein] = {'probability': probability, 'drugnome_score': drugnome_dict[gene]}
# Display the first few items of the dictionary to verify

final_dict = dict(sorted(final_dict.items(), key=lambda item: item[1]['probability'], reverse=True))

with open(r'C:\Users\George\Desktop\ISEF-2023\Outcome_data\DrugnomeAI\Drugnome_Score.json', 'w') as json_file:
    json.dump(final_dict, json_file, indent=4)







