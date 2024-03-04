import pandas as pd
import json

df = pd.read_csv(r'C:\Users\Srisharan\OneDrive\Desktop\ISEF\ISEF-2023\Outcome_data\All_predicted_proteins.csv')
df1 = pd.read_csv(r'C:\Users\Srisharan\OneDrive\Desktop\ISEF\ISEF-2023\Datas\labels\alzheimer_disease\alzheimer_disease.txt')
df2 = pd.read_csv(r'C:\Users\Srisharan\OneDrive\Desktop\idmapping_2024_03_04.tsv', sep='\t')

list_of_known = []
ac = df1.to_numpy()
print(list_of_known)
print(df2)

for i in ac:
    list_of_known.append(i[0])
    


map_dict = {}
top_gene_use = 220
unknown = 0

for index, row in df2.iterrows():
    gene_name = (row['Gene Names']).split(" ")
    gene_name = gene_name[0]

    if gene_name in map_dict.keys():
        if row['From'] in list_of_known:
            map_dict[gene_name]['all_predicts'] += 1
            map_dict[gene_name]['known'] += 1
        else:
            unknown += 1
            map_dict[gene_name]['all_predicts'] += 1
            map_dict[gene_name]['predicted_unknown'] += 1
    
    else:
        map_dict[gene_name] = {'all_predicts':0, 'predicted_unknown':0, 'known':0, 'Protein':row['From'], 'Probability':df.iloc[index, 1]}

        if row['From'] in list_of_known:
            map_dict[gene_name]['all_predicts'] += 1
            map_dict[gene_name]['known'] += 1
        else:
            unknown += 1
            map_dict[gene_name]['all_predicts'] += 1
            map_dict[gene_name]['predicted_unknown'] += 1
        

    if index >= top_gene_use:
        break


print(map_dict)
print(unknown)

with open(r'C:\Users\Srisharan\OneDrive\Desktop\ISEF\ISEF-2023\Outcome_data\gsea_gene.json', 'w') as json_file:
    json.dump(map_dict, json_file, indent=4)